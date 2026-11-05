"""Check whether the original four-feature inputs can be reused for sector ranking."""
import argparse
import json
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path
import sys

import numpy as np
import pandas as pd

STEP = Path(__file__).resolve().parent
ROOT = STEP.parents[2]
sys.path.insert(0, str(ROOT))

from calendar_utils import TICKERS, schedule
from compare_predictions import MODEL_NAMES, align_predictions
from models.common import FEATURES, snapshot_id
from prepare_data import DATE_COLUMNS, digest, unpack
from price_basis import BASIS, audit
from study_config import TEST_YEARS, TRAIN_YEARS, VALIDATION_YEARS, split_and_purge

SPLITS = {'train': (TRAIN_YEARS, 'original_training'),
          'validation': (VALIDATION_YEARS, 'development'),
          'test': (TEST_YEARS, 'exploratory')}
SAMPLES = ROOT / 'prepared_data/weekly_candidates.csv'
PREPARATION = ROOT / 'prepared_data/preparation.json'
DAILY = ROOT / 'data/daily_prices.csv'


def require(condition, message):
    if not condition:
        raise ValueError(message)


def check_samples(candidates):
    require(not candidates.duplicated(['SignalDate', 'Ticker']).any(), 'Duplicate ETF-week')
    require(candidates.groupby('SignalDate').Ticker.agg(lambda tickers: set(tickers) == set(TICKERS)).all(),
            'Incomplete candidate week')
    frames = unpack(candidates)
    included = pd.concat(frames, ignore_index=True)
    eligible = candidates.loc[candidates.FeaturesComplete & candidates.HoldingPeriodComplete].copy()
    complete = eligible.groupby('SignalDate').Ticker.transform('size').eq(len(TICKERS))
    reassigned, _ = split_and_purge(eligible.loc[complete])
    expected = reassigned.Split.reindex(candidates.index, fill_value='excluded')
    pd.testing.assert_series_equal(candidates.Split, expected)
    daily = pd.read_csv(DAILY, parse_dates=['Date']).set_index(['Date', 'Ticker']).sort_index()
    dates = daily.index.get_level_values('Date')
    calendar = schedule(dates.min(), dates.max())
    schedule_cols = list(calendar.columns)
    observed = candidates[schedule_cols].drop_duplicates().sort_values('SignalDate').reset_index(drop=True)
    pd.testing.assert_frame_equal(observed, calendar, check_dtype=False)
    require(included.FeatureMaxDate.eq(included.SignalDate).all(), 'Unexpected feature cutoff')
    require((included.SignalCloseUTC.lt(included.EntryOpenUTC) &
             included.EntryOpenUTC.lt(included.ExitOpenUTC)).all(), 'Invalid trade timestamps')
    require(included.TargetBasis.eq(BASIS).all(), 'Unexpected price basis')

    def at_date(column, date_field):
        return daily[column].reindex(pd.MultiIndex.from_frame(included[[date_field, 'Ticker']])).to_numpy()

    close = daily.Close.unstack('Ticker')
    volume = daily.Volume.unstack('Ticker')
    expected_features = {
        'return_5': close / close.shift(5) - 1,
        'volatility_5': close.pct_change(fill_method=None).rolling(5).std(ddof=1),
        'price_ma_5': close / close.rolling(5).mean(),
        'activity_5': volume / volume.rolling(5).mean(),
    }
    keys = pd.MultiIndex.from_frame(included[['SignalDate', 'Ticker']])
    for feature, values in expected_features.items():
        np.testing.assert_allclose(included[feature], values.stack().reindex(keys),
                                   rtol=1e-10, atol=1e-12, err_msg=feature)
    np.testing.assert_allclose(included.TargetPriceReturn,
                               at_date('Open', 'ExitDate') / at_date('Open', 'EntryDate') - 1,
                               rtol=1e-10, atol=1e-12, err_msg='TargetPriceReturn')

    windows = []
    for frame, (name, (years, use)) in zip(frames, SPLITS.items()):
        require(frame.SignalDate.dt.year.isin(years).all() and frame.ExitDate.dt.year.isin(years).all(),
                f'{name}: signal or label outside the stage')
        windows.append(dict(split=name, use=use, weeks=frame.SignalDate.nunique(), rows=len(frame),
                            first_signal=frame.SignalCloseUTC.min().isoformat(),
                            last_signal=frame.SignalCloseUTC.max().isoformat(),
                            last_label_exit=frame.ExitOpenUTC.max().isoformat()))
    return included, windows


def run(output):
    preparation = json.loads(PREPARATION.read_text())
    source_hashes = {str(Path(p).relative_to(ROOT)): h
                     for p, h in audit(ROOT, write=False)['input_hashes'].items()}
    require(preparation['input_hashes'] == source_hashes, 'Daily data snapshot changed')
    for path, expected in preparation['code_hashes'].items():
        require(digest(ROOT / path) == expected, f'Sample preparation code changed: {path}')
    sample_hash = digest(SAMPLES)
    require(preparation['samples_sha256'] == sample_hash, 'Weekly sample fingerprint changed')

    candidates = pd.read_csv(SAMPLES, parse_dates=DATE_COLUMNS)
    reference, windows = check_samples(candidates)
    source_id = snapshot_id(source_hashes)
    require(reference.PriceBasisAuditId.eq(source_id).all(), 'Mixed source snapshots in samples')
    parameters, availability = {}, {}
    inputs = {**source_hashes, **preparation['code_hashes'],
              str(SAMPLES.relative_to(ROOT)): sample_hash,
              str(PREPARATION.relative_to(ROOT)): digest(PREPARATION)}
    for name in MODEL_NAMES:
        folder = ROOT / 'results' / name
        manifest_path = folder / 'debug_run.json'
        predictions_path = folder / 'predictions.csv'
        manifest = json.loads(manifest_path.read_text())
        require(manifest['input_hashes'] == source_hashes and manifest['samples_sha256'] == sample_hash,
                f'{name}: model results are stale')
        predictions = pd.read_csv(predictions_path, parse_dates=DATE_COLUMNS)
        aligned = align_predictions(predictions, reference, name, source_id)
        availability[name] = pd.to_datetime(aligned.loc[aligned.Split.eq('test'), 'ModelAvailableAsOf'],
                                            utc=True).max().isoformat()
        parameters[name] = dict(selected_on_2024=manifest['selected'],
                                original_seed=None if name == 'ridge' else 42)
        inputs.update({str(p.relative_to(ROOT)): digest(p) for p in (manifest_path, predictions_path)})

    excluded = candidates.loc[~candidates.Split.isin(SPLITS)]
    excluded_weeks = (excluded.groupby(['SignalDate', 'EntryDate', 'ExitDate', 'Split'])
                      .agg(etfs=('Ticker', 'size'), features_complete=('FeaturesComplete', 'all'),
                           holding_complete=('HoldingPeriodComplete', 'all')).reset_index())
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    script_sha256=digest(Path(__file__)),
                    input_sha256=inputs, source_snapshot_id=source_id, feature_columns=FEATURES,
                    target=BASIS, time_windows=windows, etfs=TICKERS,
                    parameters={'fit': False, 'seed': None, 'rtol': 1e-10, 'atol': 1e-12},
                    reused_models=parameters, original_model_available_as_of=availability,
                    inherited_future_perturbation_checks=preparation['future_perturbation_checks'],
                    runtime={'python': sys.version.split()[0], **{p: version(p) for p in
                             ('pandas', 'numpy', 'exchange_calendars', 'scikit-learn')}})
    output.mkdir()
    pd.DataFrame(windows).to_csv(output / 'split_windows.csv', index=False)
    excluded_weeks.to_csv(output / 'excluded_weeks.csv', index=False)
    (output / 'audit.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'Verified {len(reference)} retained ETF-week rows and {len(excluded_weeks)} excluded weeks.')
    print(f'Aligned {len(MODEL_NAMES)} saved prediction files. Audit saved to {output}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='run_01', help='New output folder name under this step.')
    args = parser.parse_args()
    output = (STEP / args.run_name).resolve()
    if output.parent != STEP or output.exists():
        parser.error('Choose a new output folder directly under the audit step.')
    run(output)
