"""Compare original and 20-session features on identical ETF ranking weeks."""
import argparse
from datetime import datetime, timezone
from importlib import import_module
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

STEP = Path(__file__).resolve().parent
ROOT = STEP.parents[2]
sys.path.insert(0, str(ROOT))

from calendar_utils import CAL, TICKERS
from models.common import FEATURES, snapshot_id
from prepare_data import digest
from study_config import END_EXCLUSIVE, START, TRAIN_YEARS, VALIDATION_YEARS

ranking = import_module('experiments.sector_ranking.01_objective_comparison.run')
WINDOW = 20
NEW_FEATURES = ['relative_spy_return_20', 'volatility_20', 'relative_volume_20']
EXTENDED = FEATURES + NEW_FEATURES
GROUPS = {'A': (False, FEATURES), 'B': (False, EXTENDED),
          'C': (True, FEATURES), 'D': (True, EXTENDED)}
COMPARISONS = [('training_objective_original_features', 'A', 'C'),
               ('training_objective_extended_features', 'B', 'D'),
               ('feature_increment_regression', 'A', 'B'),
               ('feature_increment_ranker', 'C', 'D')]
KEYS = ['SignalDate', 'Ticker']


def feature_panels(close, volume):
    etf = close[TICKERS]
    period_return = close.div(close.shift(WINDOW)).sub(1)
    period_return = period_return.where(close.rolling(WINDOW + 1).count().eq(WINDOW + 1))
    relative = period_return[TICKERS].sub(period_return.SPY, axis='index')
    volatility = etf.pct_change(fill_method=None).rolling(WINDOW).std(ddof=1)
    average_volume = volume.rolling(WINDOW).mean()
    activity = volume.div(average_volume.where(average_volume.gt(0)))
    return dict(zip(NEW_FEATURES, (relative, volatility, activity)))


def aligned_bars():
    prices = pd.concat([pd.read_csv(ROOT / 'data' / name, parse_dates=['Date'])
                        for name in ('daily_prices.csv', 'spy_daily_prices.csv')], ignore_index=True)
    sessions = CAL.sessions_in_range(START, pd.Timestamp(END_EXCLUSIVE) - pd.Timedelta(days=1))
    sessions = sessions.rename('SignalDate')
    close = prices.pivot(index='Date', columns='Ticker', values='Close').reindex(sessions)
    volume = prices.pivot(index='Date', columns='Ticker', values='Volume').reindex(sessions)[TICKERS]
    return close, volume


def build_features(base, close, volume):
    panels = feature_panels(close, volume)
    dates = pd.DatetimeIndex(base.SignalDate.unique()).sort_values()
    if not dates.isin(close.index).all():
        raise ValueError('A signal date is outside the market calendar')
    for date in dates:
        truncated = feature_panels(close.loc[:date], volume.loc[:date])
        altered_close = close.copy()
        altered_volume = volume.copy()
        future = close.index > date
        altered_close.loc[future, TICKERS] *= 1.7
        altered_close.loc[future, 'SPY'] *= 0.6
        altered_volume.loc[future] = altered_volume.loc[future] * 2 + 100
        altered = feature_panels(altered_close, altered_volume)
        for feature, panel in panels.items():
            np.testing.assert_array_equal(panel.loc[date], truncated[feature].loc[date],
                                          err_msg=f'{date}: {feature} used data beyond the signal')
            np.testing.assert_array_equal(panel.loc[date], altered[feature].loc[date],
                                          err_msg=f'{date}: future prices or volume affected {feature}')
    feature_rows = pd.concat({name: panel.loc[dates].stack() for name, panel in panels.items()}, axis=1)
    enriched = base.join(feature_rows, on=KEYS, validate='one_to_one')
    return enriched, len(dates)


def sample_coverage(enriched, sessions, samples_path):
    counts = (~np.isfinite(enriched.set_index(KEYS)[NEW_FEATURES])).groupby('SignalDate').sum()
    counts = counts.add_prefix('unavailable_etfs_')
    candidates = pd.read_csv(samples_path, parse_dates=['SignalDate', 'EntryDate', 'ExitDate'])
    candidates = candidates.loc[candidates.SignalDate.dt.year.isin((*TRAIN_YEARS, *VALIDATION_YEARS))]
    coverage = (candidates.groupby(['SignalDate', 'Split', 'EntryDate', 'ExitDate'])
                .agg(candidate_etfs=('Ticker', 'size'), original_features_complete=('FeaturesComplete', 'all'),
                     original_holding_complete=('HoldingPeriodComplete', 'all')).reset_index())
    coverage = coverage.join(counts, on='SignalDate', validate='one_to_one')
    coverage['available_sessions'] = sessions.searchsorted(coverage.SignalDate, side='right')
    original = coverage.Split.isin(['train', 'validation'])
    if not counts.index.equals(pd.DatetimeIndex(coverage.loc[original, 'SignalDate'])):
        raise ValueError('Common-sample coverage does not match the original fitting and development weeks')
    missing = coverage[counts.columns].gt(0).any(axis=1)
    coverage['kept'] = original & ~missing
    coverage['kept_etfs'] = np.where(coverage.kept, len(TICKERS), 0)
    coverage['reason'] = np.select(
        [coverage.kept, coverage.Split.eq('purged_unrealized_label'),
         ~coverage.original_features_complete, ~coverage.original_holding_complete,
         original & (coverage.available_sessions < WINDOW + 1), original & missing],
        ['kept', 'original_cross_stage_label', 'original_feature_warmup', 'original_incomplete_holding',
         '20_session_warmup', '20_session_missing_or_invalid_feature'], default='original_excluded')
    kept = enriched.loc[enriched.SignalDate.isin(coverage.loc[coverage.kept, 'SignalDate'])].copy()
    if (kept.duplicated(KEYS).any() or not np.isfinite(kept[EXTENDED]).all().all()
            or not kept.groupby('SignalDate').Ticker.agg(lambda tickers: set(tickers) == set(TICKERS)).all()):
        raise ValueError('Common sample has duplicates, unavailable features, or incomplete ETF groups')
    kept['NewFeatureMaxDate'] = kept.SignalDate
    return kept, coverage


def paired_comparisons(weekly):
    ic = weekly.pivot(index='SignalDate', columns='method', values='rank_ic')
    overlap = weekly.pivot(index='SignalDate', columns='method', values='top3_overlap')
    rows = []
    for question, reference, compared in COMPARISONS:
        paired = ic[compared] - ic[reference]
        rows.append(dict(question=question, reference=reference, compared=compared,
                         mean_rank_ic_delta=ic[compared].mean() - ic[reference].mean(),
                         common_valid_weeks=int(paired.notna().sum()), mean_paired_rank_ic_delta=paired.mean(),
                         mean_top3_overlap_delta=(overlap[compared] - overlap[reference]).mean()))
    return pd.DataFrame(rows)


def run(output):
    (original_train, original_dev, _), hashes = ranking.read_inputs()
    base = pd.concat([original_train, original_dev], ignore_index=True)
    close, volume = aligned_bars()
    enriched, checked_dates = build_features(base, close, volume)
    samples_path = ROOT / 'prepared_data/weekly_candidates.csv'
    kept, coverage = sample_coverage(enriched, close.index, samples_path)
    train, group_sizes = ranking.order_and_label(kept.loc[kept.Split.eq('train')])
    development, _ = ranking.order_and_label(kept.loc[kept.Split.eq('validation')])
    for frame, years in ((train, TRAIN_YEARS), (development, VALIDATION_YEARS)):
        if (frame.empty or not frame.SignalDate.dt.year.isin(years).all()
                or not frame.ExitDate.dt.year.isin(years).all()):
            raise ValueError('Empty partition or a signal or label crossed its stage')
    if train.ExitOpenUTC.max() >= development.SignalCloseUTC.min():
        raise ValueError('Training labels were not realized before the first development signal')

    parameter_path = ranking.HISTORICAL / 'xgboost_parameters.json'
    debug_path = ranking.HISTORICAL / 'debug_run.json'
    parameters = json.loads(parameter_path.read_text())
    debug = json.loads(debug_path.read_text())
    if (debug['input_hashes'] != hashes or debug['samples_sha256'] != digest(samples_path)
            or parameters['features'] != FEATURES or parameters['selected'] != debug['selected']):
        raise ValueError('Original XGBoost parameter provenance does not match the sample snapshot')
    common = {key: parameters['parameters'][key] for key in ranking.COMMON_PARAMETERS}
    selected, grids = {}, []
    scores = {'return_5_baseline': development.return_5.to_numpy()}
    for group, (is_ranker, features) in GROUPS.items():
        scores[group], grid, fit = ranking.search(train, development, group_sizes, common,
                                                  is_ranker, features=features)
        selected[group] = dict(features=features, parameters=fit)
        grids.append(grid.assign(method=group, objective=fit['objective'], n_features=len(features)))

    score_rows = development[[*KEYS, ranking.RETURN, 'Relevance']].copy()
    weekly_frames, summary_rows = [], []
    for method, score in scores.items():
        score_rows[f'{method}_score'] = score.astype(float)
        metrics = ranking.weekly_metrics(development, score)
        weekly_frames.append(metrics.assign(method=method))
        summary_rows.append(dict(method=method, period='2024_development', **ranking.summarize(metrics)))
    weekly = pd.concat(weekly_frames, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    deltas = paired_comparisons(weekly)
    windows = {split: dict(original_weeks=original.SignalDate.nunique(), common_weeks=frame.SignalDate.nunique(),
                           rows=len(frame), first_signal=frame.SignalCloseUTC.min().isoformat(),
                           last_signal=frame.SignalCloseUTC.max().isoformat(),
                           last_exit=frame.ExitOpenUTC.max().isoformat())
               for split, original, frame in [('fit_2021_2023', original_train, train),
                                               ('development_2024', original_dev, development)]}
    paths = [samples_path, ROOT / 'prepared_data/preparation.json', parameter_path, debug_path,
             STEP / 'DESIGN.md', Path(__file__), Path(ranking.__file__), ROOT / 'models/xgboost.py',
             ROOT / 'run_feature_ablation.py', ROOT / 'price_basis.py', ROOT / 'requirements.txt']
    config = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='2024_development_selected_and_evaluated_on_same_window', scored_2025=False,
                  xgboost_version=ranking.xgb.__version__, source_snapshot_id=snapshot_id(hashes),
                  input_sha256={**hashes, **{str(p.relative_to(ROOT)): digest(p) for p in paths}},
                  windows=windows, original_features=FEATURES, new_features=NEW_FEATURES,
                  ticker_universe=TICKERS, target_basis=development.TargetBasis.iloc[0],
                  feature_timing_checks=dict(signal_dates=checked_dates, cut_at_signal=True,
                                             perturbed_after_signal=True, window_sessions=WINDOW),
                  ranker_group='SignalDate', group_size=len(TICKERS), seed=common['random_state'],
                  selection='mean weekly Rank IC over valid weeks; lower depth, then fewer trees on ties',
                  grid={'max_depth': ranking.DEPTHS, 'n_estimators': ranking.TREE_COUNTS},
                  baseline='return_5 descending', selected=selected)
    sample_cols = ['Split', *KEYS, 'SignalCloseUTC', 'EntryDate', 'EntryOpenUTC', 'ExitDate', 'ExitOpenUTC',
                   'NewFeatureMaxDate', *EXTENDED, ranking.RETURN, 'Relevance']
    output.mkdir()
    pd.concat([train, development], ignore_index=True)[sample_cols].to_csv(output / 'feature_samples.csv', index=False)
    coverage.to_csv(output / 'sample_coverage.csv', index=False)
    score_rows.to_csv(output / 'scores.csv', index=False)
    weekly.to_csv(output / 'weekly_metrics.csv', index=False)
    summary.to_csv(output / 'summary.csv', index=False)
    deltas.to_csv(output / 'deltas.csv', index=False)
    pd.concat(grids, ignore_index=True).to_csv(output / 'grid.csv', index=False)
    (output / 'experiment.json').write_text(json.dumps(config, indent=2) + '\n')
    print(pd.DataFrame(windows).T.to_string())
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='run_01', help='New output folder name under this step.')
    args = parser.parse_args()
    output = (STEP / args.run_name).resolve()
    if output.parent != STEP or output.exists():
        parser.error('Choose a new output folder directly under the medium-term feature step.')
    run(output)
