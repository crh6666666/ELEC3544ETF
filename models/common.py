"""Shared model evaluation and offline, isolated debugging."""
import argparse
import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

FEATURES = ['return_5', 'volatility_5', 'price_ma_5', 'activity_5']
ROOT = Path(__file__).resolve().parent.parent


@dataclass
class FitResult:
    model: object
    grid: list
    selected: dict


def snapshot_id(hashes):
    return hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()


def output_metadata(train, test, hashes):
    return dict(price_basis_audit_id=snapshot_id(hashes),
        experiment_status=f'exploratory_{pd.Timestamp(train.SignalDate.min()).year}_{pd.Timestamp(test.SignalDate.max()).year}')


def evaluate_models(models, train, validation, test, features, hashes):
    """Evaluate already fitted models without fitting on validation or test."""
    from calendar_utils import TICKERS

    predictions, metrics, weekly = [], [], []
    snapshot = snapshot_id(hashes)
    for split, frame in [('train', train), ('validation', validation), ('test', test)]:
        assert not frame.duplicated(['SignalDate', 'Ticker']).any()
        assert frame.groupby('SignalDate').Ticker.nunique().eq(len(TICKERS)).all()
        for name, model in models.items():
            values = model.predict(frame[features])
            a = frame.copy()
            a['Prediction'] = values
            a['ModelId'] = name
            a['FitAsOf'] = str(train.ExitOpenUTC.max())
            a['TrainingLabelsThrough'] = str(train.ExitOpenUTC.max())
            a['SelectionLabelsThrough'] = str(validation.ExitOpenUTC.max())
            a['ModelAvailableAsOf'] = str(max(train.ExitOpenUTC.max(), validation.ExitOpenUTC.max()))
            a['GeneratedAtUTC'] = datetime.now(timezone.utc).isoformat()
            a['PredictionUse'] = ('retrospective_test' if split == 'test' else
                                  'parameter_selection' if split == 'validation' else 'in_sample_fit')
            a['FeatureVersion'] = 'four_features_v1'
            a['SourceSnapshotId'] = snapshot
            predictions.append(a)
            correlations = []
            for date, group in a.groupby('SignalDate'):
                corr = (group.Prediction.rank().corr(group.TargetPriceReturn.rank())
                        if group.Prediction.nunique() > 1 and group.TargetPriceReturn.nunique() > 1
                        else np.nan)
                correlations.append(corr)
                weekly.append(dict(ModelId=name, Split=split, SignalDate=date, rank_ic=corr))
            valid = [v for v in correlations if np.isfinite(v)]
            metrics.append(dict(ModelId=name, Split=split, rows=len(a),
                weeks=a.SignalDate.nunique(), mae=mean_absolute_error(a.TargetPriceReturn, values),
                rmse=np.sqrt(mean_squared_error(a.TargetPriceReturn, values)),
                direction_accuracy=float((np.sign(values) == np.sign(a.TargetPriceReturn)).mean()),
                always_up_accuracy=float((a.TargetPriceReturn > 0).mean()),
                mean_weekly_rank_ic=float(np.mean(valid)) if valid else np.nan,
                defined_rank_ic_weeks=len(valid)))
    # Keep each model's prediction dtype until its own CSV has been written.
    return predictions, pd.DataFrame(metrics), pd.DataFrame(weekly)


def load_debug_samples():
    from prepare_data import load_samples
    frames, hashes = load_samples()
    return (*frames, hashes)


def run_debug(name, fit_model, save_outputs):
    parser = argparse.ArgumentParser(description=f'Offline {name} training on existing weekly samples.')
    parser.add_argument('--output', type=Path, default=ROOT/'results'/name,
                        help='Separate output directory; existing debug outputs will be replaced.')
    args = parser.parse_args()
    out = args.output.resolve()
    protected = [ROOT/p for p in ('data', 'prepared_data', 'models', '.venv')]
    if out == ROOT or any(out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        parser.error('Choose a separate debug output directory outside source data and existing results.')
    train, validation, test, hashes = load_debug_samples()
    result = fit_model(train, validation, FEATURES)
    predictions, metrics, weekly = evaluate_models(
        {name: result.model}, train, validation, test, FEATURES, hashes)
    out.mkdir(parents=True, exist_ok=True)
    save_outputs(result, FEATURES, out, output_metadata(train, test, hashes))
    pd.concat(predictions).to_csv(out/'predictions.csv', index=False)
    metrics.to_csv(out/'metrics.csv', index=False)
    weekly.to_csv(out/'weekly_rank_ic.csv', index=False)
    from study_config import SPLIT_DESIGN
    manifest = dict(model=name, selected=result.selected, split_design=SPLIT_DESIGN,
        input_hashes=hashes,
        samples_sha256=hashlib.sha256((ROOT/'prepared_data/weekly_candidates.csv').read_bytes()).hexdigest(),
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        note='Debug run on existing samples; price-only retrospective study, not independent data certification.')
    (out/'debug_run.json').write_text(json.dumps(manifest, indent=2))
    print(f'{name}: selected {result.selected}')
    print(f'Output: {out}')
    print(metrics.to_string(index=False))
