"""Compare two XGBoost objectives for weekly ETF ranking on the 2024 development set."""
import argparse
from datetime import datetime, timezone
from itertools import product
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xgboost as xgb

STEP = Path(__file__).resolve().parent
ROOT = STEP.parents[2]
sys.path.insert(0, str(ROOT))

from calendar_utils import TICKERS
from compare_predictions import align_predictions
from models.common import FEATURES, snapshot_id
from models.xgboost import DEPTHS, TREE_COUNTS
from prepare_data import DATE_COLUMNS, digest
from run_feature_ablation import read_inputs

TOP_K = 3
RETURN = 'TargetPriceReturn'
HISTORICAL = ROOT / 'results/xgboost'
COMMON_PARAMETERS = ('learning_rate', 'min_child_weight', 'reg_lambda', 'subsample',
                     'colsample_bytree', 'tree_method', 'random_state', 'n_jobs')
RANKER_PARAMETERS = dict(objective='rank:pairwise', lambdarank_pair_method='topk',
                         lambdarank_num_pair_per_sample=len(TICKERS))
SCORE_COLUMNS = {'return_5_baseline': 'Return5Score', 'xgb_regression': 'RegressionScore',
                 'xgb_ranker': 'RankingScore', 'historical_rmse_reference': 'HistoricalRMSEScore'}


def order_and_label(frame):
    frame = frame.sort_values(['SignalDate', 'Ticker']).reset_index(drop=True)
    if not (frame.FeatureMaxDate.le(frame.SignalDate) & frame.SignalCloseUTC.lt(frame.EntryOpenUTC)
            & frame.EntryOpenUTC.lt(frame.ExitOpenUTC) & frame.HoldingPeriodComplete).all():
        raise ValueError('Features or trading timestamps violate the original weekly schedule')
    frame['Relevance'] = frame.groupby('SignalDate')[RETURN].rank(method='dense').astype(int) - 1
    groups = frame.groupby('SignalDate', sort=False)
    sizes = groups.size().to_numpy()
    if not np.all(sizes == len(TICKERS)):
        raise ValueError('Every ranking group must have 11 contiguous ETF rows')
    for _, week in groups:
        unique_returns = np.unique(week[RETURN])
        np.testing.assert_array_equal(week.Relevance, np.searchsorted(unique_returns, week[RETURN]))
    return frame, sizes


def weekly_metrics(frame, score):
    if len(score) != len(frame) or not np.isfinite(score).all():
        raise ValueError('Scores must be finite and aligned with the ETF-week rows')
    rows = []
    for date, week in frame.assign(Score=score).groupby('SignalDate', sort=True):
        score_constant = week.Score.nunique() == 1
        actual_constant = week[RETURN].nunique() == 1
        reasons = [reason for flag, reason in [(score_constant, 'constant_score'),
                                              (actual_constant, 'constant_actual')] if flag]
        ic = (np.nan if reasons else
              week.Score.rank(method='average').corr(week[RETURN].rank(method='average')))
        predicted = week.sort_values(['Score', 'Ticker'], ascending=[False, True]).Ticker.iloc[:TOP_K]
        actual = week.sort_values([RETURN, 'Ticker'], ascending=[False, True]).Ticker.iloc[:TOP_K]
        hits = len(set(predicted) & set(actual))
        rows.append(dict(SignalDate=date, rank_ic=ic, rank_ic_undefined_reason='+'.join(reasons),
                         constant_score=score_constant, constant_actual=actual_constant,
                         predicted_top3=' '.join(predicted), actual_top3=' '.join(actual),
                         top3_hits=hits, top3_overlap=hits / TOP_K))
    return pd.DataFrame(rows)


def summarize(weekly):
    return dict(mean_weekly_rank_ic=weekly.rank_ic.mean(),
                mean_top3_overlap=weekly.top3_overlap.mean(), weeks=len(weekly),
                valid_rank_ic_weeks=int(weekly.rank_ic.notna().sum()),
                constant_score_weeks=int(weekly.constant_score.sum()),
                constant_actual_weeks=int(weekly.constant_actual.sum()))


def search(train, development, sizes, common, ranker, features=FEATURES):
    model_type, objective_params = ((xgb.XGBRanker, RANKER_PARAMETERS) if ranker else
                                    (xgb.XGBRegressor, {'objective': 'reg:squarederror'}))
    label = train.Relevance if ranker else train[RETURN]
    group_arg = {'group': sizes} if ranker else {}
    grid, predictions = [], []
    for depth, trees in product(DEPTHS, TREE_COUNTS):
        model = model_type(**common, **objective_params, max_depth=depth, n_estimators=trees)
        model.fit(train[features], label, **group_arg)
        booster = model.get_booster()
        trained_objective = json.loads(booster.save_config())['learner']['objective']['name']
        if booster.feature_names != features or trained_objective != objective_params['objective']:
            raise ValueError('Unexpected features or trained XGBoost objective')
        score = model.predict(development[features])
        predictions.append(score)
        grid.append(dict(max_depth=depth, n_estimators=trees,
                         **summarize(weekly_metrics(development, score))))
    grid = pd.DataFrame(grid)
    defined = grid.dropna(subset=['mean_weekly_rank_ic'])
    if defined.empty:
        raise ValueError(f'No configuration has a defined development Rank IC for {model_type.__name__}')
    best = defined.sort_values(['mean_weekly_rank_ic', 'max_depth', 'n_estimators'],
                               ascending=[False, True, True]).index[0]
    grid['selected'] = grid.index == best
    selected = {key: int(grid.loc[best, key]) for key in ('max_depth', 'n_estimators')}
    return predictions[best], grid, {**common, **objective_params, **selected}


def run(output):
    (train, development, _), hashes = read_inputs()
    train, sizes = order_and_label(train)
    development, _ = order_and_label(development)
    if (not train.SignalDate.dt.year.isin([2021, 2022, 2023]).all()
            or not train.ExitDate.dt.year.isin([2021, 2022, 2023]).all()
            or not development.SignalDate.dt.year.eq(2024).all()
            or not development.ExitDate.dt.year.eq(2024).all()
            or train.ExitOpenUTC.max() >= development.SignalCloseUTC.min()):
        raise ValueError('Training and development signals or realized labels cross their time windows')

    paths = [ROOT / 'prepared_data' / name for name in ('weekly_candidates.csv', 'preparation.json')]
    prediction_path, debug_path, parameter_path = paths_from_history = [HISTORICAL / name for name in
        ('predictions.csv', 'debug_run.json', 'xgboost_parameters.json')]
    debug = json.loads(debug_path.read_text())
    if debug['input_hashes'] != hashes or debug['samples_sha256'] != digest(paths[0]):
        raise ValueError('Historical XGBoost results have stale data or sample fingerprints')
    historical_params = json.loads(parameter_path.read_text())
    if historical_params['features'] != FEATURES or historical_params['selected'] != debug['selected']:
        raise ValueError('Historical parameter file disagrees with its feature or selection records')
    old = pd.read_csv(prediction_path, parse_dates=DATE_COLUMNS)
    old = align_predictions(old.loc[old.Split.eq('validation')], development, 'xgboost', snapshot_id(hashes))
    common = {key: historical_params['parameters'][key] for key in COMMON_PARAMETERS}

    scores = {'return_5_baseline': development.return_5.to_numpy()}
    grids, selected = [], {}
    for method, ranker in [('xgb_regression', False), ('xgb_ranker', True)]:
        scores[method], grid, selected[method] = search(train, development, sizes, common, ranker)
        grids.append(grid.assign(method=method))
    scores['historical_rmse_reference'] = old.Prediction.to_numpy()

    per_etf = development[['SignalDate', 'EntryDate', 'ExitDate', 'SignalCloseUTC', 'EntryOpenUTC',
                           'ExitOpenUTC', 'FeatureMaxDate', 'Ticker', *FEATURES, RETURN, 'Relevance']].copy()
    weekly, summary, by_method = [], [], {}
    for method, score in scores.items():
        per_etf[SCORE_COLUMNS[method]] = score.astype(float)
        per_week = weekly_metrics(development, score)
        by_method[method] = per_week.set_index('SignalDate')
        weekly.append(per_week.assign(method=method))
        summary.append(dict(method=method, period='2024_development', **summarize(per_week)))

    deltas = []
    ranker_weekly = by_method['xgb_ranker']
    for method in ('return_5_baseline', 'xgb_regression', 'historical_rmse_reference'):
        reference = by_method[method]
        paired_ic = (ranker_weekly.rank_ic - reference.rank_ic).dropna()
        deltas.append(dict(reference=method, common_valid_weeks=len(paired_ic),
                           mean_rank_ic_delta=ranker_weekly.rank_ic.mean() - reference.rank_ic.mean(),
                           mean_paired_rank_ic_delta=paired_ic.mean(),
                           mean_top3_overlap_delta=(ranker_weekly.top3_overlap - reference.top3_overlap).mean()))

    paths.extend([*paths_from_history, STEP / 'DESIGN.md', Path(__file__),
                  ROOT / 'models/xgboost.py', ROOT / 'compare_predictions.py', ROOT / 'run_feature_ablation.py'])
    config = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='2024_development_selected_and_evaluated_on_same_window',
                  xgboost_version=xgb.__version__, features=FEATURES, target_basis=development.TargetBasis.iloc[0],
                  ticker_universe=TICKERS, source_snapshot_id=snapshot_id(hashes),
                  input_sha256={**hashes, **{str(p.relative_to(ROOT)): digest(p) for p in paths}},
                  windows={name: dict(rows=len(frame), weeks=frame.SignalDate.nunique(),
                                      first_signal=frame.SignalCloseUTC.min().isoformat(),
                                      last_signal=frame.SignalCloseUTC.max().isoformat(),
                                      last_exit=frame.ExitOpenUTC.max().isoformat())
                           for name, frame in [('fit_2021_2023', train), ('development_2024', development)]},
                  ranking_group='SignalDate', ranking_fit_argument='group', group_size=len(TICKERS),
                  fit_groups=len(sizes), label='zero-based dense rank within week, ascending actual price return',
                  top_k=TOP_K, top_k_tiebreak='Ticker ascending', ic_ties='average ranks',
                  selection='maximum mean valid weekly Rank IC; then lower depth, then fewer trees',
                  grid={'max_depth': DEPTHS, 'n_estimators': TREE_COUNTS},
                  seed=common['random_state'], selected=selected,
                  historical_rmse_reference=historical_params['selected'], scored_2025=False)
    output.mkdir()
    per_etf.to_csv(output / 'scores.csv', index=False)
    pd.concat(weekly, ignore_index=True).to_csv(output / 'weekly_metrics.csv', index=False)
    pd.DataFrame(summary).to_csv(output / 'summary.csv', index=False)
    pd.DataFrame(deltas).to_csv(output / 'deltas.csv', index=False)
    pd.concat(grids, ignore_index=True).to_csv(output / 'grid.csv', index=False)
    (output / 'experiment.json').write_text(json.dumps(config, indent=2) + '\n')
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='run_01', help='New output folder name under this step.')
    args = parser.parse_args()
    output = (STEP / args.run_name).resolve()
    if output.parent != STEP or output.exists():
        parser.error('Choose a new output folder directly under the comparison step.')
    run(output)
