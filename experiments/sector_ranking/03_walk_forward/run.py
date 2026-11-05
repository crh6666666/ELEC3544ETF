"""Train, select and score the four ranking groups in chronological windows."""
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

from calendar_utils import TICKERS
from models.common import snapshot_id
from plot_portfolio import plt
from prepare_data import DATE_COLUMNS, digest

medium = import_module('experiments.sector_ranking.02_medium_term_features.run')
ranking = medium.ranking
WINDOWS = {
    'W1': ('2021-01-01', '2023-01-01', '2023-07-01', '2024-01-01'),
    'W2': ('2021-01-01', '2023-07-01', '2024-01-01', '2024-07-01'),
    'W3': ('2021-01-01', '2024-01-01', '2024-07-01', '2025-01-01'),
}
PHASES = {'fit': 52, 'selection': 20, 'evaluation': 20}
BASELINE = 'return_5_baseline'
METHODS = (BASELINE, *medium.GROUPS)
EPS = 1e-12
INPUT = medium.STEP / 'run_01'


def load_features():
    _, hashes = ranking.read_inputs()
    manifest_path = INPUT / 'experiment.json'
    manifest = json.loads(manifest_path.read_text())
    for path, sha in manifest['input_sha256'].items():
        if not (ROOT / path).is_file() or digest(ROOT / path) != sha:
            raise ValueError(f'Step 2 input changed or is missing: {path}; inspect the snapshot and rerun Step 2')
    if manifest['source_snapshot_id'] != snapshot_id(hashes):
        raise ValueError('Step 2 and current data snapshots differ')
    samples_path = ROOT / 'prepared_data/weekly_candidates.csv'
    candidates = pd.read_csv(samples_path, parse_dates=DATE_COLUMNS)
    candidates = candidates[candidates.SignalDate.dt.year.lt(2025)].copy()
    features_path = INPUT / 'feature_samples.csv'
    coverage_path = INPUT / 'sample_coverage.csv'
    feature_rows = pd.read_csv(features_path, parse_dates=['SignalDate', 'NewFeatureMaxDate'])
    coverage = pd.read_csv(coverage_path, parse_dates=['SignalDate']).set_index('SignalDate')
    dates = coverage.index[coverage.kept]
    expected = candidates.loc[candidates.SignalDate.isin(dates)].sort_values(medium.KEYS).reset_index(drop=True)
    feature_rows = feature_rows.sort_values(medium.KEYS).reset_index(drop=True)
    pd.testing.assert_frame_equal(feature_rows[medium.KEYS + medium.FEATURES + [ranking.RETURN]],
                                  expected[medium.KEYS + medium.FEATURES + [ranking.RETURN]])
    if not feature_rows.NewFeatureMaxDate.eq(feature_rows.SignalDate).all():
        raise ValueError('Invalid Step 2 feature cutoff')
    close, volume = medium.aligned_bars()
    keys = pd.MultiIndex.from_frame(feature_rows[medium.KEYS])
    for name, panel in medium.feature_panels(close, volume).items():
        np.testing.assert_allclose(feature_rows[name], panel.loc[dates].stack().reindex(keys),
                                   rtol=1e-10, atol=1e-12, err_msg=f'Step 2 feature changed: {name}')
    base = candidates.merge(feature_rows[medium.KEYS + medium.NEW_FEATURES + ['NewFeatureMaxDate']],
                            on=medium.KEYS, how='left', validate='one_to_one')
    base['InputReason'] = base.SignalDate.map(coverage.reason)
    if base.InputReason.isna().any() or not base.groupby('SignalDate').Ticker.agg(
            lambda tickers: len(tickers) == len(TICKERS) and set(tickers) == set(TICKERS)).all():
        raise ValueError('Step 2 coverage or original ETF groups are incomplete')
    return base, hashes, [manifest_path, samples_path, features_path, coverage_path]


def partition(base):
    stages, audit_rows, summaries = {}, [], []
    for window, boundaries in WINDOWS.items():
        stage_frames = {}
        audit_frames = []
        for phase, start, end in zip(PHASES, boundaries, boundaries[1:]):
            nominal = base.loc[base.SignalDate.ge(start) & base.SignalDate.lt(end)].copy()
            eligible = nominal.InputReason.eq('kept')
            crosses = nominal.ExitDate.ge(end)
            reason = np.where(crosses, 'label_exits_outside_stage', nominal.InputReason)
            audit = nominal[['SignalDate', 'SignalCloseUTC', 'EntryDate', 'EntryOpenUTC',
                             'ExitDate', 'ExitOpenUTC', 'InputReason']].assign(
                                 window=window, phase=phase, nominal_start=start, nominal_end_exclusive=end,
                                 kept=eligible & ~crosses, reason=reason)
            audit_frames.append(audit)
            stage_frames[phase] = nominal.loc[audit.kept].copy()
        for (phase, minimum), audit in zip(PHASES.items(), audit_frames):
            frame = stage_frames[phase]
            weeks = frame.SignalDate.nunique()
            if weeks < minimum:
                reasons = audit.loc[~audit.kept].drop_duplicates('SignalDate').reason.value_counts().to_dict()
                raise ValueError(f'{window}/{phase} has {weeks} weeks (minimum {minimum}); exclusions: {reasons}')
            if not np.isfinite(frame[medium.EXTENDED + [ranking.RETURN]]).all().all():
                raise ValueError(f'{window}/{phase}: nonfinite inputs')
            week_audit = (audit.groupby(list(audit.columns), dropna=False).size()
                          .rename('etfs').reset_index())
            if not week_audit.etfs.eq(len(TICKERS)).all():
                raise ValueError('A week was split across stages or eligibility decisions')
            audit_rows.append(week_audit)
            summaries.append(dict(window=window, phase=phase, nominal_start=audit.nominal_start.iloc[0],
                                  nominal_end_exclusive=audit.nominal_end_exclusive.iloc[0],
                                  first_signal=frame.SignalCloseUTC.min(), last_signal=frame.SignalCloseUTC.max(),
                                  first_entry=frame.EntryOpenUTC.min(), last_exit=frame.ExitOpenUTC.max(),
                                  weeks=weeks, rows=len(frame),
                                  excluded_weeks=int((~week_audit.kept).sum()),
                                  boundary_weeks=int(week_audit.reason.eq('label_exits_outside_stage').sum())))
        fit, selection, evaluation = (stage_frames[phase] for phase in PHASES)
        if not (fit.ExitOpenUTC.max() < selection.SignalCloseUTC.min()
                and selection.ExitOpenUTC.max() < evaluation.SignalCloseUTC.min()):
            raise ValueError(f'{window}: a label was still unrealized at a later phase signal')
        stages[window] = stage_frames
    eval_keys = pd.concat([frames['evaluation'][medium.KEYS] for frames in stages.values()])
    if eval_keys.duplicated(medium.KEYS).any():
        raise ValueError('Evaluation windows reuse a signal week')
    return stages, pd.concat(audit_rows, ignore_index=True), pd.DataFrame(summaries)


def evaluate(stages, common):
    scores, weekly, grids, selected = [], [], [], {}
    for window, phases in stages.items():
        fit, sizes = ranking.order_and_label(phases['fit'])
        selection, _ = ranking.order_and_label(phases['selection'])
        evaluation, _ = ranking.order_and_label(phases['evaluation'])
        score_cols = [*DATE_COLUMNS, 'NewFeatureMaxDate', 'Ticker', ranking.RETURN, 'TargetBasis']
        rows = evaluation[score_cols].assign(Window=window, ScoreAsOfUTC=evaluation.SignalCloseUTC,
                                             FitLabelsThroughUTC=fit.ExitOpenUTC.max(),
                                             SelectionLabelsThroughUTC=selection.ExitOpenUTC.max())
        rows[f'{BASELINE}_score'] = evaluation.return_5.to_numpy()
        selected[window] = {}
        for method, (is_ranker, features) in medium.GROUPS.items():
            validation_scores, grid, params = ranking.search(fit, selection, sizes, common, is_ranker, features)
            model_type = ranking.xgb.XGBRanker if is_ranker else ranking.xgb.XGBRegressor
            group_arg = {'group': sizes} if is_ranker else {}
            target = fit.Relevance if is_ranker else fit[ranking.RETURN]
            model = model_type(**params).fit(fit[features], target, **group_arg)
            np.testing.assert_array_equal(model.predict(selection[features]), validation_scores)
            rows[f'{method}_score'] = model.predict(evaluation[features]).astype(float)
            selected[window][method] = dict(features=features, parameters=params)
            grids.append(grid.assign(window=window, method=method))
        for method in METHODS:
            metrics = ranking.weekly_metrics(evaluation, rows[f'{method}_score'].to_numpy())
            weekly.append(metrics.assign(window=window, method=method))
        scores.append(rows)
    return (pd.concat(scores, ignore_index=True), pd.concat(weekly, ignore_index=True),
            pd.concat(grids, ignore_index=True), selected)


def compare(weekly):
    paired = weekly.loc[weekly.method.ne(BASELINE)].merge(
        weekly.loc[weekly.method.eq(BASELINE), ['window', 'SignalDate', 'rank_ic', 'top3_overlap']],
        on=['window', 'SignalDate'], suffixes=('', '_baseline'), validate='many_to_one')
    for metric in ('rank_ic', 'top3_overlap'):
        paired[f'{metric}_delta'] = paired[metric] - paired[f'{metric}_baseline']
    summary, paired_summary = [], []
    periods = {window: weekly.loc[weekly.window.eq(window)] for window in WINDOWS}
    periods['all_evaluation'] = weekly
    periods['2024_evaluation'] = weekly.loc[weekly.SignalDate.dt.year.eq(2024)]
    for period, frame in periods.items():
        for method, rows in frame.groupby('method', sort=False):
            stats = ranking.summarize(rows)
            summary.append(dict(period=period, method=method, **stats,
                                undefined_rank_ic_weeks=len(rows) - stats['valid_rank_ic_weeks']))
        period_pairs = paired.loc[paired.SignalDate.isin(frame.SignalDate)]
        for method, rows in period_pairs.groupby('method', sort=False):
            for metric in ('rank_ic', 'top3_overlap'):
                values = rows[f'{metric}_delta'].dropna()
                trimmed = values.sort_values(ascending=False).iloc[3:]
                paired_summary.append(dict(period=period, method=method, metric=metric, weeks=len(rows),
                                           valid_weeks=len(values), undefined_weeks=len(rows) - len(values),
                                           mean_delta=values.mean(), wins=int(values.gt(EPS).sum()),
                                           losses=int(values.lt(-EPS).sum()), ties=int(values.abs().le(EPS).sum()),
                                           top3_delta_sum=values.nlargest(3).sum(),
                                           delta_sum=values.sum(), mean_excluding_best_3=trimmed.mean()))
    return paired, pd.DataFrame(summary), pd.DataFrame(paired_summary)


def failure(scores, weekly, paired):
    worst = (paired.loc[paired.method.eq('D')].dropna(subset='rank_ic_delta')
             .sort_values(['rank_ic_delta', 'SignalDate']).iloc[0])
    date = worst.SignalDate
    rows = scores.loc[scores.SignalDate.eq(date)].sort_values(
        [ranking.RETURN, 'Ticker'], ascending=[False, True]).copy()
    rows['ActualRank'] = np.arange(1, len(rows) + 1)
    for method in METHODS:
        order = rows.sort_values([f'{method}_score', 'Ticker'], ascending=[False, True]).index
        rows.loc[order, f'{method}_rank'] = np.arange(1, len(rows) + 1)
    return rows, weekly.loc[weekly.SignalDate.eq(date)]


def plot_deltas(paired, path):
    fig, axes = plt.subplots(4, 1, figsize=(12, 7), sharex=True)
    for ax, (method, data) in zip(axes, paired.groupby('method', sort=False)):
        for (_, group), color in zip(data.groupby('window', sort=False), ['#346b9d', '#227a62', '#a35d22']):
            ax.plot(group.SignalDate, group.rank_ic_delta, '.-', color=color, lw=.8, ms=4)
        ax.axhline(0, color='#566271', lw=.6)
        ax.set_ylabel(method, rotation=0, labelpad=18)
        ax.set_ylim(-2, 2)
        ax.grid(axis='y', alpha=.18)
        ax.spines[['right', 'top']].set_visible(False)
    fig.suptitle('Weekly Rank IC difference vs. return_5 baseline', x=.1, ha='left', fontsize=16)
    axes[-1].set_xlabel('Evaluation signal date | W1: blue · W2: green · W3: brown')
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run(output):
    base, hashes, inputs = load_features()
    stages, boundary_audit, windows = partition(base)
    parameter_path = ranking.HISTORICAL / 'xgboost_parameters.json'
    historical = json.loads(parameter_path.read_text())
    common = {key: historical['parameters'][key] for key in ranking.COMMON_PARAMETERS}
    scores, weekly, grids, selected = evaluate(stages, common)
    paired, summary, paired_summary = compare(weekly)
    failure_rows, failure_metrics = failure(scores, weekly, paired)
    inputs.extend([Path(__file__), STEP / 'DESIGN.md', Path(medium.__file__), Path(ranking.__file__), parameter_path])
    output.mkdir()
    tables = dict(eval_scores=scores, weekly_metrics=weekly, weekly_deltas=paired, summary=summary,
                  paired_summary=paired_summary, phase_windows=windows, phase_audit=boundary_audit,
                  boundary_exclusions=boundary_audit.loc[boundary_audit.reason.eq('label_exits_outside_stage')],
                  grid=grids, failure_ranks=failure_rows, failure_metrics=failure_metrics)
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    plot_deltas(paired, output / 'weekly_rank_ic_deltas.png')
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    status='retrospective_walk_forward_previously_examined_2023_2024', scored_2025=False,
                    windows=WINDOWS, min_phase_weeks=PHASES, seed=common['random_state'],
                    xgboost_version=ranking.xgb.__version__, grid=dict(max_depth=ranking.DEPTHS,
                    n_estimators=ranking.TREE_COUNTS), selection='maximum mean valid weekly Rank IC',
                    selected=selected, source_snapshot_id=snapshot_id(hashes),
                    input_sha256={**hashes, **{str(p.relative_to(ROOT)): digest(p) for p in inputs}},
                    output_sha256={p.name: digest(p) for p in output.iterdir()},
                    score_time='historical replay as of signal close; run generated retrospectively')
    (output / 'experiment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(windows.to_string(index=False))
    print(summary.to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-name', default='run_01')
    args = parser.parse_args()
    output = (STEP / args.run_name).resolve()
    if output.parent != STEP or output.exists():
        parser.error('Choose a new output folder directly under this step.')
    run(output)
