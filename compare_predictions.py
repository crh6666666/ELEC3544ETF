"""Compare saved model predictions with train-only constant baselines.

Run directly to export CSV/Markdown tables and PNG/SVG figures without training.
"""
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from models.common import FEATURES, ROOT, snapshot_id
from prepare_data import DATE_COLUMNS, digest, load_samples

MODEL_NAMES = ('ridge', 'random_forest', 'xgboost')
SPLITS = ('train', 'validation', 'test')
LABELS = {'zero_return': 'Zero return', 'train_mean': 'Training mean',
          'ridge': 'Ridge', 'random_forest': 'Random Forest',
          'xgboost': 'XGBoost', 'always_up': 'Always up (direction only)'}
ALIGN_COLUMNS = ['Split', 'SignalDate', 'EntryDate', 'ExitDate', 'Ticker',
                 'SignalCloseUTC', 'EntryOpenUTC', 'ExitOpenUTC',
                 'TargetPriceReturn', 'TargetBasis', 'PriceBasisAuditId', *FEATURES]


def align_predictions(frame, reference, model_name, source_id):
    """Reject missing/duplicate samples, changed labels, or mixed source snapshots."""
    keys = ['Split', 'SignalDate', 'Ticker']
    if frame.empty or frame.duplicated(keys).any():
        raise ValueError(f'{model_name}: empty or duplicate predictions')
    if not frame.ModelId.eq(model_name).all():
        raise ValueError(f'{model_name}: incorrect ModelId')
    if not frame.SourceSnapshotId.eq(source_id).all():
        raise ValueError(f'{model_name}: source snapshot changed; rerun the model')
    if not np.isfinite(frame[['Prediction', 'TargetPriceReturn']]).all().all():
        raise ValueError(f'{model_name}: nonfinite prediction or label')
    ordered = frame.sort_values(keys).reset_index(drop=True)
    expected = reference.sort_values(keys).reset_index(drop=True)
    try:
        pd.testing.assert_frame_equal(ordered[ALIGN_COLUMNS], expected[ALIGN_COLUMNS],
                                      check_dtype=False, rtol=1e-12, atol=1e-14)
    except AssertionError as exc:
        raise ValueError(f'{model_name}: samples, features or labels do not align') from exc
    test = ordered[ordered.Split == 'test']
    availability = pd.to_datetime(test.ModelAvailableAsOf, utc=True)
    if not availability.lt(test.SignalCloseUTC).all():
        raise ValueError(f'{model_name}: model unavailable before a test signal')
    return ordered


def build_baselines(frames, source_id):
    """One pooled mean over training ETF-week labels; never update it on later data."""
    train = frames[0]
    mean = float(train.TargetPriceReturn.mean())
    fitted_through = train.ExitOpenUTC.max().isoformat()
    generated = datetime.now(timezone.utc).isoformat()
    predictions, parameters = {}, {}
    for name, value in [('zero_return', 0.0), ('train_mean', mean)]:
        parts = []
        for split, frame in zip(SPLITS, frames):
            a = frame.copy()
            a['ModelId'] = name
            a['Prediction'] = value
            a['SourceSnapshotId'] = source_id
            a['GeneratedAtUTC'] = generated
            a['TrainingLabelsThrough'] = fitted_through if name == 'train_mean' else ''
            a['ModelAvailableAsOf'] = fitted_through if name == 'train_mean' else ''
            a['PredictionUse'] = ('in_sample_reference' if split == 'train'
                                  else f'retrospective_{split}')
            parts.append(a)
        predictions[name] = pd.concat(parts, ignore_index=True)
        parameters[name] = dict(prediction=value, fit_partition=(
            'train_only_pooled_across_etfs' if name == 'train_mean' else 'none'),
            training_rows=len(train) if name == 'train_mean' else 0,
            training_labels_through=fitted_through if name == 'train_mean' else None,
            source_snapshot_id=source_id,
            note='Training metrics are in-sample references, not historical tradable signals.')
    return predictions, parameters


def score_predictions(frame):
    """Use the existing sign convention, including sign(0)=0, and weekly Rank IC."""
    rows, weekly = [], []
    for split in SPLITS:
        a = frame[frame.Split == split]
        if a.empty:
            raise ValueError(f'Missing partition: {split}')
        values = a.Prediction.to_numpy(dtype=float)
        actual = a.TargetPriceReturn.to_numpy(dtype=float)
        errors = values - actual
        correlations = []
        for date, group in a.groupby('SignalDate', sort=True):
            corr = (group.Prediction.rank().corr(group.TargetPriceReturn.rank())
                    if group.Prediction.nunique() > 1 and group.TargetPriceReturn.nunique() > 1
                    else np.nan)
            correlations.append(corr)
            weekly.append(dict(ModelId=a.ModelId.iloc[0], Split=split,
                               SignalDate=date, rank_ic=corr))
        valid = [c for c in correlations if np.isfinite(c)]
        rows.append(dict(ModelId=a.ModelId.iloc[0], Split=split, rows=len(a),
            weeks=a.SignalDate.nunique(), mae=float(np.abs(errors).mean()),
            rmse=float(np.sqrt(np.mean(errors ** 2))),
            direction_accuracy=float((np.sign(values) == np.sign(actual)).mean()),
            always_up_accuracy=float((actual > 0).mean()),
            mean_weekly_rank_ic=float(np.mean(valid)) if valid else np.nan,
            defined_rank_ic_weeks=len(valid)))
    return pd.DataFrame(rows), pd.DataFrame(weekly)


def comparison_metrics(predictions):
    metrics = pd.concat([score_predictions(f)[0] for f in predictions.values()], ignore_index=True)
    direction = metrics[metrics.ModelId == 'zero_return'].copy()
    direction['ModelId'] = 'always_up'
    direction['direction_accuracy'] = direction.always_up_accuracy
    direction[['mae', 'rmse', 'mean_weekly_rank_ic']] = np.nan
    direction['defined_rank_ic_weeks'] = 0
    metrics = pd.concat([metrics, direction], ignore_index=True)
    for split in SPLITS:
        mask = metrics.Split == split
        group = metrics[mask].set_index('ModelId')
        for baseline in ('zero_return', 'train_mean', 'ridge'):
            if baseline in group.index:
                denominator = group.loc[baseline, 'rmse']
                # Positive means a reduction in RMSE; NaN if the denominator is zero.
                metrics.loc[mask, f'rmse_reduction_vs_{baseline}_pct'] = (
                    100 * (1 - metrics.loc[mask, 'rmse'] / denominator)
                    if denominator > 0 else np.nan)
    return metrics


def report_table(metrics, split):
    a = metrics[metrics.Split == split]
    return pd.DataFrame({
        'Model': a.ModelId.map(LABELS), 'Rows': a.rows, 'Weeks': a.weeks,
        'MAE (pp)': a.mae * 100, 'RMSE (pp)': a.rmse * 100,
        'Direction accuracy (%)': a.direction_accuracy * 100,
        'Mean weekly Rank IC': a.mean_weekly_rank_ic,
        'Defined IC weeks': a.defined_rank_ic_weeks,
        'RMSE reduction vs training mean (%)': a.rmse_reduction_vs_train_mean_pct,
    })


def markdown_table(table):
    lines = ['| ' + ' | '.join(table.columns) + ' |',
             '| ' + ' | '.join(['---'] * len(table.columns)) + ' |']
    for row in table.itertuples(index=False, name=None):
        cells = [('N/A' if pd.isna(v) else f'{v:.4f}' if isinstance(v, float) else str(v))
                 for v in row]
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def write_report(metrics, parameters, path):
    mean = parameters['train_mean']['prediction']
    text = ['# 行业 ETF 周收益预测：统一比较表', '',
        f'训练均值：{mean:.12f}（收益小数），即每周 {mean * 100:.6f}%；'
        f'仅使用 {parameters["train_mean"]["training_rows"]} 条训练集标签，'
        '对所有 ETF 及所有分区使用同一个固定值。', '',
        'MAE/RMSE 的原始 CSV 单位为收益小数；本表误差单位为百分点（pp）。'
        'RMSE 降幅为 100 × (1 − 模型 RMSE / 基准 RMSE)，正值表示改善。', '',
        '方向准确率沿用现有定义 sign(预测)=sign(实际)，sign(0)=0；'
        '零收益预测仅在实际收益恰为零时方向正确，不把零当作上涨。'
        'Always up 只评价实际收益严格大于零的比例，没有收益幅度预测。', '',
        '常数预测没有横截面排序，Rank IC 记为 N/A，有效 IC 周数为 0。'
        '训练指标为样本内参考；验证用于选参；2025 结果已查看，属于回顾性探索。'
        '收益不含现金分红；同周 ETF 记录并非独立时间样本。', '']
    test = metrics[metrics.Split == 'test'].set_index('ModelId')
    models = test.loc[[name for name in MODEL_NAMES if name in test.index]]
    best_rmse = models.rmse.idxmin()
    best_mae = models.mae.idxmin()
    up = test.loc['always_up', 'direction_accuracy']
    beat_up = models.index[models.direction_accuracy > up].tolist()
    val = metrics[metrics.Split == 'validation'].set_index('ModelId')
    beat_mean = [LABELS[name] for name in models.index
                 if val.loc[name, 'rmse'] < val.loc['train_mean', 'rmse']]
    text += ['## 初步观察（自动按当前结果生成）', '',
        f'- 在本次比较的模型中，{LABELS[best_rmse]} 的测试 RMSE 最低，'
        f'为 {models.loc[best_rmse, "rmse"] * 100:.6f} 个百分点；'
        f'相对训练均值基准的 RMSE 降幅为 '
        f'{models.loc[best_rmse, "rmse_reduction_vs_train_mean_pct"]:.4f}%。',
        f'- 测试 MAE 最低的模型为 {LABELS[best_mae]}，'
        f'为 {models.loc[best_mae, "mae"] * 100:.6f} 个百分点；'
        'MAE 与 RMSE 对误差的权重不同，应分别报告。',
        f'- 始终上涨的测试方向准确率为 {up:.4%}；'
        + ('超过这一参考的模型：' + '、'.join(LABELS[name] for name in beat_up) + '。'
           if beat_up else '本次比较的模型均未超过这一参考。'),
        '- 验证集 RMSE 优于训练均值基准的模型：'
        + ('、'.join(beat_mean) if beat_mean else '无') + '。应同时报告验证与测试结果。',
        '- 以上为当前区间的描述性比较，不能据此认定显著改善或直接推断组合盈利。', '']
    for split, label in [('test', '2025 评估'), ('validation', '2024 验证'),
                         ('train', '2021—2023 训练')]:
        text += [f'## {label}', '', markdown_table(report_table(metrics, split)), '']
    path.write_text('\n'.join(text), encoding='utf-8')


def plot_comparison(metrics, path):
    # Reuse the project's plotting setup; it configures a headless backend and local fonts.
    from plot_results import configure_fonts
    import matplotlib.pyplot as plt

    configure_fonts()
    plt.rcParams['axes.titleweight'] = 600
    a = metrics[metrics.Split == 'test'].set_index('ModelId')
    names = [name for name in a.index if name != 'always_up']
    labels = [LABELS[name] for name in names]
    palette = dict(zip(LABELS, ['#94A3B8', '#64748B', '#118C88', '#416CC5', '#C98522', '#D05E47']))
    colors = [palette[name] for name in names]
    fig, axes = plt.subplots(1, 3, figsize=(16, 6.5), facecolor='#F6F8FB')
    fig.subplots_adjust(left=.055, right=.98, top=.72, bottom=.25, wspace=.33)
    fig.text(.055, .935, '行业 ETF · 模型与简单基准', fontsize=23, weight=600)
    fig.text(.055, .865, f'2025 评估 / {int(a.iloc[0].weeks)} 周 / '
             f'{int(a.iloc[0].rows)} 条 ETF–周记录 / 回顾性探索', fontsize=12, color='#64748B')
    x = np.arange(len(names))
    ax = axes[0]
    for offset, field, label, color in [(-.19, 'mae', 'MAE', '#98ABB9'),
                                        (.19, 'rmse', 'RMSE', '#416CC5')]:
        bars = ax.bar(x + offset, a.loc[names, field] * 100, width=.36, label=label, color=color)
        ax.bar_label(bars, fmt='%.3f', padding=3, fontsize=8)
    ax.set_title('收益幅度误差：越低越好', loc='left', pad=15)
    ax.set_ylabel('误差（百分点）')
    ax.set_ylim(0, a.rmse.max() * 100 * 1.22)
    ax.legend(frameon=False, ncol=2, loc='upper left')
    ax = axes[1]
    bars = ax.bar(x, a.loc[names, 'direction_accuracy'] * 100, color=colors[:len(names)])
    ax.bar_label(bars, fmt='%.2f%%', padding=4, fontsize=9)
    up = float(a.loc['always_up', 'direction_accuracy'] * 100)
    ax.axhline(up, color='#D05E47', linestyle='--', lw=1.4, label=f'始终上涨 {up:.2f}%')
    ax.set_title('方向准确率：越高越好', loc='left', pad=15)
    ax.set_ylabel('准确率（%）')
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc='upper left', fontsize=9)
    ax = axes[2]
    for i, name in enumerate(names):
        value = a.loc[name, 'mean_weekly_rank_ic']
        if pd.isna(value):
            ax.text(i, .004, 'N/A', ha='center', fontsize=10, color='#64748B')
        else:
            ax.bar(i, value, color=colors[i])
            ax.annotate(f'{value:.4f}', (i, value), xytext=(0, 5 if value >= 0 else -5),
                        textcoords='offset points', ha='center',
                        va='bottom' if value >= 0 else 'top', fontsize=9)
    ax.axhline(0, color='#94A3B8', lw=.8)
    ax.set_ylim(min(-.02, a.mean_weekly_rank_ic.min() - .015),
                max(.08, a.mean_weekly_rank_ic.max() + .025))
    ax.set_title('周度横截面排序', loc='left', pad=15)
    ax.set_ylabel('平均 Rank IC')
    for ax in axes:
        ax.set_facecolor('white')
        ax.set_xticks(x, labels, rotation=25, ha='right', fontsize=9)
        ax.set_xlim(-.6, len(names) - .4)
        ax.grid(axis='y', alpha=.18)
        ax.set_axisbelow(True)
    fig.text(.055, .085, 'Training mean 仅由训练集计算；常数预测的 Rank IC 不适用。'
             '零收益方向判断使用 sign(0)=0。', fontsize=10, color='#64748B')
    fig.text(.055, .04, '价格收益不含现金分红；预测指标不能直接解释为组合收益；'
             '图中差异未构成统计显著性证据。', fontsize=10, color='#64748B')
    for suffix in ('png', 'svg'):
        fig.savefig(path.with_suffix('.' + suffix), dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    parser.add_argument('--output', type=Path, help='Table output folder; defaults to --results.')
    parser.add_argument('--figures', type=Path, help='Figure output folder.')
    parser.add_argument('--models', nargs='+', choices=MODEL_NAMES, default=list(MODEL_NAMES))
    parser.add_argument('--no-plot', action='store_true', help='Export tables without figures.')
    args = parser.parse_args()
    names = list(dict.fromkeys(args.models))
    results = args.results.resolve()
    out = (args.output or results).resolve()
    figures = (args.figures or (ROOT/'figures' if out == ROOT/'results' else out/'figures')).resolve()
    protected = [ROOT/p for p in ('data', 'prepared_data', 'models', '.venv')]
    for folder in (out, figures):
        if folder == ROOT or any(folder.is_relative_to(p) or p.is_relative_to(folder) for p in protected):
            parser.error('Choose output folders outside source data, models and environment.')
    frames, hashes = load_samples()
    source_id = snapshot_id(hashes)
    reference = pd.concat(frames, ignore_index=True)
    sample_hash = digest(ROOT/'prepared_data/weekly_candidates.csv')
    predictions, parameters = build_baselines(frames, source_id)
    input_files = {str(ROOT/'prepared_data/weekly_candidates.csv'): sample_hash}
    for name in names:
        folder = results/name
        manifest = json.loads((folder/'debug_run.json').read_text())
        if manifest['input_hashes'] != hashes or manifest['samples_sha256'] != sample_hash:
            raise ValueError(f'{name}: stale model outputs; rerun this model')
        frame = pd.read_csv(folder/'predictions.csv', parse_dates=DATE_COLUMNS)
        predictions[name] = align_predictions(frame, reference, name, source_id)
        for filename in ('predictions.csv', 'debug_run.json'):
            input_files[str(folder/filename)] = digest(folder/filename)
    metrics = comparison_metrics(predictions)
    # All inputs are validated before exporting any comparison files.
    out.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    for name in parameters:
        folder = out/'baselines'/name
        folder.mkdir(parents=True, exist_ok=True)
        predictions[name].to_csv(folder/'predictions.csv', index=False)
        scores, weekly = score_predictions(predictions[name])
        scores.to_csv(folder/'metrics.csv', index=False)
        weekly.to_csv(folder/'weekly_rank_ic.csv', index=False)
        (folder/'parameters.json').write_text(json.dumps(parameters[name], indent=2))
    suffix = '' if set(names) == set(MODEL_NAMES) else '_' + '_'.join(names)
    stem = 'prediction_comparison' + suffix
    metrics.to_csv(out/f'{stem}_metrics.csv', index=False)
    report_table(metrics, 'test').to_csv(out/f'{stem}_test.csv', index=False, na_rep='N/A')
    write_report(metrics, parameters, out/f'{stem}.md')
    if not args.no_plot:
        plot_comparison(metrics, figures/stem)
    audit = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                 models=names, source_snapshot_id=source_id, input_files_sha256=input_files,
                 script_sha256=digest(Path(__file__)), baselines=parameters,
                 experiment_status='retrospective_exploratory',
                 note='Metrics recomputed from aligned saved predictions; no model refitting or selection.')
    (out/f'{stem}_audit.json').write_text(json.dumps(audit, indent=2))
    print(report_table(metrics, 'test').to_string(index=False, na_rep='N/A'))
    print(f'Tables: {out/stem}.md and CSV files')
    if not args.no_plot:
        print(f'Figures: {figures/stem}.png and .svg')


if __name__ == '__main__':
    main()
