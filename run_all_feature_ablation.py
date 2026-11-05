"""All four leave-one-feature-out experiments for three fixed regressors.

Train and evaluate 15 models, export paired predictions, and simulate portfolios.
"""
import argparse
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

from run_feature_ablation import (
    ROOT, FEATURES, NAMES, PARAM_FILES, read_inputs, estimator, save_model,
    align_predictions, build_baselines, score_predictions, markdown_table,
    evaluate_models, output_metadata, snapshot_id, digest, DATE_COLUMNS,
    SPLIT_DESIGN, np, pd, plt,
)
from matplotlib.ticker import PercentFormatter

VARIANTS = {'full': list(FEATURES),
            **{f'without_{f}': [x for x in FEATURES if x != f] for f in FEATURES}}
LABELS = {'full': 'All four', 'without_return_5': 'No recent return',
          'without_volatility_5': 'No volatility', 'without_price_ma_5': 'No MA ratio',
          'without_activity_5': 'No volume activity'}
ZH = {'full': '完整四特征', 'without_return_5': '删除近期收益',
      'without_volatility_5': '删除波动率', 'without_price_ma_5': '删除均线比值',
      'without_activity_5': '删除成交活跃度'}
COLORS = ['#334155', '#c87813', '#167d6e', '#8b55be', '#307bc0']


def plot_errors(metrics, delta, out):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    bound = max(abs(delta.loc[delta.Split.isin(['validation', 'test']), 'rmse_change_pct']).max(), .01)
    for ax, split, title in zip(axes, ['validation', 'test'], ['2024 validation (development)', '2025 test (exploratory)']):
        a = delta[delta.Split == split].pivot(index='ModelId', columns='RemovedFeature', values='rmse_change_pct').loc[list(NAMES), FEATURES]
        im = ax.imshow(a, cmap='RdBu_r', vmin=-bound, vmax=bound, aspect='auto')
        for row in range(3):
            for col in range(4):
                value = a.iloc[row, col]
                ax.text(col, row, f'{value:+.3f}%', ha='center', va='center',
                        color='white' if abs(value) > bound*.6 else '#172033', fontsize=10)
        ax.set_xticks(range(4), ['Recent return', 'Volatility', 'MA ratio', 'Volume activity'], rotation=22, ha='right')
        ax.set_yticks(range(3), ['Ridge', 'Random Forest', 'XGBoost'])
        ax.set_title(title)
    fig.colorbar(im, ax=list(axes), label='RMSE change versus full model (%)', fraction=.035, pad=.03)
    fig.suptitle('Feature removal: blue = lower error; red = higher error', fontsize=14)
    fig.subplots_adjust(left=.11, right=.85, bottom=.25, top=.79, wspace=.35)
    for ext in ['png', 'svg']:
        fig.savefig(out/f'ablation_rmse_changes.{ext}', dpi=180)
    plt.close(fig)


def portfolios(out):
    summaries, curves, trades = [], [], []
    for variant in VARIANTS:
        folder = out/variant/'figures'
        subprocess.run([sys.executable, '-B', str(ROOT/'plot_portfolio.py'),
                        '--results', str(out/variant), '--output', str(folder),
                        '--cost-bps', '10'], cwd=ROOT, check=True, capture_output=True, text=True)
        for suffix, collection in [('summary', summaries), ('curves', curves), ('trades', trades)]:
            collection.append(pd.read_csv(folder/f'portfolio_simulation_2025_{suffix}.csv').assign(Variant=variant))
        print(f'Portfolio complete: {variant}', flush=True)
    summary = pd.concat(summaries, ignore_index=True)
    summary['TotalPriceReturn_pct'] = summary.TotalPriceReturn*100
    summary['MaxDrawdown_pct'] = summary.MaxDrawdown*100
    summary.to_csv(out/'portfolio_comparison_2025.csv', index=False)
    curves = pd.concat(curves, ignore_index=True)
    curves.to_csv(out/'portfolio_daily_curves_2025.csv', index=False)
    pd.concat(trades, ignore_index=True).to_csv(out/'portfolio_trades_2025.csv', index=False)
    for strategy in ['spy', 'equal_weight']:
        a = summary[summary.Strategy == strategy]
        assert a.TotalPriceReturn.nunique() == 1 and a.MaxDrawdown.nunique() == 1
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})
    for col, (model, title) in enumerate(zip(NAMES, ['Ridge', 'Random Forest', 'XGBoost'])):
        for (variant, label), color in zip(LABELS.items(), COLORS):
            a = curves[(curves.Strategy == model)&(curves.Variant == variant)]
            dates = pd.to_datetime(a.Date)
            axes[0, col].plot(dates, a.CumulativeReturn, label=label, color=color, lw=1.6)
            axes[1, col].plot(dates, a.Drawdown, color=color, lw=1.4)
        axes[0, col].set_title(title)
        for row in range(2):
            ax = axes[row, col]
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.grid(alpha=.2)
            ax.spines[['top', 'right']].set_visible(False)
        axes[1, col].tick_params(axis='x', labelrotation=25)
    axes[0, 0].set_ylabel('Cumulative price return')
    axes[1, 0].set_ylabel('Drawdown')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.5, .94), ncol=5, frameon=False)
    fig.suptitle('All feature ablations | 2025 exploratory portfolio simulation', fontsize=17)
    fig.tight_layout(rect=[0, .09, 1, .89])
    fig.text(.05, .045, '2025-01-06 to 2025-12-29 | Weekly: 1/11 per ETF if prediction > 0, otherwise cash | One-way cost: 10 bps', fontsize=10)
    fig.text(.05, .017, 'Daily open valuations; excludes dividends; cash interest = 0. Initial entry and final exit included. Previously inspected test data.', fontsize=10)
    for ext in ['png', 'svg']:
        fig.savefig(out/f'portfolio_all_ablations_2025.{ext}', dpi=180)
    plt.close(fig)
    return summary


def run(out):
    if out.exists() or not out.is_relative_to(ROOT/'experiments'):
        raise ValueError('Choose a new output directory under experiments/.')
    frames, hashes = read_inputs()
    train, validation, test = frames
    reference = pd.concat(frames, ignore_index=True)
    params = {n: json.loads((ROOT/'results'/n/PARAM_FILES[n]).read_text()) for n in NAMES}
    protected = [p for folder in ['data', 'prepared_data', 'results', 'models', 'experiments/return_5_ablation']
                 for p in (ROOT/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    before = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    out.mkdir(parents=True)
    predictions, metrics, weekly_ic, checks = [], [], [], []
    for variant, features in VARIANTS.items():
        for name in NAMES:
            model = estimator(name, params[name])
            model.fit(train[features], train.TargetPriceReturn)
            assert model.n_features_in_ == len(features)
            parts, scores, weekly = evaluate_models({name: model}, *frames, features, hashes)
            pred = pd.concat(parts, ignore_index=True)
            pred['Variant'] = variant
            pred['FeatureVersion'] = f'leave_one_out_{variant}_v1'
            pred['UsedFeatures'] = ','.join(features)
            pred = align_predictions(pred, reference, name, snapshot_id(hashes))
            if variant == 'full':
                original = pd.read_csv(ROOT/'results'/name/'predictions.csv', parse_dates=DATE_COLUMNS)
                original = align_predictions(original, reference, name, snapshot_id(hashes))
                maxdiff = float(np.max(np.abs(pred.Prediction-original.Prediction)))
                exact32 = name == 'xgboost' and np.array_equal(pred.Prediction.to_numpy(dtype=np.float32), original.Prediction.to_numpy(dtype=np.float32))
                assert exact32 or maxdiff < 1e-12
                checks.append(dict(model=name, full_max_prediction_difference=maxdiff, exact_after_float32_recovery=exact32))
            # Check the existing two-variant experiment is reproduced as well.
            old = ROOT/'experiments/return_5_ablation'/variant/name/'predictions.csv'
            if old.exists():
                prior = pd.read_csv(old, parse_dates=DATE_COLUMNS)
                prior = align_predictions(prior, reference, name, snapshot_id(hashes))
                np.testing.assert_allclose(pred.Prediction, prior.Prediction, atol=1e-12, rtol=0)
            pred['Prediction'] = pred.Prediction.astype(float)
            dest = out/variant/name
            dest.mkdir(parents=True)
            save_model(name, model, features, params[name], dest, output_metadata(train, test, hashes))
            scores['Variant'] = weekly['Variant'] = variant
            pred.to_csv(dest/'predictions.csv', index=False)
            scores.to_csv(dest/'metrics.csv', index=False)
            weekly.to_csv(dest/'weekly_rank_ic.csv', index=False)
            saved = pd.read_csv(dest/'predictions.csv')
            rescored, _ = score_predictions(saved)
            for field in ['mae', 'rmse', 'direction_accuracy', 'mean_weekly_rank_ic']:
                np.testing.assert_allclose(rescored[field], scores[field], atol=1e-12, rtol=1e-10)
            predictions.append(pred); metrics.append(scores); weekly_ic.append(weekly)
            print(f'Trained and verified: {name} / {variant}', flush=True)
    metrics = pd.concat(metrics, ignore_index=True)
    predictions = pd.concat(predictions, ignore_index=True)
    pd.concat(weekly_ic).to_csv(out/'weekly_rank_ic_all.csv', index=False)
    baselines, _ = build_baselines(frames, snapshot_id(hashes))
    base_metrics = pd.concat([score_predictions(p)[0] for p in baselines.values()])
    base_metrics.to_csv(out/'baseline_metrics.csv', index=False)
    base_rmse = base_metrics[base_metrics.ModelId == 'train_mean'].set_index('Split').rmse
    metrics['rmse_reduction_vs_train_mean_pct'] = 100*(1-metrics.rmse/metrics.Split.map(base_rmse))
    metrics.to_csv(out/'metrics_all.csv', index=False)
    readable = metrics.assign(FeatureGroup=metrics.Variant.map(ZH), mae_pp=metrics.mae*100,
                              rmse_pp=metrics.rmse*100, direction_accuracy_pct=metrics.direction_accuracy*100)
    columns = ['ModelId', 'FeatureGroup', 'Split', 'rows', 'weeks', 'mae_pp', 'rmse_pp',
               'direction_accuracy_pct', 'mean_weekly_rank_ic', 'rmse_reduction_vs_train_mean_pct']
    readable[columns].to_csv(out/'comparison_readable.csv', index=False, encoding='utf-8-sig')
    predictions.to_csv(out/'predictions_all.csv', index=False)
    keys = ['Split', 'SignalDate', 'Ticker']
    paired = reference.copy()
    for (variant, name), group in predictions.groupby(['Variant', 'ModelId']):
        paired = paired.merge(group[keys+['Prediction']].rename(columns={'Prediction':f'{name}_{variant}'}), on=keys, validate='one_to_one')
    assert len(paired) == len(reference) and len(predictions) == len(reference)*15
    paired.to_csv(out/'predictions_paired.csv', index=False)
    paired[paired.Split == 'test'].to_csv(out/'predictions_paired_2025.csv', index=False)
    full = metrics[metrics.Variant == 'full'].set_index(['ModelId', 'Split'])
    deltas, weekly_errors = [], []
    for feature in FEATURES:
        variant = f'without_{feature}'
        removed = metrics[metrics.Variant == variant].set_index(['ModelId', 'Split'])
        a = pd.DataFrame(index=full.index)
        a['RemovedFeature'] = feature
        a['full_rmse_pp'] = full.rmse*100
        a['removed_rmse_pp'] = removed.rmse*100
        a['delta_rmse_pp'] = (removed.rmse-full.rmse)*100
        a['rmse_change_pct'] = 100*(removed.rmse/full.rmse-1)
        a['delta_rank_ic'] = removed.mean_weekly_rank_ic-full.mean_weekly_rank_ic
        deltas.append(a.reset_index())
        for name in NAMES:
            errors = paired[keys].copy()
            errors['full_mse'] = (paired[f'{name}_full']-paired.TargetPriceReturn)**2
            errors['removed_mse'] = (paired[f'{name}_{variant}']-paired.TargetPriceReturn)**2
            w = errors.groupby(['Split','SignalDate'])[['full_mse','removed_mse']].mean().reset_index()
            w['delta_mse'] = w.removed_mse-w.full_mse
            w['ModelId'] = name; w['RemovedFeature'] = feature
            weekly_errors.append(w)
    delta = pd.concat(deltas, ignore_index=True)
    delta.to_csv(out/'ablation_deltas.csv', index=False)
    weeks = pd.concat(weekly_errors, ignore_index=True)
    weeks.to_csv(out/'weekly_paired_errors.csv', index=False)
    weeks['RemovedBetter'] = weeks.delta_mse < 0
    stability = weeks.groupby(['ModelId','RemovedFeature','Split']).agg(
        weeks=('SignalDate','count'), weeks_removed_better=('RemovedBetter','sum'),
        mean_delta_mse=('delta_mse','mean'), median_delta_mse=('delta_mse','median')).reset_index()
    stability['weeks_removed_better_pct'] = stability.weeks_removed_better/stability.weeks*100
    stability.to_csv(out/'weekly_stability_summary.csv', index=False)
    weeks['Quarter'] = pd.to_datetime(weeks.SignalDate).dt.to_period('Q').astype(str)
    weeks.groupby(['ModelId','RemovedFeature','Split','Quarter'])[['full_mse','removed_mse','delta_mse']].mean().reset_index().to_csv(out/'quarterly_paired_errors.csv', index=False)
    corr = train[FEATURES].corr()
    corr.to_csv(out/'training_feature_correlations.csv')
    plot_errors(metrics, delta, out)
    portfolio = portfolios(out)
    report = ['# 四特征逐一删除消融：完整结果', '',
              '三种回归模型 × 五组特征，共15组实际重新训练；每次仅删除一个特征，没有搜索全部特征组合。',
              '训练2021—2023（1683条/153周）、验证2024（561条/51周）、测试2025（561条/51周）。',
              '固定原完整模型在2024验证选出的参数；仅训练集拟合，Ridge标准化也仅在训练集拟合。不重新选参，不合并训练/验证。',
              '2024为开发分析，2025为已查看数据的探索性比较。未做显著性检验，不能解释为因果贡献。',
              'MAE/RMSE（pp）为百分点；原始预测和metrics_all.csv为收益小数；ΔRMSE=删除后−完整，负值表示删后误差下降。',
              '预测表保留全部四个原始特征用于核对；实际模型输入见UsedFeatures、Variant及各模型参数文件。', '', '## 主要发现', '']
    for name in NAMES:
        a = delta[(delta.ModelId == name)&(delta.Split == 'validation')]
        better = a[a.delta_rmse_pp < 0]
        description = '、'.join(ZH[f'without_{f}'] for f in better.RemovedFeature) or '无'
        report.append(f'- {name}：2024验证集删除后RMSE下降的实验：{description}。')
    report += ['- 训练集近期收益与均线比值相关系数约0.840；单项消融衡量的是其他特征仍在时的额外贡献。',
               '- 同周ETF存在共同市场暴露；按周结果与季度汇总另存，避免把561条记录当作561个独立时间样本。', '']
    for split in ['validation','test','train']:
        report += [f'## {split}', '', markdown_table(readable.loc[readable.Split == split, columns]), '',
                   '删除后相对完整模型的变化：', '', markdown_table(delta[delta.Split == split]), '']
    report += ['## 周度稳定性（2024验证）', '', markdown_table(stability[stability.Split == 'validation']), '',
               '## 2025探索性收益模拟', '',
               '沿用既有规则：预测严格大于0时每只ETF占1/11，否则留现金；每周调仓，单边成本10 bps。',
               '2025-01-06至2025-12-29，每日开盘估值、最后开盘清仓；不含分红、税、现金利息和真实执行摩擦。',
               '收益变化也受现金暴露和换手成本影响，不能全部归于预测误差变化。', '',
               markdown_table(portfolio[['Strategy','Variant','TotalPriceReturn_pct','MaxDrawdown_pct']]), '',
               '## 数据与复现', '',
               'predictions_paired.csv：2805条样本，每行真实收益与15组预测；2025专表为predictions_paired_2025.csv。',
               '各特征组目录保存三种模型逐条预测、指标、参数、周度Rank IC及收益模拟图/交易流水。',
               'comparison_readable.csv、ablation_deltas.csv为指标汇总与差异；baseline_metrics.csv为简单基准。',
               'weekly_paired_errors.csv、weekly_stability_summary.csv及quarterly_paired_errors.csv为描述性稳定性数据。',
               'ablation_rmse_changes.png展示验证/测试误差变化；portfolio_all_ablations_2025.png展示全部收益/回撤曲线。', '',
               '在项目根目录执行（需选新的输出目录）：', '', '```sh',
               '.venv/bin/python -B run_all_feature_ablation.py --output experiments/all_feature_ablation_new', '```', '']
    (out/'REPORT.md').write_text('\n'.join(report))
    assert all(digest(ROOT/p) == h for p,h in before.items()), 'An existing file changed'
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    status='retrospective_exploratory', variants=VARIANTS, models=list(NAMES),
                    parameter_rule='fixed_original_full_validation_selected_no_retuning',
                    source_parameters=params, split_design=SPLIT_DESIGN, input_hashes=hashes,
                    protected_files_sha256=before, protected_files_unchanged=True,
                    full_reproduction_checks=checks, previous_two_variants_reproduced=True,
                    saved_prediction_metrics_recomputed=True, paired_rows=len(paired),
                    prediction_rows=len(predictions), script_sha256=digest(Path(__file__)))
    (out/'experiment.json').write_text(json.dumps(manifest, indent=2))
    (out/'output_checksums.json').write_text(json.dumps(
        {str(p.relative_to(out)):digest(p) for p in sorted(out.rglob('*')) if p.is_file()}, indent=2))
    archive = out.with_name(out.name+'_bundle.zip')
    if archive.exists():
        raise ValueError(f'Archive already exists: {archive}')
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        for p in sorted(out.rglob('*')):
            if p.is_file(): z.write(p,p.relative_to(out.parent))
        for file in ['run_all_feature_ablation.py','run_feature_ablation.py']:
            z.write(ROOT/file,file)
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
    print(readable[readable.Split.isin(['validation','test'])][columns].to_string(index=False))
    print(f'Output: {out}\nArchive: {archive}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/all_feature_ablation')
    run(parser.parse_args().output.resolve())
