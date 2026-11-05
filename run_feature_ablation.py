"""Paired, fixed-parameter retraining: all features versus no return_5.

Existing source, samples and results are read only. Run with this project's Python.
"""
import argparse
import hashlib
import importlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/elec3544etf_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from compare_predictions import align_predictions, build_baselines, markdown_table, score_predictions
from models.common import FEATURES, ROOT, FitResult, evaluate_models, output_metadata, snapshot_id
from prepare_data import DATE_COLUMNS, digest, unpack
from price_basis import audit
from study_config import SPLIT_DESIGN

NAMES = ('ridge', 'random_forest', 'xgboost')
VARIANTS = {'full': FEATURES, 'without_return_5': [f for f in FEATURES if f != 'return_5']}
PARAM_FILES = {'ridge': 'model_parameters.json',
               'random_forest': 'random_forest_parameters.json',
               'xgboost': 'xgboost_parameters.json'}


def read_inputs():
    """Validate cache without invoking the preparation function that can rewrite it."""
    report = audit(ROOT, write=False)
    hashes = {str(Path(p).relative_to(ROOT)): h for p, h in report['input_hashes'].items()}
    manifest = json.loads((ROOT/'prepared_data/preparation.json').read_text())
    assert manifest['input_hashes'] == hashes, 'Source data changed'
    for path, value in manifest['code_hashes'].items():
        assert digest(ROOT/path) == value, f'Stale sample generation code: {path}'
    assert digest(ROOT/'prepared_data/weekly_candidates.csv') == manifest['samples_sha256']
    frames = unpack(pd.read_csv(ROOT/'prepared_data/weekly_candidates.csv', parse_dates=DATE_COLUMNS))
    return frames, hashes


def estimator(name, params):
    if name == 'ridge':
        return make_pipeline(StandardScaler(), Ridge(alpha=params['alpha']))
    if name == 'random_forest':
        return RandomForestRegressor(**params['parameters'])
    return XGBRegressor(**params['parameters'])


def save_model(name, model, features, source_params, out, metadata):
    # Export using existing formats, adapting Ridge's named pipeline for reuse.
    if name == 'ridge':
        from sklearn.pipeline import Pipeline
        model = Pipeline([('scale', model['standardscaler']), ('ridge', model['ridge'])])
        selected = {'alpha': source_params['alpha']}
    else:
        selected = {k: v for k, v in source_params['selected'].items() if k != 'validation_mse'}
    module = importlib.import_module(f'models.{name}')
    module.save_outputs(FitResult(model, [], selected), features, out, metadata)
    # No validation search occurred in this experiment; remove misleading empty grid.
    for file in out.glob('*validation_grid.csv'):
        file.unlink()


def run(out):
    if out.exists():
        raise ValueError('Choose a new experiment directory to preserve previous results.')
    if not out.is_relative_to(ROOT/'experiments'):
        raise ValueError('Output must be a new directory under this project experiments/.')
    frames, hashes = read_inputs()
    train, validation, test = frames
    reference = pd.concat(frames, ignore_index=True)
    source_params = {n: json.loads((ROOT/'results'/n/PARAM_FILES[n]).read_text()) for n in NAMES}
    input_files = [ROOT/'prepared_data/weekly_candidates.csv']
    input_files += [ROOT/'results'/n/f for n in NAMES
                    for f in (PARAM_FILES[n], 'predictions.csv')]
    before = {str(p.relative_to(ROOT)): digest(p) for p in input_files}
    out.mkdir(parents=True)
    all_metrics, all_predictions, checks = [], [], []
    for variant, features in VARIANTS.items():
        for name in NAMES:
            model = estimator(name, source_params[name])
            model.fit(train[features], train.TargetPriceReturn)
            predictions, metrics, weekly = evaluate_models(
                {name: model}, *frames, features, hashes)
            pred = pd.concat(predictions, ignore_index=True)
            pred['Variant'] = variant
            pred['FeatureVersion'] = f'ablation_{variant}_v1'
            pred['UsedFeatures'] = ','.join(features)
            # All original feature columns remain as reference data, not model inputs.
            pred = align_predictions(pred, reference, name, snapshot_id(hashes))
            if variant == 'full':
                original = pd.read_csv(ROOT/'results'/name/'predictions.csv', parse_dates=DATE_COLUMNS)
                original = align_predictions(original, reference, name, snapshot_id(hashes))
                difference = float(np.max(np.abs(pred.Prediction - original.Prediction)))
                # Existing XGBoost CSV serialized float32 predictions with fewer digits;
                # recover that dtype before checking exact reproducibility.
                same_float32 = (name == 'xgboost' and np.array_equal(
                    pred.Prediction.to_numpy(dtype=np.float32),
                    original.Prediction.to_numpy(dtype=np.float32)))
                assert same_float32 or difference < 1e-12, f'{name}: full model does not reproduce original'
                checks.append({'model': name, 'full_max_prediction_difference': difference,
                               'original_float32_predictions_match_exactly': same_float32})
            pred['Prediction'] = pred.Prediction.astype(float)
            dest = out/variant/name
            dest.mkdir(parents=True)
            save_model(name, model, features, source_params[name], dest,
                       output_metadata(train, test, hashes))
            metrics['Variant'] = variant
            weekly['Variant'] = variant
            pred.to_csv(dest/'predictions.csv', index=False)
            metrics.to_csv(dest/'metrics.csv', index=False)
            weekly.to_csv(dest/'weekly_rank_ic.csv', index=False)
            all_predictions.append(pred)
            all_metrics.append(metrics)
            print(f'Completed {name}: {variant}', flush=True)
    metrics = pd.concat(all_metrics, ignore_index=True)
    baselines, _ = build_baselines(frames, snapshot_id(hashes))
    base_metrics = pd.concat([score_predictions(p)[0] for p in baselines.values()])
    base_metrics.to_csv(out/'baseline_metrics.csv', index=False)
    mean_rmse = base_metrics[base_metrics.ModelId == 'train_mean'].set_index('Split').rmse
    metrics['rmse_reduction_vs_train_mean_pct'] = 100*(1-metrics.rmse/metrics.Split.map(mean_rmse))
    metrics.to_csv(out/'metrics_all.csv', index=False)
    report = metrics.copy()
    report['mae_pp'] = report.mae*100
    report['rmse_pp'] = report.rmse*100
    report['direction_accuracy_pct'] = report.direction_accuracy*100
    columns = ['ModelId', 'Variant', 'Split', 'rows', 'weeks', 'mae_pp', 'rmse_pp',
               'direction_accuracy_pct', 'mean_weekly_rank_ic', 'rmse_reduction_vs_train_mean_pct']
    report[columns].to_csv(out/'comparison_readable.csv', index=False)
    predictions = pd.concat(all_predictions, ignore_index=True)
    predictions.to_csv(out/'predictions_all.csv', index=False)
    paired = reference.copy()
    keys = ['Split', 'SignalDate', 'Ticker']
    for (variant, name), group in predictions.groupby(['Variant', 'ModelId']):
        paired = paired.merge(group[keys+['Prediction']].rename(
            columns={'Prediction': f'{name}_{variant}'}), on=keys, validate='one_to_one')
    assert len(paired) == len(reference)
    paired.to_csv(out/'predictions_paired.csv', index=False)
    paired[paired.Split == 'test'].to_csv(out/'predictions_paired_2025.csv', index=False)
    full = metrics[metrics.Variant == 'full'].set_index(['ModelId', 'Split'])
    removed = metrics[metrics.Variant != 'full'].set_index(['ModelId', 'Split'])
    delta = pd.DataFrame(index=full.index)
    delta['full_rmse_pp'] = full.rmse*100
    delta['without_return_5_rmse_pp'] = removed.rmse*100
    delta['delta_rmse_pp'] = (removed.rmse-full.rmse)*100
    delta['rmse_change_pct'] = 100*(removed.rmse/full.rmse-1)
    delta['delta_rank_ic'] = removed.mean_weekly_rank_ic-full.mean_weekly_rank_ic
    delta.reset_index().to_csv(out/'ablation_deltas.csv', index=False)
    # Weekly paired errors show whether an aggregate change is concentrated in a few weeks.
    weekly_errors = []
    for name in NAMES:
        a = paired[keys+['TargetPriceReturn', f'{name}_full', f'{name}_without_return_5']].copy()
        for variant in VARIANTS:
            a[f'{variant}_mse'] = (a[f'{name}_{variant}']-a.TargetPriceReturn)**2
        w = a.groupby(['Split', 'SignalDate'])[['full_mse', 'without_return_5_mse']].mean()
        w['delta_mse'] = w.without_return_5_mse-w.full_mse
        weekly_errors.append(w.reset_index().assign(ModelId=name))
    pd.concat(weekly_errors).to_csv(out/'weekly_paired_errors.csv', index=False)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, split, title in zip(axes, ['validation', 'test'], ['2024 validation (development)', '2025 test (exploratory)']):
        for offset, (variant, label, color) in enumerate([
                ('full', 'All four features', '#3b65b8'),
                ('without_return_5', 'Without return_5', '#e69b3a')]):
            values = report[(report.Split == split)&(report.Variant == variant)].set_index('ModelId').loc[list(NAMES), 'rmse_pp']
            bars = ax.bar(np.arange(3)+(offset-.5)*.36, values, .36, label=label, color=color)
            ax.bar_label(bars, fmt='%.4f', padding=3, fontsize=9)
        ax.axhline(mean_rmse[split]*100, color='#64748b', linestyle='--', label='Training mean baseline')
        ax.set_xticks(np.arange(3), ['Ridge', 'Random Forest', 'XGBoost'])
        ax.set_title(title)
        ax.set_ylabel('RMSE (percentage points; lower is better)')
        ax.set_ylim(0, report[report.Split == split].rmse_pp.max()*1.18)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0].legend(fontsize=8, loc='lower left')
    fig.suptitle('Feature ablation: fixed parameters, separately retrained models')
    fig.tight_layout()
    fig.savefig(out/'ablation_rmse.png', dpi=180)
    fig.savefig(out/'ablation_rmse.svg')
    plt.close(fig)
    text = ['# return_5 特征消融实验', '',
            '三种模型 × 两组特征，均实际重新训练。训练：2021—2023；验证：2024；测试：2025。',
            '固定原完整模型通过2024验证选出的参数；本次不重新选参，也不合并训练和验证集。',
            'MAE/RMSE（pp）为收益百分点；原始预测和 metrics_all.csv 使用收益小数。',
            'ΔRMSE = 删除后 − 完整；正值表示删除后变差。Rank IC 越大越好。',
            '2024是开发分析，2025已查看、仅为探索性比较；不代表因果贡献或显著性。',
            '预测文件保留四列原特征作核对，实际输入由 Variant、UsedFeatures 和参数文件明确记录。', '']
    for split in ['validation', 'test', 'train']:
        text += [f'## {split}', '', markdown_table(report.loc[report.Split == split, columns]), '',
                 markdown_table(delta.reset_index().query('Split == @split')), '']
    text += ['## 文件', '',
             '- predictions_paired.csv：所有分区、真实收益、六组预测，同一行直接比较。',
             '- predictions_paired_2025.csv：2025逐条对照。',
             '- full/ 和 without_return_5/：各模型预测、指标、参数和周度Rank IC。',
             '- baseline_metrics.csv：零收益、训练均值基准。',
             '- weekly_paired_errors.csv：每周配对均方误差。',
             '- ablation_deltas.csv：删除特征前后差异。',
             '- ablation_rmse.png / .svg：验证和测试RMSE对比。', '']
    (out/'REPORT.md').write_text('\n'.join(text))
    assert all(digest(ROOT/p) == h for p, h in before.items()), 'Existing inputs modified'
    # Recompute saved predictions with an independent scoring path.
    for variant in VARIANTS:
        for name in NAMES:
            saved = pd.read_csv(out/variant/name/'predictions.csv')
            rescored, _ = score_predictions(saved)
            expected = metrics[(metrics.Variant == variant)&(metrics.ModelId == name)].set_index('Split')
            actual = rescored.set_index('Split')
            for metric in ['mae', 'rmse', 'direction_accuracy', 'mean_weekly_rank_ic']:
                np.testing.assert_allclose(actual[metric], expected.loc[actual.index, metric], atol=1e-12, rtol=1e-10)
    manifest = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                    status='retrospective_exploratory', split_design=SPLIT_DESIGN,
                    variants=VARIANTS, parameter_rule='fixed_original_full_validation_selected_no_retuning',
                    source_parameters=source_params, input_hashes=hashes, input_files_sha256=before,
                    script_sha256=digest(Path(__file__)), verification=checks,
                    paired_rows=len(paired), existing_inputs_unchanged=True,
                    saved_prediction_metrics_recomputed=True)
    (out/'experiment.json').write_text(json.dumps(manifest, indent=2))
    print(delta.to_string())
    print(f'Output: {out}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'experiments/return_5_ablation')
    run(parser.parse_args().output.resolve())
