"""Combine the two independently generated portfolio illustrations."""
import os
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', '/tmp/elec3544etf_mpl')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import pandas as pd

out = Path(__file__).resolve().parent
summaries, curves, trades = [], [], []
for variant in ['full', 'without_return_5']:
    folder = out/variant/'figures'
    for suffix, collection in [('summary', summaries), ('curves', curves), ('trades', trades)]:
        collection.append(pd.read_csv(folder/f'portfolio_simulation_2025_{suffix}.csv').assign(Variant=variant))
summary = pd.concat(summaries, ignore_index=True)
summary['TotalPriceReturn_pct'] = summary.TotalPriceReturn*100
summary['MaxDrawdown_pct'] = summary.MaxDrawdown*100
summary.to_csv(out/'portfolio_comparison_2025.csv', index=False)
curves = pd.concat(curves, ignore_index=True)
curves.to_csv(out/'portfolio_daily_curves_2025.csv', index=False)
pd.concat(trades, ignore_index=True).to_csv(out/'portfolio_trades_2025.csv', index=False)
fig, axes = plt.subplots(2, 3, figsize=(15, 7), sharex=True,
                         gridspec_kw={'height_ratios': [2, 1]})
for col, (model, label) in enumerate([('ridge', 'Ridge'), ('random_forest', 'Random Forest'), ('xgboost', 'XGBoost')]):
    for variant, legend, color in [('full', 'All four features', '#3b65b8'),
                                    ('without_return_5', 'Without return_5', '#e69b3a')]:
        a = curves[(curves.Strategy == model)&(curves.Variant == variant)]
        dates = pd.to_datetime(a.Date)
        axes[0, col].plot(dates, a.CumulativeReturn, label=legend, color=color)
        axes[1, col].plot(dates, a.Drawdown, color=color)
    axes[0, col].set_title(label)
    for row in range(2):
        ax = axes[row, col]
        ax.yaxis.set_major_formatter(PercentFormatter(1))
        ax.grid(alpha=.2)
        ax.spines[['top', 'right']].set_visible(False)
    axes[0, col].legend(fontsize=9)
    axes[1, col].tick_params(axis='x', labelrotation=25)
axes[0, 0].set_ylabel('Cumulative price return')
axes[1, 0].set_ylabel('Drawdown')
fig.suptitle('2025 exploratory simulation: all features versus removing return_5', fontsize=16)
fig.tight_layout(rect=[0, .09, 1, .95])
fig.text(.04, .045, '2025-01-06 to 2025-12-29 | Weekly: 1/11 per ETF if prediction > 0, otherwise cash | One-way costs: 10 bps', fontsize=10)
fig.text(.04, .015, 'Daily open valuations; excludes dividends; cash interest = 0. Includes initial entry and final exit. Previously inspected test data.', fontsize=10)
for extension in ['png', 'svg']:
    fig.savefig(out/f'portfolio_ablation_2025.{extension}', dpi=180)
plt.close(fig)
print(summary[['Strategy', 'Variant', 'TotalPriceReturn_pct', 'MaxDrawdown_pct']].to_string(index=False))
