"""Offline 5/20/60-session momentum baselines on saved evaluation samples.

Run: python3 run_momentum_comparison.py
Uses the original accounting functions, without retraining or selecting a winner.
"""
import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'experiments/momentum_baselines'
SOURCE = ROOT / 'experiments/sector_ranking/03_walk_forward/run_01/eval_scores.csv'


def main():
    OUT.mkdir(exist_ok=True)
    # Execute only the project's original accounting functions: no model imports.
    tree = ast.parse((ROOT / 'plot_portfolio.py').read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ('rebalance', 'simulate')]
    assert len(functions) == 2
    namespace = {'np': np, 'pd': pd}
    exec(compile(ast.Module(body=functions, type_ignores=[]), 'plot_portfolio.py', 'exec'), namespace)
    simulate = namespace['simulate']
    bars = pd.read_csv(ROOT / 'data/daily_prices.csv', parse_dates=['Date'])
    assert not bars.duplicated(['Date', 'Ticker']).any()
    close = bars.pivot(index='Date', columns='Ticker', values='Close').sort_index()
    opens = bars.pivot(index='Date', columns='Ticker', values='Open').sort_index()
    assert close.notna().all().all() and opens.notna().all().all()
    samples = pd.read_csv(SOURCE, parse_dates=['SignalDate', 'EntryDate', 'ExitDate'])
    test = pd.read_csv(ROOT / 'results/ridge/predictions.csv',
                       parse_dates=['SignalDate', 'EntryDate', 'ExitDate'])
    test = test.loc[test.Split.eq('test')].copy()
    test['Window'] = '2025_exploratory'
    samples = pd.concat([samples, test], ignore_index=True)
    assert not samples.duplicated(['SignalDate', 'Ticker']).any()
    for horizon in (5, 20, 60):
        panel = close / close.shift(horizon) - 1
        samples[f'momentum_{horizon}'] = [panel.at[r.SignalDate, r.Ticker]
                                         for r in samples.itertuples()]
        # Independently recompute from truncated history (future data cannot enter).
        for r in samples.itertuples():
            history = close.loc[:r.SignalDate, r.Ticker]
            expected = history.iloc[-1] / history.iloc[-horizon-1] - 1
            np.testing.assert_allclose(panel.at[r.SignalDate, r.Ticker], expected)
    assert np.isfinite(samples[[f'momentum_{h}' for h in (5, 20, 60)]]).all().all()
    reference = samples.return_5.fillna(samples.return_5_baseline_score)
    np.testing.assert_allclose(samples.momentum_5, reference, rtol=1e-10, atol=1e-12)
    actual = [opens.at[r.ExitDate, r.Ticker] / opens.at[r.EntryDate, r.Ticker] - 1
              for r in samples.itertuples()]
    np.testing.assert_allclose(actual, samples.TargetPriceReturn, rtol=1e-10, atol=1e-12)
    weekly, portfolios = [], []
    for window, frame in samples.groupby('Window'):
        schedule = frame[['SignalDate', 'EntryDate', 'ExitDate']].drop_duplicates().sort_values('EntryDate')
        assert np.array_equal(schedule.ExitDate.iloc[:-1], schedule.EntryDate.iloc[1:])
        prices = opens.loc[schedule.EntryDate.min():schedule.ExitDate.max()]
        strategies = {f'momentum_{h}': f'momentum_{h}' for h in (5, 20, 60)}
        if window != '2025_exploratory':
            strategies.update({m: f'{m}_score' for m in ('A', 'B', 'C', 'D')})
        for method, col in strategies.items():
            targets = []
            for date, group in frame.groupby('SignalDate'):
                assert len(group) == len(close.columns)
                chosen = group.sort_values([col, 'Ticker'], ascending=[False, True]).Ticker.iloc[:3]
                truth = group.sort_values(['TargetPriceReturn', 'Ticker'], ascending=[False, True]).Ticker.iloc[:3]
                ic = group[col].rank().corr(group.TargetPriceReturn.rank())
                weekly.append(dict(Window=window, SignalDate=date, Method=method,
                                   RankIC=ic, Top3Overlap=len(set(chosen) & set(truth))/3))
                targets.append(pd.Series({t: float(t in set(chosen))/3 for t in close.columns},
                                         name=group.EntryDate.iloc[0]))
            weights = pd.DataFrame(targets)[prices.columns]
            for cost in (0, 5, 10, 20):
                curve, trades = simulate(prices, weights, cost)
                portfolios.append(dict(Window=window, Method=method, CostBps=cost,
                                       NetPriceReturn=curve.NAV.iloc[-1]-1,
                                       MaxDrawdown=curve.Drawdown.min(), Fees=trades.Fee.sum()))
        equal = pd.DataFrame(1/len(prices.columns), index=schedule.EntryDate, columns=prices.columns)
        for cost in (0, 5, 10, 20):
            curve, trades = simulate(prices, equal, cost)
            portfolios.append(dict(Window=window, Method='equal_weight', CostBps=cost,
                                   NetPriceReturn=curve.NAV.iloc[-1]-1,
                                   MaxDrawdown=curve.Drawdown.min(), Fees=trades.Fee.sum()))
    weekly = pd.DataFrame(weekly)
    metrics = weekly.groupby(['Window', 'Method']).agg(
        Weeks=('RankIC', 'size'), MeanRankIC=('RankIC', 'mean'),
        Top3Overlap=('Top3Overlap', 'mean')).reset_index()
    portfolios = pd.DataFrame(portfolios)
    # Verify the reproduced original baseline and model portfolios match saved outputs.
    original = pd.read_csv(ROOT / 'experiments/sector_ranking/04_portfolio_comparison/run_01/summary.csv')
    for row in portfolios.loc[portfolios.Window.ne('2025_exploratory')].itertuples():
        name = 'return_5_baseline' if row.Method == 'momentum_5' else row.Method
        if name not in ['return_5_baseline', 'A', 'B', 'C', 'D']:
            continue
        old = original.loc[original.Window.eq(row.Window) & original.Strategy.eq(name)
                           & original.CostBps.eq(row.CostBps)]
        assert len(old) == 1
        np.testing.assert_allclose(row.NetPriceReturn, old.TotalPriceReturn.iloc[0], atol=1e-10)
    for name, table in [('scores', samples), ('weekly_metrics', weekly),
                        ('ranking_summary', metrics), ('portfolio_summary', portfolios)]:
        table.to_csv(OUT / f'{name}.csv', index=False)
    manifest = dict(status='retrospective_exploratory', horizons=[5, 20, 60],
                    convention='close_t / close_t_minus_h - 1; signal close, next session open entry',
                    holdings='Top 3, each 1/3; weekly rebalance; terminal liquidation',
                    returns='price only; cash dividends excluded',
                    tests='truncated-history scores, original 5d feature, open labels, original portfolios reconciled',
                    hashes={str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in [SOURCE, ROOT/'data/daily_prices.csv', ROOT/'plot_portfolio.py', Path(__file__)]})
    (OUT / 'experiment.json').write_text(json.dumps(manifest, indent=2))
    print(metrics.to_string(index=False))
    print(portfolios.loc[portfolios.CostBps.eq(10)].to_string(index=False))


if __name__ == '__main__':
    main()
