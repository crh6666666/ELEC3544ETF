"""Backtest frozen walk-forward scores under common holdings and cost rules."""
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
from plot_portfolio import simulate
from prepare_data import DATE_COLUMNS, digest

walk = import_module('experiments.sector_ranking.03_walk_forward.run')
COSTS = (0, 5, 10, 20)
MAIN_COST = 10
EQUAL = 'equal_weight_11'
POSITIVE = 'B_positive'
MATCHED = 'B_cash_matched'
PRIMARY = (*walk.METHODS, EQUAL)
KEYS = ['Window', 'SignalDate']
DAILY = ROOT / 'data/daily_prices.csv'


def load_scores(source):
    manifest = json.loads((source / 'experiment.json').read_text())
    for path, sha in manifest['input_sha256'].items():
        if not (ROOT / path).is_file() or digest(ROOT / path) != sha:
            raise ValueError(f'Step 3 input changed or is missing: {path}; inspect and rerun Step 3')
    for name in ('eval_scores.csv', 'weekly_metrics.csv', 'phase_audit.csv'):
        if digest(source / name) != manifest['output_sha256'][name]:
            raise ValueError(f'Step 3 output changed: {name}')
    time_cols = ['ScoreAsOfUTC', 'FitLabelsThroughUTC', 'SelectionLabelsThroughUTC']
    scores = pd.read_csv(source / 'eval_scores.csv', parse_dates=[*DATE_COLUMNS, 'NewFeatureMaxDate', *time_cols])
    dates = ['SignalDate', 'SignalCloseUTC', 'EntryDate', 'EntryOpenUTC', 'ExitDate', 'ExitOpenUTC']
    approved = pd.read_csv(source / 'phase_audit.csv', parse_dates=dates)
    approved = approved.loc[approved.phase.eq('evaluation') & approved.kept, ['window', *dates]]
    observed = scores[['Window', *dates]].drop_duplicates().rename(columns={'Window': 'window'})
    pd.testing.assert_frame_equal(observed.sort_values('SignalDate').reset_index(drop=True),
                                  approved.sort_values('SignalDate').reset_index(drop=True))
    checks = (scores.FitLabelsThroughUTC.lt(scores.SelectionLabelsThroughUTC)
              & scores.SelectionLabelsThroughUTC.lt(scores.ScoreAsOfUTC)
              & scores.FeatureMaxDate.le(scores.SignalDate) & scores.NewFeatureMaxDate.le(scores.SignalDate)
              & scores.ScoreAsOfUTC.eq(scores.SignalCloseUTC) & scores.ScoreAsOfUTC.lt(scores.EntryOpenUTC)
              & scores.EntryOpenUTC.lt(scores.ExitOpenUTC))
    if (not checks.all() or scores.duplicated(['SignalDate', 'Ticker']).any()
            or not scores.groupby(KEYS).Ticker.agg(lambda tickers: set(tickers) == set(TICKERS)).all()
            or not np.isfinite(scores[[f'{name}_score' for name in walk.METHODS]]).all().all()):
        raise ValueError('Scores fail the ETF universe, timing or finiteness checks')
    return scores, manifest


def target_weights(scores):
    """Return target percentages on each signal and the original scores for audit."""
    holdings = []
    for name in walk.METHODS:
        col = f'{name}_score'
        ordered = scores.sort_values([*KEYS, col, 'Ticker'], ascending=[True, True, False, True])
        top = ordered.groupby(KEYS).cumcount().lt(3)
        holdings.append(ordered.assign(Strategy=name, Weight=top.astype(float)/3, Score=ordered[col]))
    positive = scores.B_score.gt(0)
    invested = positive.groupby([scores.Window, scores.SignalDate]).transform('mean')
    holdings.extend([scores.assign(Strategy=EQUAL, Weight=1/len(TICKERS), Score=np.nan),
                     scores.assign(Strategy=POSITIVE, Weight=positive.astype(float)/len(TICKERS),
                                   Score=scores.B_score),
                     scores.assign(Strategy=MATCHED, Weight=invested/len(TICKERS), Score=np.nan)])
    cols = [*KEYS, 'EntryDate', 'ExitDate', 'ScoreAsOfUTC', 'EntryOpenUTC', 'ExitOpenUTC',
            'Strategy', 'Ticker', 'Score', 'Weight']
    result = pd.concat(holdings, ignore_index=True)[cols]
    targets = result.groupby([*KEYS, 'Strategy']).Weight.agg(['sum', lambda w: w.gt(0).sum()])
    invested = targets['sum'].unstack('Strategy')
    np.testing.assert_allclose(invested[list(PRIMARY)], 1, atol=1e-12)
    np.testing.assert_allclose(invested[POSITIVE], invested[MATCHED], atol=1e-12)
    top_counts = targets['<lambda_0>'].unstack('Strategy')
    if not top_counts[list(walk.METHODS)].eq(3).all().all() or not top_counts[EQUAL].eq(len(TICKERS)).all():
        raise ValueError('Unexpected number of eligible holdings')
    result['InvestedWeight'] = result.groupby([*KEYS, 'Strategy']).Weight.transform('sum')
    return result


def position_ledger(prices, weights, curve, trades, cost_bps):
    """Reconcile each original simulator trade with asset-level changes in units."""
    shares = np.zeros(len(TICKERS))
    rows, pretrade = [], []
    last = prices.index[-1]
    for event in trades.itertuples():
        date = event.Date
        opens = prices.loc[date].to_numpy()
        target = np.zeros(len(TICKERS)) if date == last else weights.loc[date].to_numpy()
        nav = curve.at[date, 'NAV']
        new_shares = nav * target / opens
        notional = (new_shares - shares) * opens
        fees = np.abs(notional) * cost_bps / 10000
        np.testing.assert_allclose([np.abs(notional).sum(), fees.sum(), 1-target.sum()],
                                   [event.TradedNotional, event.Fee, event.CashWeight], atol=1e-12)
        before = nav + event.Fee
        pretrade.append(before)
        rows.append(pd.DataFrame(dict(Date=date, Ticker=TICKERS, Open=opens, SharesBefore=shares,
                                      SharesAfter=new_shares, Weight=target, SignedNotional=notional, Fee=fees)))
        shares = new_shares
    np.testing.assert_array_equal(shares, 0)
    trades = trades.assign(NAVBeforeTrade=pretrade)
    trades['TwoWayTurnover'] = trades.TradedNotional / trades.NAVBeforeTrade
    return pd.concat(rows, ignore_index=True), trades


def simulate_windows(scores, holdings):
    opens = pd.read_csv(DAILY, parse_dates=['Date']).pivot(index='Date', columns='Ticker', values='Open')
    schedule = scores[[*KEYS, 'EntryDate', 'ExitDate']].drop_duplicates().sort_values(['Window', 'EntryDate'])
    if schedule.duplicated(['Window', 'SignalDate']).any() or schedule.EntryDate.duplicated().any():
        raise ValueError('Inconsistent or duplicate holding dates')
    prev_exit = schedule.groupby('Window').ExitDate.shift()
    if prev_exit.gt(schedule.EntryDate).any():
        raise ValueError('Overlapping holding intervals within a window')
    schedule['Segment'] = (prev_exit.ne(schedule.EntryDate)).groupby(schedule.Window).cumsum()
    holdings = holdings.merge(schedule, on=[*KEYS, 'EntryDate', 'ExitDate'], validate='many_to_one')
    curves, ledgers, positions, summaries, spot_checks = [], [], [], [], []
    for (window, segment), weeks in schedule.groupby(['Window', 'Segment']):
        start, end = weeks.EntryDate.iloc[0], weeks.ExitDate.iloc[-1]
        sessions = CAL.sessions_in_range(start, end).rename('Date')
        prices = opens.reindex(index=sessions, columns=TICKERS)
        labels = scores.loc[scores.SignalDate.isin(weeks.SignalDate)]
        realized = np.array([prices.at[r.ExitDate, r.Ticker]/prices.at[r.EntryDate, r.Ticker]-1
                             for r in labels.itertuples()])
        np.testing.assert_allclose(realized, labels[walk.ranking.RETURN], rtol=1e-10, atol=1e-12)
        sample = holdings.loc[holdings.Window.eq(window) & holdings.Segment.eq(segment)]
        for strategy, strategy_holdings in sample.groupby('Strategy', sort=False):
            weights = strategy_holdings.pivot(index='EntryDate', columns='Ticker', values='Weight')[TICKERS]
            for cost in COSTS:
                curve, trades = simulate(prices, weights, cost)
                tickets, trades = position_ledger(prices, weights, curve, trades, cost)
                context = dict(Window=window, Segment=segment, Strategy=strategy, CostBps=cost)
                trades['Event'] = 'rebalance'
                trades.loc[trades.Date.eq(start), 'Event'] = 'initial_buy'
                trades.loc[trades.Date.eq(end), 'Event'] = 'final_liquidation'
                curves.append(curve.reset_index().assign(**context))
                ledgers.append(trades.assign(**context))
                positions.append(tickets.assign(**context))
                summaries.append(dict(**context, StartOpen=start, EndOpen=end, Weeks=len(weeks),
                                      TotalPriceReturn=curve.CumulativeReturn.iloc[-1],
                                      MaxDrawdown=curve.Drawdown.min(),
                                      TradedNotionalInInitialNAVUnits=trades.TradedNotional.sum(),
                                      TwoWayTurnover=trades.TwoWayTurnover.sum(),
                                      FeesInInitialNAVUnits=trades.Fee.sum()))
                if cost == MAIN_COST:
                    events = {'first_week': start, 'last_week': weeks.EntryDate.iloc[-1], 'final_exit': end}
                    if len(weeks) > 1:
                        events['first_rebalance'] = weeks.EntryDate.iloc[1]
                    for event, date in events.items():
                        traded = trades.loc[trades.Date.eq(date)].iloc[0]
                        held = strategy_holdings.loc[strategy_holdings.EntryDate.eq(date)]
                        spot_checks.append(dict(**context, Check=event, Date=date,
                                                SignalDate=held.SignalDate.min(), ScoreAsOfUTC=held.ScoreAsOfUTC.min(),
                                                HeldETFs=' '.join(sorted(held.loc[held.Weight.gt(0), 'Ticker'])),
                                                InvestedWeight=held.Weight.sum(), CashWeight=traded.CashWeight,
                                                TradedNotional=traded.TradedNotional, Fee=traded.Fee))
    return dict(weekly_holdings=holdings, daily_nav=pd.concat(curves, ignore_index=True),
                trades=pd.concat(ledgers, ignore_index=True), asset_trades=pd.concat(positions, ignore_index=True),
                summary=pd.DataFrame(summaries), spot_checks=pd.DataFrame(spot_checks))


def comparison_tables(summary):
    primary = summary.loc[summary.Strategy.isin(PRIMARY)].copy()
    primary['ReturnRank'] = primary.groupby(['Window', 'Segment', 'CostBps']).TotalPriceReturn.rank(
        ascending=False, method='min').astype(int)
    cash = summary.loc[summary.Strategy.isin([POSITIVE, MATCHED, EQUAL])].copy()
    cash['DrawdownReductionVsFullyInvested'] = (
        cash.MaxDrawdown - cash.groupby(['Window', 'Segment', 'CostBps']).MaxDrawdown.transform(
            lambda values: values[cash.loc[values.index, 'Strategy'].eq(EQUAL)].iloc[0]))
    cash_wide = cash.pivot(index=['Window', 'Segment', 'CostBps'], columns='Strategy',
                           values=['TotalPriceReturn', 'MaxDrawdown'])
    differences = pd.DataFrame({
        'PositiveMinusCashMatchedReturn': cash_wide['TotalPriceReturn'][POSITIVE] - cash_wide['TotalPriceReturn'][MATCHED],
        'PositiveMinusCashMatchedDrawdown': cash_wide['MaxDrawdown'][POSITIVE] - cash_wide['MaxDrawdown'][MATCHED],
        'PositiveMinusFullyInvestedDrawdown': cash_wide['MaxDrawdown'][POSITIVE] - cash_wide['MaxDrawdown'][EQUAL],
        'CashMatchedMinusFullyInvestedDrawdown': cash_wide['MaxDrawdown'][MATCHED] - cash_wide['MaxDrawdown'][EQUAL],
    }).reset_index()
    return dict(top3_comparison=primary.loc[primary.CostBps.eq(MAIN_COST)],
                cost_sensitivity=primary, cash_matching=cash, cash_differences=differences)


def run(source, output):
    scores, source_config = load_scores(source)
    tables = simulate_windows(scores, target_weights(scores))
    tables.update(comparison_tables(tables['summary']))
    output.mkdir()
    for name, table in tables.items():
        table.to_csv(output / f'{name}.csv', index=False)
    inputs = [source / name for name in ('experiment.json', 'eval_scores.csv', 'phase_audit.csv', 'weekly_metrics.csv')]
    inputs.extend([DAILY, Path(__file__), STEP / 'DESIGN.md', ROOT / 'plot_portfolio.py'])
    config = dict(generated_at_utc=datetime.now(timezone.utc).isoformat(),
                  status='retrospective_price_return_simulation_hypothetical_costs', scored_2025=False,
                  source_snapshot_id=source_config['source_snapshot_id'],
                  input_sha256={str(p.relative_to(ROOT)): digest(p) for p in inputs},
                  output_sha256={p.name: digest(p) for p in output.iterdir()},
                  source_features={method: source_config['selected']['W1'][method]['features']
                                   for method in walk.medium.GROUPS},
                  model_parameters_by_window=source_config['selected'], source_seed=source_config['seed'],
                  strategy_rules=dict(top3='100% split equally among top three; ticker ascending on ties',
                                      equal_weight='100% split equally among 11 ETFs',
                                      positive='B score > 0 gives 1/11 weight, remainder cash',
                                      matched='B positive strategy total weight spread across 11 ETFs'),
                  costs_bps=COSTS, main_cost_bps=MAIN_COST, initial_nav=1, fractional_shares=True,
                  valuation='split-adjusted daily Open, excluding cash dividends',
                  windows=tables['summary'][['Window', 'Segment', 'StartOpen', 'EndOpen', 'Weeks']]
                  .drop_duplicates().astype(str).to_dict('records'),
                  cash_interest=0, rebalance_on='EntryDate open', liquidate_at='last ExitDate open',
                  turnover='sum of absolute asset notional / pre-trade NAV, both sides, including entry and exit')
    (output / 'experiment.json').write_text(json.dumps(config, indent=2) + '\n')
    print(tables['top3_comparison'][['Window', 'Strategy', 'TotalPriceReturn', 'MaxDrawdown']].to_string(index=False))
    print(tables['cash_differences'].loc[lambda frame: frame.CostBps.eq(MAIN_COST)].to_string(index=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-run', default='run_01')
    parser.add_argument('--run-name', default='run_01')
    args = parser.parse_args()
    source = (walk.STEP / args.source_run).resolve()
    output = (STEP / args.run_name).resolve()
    if source.parent != walk.STEP or output.parent != STEP or output.exists():
        parser.error('Choose a Step 3 run and a new output folder under Step 4.')
    run(source, output)
