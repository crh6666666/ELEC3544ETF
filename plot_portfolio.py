"""Daily open-marked, price-only portfolio illustration from saved predictions."""
import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'elec3544etf_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from calendar_utils import CAL, TICKERS
from models.common import snapshot_id
from price_basis import audit

ROOT = Path(__file__).resolve().parent
MODELS = ('ridge', 'random_forest', 'xgboost')
STYLES = {
    'spy': ('SPY buy & hold', '#475569'),
    'equal_weight': ('Sector equal weight', '#159b78'),
    'ridge': ('Ridge + cash', '#8745ec'),
    'random_forest': ('Random forest + cash', '#e18013'),
    'xgboost': ('XGBoost + cash', '#148ac0'),
}


def load_inputs(results, names):
    """Reject stale/misaligned predictions instead of silently mixing experiments."""
    report = audit(ROOT, write=False)
    hashes = {str(Path(p).relative_to(ROOT)): h for p, h in report['input_hashes'].items()}
    source_id = snapshot_id(hashes)
    frames, reference = {}, None
    for name in names:
        frame = pd.read_csv(results/name/'predictions.csv',
                            parse_dates=['SignalDate', 'EntryDate', 'ExitDate', 'FeatureMaxDate'])
        frame = frame.loc[frame.Split == 'test'].sort_values(['SignalDate', 'Ticker']).reset_index(drop=True)
        if frame.empty or frame.duplicated(['SignalDate', 'Ticker']).any():
            raise ValueError(f'{name}: empty/duplicate test predictions')
        if not np.isfinite(frame[['Prediction', 'TargetPriceReturn']]).all().all():
            raise ValueError(f'{name}: nonfinite predictions or labels')
        if not frame.SourceSnapshotId.eq(source_id).all():
            raise ValueError(f'{name}: source snapshot changed; rerun model first')
        if not frame.groupby('SignalDate').Ticker.agg(lambda x: set(x) == set(TICKERS)).all():
            raise ValueError(f'{name}: incomplete ETF universe')
        if not ((frame.FeatureMaxDate <= frame.SignalDate) &
                (frame.SignalDate < frame.EntryDate) & (frame.EntryDate < frame.ExitDate)).all():
            raise ValueError(f'{name}: invalid feature/signal/trade dates')
        available = pd.to_datetime(frame.ModelAvailableAsOf, utc=True)
        signals = pd.to_datetime(frame.SignalCloseUTC, utc=True)
        if not (available <= signals).all():
            raise ValueError(f'{name}: model selected after signal')
        actual = frame[['SignalDate', 'EntryDate', 'ExitDate', 'Ticker',
                        'TargetPriceReturn', 'SourceSnapshotId']]
        if reference is not None:
            pd.testing.assert_frame_equal(reference, actual, rtol=1e-12, atol=1e-14)
        reference = actual
        frames[name] = frame

    schedule = reference[['SignalDate', 'EntryDate', 'ExitDate']].drop_duplicates().sort_values('EntryDate')
    if schedule.EntryDate.duplicated().any() or not np.array_equal(
            schedule.ExitDate.iloc[:-1].to_numpy(), schedule.EntryDate.iloc[1:].to_numpy()):
        raise ValueError('Weekly holding periods must be unique and contiguous')
    daily = pd.read_csv(ROOT/'data/daily_prices.csv', parse_dates=['Date'])
    spy = pd.read_csv(ROOT/'data/spy_daily_prices.csv', parse_dates=['Date'])
    daily = pd.concat([daily, spy], ignore_index=True)
    if daily.duplicated(['Date', 'Ticker']).any():
        raise ValueError('Duplicate daily price rows')
    dates = CAL.sessions_in_range(schedule.EntryDate.min(), schedule.ExitDate.max())
    prices = daily.pivot(index='Date', columns='Ticker', values='Open').reindex(
        index=dates, columns=[*TICKERS, 'SPY'])
    if not np.isfinite(prices).all().all() or not (prices > 0).all().all():
        raise ValueError('Missing/nonpositive open prices during simulation')
    for name, frame in frames.items():
        actual_returns = np.array([prices.loc[r.ExitDate, r.Ticker]/prices.loc[r.EntryDate, r.Ticker]-1
                                   for r in frame.itertuples()])
        if not np.allclose(actual_returns, frame.TargetPriceReturn, rtol=1e-10, atol=1e-12):
            raise ValueError(f'{name}: label returns disagree with price data')
    return frames, schedule, prices, hashes


def rebalance(holdings, cash, weights, cost_rate):
    """Target weights apply to NAV after fees; solve the self-financing trade."""
    nav = float(holdings.sum() + cash)
    initial_turnover = float(np.abs(nav*weights-holdings).sum())
    if cost_rate == 0 or initial_turnover == 0:
        return nav*weights, nav*(1-float(weights.sum())), initial_turnover, 0.0
    # fee = rate * sum(abs(post-fee target holdings - current holdings)).
    lo, hi = 0.0, nav
    for _ in range(60):
        fee = (lo+hi)/2
        turnover = float(np.abs((nav-fee)*weights-holdings).sum())
        if fee < cost_rate*turnover:
            lo = fee
        else:
            hi = fee
    fee = (lo+hi)/2 if cost_rate else 0.0
    target = (nav-fee)*weights
    turnover = float(np.abs(target-holdings).sum())
    cash = (nav-fee)*(1-float(weights.sum()))
    return target, cash, turnover, fee


def simulate(prices, targets, cost_bps):
    """Mark positions at each daily open; trade only at specified opens.

    Initial NAV=1, no leverage, fractional shares, zero cash interest. All remaining
    holdings are liquidated on the last open, including benchmark holdings.
    """
    if not np.isfinite(cost_bps) or not 0 <= cost_bps < 10000:
        raise ValueError('cost_bps must be finite and between 0 and 10000 (exclusive)')
    if prices.empty or not prices.index.is_unique or not prices.index.is_monotonic_increasing:
        raise ValueError('Prices need a nonempty unique increasing date index')
    if not np.isfinite(prices).all().all() or not (prices > 0).all().all():
        raise ValueError('Invalid price values')
    if not targets.index.is_unique or not targets.index.isin(prices.index).all():
        raise ValueError('Trade dates must be unique observed opens')
    if not targets.columns.equals(prices.columns):
        raise ValueError('Weight/price asset columns do not match')
    if (not np.isfinite(targets).all().all() or (targets < 0).any().any()
            or (targets.sum(axis=1) > 1+1e-12).any()):
        raise ValueError('Weights must be finite, nonnegative and sum to at most one')
    holdings, cash = np.zeros(len(prices.columns)), 1.0
    navs, trades = [], []
    values = prices.to_numpy()
    for i, date in enumerate(prices.index):
        if i:
            holdings *= values[i]/values[i-1]
        weights = None
        if date == prices.index[-1]:
            weights = np.zeros(len(holdings))
        elif date in targets.index:
            weights = targets.loc[date].to_numpy(dtype=float)
        if weights is not None:
            holdings, cash, turnover, fee = rebalance(holdings, cash, weights, cost_bps/10000)
            trades.append(dict(Date=date, TradedNotional=turnover, Fee=fee,
                               CashWeight=cash/(holdings.sum()+cash)))
        navs.append(float(holdings.sum()+cash))
    nav = pd.Series(navs, index=prices.index, name='NAV')
    # Include pre-trade initial capital in the high-water mark (initial fee counts).
    peak = np.maximum.accumulate(np.r_[1.0, nav.to_numpy()])[1:]
    curve = pd.DataFrame({'NAV': nav, 'CumulativeReturn': nav-1, 'Drawdown': nav/peak-1})
    return curve, pd.DataFrame(trades)


def plot_curves(curves, output, stem, cost_bps):
    fig, (ax, dd) = plt.subplots(2, 1, figsize=(15, 9), sharex=True,
                                gridspec_kw={'height_ratios': [2, 1], 'hspace': .06})
    for name, curve in curves.items():
        label, color = STYLES[name]
        ax.plot(curve.index, curve.CumulativeReturn, label=label, color=color, lw=1.8)
        dd.plot(curve.index, curve.Drawdown, color=color, lw=1.6)
    for panel in (ax, dd):
        panel.yaxis.set_major_formatter(PercentFormatter(1))
        panel.grid(alpha=.20)
        panel.spines[['top', 'right']].set_visible(False)
        panel.axhline(0, color='#94a3b8', lw=.6)
    first = next(iter(curves.values()))
    ax.set_title(f'{first.index[0].year} exploratory simulation | fixed models, hypothetical costs',
                 loc='left', fontsize=17, weight='bold', pad=14)
    ax.set_ylabel('Cumulative price return')
    dd.set_ylabel('Drawdown')
    ax.legend(loc='upper left', ncol=2, frameon=False)
    dd.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    dd.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    dd.set_xlim(first.index[0], first.index[-1])
    fig.subplots_adjust(left=.085, right=.985, top=.92, bottom=.15)
    fig.text(.085, .075, f'{first.index[0]:%Y-%m-%d} to {first.index[-1]:%Y-%m-%d} | '
             f'Daily open valuations | One-way cost: {cost_bps:g} bps, including entry and final exit',
             fontsize=9, color='#475569')
    fig.text(.085, .045, 'Models: 1/11 per ETF when prediction > 0, otherwise cash; weekly rebalance. '
             'Cash interest = 0. Price returns exclude dividends.', fontsize=9, color='#475569')
    fig.text(.085, .02, 'Retrospective illustration using previously inspected test data; '
             'not an independent holdout or executable trading performance.', fontsize=9, color='#475569')
    for suffix in ('png', 'svg'):
        path = output/f'{stem}.{suffix}'
        fig.savefig(path, dpi=180)
        print(path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, default=ROOT/'results')
    parser.add_argument('--output', type=Path, default=ROOT/'figures')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--cost-bps', type=float, default=10.0,
                        help='Hypothetical one-way cost per traded notional; default 10 bps.')
    args = parser.parse_args()
    if not np.isfinite(args.cost_bps) or not 0 <= args.cost_bps < 10000:
        parser.error('--cost-bps must be finite and between 0 and 10000 (exclusive)')
    out = args.output.resolve()
    protected = [ROOT/p for p in ('data', 'prepared_data', 'models', '.venv')]
    if out == ROOT or any(out.is_relative_to(p) or p.is_relative_to(out) for p in protected):
        parser.error('Choose a separate figure output directory.')
    names = list(dict.fromkeys(args.models))
    frames, schedule, prices, hashes = load_inputs(args.results, names)
    targets = {}
    targets['spy'] = pd.DataFrame(0.0, index=pd.DatetimeIndex([prices.index[0]]), columns=prices.columns)
    targets['spy'].loc[:, 'SPY'] = 1.0
    targets['equal_weight'] = pd.DataFrame(0.0, index=pd.DatetimeIndex(schedule.EntryDate), columns=prices.columns)
    targets['equal_weight'].loc[:, TICKERS] = 1/len(TICKERS)
    for name in names:
        weights = (frames[name].pivot(index='EntryDate', columns='Ticker', values='Prediction') > 0).astype(float)/len(TICKERS)
        targets[name] = weights.reindex(columns=prices.columns, fill_value=0.0)
    curves, trades, summaries = {}, [], []
    for name, weights in targets.items():
        curve, ledger = simulate(prices, weights, args.cost_bps)
        curves[name] = curve
        trades.append(ledger.assign(Strategy=name))
        summaries.append(dict(Strategy=name, TotalPriceReturn=curve.CumulativeReturn.iloc[-1],
                              MaxDrawdown=curve.Drawdown.min(),
                              FeesInInitialNAVUnits=ledger.Fee.sum(),
                              TradedNotionalInInitialNAVUnits=ledger.TradedNotional.sum()))
    out.mkdir(parents=True, exist_ok=True)
    year = prices.index[0].year
    suffix = '' if names == list(MODELS) else '_'+ '_'.join(names)
    stem = f'portfolio_simulation_{year}{suffix}'
    plot_curves(curves, out, stem, args.cost_bps)
    pd.concat([c.assign(Strategy=n) for n, c in curves.items()]).rename_axis('Date').reset_index().to_csv(out/f'{stem}_curves.csv', index=False)
    pd.concat(trades, ignore_index=True).to_csv(out/f'{stem}_trades.csv', index=False)
    pd.DataFrame(summaries).to_csv(out/f'{stem}_summary.csv', index=False)
    manifest = dict(status='retrospective_exploratory_previously_inspected_test',
                    start=str(prices.index[0].date()), end=str(prices.index[-1].date()),
                    valuation='daily split-adjusted Open; excludes cash dividends',
                    model_rule='each ETF: weight 1/11 if Prediction > 0, else cash; weekly rebalance',
                    benchmark_rules='SPY buy and hold; sector equal weight rebalanced weekly',
                    cost_bps_one_way=args.cost_bps, charge_initial_and_final_trades=True,
                    cash_interest=0, fractional_shares=True, leverage=False,
                    source_hashes=hashes,
                    predictions_sha256={n: hashlib.sha256((args.results/n/'predictions.csv').read_bytes()).hexdigest() for n in names},
                    note='Rules are illustrative, not tuned to maximize the displayed test return. '
                         'No intraday execution, dividend account, tax or independent source certification.')
    (out/f'{stem}_assumptions.json').write_text(json.dumps(manifest, indent=2))
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == '__main__':
    main()
