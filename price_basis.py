"""Fail-closed reconciliation of the frozen Yahoo PRICE-only snapshot.

This verifies source consistency, not a historical cash-dividend account or
independent exchange-price certification. No price or dividend is rewritten.
"""
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_utils import CAL, TICKERS

BASIS = 'source_split_adjusted_open_price_return_excludes_cash_dividends'
STATUS = 'SNAPSHOT_RECONCILED_PRICE_ONLY'
SHARE_UNIT = 'source_split_adjusted_share_units_not_historical_actual_shares'
CONTRACT = dict(version='price_only_v3', target_basis=BASIS,
    feature_prices='source Close; Volume used only as a within-window ratio',
    label_prices='source Open', risk_prices='same open-to-open price returns',
    account_prices='source Open', benchmark_prices='source Open',
    adjusted_close_usage='audit only; never a model or account price',
    cash_dividends_included=False, share_unit=SHARE_UNIT,
    payment_dates_verified=False, historical_point_in_time_quotes=False,
    split_handling='source OHLC/Volume coordinate; audited known splits; never adjust twice')


def require(condition, message):
    if not bool(condition):
        raise ValueError('Price-basis audit failed: ' + message)


def read_raw(path):
    result = json.loads(path.read_text())['chart']['result'][0]
    ticker = result['meta']['symbol']
    dates = pd.to_datetime(result['timestamp'], unit='s', utc=True).tz_convert(
        'America/New_York').strftime('%Y-%m-%d')
    quote = result['indicators']['quote'][0]
    bars = pd.DataFrame({k.title(): quote[k] for k in
                         ['open', 'high', 'low', 'close', 'volume']})
    bars['Adj Close'] = result['indicators']['adjclose'][0]['adjclose']
    bars['Date'] = dates
    bars['Ticker'] = ticker
    bars['Currency'] = result['meta']['currency']
    events = []
    for kind, entries in result.get('events', {}).items():
        for key, event in entries.items():
            events.append(dict(Ticker=ticker, EventType=kind, EventKey=str(key), **event))
    return bars, events


def reconcile_factors(bars, events):
    """Reconcile *relative* AdjClose/Close factors, including batch seams.

    Absolute factors also contain distributions after this study's end and
    cannot be reconstructed from the study-period events alone.
    """
    audit_rows = []
    for ticker, group in bars.groupby('Ticker'):
        group = group.sort_values('Date').set_index('Date')
        factor = group['Adj Close'] / group.Close
        previous_close = group.Close.shift()
        amounts = pd.Series(0., index=group.index)
        selected = events[(events.Ticker == ticker) & (events.EventType == 'dividends')]
        for event in selected.itertuples():
            require(event.Date in group.index, f'{ticker}: event missing trading bar {event.Date}')
            require(event.amount > 0, f'{ticker}: nonpositive dividend')
            amounts.loc[event.Date] += event.amount
        require((amounts.iloc[1:] < previous_close.iloc[1:]).all(), f'{ticker}: invalid dividend units')
        actual = factor / factor.shift()
        expected = 1 / (1 - amounts / previous_close)
        errors = (actual - expected).abs()
        require(errors.iloc[1:].max() <= 2e-6,
                f'{ticker}: unexplained adjustment-factor change (including batch boundary)')
        for event in selected.itertuples():
            require(pd.notna(previous_close.loc[event.Date]), f'{ticker}: no pre-event close')
            implied = previous_close.loc[event.Date] * (1 - 1 / actual.loc[event.Date])
            require(np.isclose(implied, event.amount, rtol=3e-4, atol=2e-5),
                    f'{ticker}: dividend/price unit mismatch at {event.Date}')
            audit_rows.append(dict(Ticker=ticker, ExDate=event.Date,
                SourceDividend=float(event.amount), ImpliedDividend=float(implied),
                RelativeError=float(implied/event.amount-1),
                PreviousClose=float(previous_close.loc[event.Date]),
                FactorRatio=float(actual.loc[event.Date]),
                ExpectedFactorRatio=float(expected.loc[event.Date]),
                PaymentDateVerified=False))
    return pd.DataFrame(audit_rows)


def reconcile_splits(bars, events, evidence):
    """Accept only issuer-documented splits on already-adjusted source bars.

    Continuity is a sanity check, not an independent historical quote source.
    A large split-day gap requires manual review rather than automatic repair.
    """
    known = evidence.get('issuer_documented_split',evidence.get('issuer_post_study_split'))
    selected = events[events.EventType == 'splits']
    in_range = bars.Date.min() <= known['effective_date'] <= bars.Date.max()
    expected = {(t,known['effective_date']) for t in known['tickers']} if in_range else set()
    actual = set(zip(selected.Ticker,selected.Date))
    require(actual == expected, 'unknown, missing or unsupported in-sample split')
    checks = []
    for event in selected.itertuples():
        ratio = event.numerator/event.denominator
        require(ratio == known['new_shares_per_old_share'] and event.splitRatio == '2:1',
                'unsupported split ratio')
        group = bars[bars.Ticker == event.Ticker].sort_values('Date').set_index('Date')
        previous = group.Close.shift().loc[event.Date]
        gap = group.Open.loc[event.Date]/previous
        require(np.isfinite(gap) and .75 < gap < 1.25,
                f'{event.Ticker}: split-day price discontinuity; possible double/unadjusted split')
        checks.append(dict(Ticker=event.Ticker,Date=event.Date,Ratio=float(ratio),
            PreviousAdjustedClose=float(previous),AdjustedOpen=float(group.Open.loc[event.Date]),
            OpenToPreviousClose=float(gap),Action='no second split adjustment',
            Source=known['url']))
    return checks


def audit(root, write=True):
    root = Path(root)
    data = root/'data'
    manifest_path = data/'dataset_manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else dict(
        start='2021-01-01',end_exclusive='2024-01-01',raw_folders=['raw','raw_2022_2023'],
        supplement_tables=['added_2022_2023.csv'])
    files = [data/x for x in ['daily_prices.csv', 'spy_daily_prices.csv','corporate_actions.csv']]
    if manifest_path.exists():
        files.append(manifest_path)
    panels = [pd.read_csv(files[0]), pd.read_csv(files[1])]
    require(set(panels[0].Ticker) == set(TICKERS), 'ETF universe mismatch')
    require(set(panels[1].Ticker) == {'SPY'}, 'benchmark universe mismatch')
    bars = pd.concat(panels, ignore_index=True)
    keys = ['Ticker', 'Date']
    numbers = ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']
    require(not bars.duplicated(keys).any(), 'duplicate bars')
    require(np.isfinite(bars[numbers]).all().all(), 'nonfinite bars')
    require((bars[numbers[:-1]] > 0).all().all() and (bars.Volume >= 0).all(), 'invalid bars')
    require((bars.High >= bars[['Open', 'Low', 'Close']].max(axis=1)).all()
            and (bars.Low <= bars[['Open', 'High', 'Close']].min(axis=1)).all(), 'OHLC ordering')
    sessions = CAL.sessions_in_range(manifest['start'],pd.Timestamp(manifest['end_exclusive'])-pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    expected = pd.MultiIndex.from_product([TICKERS+['SPY'], sessions], names=keys)
    actual = pd.MultiIndex.from_frame(bars[keys])
    require(len(expected.difference(actual)) == len(actual.difference(expected)) == 0,
            'unexpected study dates or missing sessions; dataset manifest mismatch')
    raw_panels, raw_events = [], []
    for folder in manifest['raw_folders']:
        for ticker in TICKERS+['SPY']:
            path = data/folder/f'{ticker}.json'
            frame, events = read_raw(path)
            require(set(frame.Ticker) == {ticker}, f'{path}: wrong symbol')
            raw_panels.append(frame)
            raw_events.extend(events)
            files.append(path)
    raw = pd.concat(raw_panels, ignore_index=True)
    require(not raw.duplicated(keys).any(), 'overlapping raw batches need explicit reconciliation')
    left, right = bars.set_index(keys).sort_index(), raw.set_index(keys).sort_index()
    require(left.index.equals(right.index), 'CSV/raw keys differ')
    require((left.Currency == right.Currency).all() and set(left.Currency) == {'USD'}, 'currency mismatch')
    require(np.allclose(left[numbers], right[numbers], rtol=1e-12, atol=1e-10), 'CSV/raw values differ')
    for name in manifest['supplement_tables']:
        path = data/name
        added = pd.read_csv(path).set_index(keys).sort_index()
        expected_added = right.loc[right.index.get_level_values('Date') >= '2022-01-01']
        require(added.index.equals(expected_added.index), 'appended table keys differ')
        require(np.allclose(added[numbers], expected_added[numbers], rtol=1e-12, atol=1e-10),
                'appended table values differ')
        files.append(path)
    events = pd.read_csv(files[2], dtype={'EventKey': str})
    require(set(events.EventType) <= {'dividends','splits'},
            'unsupported distributions; refusing to continue')
    ekeys = ['Ticker', 'EventType', 'EventKey']
    source_events = pd.DataFrame(raw_events)
    require(not events.duplicated(ekeys).any() and not source_events.duplicated(ekeys).any(), 'duplicate events')
    left_e, right_e = events.set_index(ekeys).sort_index(), source_events.set_index(ekeys).sort_index()
    require(left_e.index.equals(right_e.index), 'event CSV/raw keys differ')
    require(np.allclose(left_e[['amount', 'date']], right_e[['amount', 'date']], rtol=0, atol=1e-12,equal_nan=True),
            'event CSV/raw values differ')
    if (events.EventType == 'splits').any():
        for col in ['numerator','denominator']:
            require(np.allclose(left_e[col],right_e[col],rtol=0,atol=1e-12,equal_nan=True),'split CSV/raw values differ')
        require(left_e.splitRatio.fillna('').equals(right_e.splitRatio.fillna('')), 'split ratio strings differ')
    events['Date'] = pd.to_datetime(events.date, unit='s', utc=True).dt.tz_convert(
        'America/New_York').dt.strftime('%Y-%m-%d')
    evidence = data/'price_basis_sources.json'
    sources = json.loads(evidence.read_text())
    split_checks = reconcile_splits(bars,events,sources)
    event_audit = reconcile_factors(bars, events)
    # A new request straddles the original batches and checks their price/share
    # units on overlapping dates. It is evidence, never a replacement dataset.
    overlap_rows = []
    for ticker in TICKERS+['SPY']:
        path = data/'basis_overlap_raw'/f'{ticker}.json'
        overlap, _ = read_raw(path)
        require(set(overlap.Ticker) == {ticker}, 'overlap symbol mismatch')
        required = CAL.sessions_in_range('2021-12-20', '2022-01-10').strftime('%Y-%m-%d')
        require(set(overlap.Date) == set(required) and not overlap.Date.duplicated().any(),
                f'{ticker}: incomplete overlap evidence')
        new = overlap.set_index('Date').sort_index()
        old = bars[bars.Ticker == ticker].set_index('Date').loc[new.index]
        require(np.allclose(new[['Open','High','Low','Close']], old[['Open','High','Low','Close']],
                            rtol=2e-6, atol=1e-6), f'{ticker}: cross-batch price scale mismatch')
        require(np.array_equal(new.Volume, old.Volume), f'{ticker}: overlap volume mismatch')
        for batch, mask in [('2021', new.index < '2022-01-01'), ('2022', new.index >= '2022-01-01')]:
            ratios = (new['Adj Close']/new.Close)/(old['Adj Close']/old.Close)
            sub = ratios.loc[mask]
            require(np.allclose(sub/sub.iloc[-1], 1, rtol=0, atol=2e-6),
                    f'{ticker}: inconsistent relative adjustment in {batch}')
        overlap_rows.append(dict(Ticker=ticker, Rows=len(new),
            MaxRelativeOHLCDifference=float((new[numbers[:4]]/old[numbers[:4]]-1).abs().max().max())))
        files.append(path)
    # Keep published 2025 split information pinned as provenance, not
    # used as a second split adjustment or as a price predictor.
    files.append(evidence)
    report = dict(status=STATUS, audited_at_utc=datetime.now(timezone.utc).isoformat(),
        contract=CONTRACT, rows=len(bars), dividends_checked=len(event_audit),
        dataset=manifest,in_sample_split_count=len(split_checks),split_checks=split_checks,
        max_dividend_relative_error=float(event_audit.RelativeError.abs().max()),
        overlap_rows=sum(x['Rows'] for x in overlap_rows), overlap_checks=overlap_rows,
        input_hashes={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        sources=sources, limitations=[
            'Same-vendor snapshot consistency, not independent exchange-price certification.',
            'Absolute adjusted-close levels include post-study actions; only relative factors are reconciled.',
            'Ex-dates are not verified cash payment dates; no dividends are credited.',
            'Prices and share counts use source split-adjusted units, not historical actual quotes/shares.',
            'Only issuer-documented source-adjusted splits supported; unknown splits/distributions fail closed.',
            'No historical point-in-time vendor snapshots or total-return account are certified.'])
    if write:
        event_audit.to_csv(data/'dividend_basis_checks.csv', index=False)
        (data/'price_basis_audit.json').write_text(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    result = audit(Path(__file__).resolve().parent)
    print(json.dumps({k:result[k] for k in ['status','rows','dividends_checked',
                                         'overlap_rows','max_dividend_relative_error']}, indent=2))
