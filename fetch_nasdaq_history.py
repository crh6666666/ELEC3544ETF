"""Freeze independent Nasdaq OHLCV evidence, then check overlap before use."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests
from calendar_utils import CAL,TICKERS

ROOT=Path(__file__).resolve().parent
END='2026-10-30'


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'data/nasdaq_20261107')
    p.add_argument('--proxy',default='http://127.0.0.1:12000');args=p.parse_args()
    out=args.output;out.mkdir(exist_ok=True)
    def fetch(ticker):
        dest=out/f'{ticker}.json'
        if dest.exists():return
        url=f'https://api.nasdaq.com/api/quote/{ticker}/historical'
        r=requests.get(url,params=dict(assetclass='etf',fromdate='2021-01-01',todate=END,limit=9999),
            headers={'User-Agent':'Mozilla/5.0','Accept':'application/json','Origin':'https://www.nasdaq.com'},
            proxies={'https':args.proxy} if args.proxy else None,timeout=40)
        r.raise_for_status();payload=r.json()
        if not payload.get('data') or not payload['data'].get('tradesTable',{}).get('rows'):
            raise ValueError(f'{ticker}: no historical bars')
        dest.write_text(json.dumps(dict(url=r.url,fetched_utc=datetime.now(timezone.utc).isoformat(),payload=payload),indent=2))
        print('Downloaded '+ticker,flush=True)
    with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(fetch,[*TICKERS,'SPY']))
    frames=[]
    for ticker in [*TICKERS,'SPY']:
        record=json.loads((out/f'{ticker}.json').read_text())['payload']['data']
        a=pd.DataFrame(record['tradesTable']['rows'])
        assert len(a)==int(record['totalRecords']),f'{ticker}: truncated history'
        a['Date']=pd.to_datetime(a.date,format='%m/%d/%Y')
        for field in ('open','high','low','close','volume'):
            cleaned=a[field].replace({'N/A':np.nan}).str.replace(',','',regex=False).str.replace('$','',regex=False)
            a[field.title()]=pd.to_numeric(cleaned,errors='raise')
        a['Ticker']=ticker;a['Currency']='USD'
        frames.append(a[['Date','Ticker','Open','High','Low','Close','Volume','Currency']])
    bars=pd.concat(frames).sort_values(['Date','Ticker']).reset_index(drop=True)
    assert not bars.duplicated(['Date','Ticker']).any()
    expected=CAL.sessions_in_range('2021-01-01',END)
    for ticker,g in bars.groupby('Ticker'):
        assert set(g.Date)==set(expected),f'{ticker}: unexpected/missing sessions'
    assert np.isfinite(bars[['Open','High','Low','Close']]).all().all()
    assert bars.loc[bars.Ticker.ne('SPY'),'Volume'].notna().all()
    assert bars[['Open','High','Low','Close']].gt(0).all().all() and bars.Volume.dropna().ge(0).all()
    valid_ohlc=bars.High.ge(bars[['Open','Low','Close']].max(axis=1)-1e-8)&bars.Low.le(bars[['Open','High','Close']].min(axis=1)+1e-8)
    assert valid_ohlc.loc[bars.Date.ge('2026-01-01')].all(),'New-period OHLC invalid'
    old=pd.concat([pd.read_csv(ROOT/'data'/name,parse_dates=['Date']) for name in ('daily_prices.csv','spy_daily_prices.csv')])
    joined=bars.merge(old,on=['Date','Ticker'],suffixes=('_nasdaq','_yahoo'),validate='one_to_one')
    joined['OpenDifference']=joined.Open_nasdaq-joined.Open_yahoo
    joined['CloseDifference']=joined.Close_nasdaq-joined.Close_yahoo
    joined['OpenRelativeDifference']=joined.OpenDifference/joined.Open_yahoo
    joined['CloseRelativeDifference']=joined.CloseDifference/joined.Close_yahoo
    joined.to_csv(out/'overlap_audit.csv',index=False)
    # Do not automatically rescale discrepancies: surface them for review first.
    report=joined.groupby('Ticker').agg(Rows=('Date','size'),
        MedianOpenRatio=('OpenRelativeDifference','median'),MaxOpenRelativeDifference=('OpenRelativeDifference',lambda x:x.abs().max()),
        MaxCloseRelativeDifference=('CloseRelativeDifference',lambda x:x.abs().max())).reset_index()
    report.to_csv(out/'overlap_summary.csv',index=False)
    bars.loc[~valid_ohlc|bars.Volume.isna()|bars.High.eq(bars.Low)].to_csv(out/'flagged_bars.csv',index=False)
    bars.to_csv(out/'daily_prices.csv',index=False)
    manifest=dict(status='DOWNLOADED_PENDING_OVERLAP_REVIEW',start=str(bars.Date.min().date()),end=END,
        rows=len(bars),sessions=len(expected),tickers=12,source='Nasdaq historical OHLCV API; no Adj Close supplied',
        hashes={f.name:hashlib.sha256(f.read_bytes()).hexdigest() for f in sorted(out.glob('*.json')) if f.name!='manifest.json'})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    print(report.to_string(index=False))


if __name__=='__main__':main()
