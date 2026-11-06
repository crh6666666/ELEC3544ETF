"""Fetch/cache issuer distribution evidence and reconcile Yahoo events.

Issuer amounts use historical share units. Convert once to Yahoo's current
split-adjusted units. No quotes or existing datasets are changed.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import requests
import numpy as np
import pandas as pd
from calendar_utils import TICKERS

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'data/issuer_evidence'
URL='https://www.ssga.com/bin/v1/ssmp/fund/dividend-distribution'


def download(ticker):
    path=OUT/f'{ticker}.json'
    if path.exists():return ticker,'cached'
    params=dict(country='us',language='en',role='intermediary',product='etfs',ticker=ticker)
    response=requests.get(URL,params=params,timeout=35)
    response.raise_for_status()
    records=response.json()
    if not isinstance(records,list) or not records or 'exDate' not in records[0]:
        raise ValueError(f'{ticker}: invalid issuer response')
    path.write_text(json.dumps(dict(ticker=ticker,url=response.url,
        fetched_utc=datetime.now(timezone.utc).isoformat(),records=records),indent=2))
    return ticker,'downloaded'


def build():
    raw=pd.read_csv(ROOT/'data/corporate_actions.csv')
    raw['ExDate']=pd.to_datetime(raw.date,unit='s',utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    end=pd.read_csv(ROOT/'data/daily_prices.csv',usecols=['Date']).Date.max()
    end=pd.Timestamp(end)
    rows=[]
    for ticker in [*TICKERS,'SPY']:
        content=json.loads((OUT/f'{ticker}.json').read_text())
        splits=raw.loc[raw.Ticker.eq(ticker)&raw.EventType.eq('splits')]
        for record in content['records']:
            ex=pd.to_datetime(record['exDate'],format='%m/%d/%Y')
            if not pd.Timestamp('2021-01-01')<=ex<=end:continue
            divisor=float((splits.loc[splits.ExDate.gt(ex), 'numerator']/splits.loc[splits.ExDate.gt(ex),'denominator']).prod())
            cash=sum(float(record.get(k,'0') or '0') if record.get(k) not in ('-',None,'') else 0.
                     for k in ('dividend','shortTeamCapital','longTeamCapital'))
            yahoo=raw.loc[raw.Ticker.eq(ticker)&raw.EventType.eq('dividends')&raw.ExDate.eq(ex),'amount']
            if len(yahoo)!=1:raise ValueError(f'{ticker}/{ex}: missing or duplicate Yahoo dividend')
            amount=cash/divisor;difference=amount-float(yahoo.iloc[0])
            # Yahoo historical dividend values are rounded before split conversion.
            tolerance=.0006/divisor+1e-6
            rows.append(dict(Ticker=ticker,ExDate=ex,PayDate=pd.to_datetime(record['payableDate'],format='%m/%d/%Y'),
                RecordDate=pd.to_datetime(record['recordDate'],format='%m/%d/%Y'),
                IssuerHistoricalAmount=cash,FutureSplitDivisor=divisor,AmountPerSourceUnit=amount,
                YahooAmount=float(yahoo.iloc[0]),Difference=difference,Tolerance=tolerance,
                Matched=abs(difference)<=tolerance,SourceURL=content['url']))
    frame=pd.DataFrame(rows)
    expected=raw.loc[raw.EventType.eq('dividends')&raw.ExDate.between('2021-01-01',end)]
    expected_keys=set(zip(expected.Ticker,expected.ExDate))
    actual_keys=set(zip(frame.Ticker,frame.ExDate))
    assert expected_keys==actual_keys,'Issuer coverage differs from Yahoo event coverage'
    assert not frame.duplicated(['Ticker','ExDate']).any()
    # Explicitly reviewed discrepancy: retain both amounts, use primary issuer
    # for the new ledger. Never widen the generic rounding tolerance to hide it.
    frame['Resolution']='within_declared_rounding_tolerance'
    known=(frame.Ticker.eq('SPY')&frame.ExDate.eq(pd.Timestamp('2021-12-17'))
           &np.isclose(frame.YahooAmount,1.633)&np.isclose(frame.IssuerHistoricalAmount,1.636431))
    frame.loc[known&~frame.Matched,'Resolution']='issuer_preferred_explicit_2021_SPY_discrepancy'
    assert (frame.Matched|known).all(),frame.loc[~(frame.Matched|known),['Ticker','ExDate','Difference']].to_string(index=False)
    assert frame.PayDate.ge(frame.ExDate).all()
    frame.to_csv(OUT/'reconciled_distributions.csv',index=False)
    report=dict(status='ISSUER_DISTRIBUTIONS_RECONCILED_NOT_DAILY_QUOTE_VERIFICATION',
        events=len(frame),tickers=frame.Ticker.nunique(),rounding_matches=int(frame.Matched.sum()),
        explicit_issuer_overrides=int((~frame.Matched).sum()),max_absolute_unit_difference=float(frame.Difference.abs().max()),
        rounding_tolerance='0.0006 historical dollars divided by future split divisor + 0.000001; predeclared',
        source='State Street issuer API, separate from Yahoo',
        limitation='Validates distribution amounts/dates and known split-unit conversion, not independent daily OHLC.',
        hashes={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(OUT.glob('*.json')) if p.name!='audit.json'})
    (OUT/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))


def main():
    p=argparse.ArgumentParser();p.add_argument('--offline',action='store_true');args=p.parse_args()
    OUT.mkdir(exist_ok=True)
    if not args.offline:
        with ThreadPoolExecutor(max_workers=3) as pool:
            for item in pool.map(download,[*TICKERS,'SPY']):print(item,flush=True)
    build()


if __name__=='__main__':main()
