"""Offline model samples: existing daily bars -> four features -> purged year splits."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from calendar_utils import CAL, TICKERS, artificial_checks, schedule
from price_basis import BASIS, CONTRACT, audit
from study_config import SPLIT_DESIGN, split_and_purge
from models.common import FEATURES

ROOT = Path(__file__).resolve().parent
OUT = ROOT/'prepared_data'
SOURCES = [ROOT/'data/daily_prices.csv']
DATE_COLUMNS = ['SignalDate','EntryDate','ExitDate','FeatureMaxDate',
                'SignalCloseUTC','EntryOpenUTC','ExitOpenUTC']


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def feat(a,s):
    idx=CAL.sessions_in_range(s-pd.Timedelta(days=20),s)
    x=a.set_index('Date').reindex(idx).iloc[-6:]
    if x[['Close','Volume']].isna().any().any():return dict.fromkeys(FEATURES,np.nan)
    c=x.Close;v=x.Volume.iloc[-5:]
    return dict(zip(FEATURES,[c.iloc[-1]/c.iloc[0]-1,c.pct_change(fill_method=None).iloc[-5:].std(ddof=1),c.iloc[-1]/c.iloc[-5:].mean(),v.iloc[-1]/v.mean() if v.mean()>0 else np.nan]))

def load_samples(force=False):
    """Rebuild stale samples from local data; never download or train a model."""
    report = audit(ROOT, write=False)
    hashes = {str(Path(p).relative_to(ROOT)): value for p, value in report['input_hashes'].items()}
    basis_id = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
    code_hashes = {p: digest(ROOT/p) for p in
                   ['prepare_data.py','calendar_utils.py','study_config.py','models/common.py']}
    samples = OUT/'weekly_candidates.csv'
    manifest_path = OUT/'preparation.json'
    if not force and samples.exists() and manifest_path.exists():
        saved = json.loads(manifest_path.read_text())
        if (saved.get('input_hashes') == hashes and saved.get('code_hashes') == code_hashes
                and saved.get('samples_sha256') == digest(samples)):
            candidates = pd.read_csv(samples, parse_dates=DATE_COLUMNS)
            return unpack(candidates), hashes
    OUT.mkdir(exist_ok=True)
    parts=[pd.read_csv(p,encoding='utf-8-sig',parse_dates=['Date']).assign(SourceFile=p.parent.parent.name) for p in SOURCES]
    d=pd.concat(parts,ignore_index=True)
    duplicates=d[d.duplicated(['Date','Ticker'],keep=False)]
    duplicates.to_csv(OUT/'duplicate_audit.csv',index=False)
    if len(duplicates):
        for _,g in duplicates.groupby(['Date','Ticker']):
            assert g[['Open','High','Low','Close','Adj Close','Volume','Currency']].nunique().max()==1,'Conflicting overlapping bars'
    d=d.drop_duplicates(['Date','Ticker']).sort_values(['Ticker','Date'])
    assert set(d.Ticker)==set(TICKERS) and set(d.Currency)=={'USD'}
    numeric=['Open','High','Low','Close','Adj Close','Volume']
    assert np.isfinite(d[numeric]).all().all()
    assert (d[numeric[:-1]]>0).all().all() and (d.Volume>=0).all()
    assert (d.High>=d[['Open','Close','Low']].max(axis=1)-1e-6).all()
    assert (d.Low<=d[['Open','Close','High']].min(axis=1)+1e-6).all()
    sessions=CAL.sessions_in_range(d.Date.min(),d.Date.max())
    expected=pd.MultiIndex.from_product([sessions,TICKERS],names=['Date','Ticker'])
    missing=expected.difference(pd.MultiIndex.from_frame(d[['Date','Ticker']]))
    assert len(missing)==0, 'Missing market bars'
    assert d.Date.isin(sessions).all()
    d['AdjustmentFactorDiagnostic']=d['Adj Close']/d.Close
    d['DollarVolumeProxy']=d.Close*d.Volume
    d.to_csv(OUT/'merged_daily.csv',index=False)
    sched=schedule(d.Date.min(),d.Date.max());sched.to_csv(OUT/'signal_entry_exit.csv',index=False)
    rows=[];leak_checks=0
    for _,s in sched.iterrows():
        for t in TICKERS:
            a=d[d.Ticker==t];f=feat(a,s.SignalDate)
            future=a.copy();future.loc[future.Date>s.SignalDate,numeric]*=7
            ff=feat(future,s.SignalDate)
            assert all((pd.isna(f[k]) and pd.isna(ff[k])) or f[k]==ff[k] for k in FEATURES)
            leak_checks+=1
            req=CAL.sessions_in_range(s.EntryDate,s.ExitDate)
            b=a.set_index('Date').reindex(req)
            complete=bool(b[numeric].notna().all().all())
            y=b.Open.iloc[-1]/b.Open.iloc[0]-1 if complete else np.nan
            rows.append({**s.to_dict(),'Ticker':t,**f,'FeatureMaxDate':s.SignalDate,'TargetPriceReturn':y,'TargetTotalReturn':np.nan,'TotalReturnVerified':False,'HoldingPeriodComplete':complete,'FeaturesComplete':all(pd.notna(v) for v in f.values()),'TargetBasis':'source_split_adjusted_open_price_return_excludes_cash_dividends','Split':'excluded'})
    candidates=pd.DataFrame(rows)
    candidates['TargetBasis']=BASIS
    candidates['PriceBasisVersion']=CONTRACT['version']
    candidates['PriceBasisAuditId']=basis_id
    good=candidates[candidates.HoldingPeriodComplete & candidates.FeaturesComplete].copy()
    complete_weeks=good.groupby('SignalDate').Ticker.agg(lambda x: len(x)==len(TICKERS) and set(x)==set(TICKERS))
    good=good[good.SignalDate.isin(complete_weeks[complete_weeks].index)].copy()
    good,purged=split_and_purge(good)
    candidates.loc[good.index,'Split']=good.Split
    candidates.to_csv(OUT/'weekly_candidates.csv',index=False)
    candidates.groupby(['SignalDate','EntryDate','ExitDate','Split'],dropna=False).agg(stocks=('Ticker','count'),complete_holding=('HoldingPeriodComplete','all'),features_complete=('FeaturesComplete','all')).reset_index().to_csv(OUT/'week_split_audit.csv',index=False)
    for name in ['train','validation','test']:assert len(good[good.Split==name])>0
    tr=good[good.Split=='train'];va=good[good.Split=='validation'];te=good[good.Split=='test']
    assert tr.ExitOpenUTC.max()<va.SignalCloseUTC.min()
    assert va.ExitOpenUTC.max()<te.SignalCloseUTC.min()
    manifest = dict(input_hashes=hashes, code_hashes=code_hashes,
        samples_sha256=digest(OUT/'weekly_candidates.csv'),
        price_basis_audit_id=basis_id, split_design=SPLIT_DESIGN,
        calendar_checks=artificial_checks(), future_perturbation_checks=leak_checks,
        counts={name: dict(rows=len(frame), weeks=int(frame.SignalDate.nunique()))
                for name,frame in [('train',tr),('validation',va),('test',te)]},
        note='Same-vendor source consistency only; no independent price certification or cash-dividend account.')
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return (tr, va, te), hashes


def unpack(candidates):
    frames = tuple(candidates[candidates.Split == name].copy()
                   for name in ('train','validation','test'))
    for frame in frames:
        if frame.empty or not np.isfinite(frame[FEATURES+['TargetPriceReturn']]).all().all():
            raise ValueError('Empty/nonfinite model samples')
        if frame.duplicated(['SignalDate','Ticker']).any():
            raise ValueError('Duplicate weekly samples')
        if not frame.groupby('SignalDate').Ticker.agg(lambda x: set(x)==set(TICKERS)).all():
            raise ValueError('Incomplete weekly ETF universe')
    tr,va,te = frames
    if not (tr.ExitOpenUTC.max() < va.SignalCloseUTC.min()
            and va.ExitOpenUTC.max() < te.SignalCloseUTC.min()):
        raise ValueError('Labels overlap a later partition signal')
    return frames


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Prepare local features and labels without training.')
    parser.add_argument('--force', action='store_true', help='Rebuild the cached samples.')
    args = parser.parse_args()
    frames, _ = load_samples(force=args.force)
    for name, frame in zip(('train','validation','test'),frames):
        print(f'{name}: {frame.SignalDate.nunique()} weeks, {len(frame)} rows')
    print(f'Prepared samples: {OUT}')
