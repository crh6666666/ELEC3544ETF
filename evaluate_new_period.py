"""Evaluate the frozen 2026 protocol once; no performance-based selection."""
import argparse
import hashlib
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd
from calendar_utils import CAL,TICKERS,schedule
from run_research import feature_panels,fit_predict,metrics,weights_for,BASELINES,SETS,bootstrap
from dividend_account import cash_simulate

ROOT=Path(__file__).resolve().parent


def issuer_events(end):
    raw=pd.read_csv(ROOT/'data/corporate_actions.csv')
    raw['ExDate']=pd.to_datetime(raw.date,unit='s',utc=True).dt.tz_convert('America/New_York').dt.tz_localize(None).dt.normalize()
    rows=[]
    for ticker in [*TICKERS,'SPY']:
        content=json.loads((ROOT/f'data/issuer_evidence/{ticker}.json').read_text())
        splits=raw.loc[raw.Ticker.eq(ticker)&raw.EventType.eq('splits')]
        for r in content['records']:
            date=pd.to_datetime(r['exDate'],format='%m/%d/%Y')
            if not pd.Timestamp('2021-01-01')<=date<=end:continue
            divisor=(splits.loc[splits.ExDate.gt(date),'numerator']/splits.loc[splits.ExDate.gt(date),'denominator']).prod()
            amount=sum(float(r[k]) if r.get(k) not in (None,'','-') else 0.
                       for k in ('dividend','shortTeamCapital','longTeamCapital'))
            rows.append(dict(Ticker=ticker,ExDate=date,PayDate=pd.to_datetime(r['payableDate'],format='%m/%d/%Y'),
                             AmountPerSourceUnit=amount/divisor))
    return pd.DataFrame(rows)


def dividend_factor(close,events):
    distributions=pd.DataFrame(0.,index=close.index,columns=close.columns)
    for r in events.itertuples():
        assert r.ExDate in close.index,'Issuer event outside observed sessions'
        distributions.at[r.ExDate,r.Ticker]+=r.AmountPerSourceUnit
    ratio=1/(1-distributions/close.shift())
    ratio.iloc[0]=1.
    assert np.isfinite(ratio).all().all() and ratio.gt(0).all().all()
    accumulated=ratio.cumprod()
    return accumulated/accumulated.iloc[-1]


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'experiments/new_period_2026')
    args=p.parse_args();out=args.output;out.mkdir(exist_ok=False)
    source=ROOT/'data/nasdaq_20261107'
    evidence=pd.read_csv(source/'overlap_summary.csv')
    assert len(evidence)==12 and evidence.Rows.eq(1255).all()
    assert evidence.MaxOpenRelativeDifference.lt(.0002).all()
    assert evidence.MaxCloseRelativeDifference.lt(.000001).all()
    nasdaq=pd.read_csv(source/'daily_prices.csv',parse_dates=['Date'])
    old=pd.concat([pd.read_csv(ROOT/'data'/name,parse_dates=['Date']) for name in ('daily_prices.csv','spy_daily_prices.csv')])
    fresh=nasdaq.loc[nasdaq.Date.ge('2026-01-01')]
    bars=pd.concat([old,fresh],ignore_index=True).sort_values(['Date','Ticker'])
    assert not bars.duplicated(['Date','Ticker']).any()
    close=bars.pivot(index='Date',columns='Ticker',values='Close').sort_index()
    opening=bars.pivot(index='Date',columns='Ticker',values='Open').reindex(close.index)
    volume=bars.pivot(index='Date',columns='Ticker',values='Volume').reindex(close.index)
    expected=CAL.sessions_in_range('2021-01-01','2026-10-30')
    assert close.index.equals(expected.rename('Date'))
    assert np.isfinite(close).all().all() and np.isfinite(opening).all().all()
    assert np.isfinite(volume[TICKERS]).all().all() # SPY volume is never a feature.
    daily_returns=close.pct_change(fill_method=None)
    assert daily_returns.iloc[1:].abs().max().max()<.5,'Possible new split or extreme event requires review'
    events=issuer_events(close.index.max());factor=dividend_factor(close,events)
    panels=feature_panels(close*factor,volume)
    sched=schedule(close.index.min(),close.index.max())
    sched=sched.loc[sched.ExitDate.le(close.index.max())]
    rows=[]
    for r in sched.itertuples():
        for ticker in TICKERS:
            features={name:panel.at[r.SignalDate,ticker] for name,panel in panels.items()}
            if not np.isfinite(list(features.values())).all():continue
            rows.append(dict(SignalDate=r.SignalDate,EntryDate=r.EntryDate,ExitDate=r.ExitDate,
                SignalCloseUTC=r.SignalCloseUTC,EntryOpenUTC=r.EntryOpenUTC,ExitOpenUTC=r.ExitOpenUTC,
                Ticker=ticker,**features,Target=(opening.at[r.ExitDate,ticker]*factor.at[r.ExitDate,ticker])/
                (opening.at[r.EntryDate,ticker]*factor.at[r.EntryDate,ticker])-1))
    base=pd.DataFrame(rows).sort_values(['SignalDate','Ticker'])
    complete=base.groupby('SignalDate').Ticker.nunique().eq(11)
    base=base.loc[base.SignalDate.isin(complete[complete].index)].copy()
    base['momentum_5']=base.return_5;base['reversal_5']=-base.return_5
    base['blend_momentum']=base.groupby('SignalDate').momentum_20.rank(pct=True)+base.groupby('SignalDate').momentum_60.rank(pct=True)
    tr=base.loc[base.SignalDate.lt('2025-01-01')&base.ExitDate.lt('2025-01-01')]
    va=base.loc[base.SignalDate.ge('2025-01-01')&base.SignalDate.lt('2026-01-01')&base.ExitDate.lt('2026-01-01')]
    te=base.loc[base.SignalDate.ge('2026-01-01')]
    assert tr.ExitOpenUTC.max()<va.SignalCloseUTC.min() and va.ExitOpenUTC.max()<te.SignalCloseUTC.min()
    # History truncation invariance for all features at each new-period signal.
    for date in te.SignalDate.unique():
        truncated=feature_panels((close*factor).loc[:date],volume.loc[:date])
        for name in panels:
            np.testing.assert_allclose(panels[name].loc[date,TICKERS],truncated[name].loc[date,TICKERS])
    selected=sorted(BASELINES,key=lambda m:(-metrics(va,va[m]).RankIC.mean(),m))[0]
    all_scores={'selected_baseline':te[selected].to_numpy()}
    grids=[]
    for name in ('extended7','extended9'):
        score,grid=fit_predict(tr,va,te,SETS[name],False)
        all_scores['reg_'+name]=score
        grids.extend([dict(Method='reg_'+name,**g) for g in grid])
    modelrows=[];pred=[];summaries=[];ledgers=[];payments=[];weekly=[]
    dates=te[['SignalDate','EntryDate','ExitDate']].drop_duplicates().sort_values('EntryDate')
    assert np.array_equal(dates.ExitDate.iloc[:-1],dates.EntryDate.iloc[1:])
    period=opening.loc[dates.EntryDate.min():dates.ExitDate.max()]
    targets={}
    for method,scores in all_scores.items():
        a=te.assign(Score=scores)
        targets[method]=weights_for(a,method,'top5',TICKERS)
        modelrows.append(metrics(te,scores).assign(Method=method))
        pred.append(a.assign(Method=method))
    targets['equal_weight']=pd.DataFrame(1/11,index=dates.EntryDate,columns=TICKERS)
    targets['SPY']=pd.DataFrame(1.,index=[dates.EntryDate.iloc[0]],columns=['SPY'])
    for method,weights in targets.items():
        curve,paid=cash_simulate(period[weights.columns],weights,events,10)
        r=curve.NAV.pct_change().fillna(curve.NAV.iloc[0]-1);vol=r.std()*np.sqrt(252)
        summaries.append(dict(Method=method,NetReturn=curve.NAV.iloc[-1]-1,MaxDrawdown=curve.Drawdown.min(),
            Volatility=vol,Sharpe=r.mean()/r.std()*np.sqrt(252),Fees=curve.Fee.sum(),
            FinalReceivable=curve.Receivable.iloc[-1],Weeks=len(dates)))
        ledgers.append(curve.reset_index().assign(Method=method));payments.append(paid.assign(Method=method))
        previous=1.
        for d in dates.itertuples():
            ending=curve.at[d.ExitDate,'NAV']
            weekly.append(dict(SignalDate=d.SignalDate,Method=method,NetWeekly=ending/previous-1));previous=ending
    wk=pd.DataFrame(weekly);pivot=wk.pivot(index='SignalDate',columns='Method',values='NetWeekly')
    intervals=[]
    for method in ('reg_extended7','reg_extended9'):
        for reference in ('selected_baseline','equal_weight','SPY'):
            for block in (2,4,8):
                mean,lo,hi=bootstrap(pivot[method]-pivot[reference],np.random.default_rng(42),block=block)
                intervals.append(dict(Method=method,Reference=reference,BlockWeeks=block,MeanWeeklyDifference=mean,CI95Low=lo,CI95High=hi))
    table=pd.DataFrame(summaries);ledger=pd.concat(ledgers)
    for name,a in dict(summary=table,daily_ledger=ledger,payments=pd.concat(payments),
                       weekly_returns=wk,bootstrap=pd.DataFrame(intervals),selection_grid=pd.DataFrame(grids),
                       predictions=pd.concat(pred),ranking_metrics=pd.concat(modelrows),issuer_events=events).items():
        a.to_csv(out/f'{name}.csv',index=False)
    rows=close.index.intersection(old.Date.unique())
    y=old.pivot(index='Date',columns='Ticker',values='Adj Close')/old.pivot(index='Date',columns='Ticker',values='Close')
    z=factor.loc[rows].div(factor.loc[rows].iloc[-1]);y=y.loc[rows].div(y.loc[rows].iloc[-1])
    (z-y).abs().max().rename('MaxNormalizedFactorDifference').to_csv(out/'factor_comparison.csv')
    paths=[ROOT/'NEW_DATA_PROTOCOL.md',ROOT/'NEW_DATA_ADDENDUM.md',Path(__file__),ROOT/'run_research.py',
           ROOT/'dividend_account.py',source/'daily_prices.csv',ROOT/'data/daily_prices.csv',ROOT/'data/spy_daily_prices.csv']
    manifest=dict(status='NEW_PERIOD_EVALUATED_UNDER_PREVIOUSLY_FROZEN_RULES',
        claim_boundary='Pipeline-unscored period, group-wide prior access unverified; source substitution disclosed',
        selected_baseline=selected,fit_weeks=tr.SignalDate.nunique(),selection_weeks=va.SignalDate.nunique(),
        evaluation_weeks=te.SignalDate.nunique(),start=str(period.index.min().date()),end=str(period.index.max().date()),
        warning='Nasdaq SPY 2026-04-20 missing volume and flat OHLC retained as supplied and flagged; SPY volume not used; independent 2026 quotes not certified',
        dividend_factor='issuer previous-close multiplier, terminal-normalized; not historical publication-vintage certified',
        hashes={str(path.relative_to(ROOT)):hashlib.sha256(path.read_bytes()).hexdigest() for path in paths})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2))
    fig,axes=plt.subplots(2,1,figsize=(11,7),sharex=True)
    for method,g in ledger.groupby('Method'):
        axes[0].plot(pd.to_datetime(g.Date),g.NAV-1,label=method)
        axes[1].plot(pd.to_datetime(g.Date),g.Drawdown,label=method)
    axes[0].set_title('2026 new-period evaluation / fixed Top5 / issuer cash dividends / 10 bps')
    axes[0].legend(fontsize=8);axes[0].set_ylabel('NAV return');axes[1].set_ylabel('Drawdown')
    for ax in axes:ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(out/'evaluation.png',dpi=160);plt.close(fig)
    print(table.to_string(index=False));print(json.dumps(manifest,indent=2))


if __name__=='__main__':main()
