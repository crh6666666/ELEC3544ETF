"""Open-marked gross cash-dividend ledger using issuer payable dates.

Source-unit shares are split-adjusted throughout; dividends are converted once.
Entitlements accrue before ex-date open trades. Payment is spendable at the first
observed open strictly AFTER payable date (conservative timing convention).
Receivables are marked at face value but cannot fund orders. No tax/interest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from plot_portfolio import rebalance
from run_research import ROOT, TICKERS, weights_for


def cash_simulate(prices,targets,events,cost_bps=10):
    shares=np.zeros(len(prices.columns));cash=1.;pending=[];rows=[];payments=[]
    for i,(date,prices_today) in enumerate(prices.iterrows()):
        amounts=prices_today.to_numpy();fee=0.;turnover=0.;accrued=0.;paid=0.
        # Shares carried from the preceding session are entitled, including when
        # sold at this ex-date open; buying at the ex-date open earns no dividend.
        for event in events.loc[events.ExDate.eq(date)].itertuples():
            if event.Ticker not in prices.columns:continue
            entitlement=float(shares[prices.columns.get_loc(event.Ticker)]*event.AmountPerSourceUnit)
            if entitlement:
                pending.append(dict(Ticker=event.Ticker,ExDate=date,PayDate=event.PayDate,Amount=entitlement))
                accrued+=entitlement
        unpaid=[]
        for event in pending:
            if event['PayDate']<date:
                cash+=event['Amount'];paid+=event['Amount']
                payments.append(dict(**event,SpendableOpen=date))
            else:unpaid.append(event)
        pending=unpaid
        receivable=sum(e['Amount'] for e in pending)
        holdings=shares*amounts
        nav_before=float(holdings.sum()+cash+receivable)
        weights=None
        if date==prices.index[-1]:weights=np.zeros(len(shares))
        elif date in targets.index:weights=targets.loc[date].to_numpy()
        if weights is not None:
            holdings,cash,turnover,fee=rebalance(holdings,cash,weights,cost_bps/10000)
            shares=holdings/amounts
        nav=float(holdings.sum()+cash+receivable)
        np.testing.assert_allclose(nav,nav_before-fee,atol=1e-12)
        assert cash>=-1e-12 and np.all(shares>=-1e-12)
        rows.append(dict(Date=date,NAV=nav,Cash=cash,Receivable=receivable,DividendAccrued=accrued,
                         DividendPaid=paid,Fee=fee,TradedNotional=turnover,
                         ShareValue=float(holdings.sum())))
    curve=pd.DataFrame(rows).set_index('Date')
    curve['Drawdown']=curve.NAV/np.maximum.accumulate(np.r_[1.,curve.NAV])[1:]-1
    np.testing.assert_allclose(curve.NAV,curve.Cash+curve.Receivable+curve.ShareValue,atol=1e-12)
    np.testing.assert_allclose(curve.DividendAccrued.sum(),curve.DividendPaid.sum()+curve.Receivable.iloc[-1],atol=1e-12)
    return curve,pd.DataFrame(payments)


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'experiments/refinement_v2')
    p.add_argument('--output',type=Path)
    args=p.parse_args();out=args.output or args.results/'cash_dividend_account';out.mkdir(exist_ok=False)
    evidence=ROOT/'data/issuer_evidence'
    report=json.loads((evidence/'audit.json').read_text())
    if report['status']!='ISSUER_DISTRIBUTIONS_RECONCILED_NOT_DAILY_QUOTE_VERIFICATION':raise ValueError('Unverified issuer events')
    for filename,digest in report['hashes'].items():
        if hashlib.sha256((evidence/filename).read_bytes()).hexdigest()!=digest:
            raise ValueError('Issuer evidence changed: '+filename)
    events=pd.read_csv(evidence/'reconciled_distributions.csv',parse_dates=['ExDate','PayDate'])
    scores=pd.read_csv(args.results/'scores.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    bars=pd.concat([pd.read_csv(ROOT/'data'/name,parse_dates=['Date'])
                    for name in ('daily_prices.csv','spy_daily_prices.csv')])
    opening=bars.pivot(index='Date',columns='Ticker',values='Open').sort_index()
    totals=[];curves=[];paid=[]
    # Frozen learned signals: isolates accounting effects without refitting.
    for (basis,window,method),a in scores.loc[scores.Method.isin(['reg_extended7','reg_extended9','rank_extended7','rank_extended9'])].groupby(['Basis','Window','Method']):
        schedule=a[['SignalDate','EntryDate','ExitDate']].drop_duplicates().sort_values('EntryDate')
        for rule in ('top3','top5','buffer3'):
            targets=weights_for(a,method,rule,TICKERS)
            prices=opening.loc[a.EntryDate.min():a.ExitDate.max(),TICKERS]
            curve,payments=cash_simulate(prices,targets,events)
            meta=dict(TrainingBasis=basis,Window=window,Method=method,Rule=rule)
            totals.append(dict(**meta,NetReturn=curve.NAV.iloc[-1]-1,MaxDrawdown=curve.Drawdown.min(),
                Fees=curve.Fee.sum(),DividendsAccrued=curve.DividendAccrued.sum(),
                DividendsPaid=curve.DividendPaid.sum(),FinalReceivable=curve.Receivable.iloc[-1]))
            curves.append(curve.reset_index().assign(**meta))
            paid.append(payments.assign(**meta))
        if method=='reg_extended9':
            for reference in ('equal_weight','SPY'):
                columns=TICKERS if reference=='equal_weight' else ['SPY']
                dates=schedule.EntryDate if reference=='equal_weight' else [schedule.EntryDate.iloc[0]]
                targets=pd.DataFrame(1/len(columns),index=dates,columns=columns)
                curve,payments=cash_simulate(opening.loc[a.EntryDate.min():a.ExitDate.max(),columns],targets,events)
                meta=dict(TrainingBasis=basis,Window=window,Method=reference,Rule='reference')
                totals.append(dict(**meta,NetReturn=curve.NAV.iloc[-1]-1,MaxDrawdown=curve.Drawdown.min(),Fees=curve.Fee.sum(),
                    DividendsAccrued=curve.DividendAccrued.sum(),DividendsPaid=curve.DividendPaid.sum(),FinalReceivable=curve.Receivable.iloc[-1]))
                curves.append(curve.reset_index().assign(**meta));paid.append(payments.assign(**meta))
    summary=pd.DataFrame(totals);summary.to_csv(out/'summary.csv',index=False)
    pd.concat(curves).to_csv(out/'daily_ledger.csv',index=False)
    pd.concat(paid).to_csv(out/'payments.csv',index=False)
    files=[evidence/'reconciled_distributions.csv',evidence/'audit.json',args.results/'scores.csv',
           ROOT/'data/daily_prices.csv',ROOT/'data/spy_daily_prices.csv',Path(__file__),ROOT/'plot_portfolio.py']
    (out/'manifest.json').write_text(json.dumps({str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest()
                                                for f in files},indent=2))
    (out/'assumptions.json').write_text(json.dumps(dict(
        gross_before_tax=True,cash_interest=0,one_way_cost_bps=10,signal_training='frozen retrospective signals, not new model test',
        entitlement='shares carried into ex-date open; entry on ex-date earns nothing; exit on ex-date retains entitlement',
        availability='first observed session open strictly after issuer payable date',
        receivable='face value in NAV; cannot finance trades; residual receivables shown separately at terminal open',
        share_units='Yahoo split-adjusted units; issuer per-share amounts converted once',
        uncertainty='Issuer historical distribution data, not point-in-time publication archive; no actual tax/settlement/friction model'),indent=2))
    print(summary.loc[summary.TrainingBasis.eq('adjusted_proxy')&summary.Window.eq('W4_2025')].to_string(index=False))


if __name__=='__main__':main()
