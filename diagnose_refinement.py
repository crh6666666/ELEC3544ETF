"""Post-run valuation attribution and bootstrap block-length sensitivity."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from run_research import ROOT, TICKERS, weights_for, bootstrap
from plot_portfolio import simulate


def main():
    p=argparse.ArgumentParser();p.add_argument('--results',type=Path,default=ROOT/'experiments/refinement_v2')
    args=p.parse_args();out=args.results
    scores=pd.read_csv(out/'scores.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    raw=pd.read_csv(ROOT/'data/daily_prices.csv',parse_dates=['Date'])
    opening=raw.pivot(index='Date',columns='Ticker',values='Open').sort_index()[TICKERS]
    factor=(raw.assign(Factor=raw['Adj Close']/raw.Close)
            .pivot(index='Date',columns='Ticker',values='Factor').reindex(opening.index)[TICKERS])
    valuation={'price':opening,'adjusted_proxy':opening*factor}
    attribution=[]
    for (training_basis,window,method),a in scores.loc[scores.Method.isin(
            ['reg_extended7','reg_extended9','rank_extended7','rank_extended9'])].groupby(['Basis','Window','Method']):
        for rule in ('top3','top5'):
            targets=weights_for(a,method,rule,TICKERS)
            for accounting,prices in valuation.items():
                prices=prices.loc[a.EntryDate.min():a.ExitDate.max()]
                curve,trades=simulate(prices,targets,10)
                attribution.append(dict(TrainingBasis=training_basis,Window=window,Method=method,
                    Rule=rule,ValuationBasis=accounting,NetReturn=curve.NAV.iloc[-1]-1,
                    MaxDrawdown=curve.Drawdown.min(),Fees=trades.Fee.sum()))
    attribution=pd.DataFrame(attribution)
    attribution.to_csv(out/'fixed_target_valuation.csv',index=False)
    policy=pd.read_csv(out/'policy_weekly_returns.csv',parse_dates=['SignalDate'])
    refs=pd.read_csv(out/'weekly_returns.csv',parse_dates=['SignalDate'])
    result=[]
    for (basis,window),group in policy.loc[policy.Method.eq('reg_extended9')&policy.Rule.eq('top5')].groupby(['Basis','Window']):
        model=group.set_index('SignalDate').NetWeekly
        equal=refs.loc[refs.Basis.eq(basis)&refs.Window.eq(window)&refs.Method.eq('equal_weight')].set_index('SignalDate').NetWeekly
        for block in (2,4,8):
            mean,lo,hi=bootstrap(model-equal,np.random.default_rng(42),block=block)
            result.append(dict(Basis=basis,Window=window,BlockWeeks=block,MeanWeeklyDifference=mean,
                               CI95Low=lo,CI95High=hi))
    pd.DataFrame(result).to_csv(out/'block_sensitivity.csv',index=False)
    print(attribution.loc[attribution.Window.eq('W4_2025')&attribution.Method.eq('reg_extended9')&attribution.Rule.eq('top5')].to_string(index=False))
    print(pd.DataFrame(result).loc[lambda x:x.Basis.eq('adjusted_proxy')].to_string(index=False))
    (out/'diagnostic_notes.json').write_text(json.dumps(dict(
        fixed_targets='Identical scores and target weights evaluated under both bases; holdings may drift differently. Not a fixed-share ledger.',
        bootstrap='reg_extended9 top5 vs equal; 2000 draws each, 2/4/8 weeks, seed42; retrospective and unadjusted.'),indent=2))


if __name__=='__main__':main()
