"""Build focused plots/tables and audit proxy labels; never select new models."""
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

ROOT=Path(__file__).resolve().parent
SOURCE=ROOT/'experiments/refinement_v2'
WINDOWS=['W1','W2','W3','W4_2025']
LABELS=['2023 H2','2024 H1','2024 H2','2025']


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=ROOT/'reports/focused_evidence')
    args=p.parse_args();out=args.output.resolve();out.mkdir(parents=True,exist_ok=False)
    prices=pd.concat([pd.read_csv(ROOT/'data'/n,parse_dates=['Date'])
                     for n in ('daily_prices.csv','spy_daily_prices.csv')])
    assert not prices.duplicated(['Date','Ticker']).any()
    opening=prices.pivot(index='Date',columns='Ticker',values='Open').sort_index()
    factor=prices.assign(Factor=prices['Adj Close']/prices.Close).pivot(index='Date',columns='Ticker',values='Factor')
    adjusted=opening*factor
    events=pd.read_csv(ROOT/'data/issuer_evidence/reconciled_distributions.csv',parse_dates=['ExDate','PayDate'])
    candidates=pd.read_csv(ROOT/'prepared_data/weekly_candidates.csv',parse_dates=['SignalDate','EntryDate','ExitDate'])
    candidates=candidates.loc[candidates.HoldingPeriodComplete & candidates.FeaturesComplete].copy()
    rows=[]
    for r in candidates.itertuples():
        eligible=events.loc[events.Ticker.eq(r.Ticker)&events.ExDate.gt(r.EntryDate)&events.ExDate.le(r.ExitDate)]
        start=opening.at[r.EntryDate,r.Ticker];end=opening.at[r.ExitDate,r.Ticker]
        exact=(end+eligible.AmountPerSourceUnit.sum())/start-1
        proxy=adjusted.at[r.ExitDate,r.Ticker]/adjusted.at[r.EntryDate,r.Ticker]-1
        rows.append(dict(SignalDate=r.SignalDate,Ticker=r.Ticker,Split=r.Split,
            PriceReturn=end/start-1,IssuerBuyHoldReturn=exact,AdjustedProxyReturn=proxy,
            Difference=exact-proxy,DistributionCount=len(eligible),
            UnpaidAtExit=float(eligible.loc[eligible.PayDate.ge(r.ExitDate),'AmountPerSourceUnit'].sum()/start)))
    audit=pd.DataFrame(rows)
    audit.to_csv(out/'label_audit.csv',index=False)
    # Audit labelled target equality, including the ex-date boundary convention.
    np.testing.assert_allclose(audit.PriceReturn,candidates.TargetPriceReturn,atol=1e-12)
    summary=dict(rows=len(audit),mean_absolute_difference=float(audit.Difference.abs().mean()),
                 max_absolute_difference=float(audit.Difference.abs().max()),
                 p99_absolute_difference=float(audit.Difference.abs().quantile(.99)),
                 definition='Issuer buy-and-hold open return plus gross dividend entitlements; no reinvestment/fees/tax; unpaid face value included',
                 boundary='Entry < ex-date <= exit; entry on ex-date earns no entitlement; selling on ex-date retains entitlement',
                 limitation='Label comparison only; does not refit or demonstrate model improvement')
    (out/'label_audit_summary.json').write_text(json.dumps(summary,indent=2))
    rankings=pd.read_csv(SOURCE/'ranking_summary.csv')
    main=rankings.loc[rankings.Basis.eq('adjusted_proxy')&rankings.Method.isin(
        ['selected_baseline','reg_extended7','reg_extended9','rank_extended7'])]
    fig,ax=plt.subplots(figsize=(10,5))
    table=main.pivot(index='Window',columns='Method',values='MeanRankIC').reindex(WINDOWS)
    table.index=LABELS;table=table.rename(columns={'selected_baseline':'Selected simple rule',
        'reg_extended7':'Regression / 7 features','reg_extended9':'Regression / 9 features','rank_extended7':'Ranking / 7 features'})
    table.plot.bar(ax=ax);ax.axhline(0,color='black',lw=.7);ax.set_ylabel('Mean weekly Rank IC')
    ax.set_xlabel('Evaluation window');ax.set_title('Ranking skill varies across periods (retrospective)')
    ax.tick_params(axis='x',rotation=0);ax.legend(fontsize=8);fig.tight_layout();fig.savefig(out/'01_ranking.png',dpi=180);plt.close(fig)
    ledger=pd.read_csv(SOURCE/'cash_dividend_verified/daily_ledger.csv',parse_dates=['Date'])
    chosen=ledger.loc[ledger.TrainingBasis.eq('adjusted_proxy')&ledger.Window.eq('W4_2025') & (
        (ledger.Method.eq('reg_extended9')&ledger.Rule.eq('top5'))|ledger.Method.isin(['equal_weight','SPY']))]
    fig,axes=plt.subplots(2,1,figsize=(10,7),sharex=True)
    names={'reg_extended9':'Regression 9 / Top5','equal_weight':'Sector equal weight','SPY':'SPY'}
    for method,g in chosen.groupby('Method'):
        g=g.sort_values('Date');axes[0].plot(g.Date,g.NAV-1,label=names[method]);axes[1].plot(g.Date,g.Drawdown,label=names[method])
    axes[0].set_title('2025 gross cash-dividend account / one-way 10 bps / retrospective')
    axes[0].set_ylabel('Cumulative NAV return');axes[1].set_ylabel('Drawdown')
    for ax in axes:ax.yaxis.set_major_formatter(PercentFormatter(1));ax.grid(alpha=.2)
    axes[0].legend(fontsize=8);fig.tight_layout();fig.savefig(out/'02_account_2025.png',dpi=180);plt.close(fig)
    portfolios=pd.read_csv(SOURCE/'portfolio_summary.csv')
    costs=portfolios.loc[portfolios.Basis.eq('adjusted_proxy')&portfolios.Window.eq('W4_2025')&portfolios.Method.eq('reg_extended9')]
    fig,ax=plt.subplots(figsize=(8,5))
    for rule,g in costs.groupby('Rule'):
        g=g.sort_values('CostBps');ax.plot(g.CostBps,g.NetReturn,marker='o',label=rule)
    ax.yaxis.set_major_formatter(PercentFormatter(1));ax.set_xlabel('Assumed one-way trading cost (bps)')
    ax.set_ylabel('Net return / adjusted-open approximation');ax.set_title('Concentration and turnover affect realised value')
    ax.legend();ax.grid(alpha=.2);fig.tight_layout();fig.savefig(out/'03_costs.png',dpi=180);plt.close(fig)
    account=pd.read_csv(SOURCE/'cash_dividend_verified/summary.csv')
    account=account.loc[account.TrainingBasis.eq('adjusted_proxy')&(
        account.Method.eq('reg_extended9')&account.Rule.eq('top5')|account.Method.isin(['equal_weight','SPY']))]
    account.to_csv(out/'main_account_table.csv',index=False)
    main.to_csv(out/'main_ranking_table.csv',index=False)
    (out/'README.md').write_text('# Focused evidence\n\n'
        'All periods are retrospective and previously inspected. These plots do not establish reliable alpha.\n\n'
        '01: ranking skill by window; 02: actual dividend-entitlement/payment ledger in 2025; '
        '03: adjusted-open proxy cost sensitivity (different accounting, labelled explicitly).\n\n'
        'Main account table includes unpaid receivables in NAV. Labels audit gross buy-and-hold returns, '
        'not a reinvested trading account. The 2025 model-vs-equal bootstrap intervals include zero.\n')
    inputs=[SOURCE/'ranking_summary.csv',SOURCE/'portfolio_summary.csv',
            SOURCE/'cash_dividend_verified/daily_ledger.csv',ROOT/'data/issuer_evidence/reconciled_distributions.csv',
            ROOT/'data/daily_prices.csv',ROOT/'data/spy_daily_prices.csv',ROOT/'prepared_data/weekly_candidates.csv',Path(__file__)]
    (out/'manifest.json').write_text(json.dumps({str(x.relative_to(ROOT)):hashlib.sha256(x.read_bytes()).hexdigest() for x in inputs},indent=2))
    print(json.dumps(summary,indent=2));print('Evidence:',out)


if __name__=='__main__':main()
