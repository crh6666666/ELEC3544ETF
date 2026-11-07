"""Unified retrospective ETF refinement study. See RESEARCH_PROTOCOL.md."""
import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from plot_portfolio import simulate
from calendar_utils import CAL, TICKERS
from price_basis import audit

ROOT = Path(__file__).resolve().parent
BASE = ['return_5', 'volatility_5', 'price_ma_5', 'activity_5']
MID = ['relative_spy_return_20', 'volatility_20', 'relative_volume_20']
SETS = {'original4': BASE, 'extended7': BASE+MID,
        'extended9': BASE+MID+['momentum_20', 'momentum_60']}
for feature in MID:
    SETS['drop_'+feature] = [x for x in SETS['extended7'] if x != feature]
SETS['own_return20'] = BASE+['momentum_20', 'volatility_20', 'relative_volume_20']
WINDOWS = {'W1': ('2021-01-01','2023-01-01','2023-07-01','2024-01-01'),
           'W2': ('2021-01-01','2023-07-01','2024-01-01','2024-07-01'),
           'W3': ('2021-01-01','2024-01-01','2024-07-01','2025-01-01'),
           'W4_2025': ('2021-01-01','2024-01-01','2025-01-01','2026-01-01')}
BASELINES = ['momentum_5','momentum_20','momentum_60','reversal_5','blend_momentum']


def metrics(frame, score):
    rows=[]
    for date, group in frame.assign(Score=np.asarray(score)).groupby('SignalDate'):
        ic = (group.Score.rank().corr(group.Target.rank())
              if group.Score.nunique()>1 and group.Target.nunique()>1 else np.nan)
        chosen=group.sort_values(['Score','Ticker'],ascending=[False,True]).Ticker.iloc[:3]
        truth=group.sort_values(['Target','Ticker'],ascending=[False,True]).Ticker.iloc[:3]
        rows.append({'SignalDate':date,'RankIC':ic,'Top3Overlap':len(set(chosen)&set(truth))/3})
    return pd.DataFrame(rows)


def feature_panels(close, volume):
    ret=close.pct_change(fill_method=None)
    p={'return_5':close/close.shift(5)-1,
       'volatility_5':ret.rolling(5).std(),
       'price_ma_5':close/close.rolling(5).mean(),
       'activity_5':volume/volume.rolling(5).mean(),
       'momentum_20':close/close.shift(20)-1,
       'momentum_60':close/close.shift(60)-1,
       'volatility_20':ret.rolling(20).std(),
       'relative_volume_20':volume/volume.rolling(20).mean()}
    p['relative_spy_return_20']=p['momentum_20'].sub(p['momentum_20'].SPY,axis=0)
    return p


def build(bars, candidates, basis):
    close=bars.pivot(index='Date',columns='Ticker',values='Close').sort_index()
    adj=bars.pivot(index='Date',columns='Ticker',values='Adj Close').reindex(close.index)
    opening=bars.pivot(index='Date',columns='Ticker',values='Open').reindex(close.index)
    volume=bars.pivot(index='Date',columns='Ticker',values='Volume').reindex(close.index)
    sessions=CAL.sessions_in_range(close.index.min(),close.index.max())
    close,adj,opening,volume=[x.reindex(sessions) for x in (close,adj,opening,volume)]
    assert all(np.isfinite(x).all().all() for x in (close,adj,opening,volume))
    if basis=='adjusted_proxy':
        opening=opening*(adj/close)
        close=adj
    panels=feature_panels(close,volume)
    frame=candidates.loc[candidates.HoldingPeriodComplete & candidates.FeaturesComplete].copy()
    for name,panel in panels.items():
        frame[name]=[panel.at[r.SignalDate,r.Ticker] for r in frame.itertuples()]
    valid=np.isfinite(frame[list(panels)]).all(axis=1)
    # All model variants share the 60-session warm-up, including original4.
    good=frame.assign(valid=valid).groupby('SignalDate').valid.all()
    frame=frame.loc[frame.SignalDate.isin(good[good].index)].copy()
    frame['Target']=[opening.at[r.ExitDate,r.Ticker]/opening.at[r.EntryDate,r.Ticker]-1
                     for r in frame.itertuples()]
    if basis=='price':
        np.testing.assert_allclose(frame.Target,frame.TargetPriceReturn,atol=1e-12)
    # Audit all feature columns at each signal using truncated history.
    for date in frame.SignalDate.unique():
        truncated=feature_panels(close.loc[:date],volume.loc[:date])
        for name,panel in panels.items():
            np.testing.assert_allclose(panel.loc[date],truncated[name].loc[date],equal_nan=True)
    frame['momentum_5']=frame.return_5
    frame['reversal_5']=-frame.return_5
    frame['blend_momentum']=frame.groupby('SignalDate').momentum_20.rank(pct=True)+frame.groupby('SignalDate').momentum_60.rank(pct=True)
    return frame.sort_values(['SignalDate','Ticker']),opening


def fit_predict(train,selection,evaluation,features,ranker):
    trials=[]
    target=train.groupby('SignalDate').Target.rank(method='dense').astype(int)-1 if ranker else train.Target
    for depth in (2,3):
        for trees in (50,100):
            cls=xgb.XGBRanker if ranker else xgb.XGBRegressor
            params=dict(n_estimators=trees,max_depth=depth,learning_rate=.03,
                        min_child_weight=10,reg_lambda=10,subsample=1.,colsample_bytree=1.,
                        tree_method='hist',random_state=42,n_jobs=1,
                        objective='rank:pairwise' if ranker else 'reg:squarederror')
            if ranker:
                params.update(lambdarank_pair_method='topk',lambdarank_num_pair_per_sample=11)
            model=cls(**params)
            kwargs={'group':train.groupby('SignalDate',sort=False).size().to_numpy()} if ranker else {}
            model.fit(train[features],target,**kwargs)
            ic=metrics(selection,model.predict(selection[features])).RankIC.mean()
            trials.append((ic,depth,trees,model))
    best=sorted(trials,key=lambda t:(-t[0] if np.isfinite(t[0]) else np.inf,t[1],t[2]))[0]
    grid=[dict(SelectionIC=x,Depth=d,Trees=n,Selected=d==best[1] and n==best[2]) for x,d,n,_ in trials]
    return best[3].predict(evaluation[features]),grid


def weights_for(frame,method,rule,columns):
    rows=[];held=set()
    for _,g in frame.groupby('SignalDate',sort=True):
        ordered=g.sort_values(['Score','Ticker'],ascending=[False,True]).Ticker.tolist()
        if rule=='positive':
            weights={t:float(g.loc[g.Ticker.eq(t),'Score'].iloc[0]>0)/len(columns) for t in columns}
        elif rule=='cash_matched':
            invested=float(g.Score.gt(0).mean())
            weights={t:invested/len(columns) for t in columns}
        else:
            k=5 if rule=='top5' else 3
            keep=held.intersection(ordered[:5]) if rule=='buffer3' else set()
            for t in ordered:
                if len(keep)>=k: break
                keep.add(t)
            held=keep
            weights={t:float(t in held)/k for t in columns}
        rows.append(pd.Series(weights,name=g.EntryDate.iloc[0]))
    return pd.DataFrame(rows)[columns]


def bootstrap(values,rng,reps=2000,block=4):
    values=np.asarray(values);values=values[np.isfinite(values)]
    if len(values)==0:return (np.nan,np.nan,np.nan)
    starts=rng.integers(0,len(values),size=(reps,int(np.ceil(len(values)/block))))
    idx=((starts[:,:,None]+np.arange(block))%len(values)).reshape(reps,-1)[:,:len(values)]
    means=values[idx].mean(axis=1)
    return float(values.mean()),float(np.quantile(means,.025)),float(np.quantile(means,.975))


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,default=ROOT/'experiments/refinement_v1')
    args=parser.parse_args();out=args.output.resolve()
    if out==ROOT or out.is_relative_to(ROOT/'data') or out.is_relative_to(ROOT/'models'):
        parser.error('Choose an experiment output directory')
    out.mkdir(parents=True,exist_ok=False)
    audit(ROOT,write=False)
    paths=[ROOT/'data'/n for n in ('daily_prices.csv','spy_daily_prices.csv','corporate_actions.csv')]
    bars=pd.concat([pd.read_csv(p,parse_dates=['Date']) for p in paths[:2]],ignore_index=True)
    assert not bars.duplicated(['Date','Ticker']).any()
    candidate_path=ROOT/'prepared_data/weekly_candidates.csv'
    candidates=pd.read_csv(candidate_path,parse_dates=['SignalDate','EntryDate','ExitDate',
                            'SignalCloseUTC','EntryOpenUTC','ExitOpenUTC'])
    rng=np.random.default_rng(42)
    all_weekly=[];all_port=[];all_grid=[];all_scores=[];all_returns=[];all_coverage=[];policy_returns=[]
    for basis in ('price','adjusted_proxy'):
        base,opening=build(bars,candidates,basis)
        for window,edges in WINDOWS.items():
            stages=[]
            for phase,start,end in zip(('fit','selection','evaluation'),edges,edges[1:]):
                nominal=base.loc[base.SignalDate.ge(start)&base.SignalDate.lt(end)]
                a=nominal.loc[nominal.ExitDate.lt(end)].copy()
                assert a.groupby('SignalDate').Ticker.nunique().eq(11).all()
                assert len(a)>0
                stages.append(a)
                all_coverage.append(dict(Basis=basis,Window=window,Phase=phase,Weeks=a.SignalDate.nunique(),Rows=len(a)))
            train,selection,evaluation=stages
            assert train.ExitOpenUTC.max()<selection.SignalCloseUTC.min()
            assert selection.ExitOpenUTC.max()<evaluation.SignalCloseUTC.min()
            selection_scores={m:metrics(selection,selection[m]).RankIC.mean() for m in BASELINES}
            selected=sorted(BASELINES,key=lambda m:(-selection_scores[m],m))[0]
            scores={m:evaluation[m].to_numpy() for m in BASELINES}
            scores['selected_baseline']=scores[selected]
            for name,features in SETS.items():
                for ranker in (False,True):
                    method=('rank_' if ranker else 'reg_')+name
                    score,grid=fit_predict(train,selection,evaluation,features,ranker)
                    scores[method]=score
                    all_grid.extend([dict(Basis=basis,Window=window,Method=method,**row) for row in grid])
            schedule=evaluation[['SignalDate','EntryDate','ExitDate']].drop_duplicates().sort_values('EntryDate')
            assert np.array_equal(schedule.ExitDate.iloc[:-1],schedule.EntryDate.iloc[1:])
            market=opening.loc[schedule.EntryDate.min():schedule.ExitDate.max()]
            for method,score in scores.items():
                a=evaluation.assign(Score=score)
                meta=dict(Basis=basis,Window=window,Method=method,SelectedBaseline=selected)
                weekly=metrics(a,score).assign(**meta)
                all_weekly.append(weekly)
                all_scores.append(a[['SignalDate','EntryDate','ExitDate','Ticker','Target','Score']].assign(**meta))
                for rule in ('top3','top5','buffer3'):
                    targets=weights_for(a,method,rule,TICKERS)
                    prices=market[TICKERS]
                    for cost in (0,5,10,20):
                        curve,trades=simulate(prices,targets,cost)
                        daily=curve.NAV.pct_change().fillna(curve.NAV.iloc[0]-1)
                        vol=daily.std(ddof=1)*np.sqrt(252)
                        all_port.append(dict(**meta,Rule=rule,CostBps=cost,NetReturn=curve.NAV.iloc[-1]-1,
                            MaxDrawdown=curve.Drawdown.min(),Volatility=vol,
                            Sharpe=daily.mean()/daily.std(ddof=1)*np.sqrt(252) if vol>0 else np.nan,
                            Fees=trades.Fee.sum(),Turnover=(trades.TradedNotional/
                                np.interp(trades.Date.astype('int64'),curve.index.astype('int64'),curve.NAV)).sum(),
                            Start=str(prices.index.min().date()),End=str(prices.index.max().date())))
                        if cost==10:
                            prior=1.
                            for r in schedule.itertuples():
                                ending=float(curve.at[r.ExitDate,'NAV'])
                                row=dict(**meta,SignalDate=r.SignalDate,NetWeekly=ending/prior-1)
                                policy_returns.append(dict(**row,Rule=rule))
                                if rule=='top3':all_returns.append(row)
                                prior=ending
            # Invested-weight matched comparison for the original extended regression.
            a=evaluation.assign(Score=scores['reg_extended7'])
            extras={'equal_weight':pd.DataFrame(1/11,index=schedule.EntryDate,columns=TICKERS),
                    'positive':weights_for(a,'reg_extended7','positive',TICKERS),
                    'cash_matched':weights_for(a,'reg_extended7','cash_matched',TICKERS),
                    'SPY':pd.DataFrame(1.,index=[schedule.EntryDate.iloc[0]],columns=['SPY'])}
            for method,targets in extras.items():
                prices=market[targets.columns]
                curve,trades=simulate(prices,targets,10)
                daily=curve.NAV.pct_change().fillna(curve.NAV.iloc[0]-1);vol=daily.std()*np.sqrt(252)
                all_port.append(dict(Basis=basis,Window=window,Method=method,Rule='reference',CostBps=10,
                    NetReturn=curve.NAV.iloc[-1]-1,MaxDrawdown=curve.Drawdown.min(),Volatility=vol,
                    Sharpe=daily.mean()/daily.std()*np.sqrt(252) if vol>0 else np.nan,Fees=trades.Fee.sum(),
                    Turnover=np.nan,Start=str(prices.index.min().date()),End=str(prices.index.max().date())))
                prior=1.
                for r in schedule.itertuples():
                    ending=float(curve.at[r.ExitDate,'NAV'])
                    all_returns.append(dict(Basis=basis,Window=window,Method=method,SignalDate=r.SignalDate,
                                            NetWeekly=ending/prior-1));prior=ending
            print(f'Finished {basis}/{window}; selected baseline={selected}',flush=True)
    weekly=pd.concat(all_weekly,ignore_index=True);port=pd.DataFrame(all_port)
    summary=weekly.groupby(['Basis','Window','Method']).agg(Weeks=('RankIC','size'),
            ValidICWeeks=('RankIC','count'),MeanRankIC=('RankIC','mean'),Top3Overlap=('Top3Overlap','mean')).reset_index()
    returns=pd.DataFrame(all_returns);intervals=[]
    for metric,source in [('RankIC',weekly),('NetWeekly',returns)]:
        for (basis,window),g in source.groupby(['Basis','Window']):
            pivot=g.pivot(index='SignalDate',columns='Method',values=metric)
            refs=['selected_baseline']+(['equal_weight'] if metric=='NetWeekly' else [])
            for ref in refs:
                for method in pivot.columns:
                    if method==ref:continue
                    mean,lo,hi=bootstrap(pivot[method]-pivot[ref],rng)
                    intervals.append(dict(Basis=basis,Window=window,Metric=metric,Method=method,Reference=ref,
                                          MeanDifference=mean,CI95Low=lo,CI95High=hi))
    policy=pd.DataFrame(policy_returns)
    policy_effects=[]
    for (basis,window,method),g in policy.groupby(['Basis','Window','Method']):
        p=g.pivot(index='SignalDate',columns='Rule',values='NetWeekly')
        equal=returns.loc[returns.Basis.eq(basis)&returns.Window.eq(window)&returns.Method.eq('equal_weight')].set_index('SignalDate').NetWeekly
        for rule in ('top3','top5','buffer3'):
            for label,reference in [('equal_weight',equal),('top3',p.top3)]:
                if rule=='top3' and label=='top3':continue
                mean,lo,hi=bootstrap(p[rule]-reference,rng)
                policy_effects.append(dict(Basis=basis,Window=window,Method=method,Rule=rule,Reference=label,
                    MeanWeeklyDifference=mean,CI95Low=lo,CI95High=hi))
    outputs={'ranking_summary':summary,'portfolio_summary':port,'weekly_metrics':weekly,
             'weekly_returns':returns,'bootstrap':pd.DataFrame(intervals),
             'policy_weekly_returns':policy,'policy_effects':pd.DataFrame(policy_effects),
             'selection_grid':pd.DataFrame(all_grid),'coverage':pd.DataFrame(all_coverage),
             'scores':pd.concat(all_scores,ignore_index=True)}
    for name,table in outputs.items():table.to_csv(out/f'{name}.csv',index=False)
    # Paired feature/objective comparisons using the same block draws per difference.
    comparisons=[('add_momentum20_60','reg_extended9','reg_extended7'),
                 ('rank_add_momentum20_60','rank_extended9','rank_extended7'),
                 ('objective_extended7','rank_extended7','reg_extended7'),
                 ('relative_vs_own20','reg_extended7','reg_own_return20')]
    comparisons += [('drop_'+f,'reg_drop_'+f,'reg_extended7') for f in MID]
    deltas=[]
    for (basis,window),g in weekly.groupby(['Basis','Window']):
        pivot=g.pivot(index='SignalDate',columns='Method',values='RankIC')
        for label,new,old in comparisons:
            mean,lo,hi=bootstrap(pivot[new]-pivot[old],rng)
            deltas.append(dict(Basis=basis,Window=window,Change=label,MeanICDifference=mean,CI95Low=lo,CI95High=hi))
    pd.DataFrame(deltas).to_csv(out/'change_effects.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(14,5))
    methods=['momentum_5','momentum_20','momentum_60','selected_baseline','reg_extended7','reg_extended9','rank_extended7','rank_extended9']
    for ax,basis in zip(axes,('price','adjusted_proxy')):
        table=port.loc[port.Basis.eq(basis)&port.Rule.eq('top3')&port.CostBps.eq(10)&port.Method.isin(methods)]
        table.pivot(index='Window',columns='Method',values='NetReturn')[methods].plot.bar(ax=ax)
        ax.set_title(basis+' / Top 3 / 10 bps');ax.set_ylabel('Net return');ax.axhline(0,color='black',lw=.5)
        ax.legend(fontsize=6)
    fig.tight_layout();fig.savefig(out/'comparison.png',dpi=160);plt.close(fig)
    config=dict(status='retrospective_exploratory_no_new_holdout',created=datetime.now(timezone.utc).isoformat(),
        windows=WINDOWS,feature_sets=SETS,baseline_selection='highest selection mean Rank IC; no evaluation selection',
        price_basis='price vs dividend-adjusted open proxy; no actual dividend payment ledger',
        bootstrap='2000 circular moving-block draws, block=4 weeks; per window; unadjusted exploratory 95% intervals',
        versions=dict(numpy=np.__version__,pandas=pd.__version__,xgboost=xgb.__version__),
        input_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
                      for p in paths+[candidate_path,Path(__file__),ROOT/'plot_portfolio.py']})
    (out/'manifest.json').write_text(json.dumps(config,indent=2))
    print(f'Results: {out}',flush=True)


if __name__=='__main__':main()
