"""Export a single, reproducible figure from saved test predictions."""
import argparse
import os
import tempfile
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir())/'elec3544etf_mpl'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import FuncFormatter, MultipleLocator
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
NAMES = ('ridge', 'random_forest', 'xgboost')
LABELS = ('Ridge', 'Random Forest', 'XGBoost')
COLORS = ('#118C88', '#416CC5', '#C98522')


def read_test_predictions(results):
    frames, rows = {}, []
    reference = None
    for name in NAMES:
        frame = pd.read_csv(results/name/'predictions.csv',
                            parse_dates=['SignalDate', 'EntryDate', 'ExitDate'])
        test = frame[frame.Split == 'test'].sort_values(['SignalDate', 'Ticker']).reset_index(drop=True)
        if test.empty or test.duplicated(['SignalDate','Ticker']).any():
            raise ValueError(f'{name}: empty or duplicated test data')
        if not np.isfinite(test[['Prediction','TargetPriceReturn']]).all().all():
            raise ValueError(f'{name}: nonfinite predictions')
        actual = test[['SignalDate','EntryDate','ExitDate','Ticker','TargetPriceReturn','SourceSnapshotId']]
        if reference is None:
            reference = actual
        else:
            pd.testing.assert_frame_equal(reference, actual, rtol=1e-12, atol=1e-14)
        error = test.Prediction-test.TargetPriceReturn
        rows.append(dict(ModelId=name, Split='test', rows=len(test),
            weeks=test.SignalDate.nunique(), mae=float(error.abs().mean()),
            rmse=float(np.sqrt((error**2).mean())),
            direction_accuracy=float((np.sign(test.Prediction)==np.sign(test.TargetPriceReturn)).mean())))
        frames[name] = test
    metrics = pd.DataFrame(rows).set_index('ModelId')
    # Read each model's own metrics so partial model runs cannot leave stale summary values.
    for name in NAMES:
        saved = pd.read_csv(results/name/'metrics.csv').set_index('Split').loc['test']
        for field in ('mae','rmse','direction_accuracy'):
            if not np.isclose(metrics.loc[name,field],saved[field],rtol=1e-10,atol=1e-12):
                raise ValueError(f'{name}: saved metrics do not match predictions')
    return frames, metrics


def configure_fonts():
    for path in ('/System/Library/Fonts/PingFang.ttc',
                 '/System/Library/Fonts/STHeiti Light.ttc',
                 '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'):
        if Path(path).exists():
            font_manager.fontManager.addfont(path)
            plt.rcParams['font.family'] = font_manager.FontProperties(fname=path).get_name()
            break
    else:
        plt.rcParams['font.family'] = ['Noto Sans CJK SC', 'Microsoft YaHei', 'DejaVu Sans']
    plt.rcParams.update({'axes.unicode_minus': False, 'font.size': 10,
        'axes.titleweight':'bold', 'axes.labelcolor':'#475569',
        'text.color':'#152338', 'xtick.color':'#64748B', 'ytick.color':'#64748B',
        'axes.edgecolor':'#D5DDE6', 'axes.spines.top':False,
        'axes.spines.right':False, 'savefig.facecolor':'#F6F8FB', 'svg.fonttype':'path'})


def main():
    parser = argparse.ArgumentParser(description='Plot saved model predictions; no training or downloads.')
    parser.add_argument('--results',type=Path,default=ROOT/'results')
    parser.add_argument('--output',type=Path,default=ROOT/'figures')
    args = parser.parse_args()
    frames, metrics = read_test_predictions(args.results)
    configure_fonts()
    first = frames['ridge']
    year_label = str(first.SignalDate.dt.year.min())
    if first.SignalDate.dt.year.max() != first.SignalDate.dt.year.min():
        year_label += '—'+str(first.SignalDate.dt.year.max())
    actual = first.TargetPriceReturn
    up_accuracy = float((actual > 0).mean())
    fig = plt.figure(figsize=(15.6,10.4),facecolor='#F6F8FB')
    fig.text(.065,.952,'行业 ETF · 三模型预测对比',fontsize=25,weight='bold')
    fig.text(.065,.913,f'{year_label} 测试集  /  {first.Ticker.nunique()} 只行业 ETF  /  '
             f'{first.SignalDate.nunique()} 周、{len(first)} 条资产–周记录',fontsize=12,color='#64748B')
    fig.text(.065,.866,'预测集中在零附近，对实际涨跌幅的捕捉仍较弱。',fontsize=13,color='#334155')
    gs = fig.add_gridspec(2,3,left=.065,right=.965,bottom=.135,top=.79,
                         hspace=.70,wspace=.30,height_ratios=[1,1.03])
    extent = max(abs(actual.min()),abs(actual.max()),
                 *(abs(frames[n].Prediction).max() for n in NAMES))*100*1.12
    percent = FuncFormatter(lambda x,_:f'{x:g}%')
    for i,(name,label,color) in enumerate(zip(NAMES,LABELS,COLORS)):
        ax=fig.add_subplot(gs[0,i],facecolor='white')
        frame=frames[name]
        ax.scatter(frame.TargetPriceReturn*100,frame.Prediction*100,
                   s=15,alpha=.43,color=color,edgecolors='none',rasterized=True,zorder=3)
        ax.plot([-extent,extent],[-extent,extent],ls='--',color='#A8B3C2',lw=1.2,zorder=1)
        ax.axhline(0,color='#D3DCE5',lw=.8);ax.axvline(0,color='#D3DCE5',lw=.8)
        ax.set(xlim=(-extent,extent),ylim=(-extent,extent),xlabel='实际周收益',ylabel='预测周收益')
        ax.set_title(label,fontsize=14,color=color,pad=15,loc='left')
        ax.xaxis.set_major_locator(MultipleLocator(10));ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_formatter(percent);ax.yaxis.set_major_formatter(percent)
        ax.grid(alpha=.22,color='#B9C4D2',lw=.6)
        ax.text(.04,.93,f'方向准确率  {metrics.loc[name,"direction_accuracy"]:.1%}',
                transform=ax.transAxes,va='top',fontsize=10,color='#475569')
    ax=fig.add_subplot(gs[1,:2],facecolor='white')
    weekly=first.groupby('EntryDate').TargetPriceReturn.mean()*100
    ax.plot(mdates.date2num(weekly.index.to_pydatetime()),weekly.values,color='#24364C',lw=1.8,label='实际均值',zorder=3)
    for name,label,color in zip(NAMES,LABELS,COLORS):
        values=frames[name].groupby('EntryDate').Prediction.mean()*100
        ax.plot(mdates.date2num(values.index.to_pydatetime()),values.values,color=color,lw=1.5,label=label,alpha=.9)
    ax.axhline(0,color='#CCD5E0',lw=.9)
    ax.set_title('每周平均收益：实际值与预测值',loc='left',fontsize=12,pad=16)
    ax.set_ylabel('11 只 ETF 的横截面平均周收益')
    ax.set_xlabel('调仓日期')
    ax.yaxis.set_major_formatter(percent)
    ticks=pd.date_range(weekly.index.min().replace(day=1),weekly.index.max(),freq='2MS')
    ax.set_xticks(mdates.date2num(ticks.to_pydatetime()),[f'{d.month:02d} 月' for d in ticks])
    ax.tick_params(axis='x',labelbottom=True)
    ax.grid(axis='y',alpha=.25,color='#B9C4D2')
    ax.legend(loc='upper right',ncol=4,frameon=False,fontsize=9,
              columnspacing=1.1,handlelength=1.8)
    ax=fig.add_subplot(gs[1,2],facecolor='white')
    x=np.arange(3);width=.31
    for shift,key,label,color in [(-width/2,'mae','MAE','#98ABB9'),(width/2,'rmse','RMSE','#416CC5')]:
        values=metrics.loc[list(NAMES),key].to_numpy()*100
        bars=ax.bar(x+shift,values,width,label=label,color=color,zorder=3)
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=8.5,color='#475569')
    ax.set_xticks(x,LABELS,fontsize=9)
    ax.set(ylim=(0,metrics.rmse.max()*100*1.26),ylabel='误差（百分点）')
    ax.set_title('预测误差：越低越好',loc='left',fontsize=12,pad=16)
    ax.grid(axis='y',alpha=.25,color='#B9C4D2',zorder=0)
    ax.legend(loc='upper left',ncol=2,frameon=False,fontsize=9)
    fig.text(.065,.075,f'方向判断参考：始终预测上涨的准确率为 {up_accuracy:.1%}。散点图虚线表示预测完全准确。',
             fontsize=10,color='#64748B')
    fig.text(.065,.043,'仅展示不含现金分红的价格收益；周均值不代表投资组合收益。同周的 11 只 ETF 并非独立时间样本。',
             fontsize=9.5,color='#64748B')
    args.output.mkdir(parents=True,exist_ok=True)
    for suffix in ('png','svg'):
        path=args.output/f'model_comparison_{year_label}.{suffix}'
        fig.savefig(path,dpi=180)
        print(path)
    metrics.to_csv(args.output/'plotted_test_metrics.csv')
    plt.close(fig)


if __name__=='__main__':
    main()
