"""Weekly US-market schedule and calendar checks."""
import exchange_calendars as xc
import pandas as pd

TICKERS="XLK XLF XLV XLY XLP XLE XLI XLB XLU XLRE XLC".split()
CAL=xc.get_calendar("XNYS",start="2017-12-01",end="2027-02-01")

def schedule(start,end):
    ss=CAL.sessions_in_range(pd.Timestamp(start)-pd.Timedelta(days=14),pd.Timestamp(end)+pd.Timedelta(days=28))
    g=pd.Series(ss,index=ss).groupby(ss.to_period('W-SUN'))
    first,last=g.first(),g.last()
    rows=[]
    for i in range(len(first)-2):
        signal,entry,exit=last.iloc[i],first.iloc[i+1],first.iloc[i+2]
        if pd.Timestamp(start)<=signal<=pd.Timestamp(end):
            rows.append(dict(SignalDate=signal,EntryDate=entry,ExitDate=exit,
                SignalCloseUTC=CAL.session_close(signal),EntryOpenUTC=CAL.session_open(entry),ExitOpenUTC=CAL.session_open(exit)))
    return pd.DataFrame(rows)

def artificial_checks():
    # Calendar-only fixtures; never appended to the real dataset.
    cases=[('2024-03-28','2024-04-01','2024-04-08'),('2024-01-12','2024-01-16','2024-01-22'),('2021-12-31','2022-01-03','2022-01-10')]
    for s,e,x in cases:
        row=schedule(s,s).iloc[0]
        assert str(row.EntryDate.date())==e and str(row.ExitDate.date())==x
    return {'good_friday':True,'monday_holiday':True,'cross_year':True}
