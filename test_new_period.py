import unittest
import numpy as np
import pandas as pd
from evaluate_new_period import dividend_factor
from run_research import feature_panels


class NewPeriodChecks(unittest.TestCase):
    def test_dividend_factor_neutralises_known_ex_drop(self):
        d=pd.bdate_range('2026-01-01',periods=4)
        c=pd.DataFrame({'AAA':[10.,10.,9.,9.]},index=d)
        e=pd.DataFrame([dict(Ticker='AAA',ExDate=d[2],AmountPerSourceUnit=1.)])
        adjusted=c*dividend_factor(c,e)
        np.testing.assert_allclose(adjusted.AAA,9.)

    def test_later_distributions_do_not_change_historical_features(self):
        d=pd.bdate_range('2025-01-01',periods=100)
        close=pd.DataFrame({'AAA':100+np.arange(100)*.1,'SPY':200+np.arange(100)*.2},index=d)
        v=pd.DataFrame(1000.,index=d,columns=close.columns)
        events=pd.DataFrame([dict(Ticker='AAA',ExDate=d[90],AmountPerSourceUnit=1.)])
        with_future=feature_panels(close*dividend_factor(close,events),v)
        without_future=feature_panels(close*dividend_factor(close,events.iloc[:0]),v)
        for name in with_future:
            np.testing.assert_allclose(with_future[name].loc[d[75],'AAA'],without_future[name].loc[d[75],'AAA'],rtol=1e-10)

    def test_unknown_ticker_event_rejected(self):
        d=pd.bdate_range('2026-01-01',periods=4)
        c=pd.DataFrame({'AAA':[10.,10.,9.,9.]},index=d)
        e=pd.DataFrame([dict(Ticker='BAD',ExDate=d[2],AmountPerSourceUnit=1.)])
        with self.assertRaises(KeyError):dividend_factor(c,e)


if __name__=='__main__':unittest.main()
