import unittest
import numpy as np
import pandas as pd
from dividend_account import cash_simulate


class DividendChecks(unittest.TestCase):
    def fixtures(self):
        dates=pd.bdate_range('2025-01-06',periods=5)
        prices=pd.DataFrame({'AAA':[10.,9.,9.,9.,9.]},index=dates)
        events=pd.DataFrame([dict(Ticker='AAA',ExDate=dates[1],PayDate=dates[2],AmountPerSourceUnit=1.)])
        return dates,prices,events

    def test_receivable_and_delayed_cash(self):
        d,p,e=self.fixtures();targets=pd.DataFrame({'AAA':[1.]},index=[d[0]])
        curve,_=cash_simulate(p,targets,e,0)
        np.testing.assert_allclose(curve.NAV,1.)
        self.assertAlmostEqual(curve.at[d[1],'Receivable'],.1)
        self.assertAlmostEqual(curve.at[d[2],'Cash'],0)
        self.assertAlmostEqual(curve.at[d[3],'Cash'],.1)

    def test_sell_on_ex_keeps_entitlement(self):
        d,p,e=self.fixtures();t=pd.DataFrame({'AAA':[1.,0.]},index=d[:2])
        curve,_=cash_simulate(p,t,e,0)
        self.assertAlmostEqual(curve.at[d[1],'Cash'],.9)
        self.assertAlmostEqual(curve.at[d[1],'Receivable'],.1)
        self.assertAlmostEqual(curve.NAV.iloc[-1],1.)

    def test_buy_on_ex_no_entitlement(self):
        d,p,e=self.fixtures();t=pd.DataFrame({'AAA':[1.]},index=[d[1]])
        curve,_=cash_simulate(p,t,e,0)
        self.assertEqual(curve.DividendAccrued.sum(),0)

    def test_unpaid_final_entitlement_remains_in_nav(self):
        d,p,e=self.fixtures();e['PayDate']=d[-1]+pd.Timedelta(days=30)
        t=pd.DataFrame({'AAA':[1.]},index=[d[0]])
        curve,_=cash_simulate(p,t,e,0)
        self.assertAlmostEqual(curve.Cash.iloc[-1],.9)
        self.assertAlmostEqual(curve.Receivable.iloc[-1],.1)
        self.assertAlmostEqual(curve.NAV.iloc[-1],1.)


if __name__=='__main__':unittest.main()
