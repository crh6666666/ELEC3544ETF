"""Behavioral checks for refinement comparison rules."""
import unittest
import numpy as np
import pandas as pd
from run_research import feature_panels, weights_for, bootstrap


class ResearchChecks(unittest.TestCase):
    def test_feature_future_invariance(self):
        dates=pd.bdate_range('2020-01-01',periods=85)
        close=pd.DataFrame({'AAA':np.exp(np.arange(85)*.001),
                            'SPY':np.exp(np.arange(85)*.0005)},index=dates)
        volume=close*100
        altered=close.copy();altered.iloc[70:]*=9
        left=feature_panels(close,volume);right=feature_panels(altered,volume)
        for name in left:
            pd.testing.assert_frame_equal(left[name].iloc[:70],right[name].iloc[:70])

    def test_buffer_retains_top5_incumbents(self):
        names=list('ABCDEF')
        rows=[]
        for date,order in [('2024-01-05','ABCDEF'),('2024-01-12','DABCEF')]:
            for rank,ticker in enumerate(order):
                rows.append(dict(SignalDate=pd.Timestamp(date),EntryDate=pd.Timestamp(date)+pd.Timedelta(days=3),
                                 Ticker=ticker,Score=6-rank))
        frame=pd.DataFrame(rows)
        top=weights_for(frame,'m','top3',names)
        buffer=weights_for(frame,'m','buffer3',names)
        self.assertGreater(top.iloc[-1].D,0)
        self.assertEqual(buffer.iloc[-1].D,0)
        np.testing.assert_allclose(buffer.sum(axis=1),1)
        self.assertEqual(set(buffer.columns[buffer.iloc[-1].gt(0)]),set('ABC'))

    def test_top5_cap_and_cash_match(self):
        names=list('ABCDEFGHIJK')
        a=pd.DataFrame(dict(SignalDate=[pd.Timestamp('2024-01-05')]*11,
                            EntryDate=[pd.Timestamp('2024-01-08')]*11,Ticker=names,
                            Score=np.arange(11)-5))
        top=weights_for(a,'m','top5',names)
        self.assertLessEqual(top.max().max(),.2)
        positive=weights_for(a,'m','positive',names)
        matched=weights_for(a,'m','cash_matched',names)
        np.testing.assert_allclose(positive.sum(axis=1),matched.sum(axis=1))

    def test_block_interval_constant_difference(self):
        mean,lo,hi=bootstrap(np.repeat(.03,25),np.random.default_rng(42))
        np.testing.assert_allclose([mean,lo,hi],.03)


if __name__=='__main__':unittest.main()
