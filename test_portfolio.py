"""Small accounting fixtures with hand-computable expected wealth."""
import unittest

import numpy as np
import pandas as pd

from plot_portfolio import simulate


class PortfolioAccountingTests(unittest.TestCase):
    def setUp(self):
        self.dates = pd.date_range('2025-01-06', periods=3, freq='B')

    def test_cash_stays_flat_without_fees(self):
        prices = pd.DataFrame({'A': [100., 150., 80.]}, index=self.dates)
        targets = pd.DataFrame({'A': [0.]}, index=self.dates[:1])
        curve, ledger = simulate(prices, targets, 10)
        np.testing.assert_allclose(curve.NAV, 1)
        np.testing.assert_allclose(curve.Drawdown, 0)
        self.assertEqual(ledger.Fee.sum(), 0)

    def test_buy_hold_return_and_drawdown(self):
        prices = pd.DataFrame({'A': [100., 110., 99.]}, index=self.dates)
        targets = pd.DataFrame({'A': [1.]}, index=self.dates[:1])
        curve, _ = simulate(prices, targets, 0)
        np.testing.assert_allclose(curve.NAV, [1, 1.1, .99])
        np.testing.assert_allclose(curve.Drawdown, [0, 0, -.1], atol=1e-14)

    def test_entry_and_liquidation_cost_are_self_financing(self):
        prices = pd.DataFrame({'A': [100., 100., 100.]}, index=self.dates)
        targets = pd.DataFrame({'A': [1.]}, index=self.dates[:1])
        curve, ledger = simulate(prices, targets, 100)
        self.assertAlmostEqual(curve.NAV.iloc[0], 1/1.01)
        self.assertAlmostEqual(curve.NAV.iloc[-1], .99/1.01)
        self.assertAlmostEqual(ledger.Fee.sum(), 1-curve.NAV.iloc[-1])
        np.testing.assert_allclose(ledger.Fee, ledger.TradedNotional*.01)
        self.assertLess(curve.Drawdown.iloc[0], 0)
        self.assertAlmostEqual(ledger.CashWeight.iloc[-1], 1)

    def test_rebalance_uses_drifted_holdings(self):
        prices = pd.DataFrame({'A': [100., 200., 200.], 'B': [100., 100., 200.]}, index=self.dates)
        targets = pd.DataFrame({'A': [.5, .5], 'B': [.5, .5]}, index=self.dates[:2])
        curve, ledger = simulate(prices, targets, 0)
        np.testing.assert_allclose(curve.NAV, [1, 1.5, 2.25])
        self.assertAlmostEqual(ledger.TradedNotional.iloc[1], .5)

    def test_negative_weights_and_invalid_cost_rejected(self):
        prices = pd.DataFrame({'A': [100., 100., 100.]}, index=self.dates)
        bad = pd.DataFrame({'A': [-.1]}, index=self.dates[:1])
        with self.assertRaises(ValueError):
            simulate(prices, bad, 10)
        with self.assertRaises(ValueError):
            simulate(prices, bad*0, float('nan'))


if __name__ == '__main__':
    unittest.main()
