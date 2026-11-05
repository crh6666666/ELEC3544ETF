"""Checks for leakage, undefined ranking, direction semantics and sample alignment."""
import unittest

import numpy as np
import pandas as pd

from compare_predictions import (ALIGN_COLUMNS, align_predictions, build_baselines,
                                 comparison_metrics, score_predictions)


def sample_frames():
    frames = []
    for split, year, targets in [('train', 2023, [-.02, .04]),
                                  ('validation', 2024, [-.5, .6]),
                                  ('test', 2025, [0., -.03])]:
        frame = pd.DataFrame({'Split': split, 'Ticker': ['A', 'B'],
                              'TargetPriceReturn': targets})
        for field, day in [('SignalDate', '01-06'), ('EntryDate', '01-09'),
                            ('ExitDate', '01-16')]:
            frame[field] = pd.Timestamp(f'{year}-{day}')
        for field, day in [('SignalCloseUTC', '01-06'), ('EntryOpenUTC', '01-09'),
                            ('ExitOpenUTC', '01-16')]:
            frame[field] = pd.Timestamp(f'{year}-{day}', tz='UTC')
        for field in ALIGN_COLUMNS:
            if field not in frame:
                frame[field] = 'basis' if field in ('TargetBasis', 'PriceBasisAuditId') else .1
        frames.append(frame)
    return tuple(frames)


class PredictionComparisonTests(unittest.TestCase):
    def test_training_mean_ignores_validation_and_test(self):
        frames = sample_frames()
        predictions, parameters = build_baselines(frames, 'snapshot')
        self.assertAlmostEqual(parameters['train_mean']['prediction'], .01)
        changed = [f.copy() for f in frames]
        changed[1]['TargetPriceReturn'] = 100.
        changed[2]['TargetPriceReturn'] = -100.
        other, _ = build_baselines(changed, 'snapshot')
        np.testing.assert_array_equal(predictions['train_mean'].Prediction,
                                      other['train_mean'].Prediction)
        self.assertTrue(predictions['zero_return'].Prediction.eq(0).all())

    def test_zero_direction_and_undefined_rank(self):
        predictions, _ = build_baselines(sample_frames(), 'snapshot')
        metrics, weekly = score_predictions(predictions['zero_return'])
        test = metrics.set_index('Split').loc['test']
        self.assertEqual(test.direction_accuracy, .5)  # Only the actual zero matches.
        self.assertAlmostEqual(test.mae, .015)
        self.assertAlmostEqual(test.rmse, np.sqrt(.03 ** 2 / 2))
        self.assertTrue(metrics.mean_weekly_rank_ic.isna().all())
        self.assertTrue(metrics.defined_rank_ic_weeks.eq(0).all())
        self.assertTrue(weekly.rank_ic.isna().all())

    def test_direction_only_baseline_has_no_error_metrics(self):
        predictions, _ = build_baselines(sample_frames(), 'snapshot')
        metrics = comparison_metrics(predictions)
        up = metrics[metrics.ModelId == 'always_up'].set_index('Split')
        self.assertEqual(up.loc['test', 'direction_accuracy'], 0.)
        self.assertTrue(up[['mae', 'rmse', 'mean_weekly_rank_ic']].isna().all().all())

    def test_alignment_rejects_changed_labels_missing_and_duplicates(self):
        frames = sample_frames()
        reference = pd.concat(frames, ignore_index=True)
        predictions, _ = build_baselines(frames, 'snapshot')
        frame = predictions['train_mean']
        aligned = align_predictions(frame.sample(frac=1, random_state=42), reference,
                                    'train_mean', 'snapshot')
        self.assertEqual(len(aligned), len(reference))
        changed = frame.copy()
        changed.loc[0, 'TargetPriceReturn'] += .01
        for bad in (changed, frame.iloc[:-1], pd.concat([frame, frame.iloc[:1]])):
            with self.subTest(rows=len(bad)), self.assertRaises(ValueError):
                align_predictions(bad, reference, 'train_mean', 'snapshot')

    def test_alignment_rejects_stale_and_future_models(self):
        frames = sample_frames()
        reference = pd.concat(frames, ignore_index=True)
        predictions, _ = build_baselines(frames, 'snapshot')
        frame = predictions['train_mean']
        with self.assertRaisesRegex(ValueError, 'snapshot'):
            align_predictions(frame, reference, 'train_mean', 'other_snapshot')
        frame.loc[frame.Split == 'test', 'ModelAvailableAsOf'] = '2026-01-01T00:00:00+00:00'
        with self.assertRaisesRegex(ValueError, 'unavailable'):
            align_predictions(frame, reference, 'train_mean', 'snapshot')


if __name__ == '__main__':
    unittest.main()
