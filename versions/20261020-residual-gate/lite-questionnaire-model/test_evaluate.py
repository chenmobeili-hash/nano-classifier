import copy
import unittest
import numpy as np
from evaluate import evaluate, folds_for
from test_lite import record


class EvaluationTests(unittest.TestCase):
    def test_model_group_never_crosses_training_and_holdout(self):
        samples = [record(i, 'f' + str(i % 2)) for i in range(8)]
        for i, sample in enumerate(samples):
            sample['claimed_model'] = 'm' + str(i % 4)
        y = np.array([i % 2 for i in range(8)])
        folds, skipped = folds_for(y, samples, 'model', 'batch_id')
        self.assertFalse(skipped)
        for _, train, test in folds:
            self.assertFalse({samples[i]['claimed_model'] for i in train} & {samples[i]['claimed_model'] for i in test})
            self.assertEqual(set(y[train]), {0, 1})

    def test_extra_holdout_duplicates_are_rejected(self):
        samples = [record(i, 'f' + str(i % 2)) for i in range(8)]
        with self.assertRaises(ValueError):
            evaluate(samples, profiles=('behavior',), epochs=1, holdout_records=[copy.deepcopy(samples[0])])

    def test_missing_group_requires_explicit_fix(self):
        samples = [record(i, 'f' + str(i % 2)) for i in range(8)]
        samples[0]['batch_id'] = None
        with self.assertRaises(ValueError):
            folds_for(np.array([i % 2 for i in range(8)]), samples, 'group', 'batch_id')

    def test_comparison_covers_every_sample_in_each_profile(self):
        samples = [record(i, 'f' + str(i % 2)) for i in range(8)]
        report = evaluate(samples, profiles=('legacy', 'behavior', 'combined'), epochs=1)
        for profile in report['feature_sets'].values():
            predictions = profile['runs'][0]['predictions']
            self.assertEqual(len(predictions), len(samples))
            self.assertEqual({r['sample_id'] for r in predictions}, {r['sample_id'] for r in samples})


if __name__ == '__main__':
    unittest.main()
