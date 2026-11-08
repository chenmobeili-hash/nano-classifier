import copy
import unittest
import numpy as np
from test_lite import record
from lite_core import features, model_feature_set
from sequence_features import (sequence_feature_map, exact_animal, best_lag_match,
                               SUMMARY_FEATURE_NAMES, SEQUENCE_FEATURE_NAMES)


class SequenceTests(unittest.TestCase):
    def test_exact_cat_does_not_match_panda(self):
        cat, panda = record(), record()
        cat['answers']['B05'] = '猫'
        panda['answers']['B05'] = '大熊猫'
        x, z = features(cat, 'summary'), features(panda, 'summary')
        names = list(SUMMARY_FEATURE_NAMES)
        self.assertEqual(x[names.index('B05_exact_猫')], 1)
        self.assertEqual(z[names.index('B05_exact_猫')], 0)
        self.assertEqual(z[names.index('B05_exact_大熊猫')], 1)
        self.assertEqual(exact_animal('“猫咪”。'), '猫')
        self.assertEqual(exact_animal('熊猫'), '大熊猫')

    def test_long_cycle_detected_without_literal_template(self):
        sample = record()
        block = '012345678990123456783456789012'
        sample['answers']['B02'] = block * 5
        stats = sequence_feature_map(sample['answers'])
        self.assertEqual(stats['B02_best_lag_match'], 1)
        self.assertEqual(stats['B02_long_repeated_block_fraction'], 1)
        self.assertGreater(stats['B02_trigram_repeat_fraction'], .7)
        self.assertEqual(best_lag_match('4738291056' * 12)[1], 10 / 120)

    def test_internal_whitespace_not_strict_but_distribution_preserved(self):
        sample = record()
        original = sequence_feature_map(sample['answers'])
        sample['answers']['B02'] = sample['answers']['B02'][:50] + '\n' + sample['answers']['B02'][50:]
        wrapped = sequence_feature_map(sample['answers'])
        self.assertEqual(wrapped['B02_strict_format'], 0)
        for key in original:
            if key != 'B02_strict_format':
                self.assertEqual(original[key], wrapped[key])

    def test_summaries_finite_for_no_valid_sequences(self):
        sample = record()
        sample['answers'].update(B02='没有数字', B03='没有抛硬币序列', B04='没有骰子序列')
        values = np.array(list(sequence_feature_map(sample['answers']).values()))
        self.assertTrue(np.isfinite(values).all())
        self.assertEqual(values.size, len(SEQUENCE_FEATURE_NAMES))

    def test_summary_ignores_A_and_collection_mode(self):
        first = record()
        second = copy.deepcopy(first)
        second.update(family_label='different', claimed_model='different', collection_context={'mode': 'PTC', 'platform': 'cmdc'}, filename='different.txt')
        for q in second['answers']:
            if q.startswith('A'):
                second['answers'][q] = '另一份可靠性测试答案。'
        np.testing.assert_array_equal(features(first, 'summary'), features(second, 'summary'))
        self.assertEqual(len(features(first, 'summary')), 139)
        self.assertEqual(len(SUMMARY_FEATURE_NAMES), 139)

    def test_v2_missing_profile_still_means_combined(self):
        self.assertEqual(model_feature_set({'feature_version': 'b-features-2'}), 'combined')
        self.assertEqual(model_feature_set({'feature_version': 'b-features-3'}), 'augmented')

    def test_augmented_preserves_all_v2_features(self):
        sample = record()
        full = features(sample, 'augmented')
        np.testing.assert_array_equal(full[:333], features(sample, 'combined'))
        self.assertEqual(len(full), 387)


if __name__ == '__main__':
    unittest.main()
