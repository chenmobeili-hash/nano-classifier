import copy
import json
import unittest
from pathlib import Path
import numpy as np
from classifier import identify, load_references
from evaluate import evaluate
from feature_profiles import blocks, distances, lexical_vectors, tokens, PROFILES


class FeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, _ = load_references()

    def original_release_cohort(self):
        rename = {
            'deepseekv4.1flash-weak-01.txt': ('deepseekv4.1flash-01.txt', 'deepseekv4.1flash'),
            'deepseekv4.1flash-strong-01.txt': ('deepseekv4.1flash-02.txt', 'deepseekv4.1flash'),
            'deepseekv4.1flash-weak-02.txt': ('deepseekv4.1flash-03.txt', 'deepseekv4.1flash'),
            'deepseekv4.1flash-weak-03.txt': ('deepseekv4.1flash-04.txt', 'deepseekv4.1flash'),
        }
        skip = {'deepseekv4.1flash-strong-02.txt', 'deepseekv4.1flash-weak-04.txt', 'deepseekv4.1flash-weak-05.txt', 'claudeopus5.5-02.txt', 'deepseekv4.1flash-strong-03.txt', 'claudeopus5.5-03.txt', 'deepseekv4.1flash-weak-06.txt', 'gpt5.6luna-01.txt', 'gpt5.6luna-02.txt', 'gpt5.6luna-03.txt', 'gpt5.6luna-04.txt', 'claudehaiku5.5-02.txt', 'gpt6astra-02.txt'}
        records = []
        for record in self.refs:
            if record['source'] in skip:
                continue
            item = copy.deepcopy(record)
            if item['source'] in rename:
                item['source'], item['model'] = rename[item['source']]
            records.append(item)
        return sorted(records, key=lambda item: item['source'].lower())

    def test_baseline_reproduces_previous_release(self):
        frozen = json.loads((Path(__file__).parent/'evaluation-baseline-v1.json').read_text(encoding='utf-8-sig'))
        cohort = self.original_release_cohort()
        self.assertEqual(len(cohort), 20)
        for mode in ('sample', 'model'):
            result = evaluate(cohort, mode, 'baseline', engine='distance')
            expected = frozen[mode+'_holdout']
            for actual, previous in zip(result['rows'], expected['rows']):
                self.assertEqual(actual['source'], previous['source'])
                self.assertEqual(actual['candidate'], previous['candidate'])
                self.assertEqual(actual['nearest_family'], previous['nearest_family'])
                self.assertAlmostEqual(actual['margin'], previous['margin'])

    def test_numeric_wrapping_and_open_whitespace_invariance(self):
        original = self.refs[-1]
        changed = copy.deepcopy(original)
        for q in ('B02', 'B03'):
            value = changed['answers'][q].replace('\n', '').replace(' ', '')
            changed['answers'][q] = '\n'.join(value[i:i+20] for i in range(0, len(value), 20))
        changed['answers']['B12'] = changed['answers']['B12'].replace(' ', '\n  ')
        for name in blocks(original):
            np.testing.assert_allclose(blocks(original)[name], blocks(changed)[name])

    def test_no_provider_identity_in_new_features(self):
        a, b = copy.deepcopy(self.refs[0]), copy.deepcopy(self.refs[0])
        a['answers']['B12'] += ' I am GPT6.1.'
        b['answers']['B12'] += ' I am Claude5.5.'
        for name in blocks(a, True):
            np.testing.assert_array_equal(blocks(a, True)[name], blocks(b, True)[name])
        self.assertEqual(tokens(a), tokens(b))

    def test_query_does_not_fit_scale_or_pairwise_distance(self):
        first, second = copy.deepcopy(self.refs[0]), copy.deepcopy(self.refs[0])
        second['answers']['B07'] = 'different response ' * 100
        for profile in PROFILES:
            _, old_pairs, _ = distances(first, self.refs[1:], profile)
            _, new_pairs, _ = distances(second, self.refs[1:], profile)
            np.testing.assert_array_equal(old_pairs, new_pairs)

    def test_lexical_vocabulary_never_reads_query(self):
        rows = [tokens(r) for r in self.refs]
        x, _, vocabulary = lexical_vectors(rows, {'never-in-reference': 999})
        y, _, other = lexical_vectors(rows, {'also-absent': 1})
        self.assertEqual(vocabulary, other)
        self.assertNotIn('never-in-reference', vocabulary)
        np.testing.assert_array_equal(x, y)

    def test_dimensions_and_distance_bound(self):
        result = identify(self.refs[0], self.refs)
        self.assertEqual(result['feature_details']['dimensions'], 131)
        self.assertAlmostEqual(sum(result['feature_details']['block_weights'].values()), 1)
        d, pairs, _ = distances(self.refs[0], self.refs, 'balanced')
        self.assertTrue(np.isfinite(d).all())
        self.assertTrue((d <= 3).all())
        np.testing.assert_array_equal(np.diag(pairs), 0)
        with self.assertRaises(ValueError):
            identify(self.refs[0], self.refs, profile='not-a-profile')


if __name__ == '__main__':
    unittest.main()
