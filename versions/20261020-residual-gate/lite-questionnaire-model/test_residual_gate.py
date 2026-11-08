"""Targeted checks of evidence locality, no metadata leakage and base equivalence."""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import copy
import unittest
import numpy as np
import residual_gate as R
import nlp_core as N
import neural_core as M
from classifier import load_references
from evaluate_residual_gate import Experiment


class ResidualTests(unittest.TestCase):
    def test_metadata_and_a_answers_do_not_enter_branch(self):
        records, _ = load_references()
        a = records[0]
        b = copy.deepcopy(a)
        b.update(source='OTHER.txt', family='Other', model='other', fingerprint='invented')
        for q in b['answers']:
            if q.startswith('A'):
                b['answers'][q] = 'An entirely different ability answer.'
        np.testing.assert_array_equal(R.buckets([a]), R.buckets([b]))

    def test_original_neighbours_after_sparse_sampling(self):
        text = ''.join(chr(0x4e00+i) for i in range(200))
        record = {'answers': {f'{group}{i:02d}': text for group in 'AB' for i in range(1, 13)}}
        out = R.buckets([record])
        positions = np.unique(np.linspace(0, len(text)-2, R.CENTERS).astype(int))
        for si, pos in enumerate(positions):
            for ki, offset in enumerate((-1, 0, 1)):
                start = pos+offset
                expected = N._bucket('B01\x1f'+text[start:start+2]) if 0 <= start and start+2 <= len(text) else -1
                self.assertEqual(out[0, 0, si, ki, 0], expected)

    def test_all_masked_input_is_finite(self):
        fake = {'w': np.zeros((N.DIMENSIONS, 5)), 'idf': np.ones(N.DIMENSIONS)}
        data = R.local_inputs(np.full((2, 12, R.CENTERS, 3, 3), -1), fake)
        self.assertEqual(float(data[1].sum()), 0)
        for variant in R.VARIANTS:
            prob, delta, _ = R.forward(data, np.zeros((2, 5)), R.init(5, variant), variant)
            self.assertTrue(np.isfinite(prob).all())
            np.testing.assert_array_equal(delta, 0)

    def test_top_family_protection_bound(self):
        rng = np.random.default_rng(9)
        base = np.tile([.34, 0, -.1, -.2, -.3], (32, 1))
        data = (rng.normal(size=(32, 12, 2, 3, 5)), np.full((32, 12, 2), 1/24))
        for variant in R.VARIANTS:
            p = R.init(5, variant)
            p['W'] += rng.normal(0, 100, p['W'].shape)
            p['b'] = np.array([-100, 100, 100, 100, 100.])
            prob, delta, _ = R.forward(data, base, p, variant)
            self.assertTrue((prob.argmax(axis=1) == 0).all())
            self.assertLessEqual(float(abs(delta).max()), .25)

    def test_cached_base_is_original_training(self):
        records, _ = load_references()
        exp = Experiment(records)
        indices = list(range(1, len(records)))
        cached = exp.base(indices)
        standard = M.train_network(records[1:])
        for key in standard:
            if key != 'meta':
                np.testing.assert_array_equal(cached['mlp'][key], standard[key])
        counts = N.count_matrix(records[1:]); idf = N.idf_from(counts)
        params = N.fit(N.apply_idf(counts, idf), exp.y[1:], exp.classes,
                       N.sample_weights(records[1:], exp.classes))
        for key in params:
            np.testing.assert_array_equal(cached['nlp'][key], params[key])


if __name__ == '__main__':
    unittest.main()
