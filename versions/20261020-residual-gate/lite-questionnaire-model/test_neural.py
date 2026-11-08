import copy
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from classifier import identify, load_references
from evaluate import evaluate
from neural_core import train_network, predict_scores, save_network, load_network, loss_and_grad, reference_signature
from test_current import test_directory


class NeuralTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, _ = load_references()
        cls.network = train_network(cls.refs)

    def test_gradient_against_finite_differences(self):
        rng = np.random.default_rng(2)
        x, y = rng.normal(size=(4, 3)), np.array([0, 1, 1, 0])
        p = {'w1': rng.normal(size=(3, 2))*.2, 'b1': rng.normal(size=2)*.2,
             'w2': rng.normal(size=(2, 2))*.2, 'b2': rng.normal(size=2)*.2}
        weights = np.array([.1, .2, .3, .4])
        _, grads = loss_and_grad(x, y, p, weights)
        for key in p:
            for index in np.ndindex(p[key].shape):
                value = p[key][index]
                p[key][index] = value+1e-6
                plus, _ = loss_and_grad(x, y, p, weights)
                p[key][index] = value-1e-6
                minus, _ = loss_and_grad(x, y, p, weights)
                p[key][index] = value
                self.assertAlmostEqual(grads[key][index], (plus-minus)/2e-6, places=6)

    def test_actual_learning_and_determinism(self):
        early = train_network(self.refs, epochs=1)
        self.assertLess(self.network['meta']['training_loss'], early['meta']['training_loss'])
        same = train_network(self.refs)
        for key in ('w1', 'b1', 'w2', 'b2', 'mean', 'scale'):
            np.testing.assert_array_equal(self.network[key], same[key])
        self.assertEqual(self.network['meta']['parameters'], 1092)

    def test_save_load_and_no_overwrite(self):
        with test_directory() as directory:
            path = directory/'network.npz'
            save_network(path, self.network)
            loaded = load_network(path)
            for r in self.refs:
                np.testing.assert_allclose(predict_scores(r, loaded), predict_scores(r, self.network), atol=1e-6)
            with self.assertRaises(FileExistsError):
                save_network(path, self.network)
            damaged = copy.deepcopy(self.network)
            damaged['w1'][0, 0] = np.nan
            bad = directory/'damaged.npz'
            save_network(bad, damaged)
            with self.assertRaises(ValueError):
                load_network(bad)

    def test_neural_scores_choose_family_not_distances(self):
        reference = self.refs[0]
        distance_result = identify(reference, self.refs, engine='distance')
        classes = self.network['meta']['classes']
        target = next(c for c in classes if c != distance_result['nearest_family'])
        scores = np.zeros(len(classes))
        scores[classes.index(target)] = 1
        with patch('classifier.predict_scores', return_value=scores):
            result = identify(reference, self.refs, model=self.network)
        self.assertEqual(result['nearest_family'], target)
        self.assertEqual(result['engine'], 'neural')
        self.assertEqual(result['ranking'][0]['score'], 1)

    def test_changed_training_data_requires_retraining(self):
        with self.assertRaisesRegex(ValueError, '重新训练'):
            identify(self.refs[0], self.refs[1:], model=self.network)
        changed = copy.deepcopy(self.refs)
        changed[0]['family'] = 'GPT'
        with self.assertRaises(ValueError):
            identify(changed[0], changed, model=self.network)

    def test_neural_evaluation_fits_only_reference_fold(self):
        def fit(refs, seed):
            return {'meta': {'signature': reference_signature(refs)}, 'seen': {r['model'] for r in refs}}
        def predict(query, refs, **kwargs):
            self.assertNotIn(query['model'], kwargs['model']['seen'])
            self.assertEqual(kwargs['model']['meta']['signature'], reference_signature(refs))
            return {'candidate': None, 'nearest_family': refs[0]['family'], 'margin': 0}
        with patch('evaluate.train_network', side_effect=fit) as training, patch('evaluate.identify', side_effect=predict):
            result = evaluate(self.refs, 'model')
        self.assertEqual(training.call_count, 12)
        self.assertEqual(result['tested'], 33)


if __name__ == '__main__':
    unittest.main()
