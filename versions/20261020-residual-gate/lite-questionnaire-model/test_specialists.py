import json
import tempfile
import subprocess
import sys
import unittest
from pathlib import Path
import numpy as np
from test_lite import record
from lite_core import FEATURE_VERSION, load_model, save_model
from specialist_core import (ARCHITECTURE, EVIDENCE_VERSION, EVIDENCE_NAMES,
                             evidence_features, combined_forward, combined_loss_and_grad,
                             fit_specialists, predict_probabilities, binary_loss_and_grad, mixture_loss_and_grad)


class SpecialistTests(unittest.TestCase):
    def test_no_a_or_metadata_leak(self):
        a, b = record(), record()
        b.update(family_label='arbitrary', claimed_model='changed', source_file='changed',
                 collection_context={'platform': 'changed', 'mode': 'changed'})
        for key in b['answers']:
            if key.startswith('A'):
                b['answers'][key] = '完全不同的答案'
        np.testing.assert_array_equal(evidence_features(a), evidence_features(b))
        self.assertEqual(evidence_features(a).shape, (len(EVIDENCE_NAMES),))

    def test_joint_gradient(self):
        rng = np.random.default_rng(8)
        x, e, y = rng.normal(size=(4, 3)), rng.normal(size=(4, 4)), np.array([0, 1, 0, 1])
        params = {'w1': rng.normal(size=(3, 2))*.1, 'b1': np.zeros(2),
                  'w2': rng.normal(size=(2, 2))*.1, 'b2': np.zeros(2),
                  'specialist_w': rng.normal(size=(4, 2))*.1,
                  'specialist_mask': np.array([[1, 0], [1, 1], [0, 1], [1, 1]])}
        _, gradients = combined_loss_and_grad(x, e, y, params)
        for key, gradient in gradients.items():
            for index in np.ndindex(gradient.shape):
                old, eps = params[key][index], 1e-5
                params[key][index] = old+eps
                plus = combined_loss_and_grad(x, e, y, params)[0]
                params[key][index] = old-eps
                minus = combined_loss_and_grad(x, e, y, params)[0]
                params[key][index] = old
                self.assertAlmostEqual(gradient[index], (plus-minus)/(2*eps), places=6)

    def test_evidence_learns_and_roundtrips(self):
        x = np.zeros((4, 3), dtype=np.float32)
        e = np.zeros((4, len(EVIDENCE_NAMES)), dtype=np.float32)
        e[:, 0] = [-2, -1, 1, 2]
        y = np.array([0, 0, 1, 1])
        val_x = np.zeros((2, 3), dtype=np.float32)
        val_e = np.zeros((2, len(EVIDENCE_NAMES)), dtype=np.float32)
        val_e[:, 0] = [-1.5, 1.5]
        params, mean, scale, history = fit_specialists(x, y, val_x, np.array([0, 1]), 2, e, val_e, epochs=200, hidden=4)
        z = np.clip((val_e-params['evidence_mean'])/params['evidence_scale'], -8, 8)
        _, probabilities, shared, adjustment = combined_forward((val_x-mean)/scale, z, params)
        np.testing.assert_array_equal(probabilities.argmax(axis=1), [0, 1])
        np.testing.assert_allclose(shared[0], shared[1])
        self.assertGreater(float(np.abs(adjustment).sum()), .1)
        np.testing.assert_array_equal(params['evidence_mean'], e.mean(axis=0))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'special.npz'
            meta = {'feature_version': FEATURE_VERSION, 'architecture': ARCHITECTURE,
                    'evidence_version': EVIDENCE_VERSION, 'feature_set': 'augmented',
                    'classes': ['a', 'b'], 'specialist_strength': .5}
            save_model(path, params, mean, scale, meta)
            loaded, m2, s2, _ = load_model(path)
            p2 = combined_forward((val_x-m2)/s2, z, loaded)[1]
            np.testing.assert_allclose(probabilities, p2)

    def test_validation_does_not_select_or_scale_evidence(self):
        rng = np.random.default_rng(9)
        x, y = rng.normal(size=(6, 4)), np.array([0, 0, 0, 1, 1, 1])
        evidence = rng.normal(size=(6, len(EVIDENCE_NAMES)))
        one, mean, scale, _ = fit_specialists(x, y, x[:2], y[:2], 2, evidence, evidence[:2], epochs=2)
        two, _, _, _ = fit_specialists(x, y, x[:2]*1000, y[:2], 2, evidence, evidence[:2]*1000, epochs=2)
        for key in ('specialist_mask', 'evidence_mean', 'evidence_scale'):
            np.testing.assert_array_equal(one[key], two[key])
        np.testing.assert_array_equal(mean, x.mean(axis=0))

    def test_independent_binary_gradient(self):
        rng = np.random.default_rng(19)
        e, target = rng.normal(size=(4, 3)), np.array([[1., 0.], [1., 0.], [0., 1.], [0., 1.]])
        w, bias, mask = rng.normal(size=(3, 2))*.2, rng.normal(size=2)*.1, np.array([[1., 0.], [1., 1.], [0., 1.]])
        sample_weights = np.ones_like(target)
        _, g_w, g_b = binary_loss_and_grad(e, target, w, bias, mask, sample_weights)
        for value, gradient in ((w, g_w), (bias, g_b)):
            for index in np.ndindex(value.shape):
                old, eps = value[index], 1e-5
                value[index] = old+eps
                plus = binary_loss_and_grad(e, target, w, bias, mask, sample_weights)[0]
                value[index] = old-eps
                minus = binary_loss_and_grad(e, target, w, bias, mask, sample_weights)[0]
                value[index] = old
                self.assertAlmostEqual(gradient[index], (plus-minus)/(2*eps), places=6)

    def test_mixture_gradient(self):
        rng = np.random.default_rng(4)
        shared, specialist = rng.normal(size=(6, 2)), rng.normal(size=(6, 2))
        mixture, y = np.array([.2, .3]), np.array([0, 1, 0, 1, 0, 1])
        _, gradient = mixture_loss_and_grad(shared, specialist, y, mixture)
        for index in range(2):
            old, eps = mixture[index], 1e-5
            mixture[index] = old+eps
            plus = mixture_loss_and_grad(shared, specialist, y, mixture)[0]
            mixture[index] = old-eps
            minus = mixture_loss_and_grad(shared, specialist, y, mixture)[0]
            mixture[index] = old
            self.assertAlmostEqual(gradient[index], (plus-minus)/(2*eps), places=6)

    def test_specialist_cli_and_score_decomposition(self):
        folder = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            commands = [
                ['make_demo.py', '--output-dir', str(root/'demo')],
                ['train.py', str(root/'demo/synthetic.jsonl'), '--classifier', 'specialist', '--epochs', '30', '--output', str(root/'special.npz')],
                ['predict.py', str(root/'demo/synthetic-answer.txt'), '--model', str(root/'special.npz'), '--show-features'],
            ]
            for command in commands:
                process = subprocess.run([sys.executable, str(folder/command[0]), *command[1:]], text=True, capture_output=True)
                self.assertEqual(process.returncode, 0, process.stderr)
            output = json.loads(process.stdout)
            self.assertEqual(output['architecture'], ARCHITECTURE)
            self.assertAlmostEqual(sum(p['score'] for p in output['closed_set_scores']), 1, places=5)
            params, mean, scale, metadata = load_model(root/'special.npz')
            sample = {'raw_response': (root/'demo/synthetic-answer.txt').read_text(encoding='utf-8')}
            _, detail = predict_probabilities([sample], params, mean, scale, metadata)
            contributions = params['specialist_mix'] * detail['standardized_evidence'][0, :, None] * params['specialist_w'] * params['specialist_mask']
            np.testing.assert_allclose(contributions.sum(axis=0)+detail['other_adjustments'][0], detail['specialist_adjustments'][0], atol=1e-5)
            self.assertTrue(((params['specialist_mix'] >= 0) & (params['specialist_mix'] <= .5)).all())


if __name__ == '__main__':
    unittest.main()
