import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from lite_core import (features, canonical_answers, prepare_data, split_data,
                       loss_and_grad, fit, forward, save_model, load_model, FEATURE_VERSION)


def record(i=0, label='f0', batch='b0'):
    answers = {f'{g}{k:02d}': f'示例{i}。' for g in 'AB' for k in range(1, 13)}
    answers.update(B01='7', B02='0123456789'*12, B03='HT'*40,
                   B04=','.join(str(k%6+1) for k in range(24)), B05='兔', B06='42')
    return {'sample_id': str(i), 'answers': answers, 'family_label': label, 'batch_id': batch}


class LiteTests(unittest.TestCase):
    def test_features_ignore_metadata(self):
        a, b = record(), record()
        b.update(family_label='other', claimed_model='SomeModel', provider='SomeProvider')
        np.testing.assert_array_equal(features(a), features(b))
        self.assertTrue(np.isfinite(features(a)).all())

    def test_missing_question(self):
        a = record()
        a['answers'].pop('B12')
        with self.assertRaises(ValueError):
            features(a)

    def test_group_split(self):
        records = [record(i, f'f{i%2}', f'b{i//2}') for i in range(12)]
        y = np.array([i%2 for i in range(12)])
        train, val, mode = split_data(y, records, mode='group')
        self.assertFalse(set(train) & set(val))
        self.assertFalse({records[i]['batch_id'] for i in train} & {records[i]['batch_id'] for i in val})
        self.assertEqual(set(y[train]), {0, 1})
        self.assertEqual(set(y[val]), {0, 1})

    def test_duplicate_conflict(self):
        a, b = record(), record()
        b['family_label'] = 'f1'
        with self.assertRaises(ValueError):
            prepare_data([a, b], 'family')

    def test_gradient(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=(4, 3))
        y = np.array([0, 1, 0, 1])
        p = {'w1': rng.normal(size=(3, 2))*.2, 'b1': np.zeros(2),
             'w2': rng.normal(size=(2, 2))*.2, 'b2': np.zeros(2)}
        _, grads = loss_and_grad(x, y, p)
        for key in p:
            for index in np.ndindex(p[key].shape):
                old = p[key][index]
                eps = 1e-5
                p[key][index] = old+eps
                plus = loss_and_grad(x, y, p)[0]
                p[key][index] = old-eps
                minus = loss_and_grad(x, y, p)[0]
                p[key][index] = old
                self.assertAlmostEqual(grads[key][index], (plus-minus)/(2*eps), places=6)

    def test_fit_and_roundtrip(self):
        x = np.array([[-2., 1.], [-1., 1.], [1., 1.], [2., 1.]], dtype=np.float32)
        y = np.array([0, 0, 1, 1])
        val = np.array([[-1.5, 1.], [1.5, 1.]], dtype=np.float32)
        params, mean, scale, _ = fit(x, y, val, np.array([0, 1]), 2, hidden=4, epochs=100)
        np.testing.assert_array_equal(mean, x.mean(axis=0))
        self.assertTrue(np.isfinite(scale).all())
        probs = forward((val-mean)/scale, params)[1]
        np.testing.assert_array_equal(probs.argmax(axis=1), [0, 1])
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'model.npz'
            save_model(path, params, mean, scale, {'feature_version': FEATURE_VERSION, 'classes': ['f0', 'f1']})
            p2, m2, s2, meta = load_model(path)
            np.testing.assert_allclose(probs, forward((val-m2)/s2, p2)[1])
            with self.assertRaises(FileExistsError):
                save_model(path, params, mean, scale, meta)

    def test_cli_pipeline(self):
        folder = Path(__file__).resolve().parent
        with tempfile.TemporaryDirectory() as d:
            directory = Path(d)
            commands = [
                ['make_demo.py', '--output-dir', str(directory/'data')],
                ['train.py', str(directory/'data/synthetic.jsonl'), '--output', str(directory/'model.npz'), '--epochs', '80'],
                ['predict.py', str(directory/'data/synthetic-answer.txt'), '--model', str(directory/'model.npz')],
            ]
            for command in commands:
                result = subprocess.run([sys.executable, str(folder/command[0]), *command[1:]], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
            output = json.loads(result.stdout)
            self.assertEqual(len(output['closed_set_scores']), 2)
            self.assertAlmostEqual(sum(x['score'] for x in output['closed_set_scores']), 1., places=5)
            report = json.loads((directory/'model.npz.report.json').read_text())
            self.assertTrue(report['split'].startswith('group:'))
            self.assertFalse(set(report['train_sample_ids']) & set(report['validation_sample_ids']))


if __name__ == '__main__':
    unittest.main()
