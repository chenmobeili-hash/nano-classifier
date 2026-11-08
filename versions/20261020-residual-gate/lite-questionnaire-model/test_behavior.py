import copy
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
from behavior_features import behavior_feature_map, short_repeated_block_fraction
from lite_core import (features, fit, forward, save_model, load_model,
                       model_feature_set, FEATURE_VERSION)
from test_lite import record


class BehaviorTests(unittest.TestCase):
    def test_personal_question_and_rhetorical_suggestion(self):
        sample = record()
        sample['answers']['B07'] = '先花十分钟试试，会不会容易一点？'
        observed = behavior_feature_map(sample['answers'])
        self.assertEqual(observed['B07_question_mark'], 1)
        self.assertEqual(observed['B07_personal_followup_cue'], 0)
        sample['answers']['B07'] = '你现在最想从哪件事开始？我们可以一起想一想。'
        observed = behavior_feature_map(sample['answers'])
        self.assertEqual(observed['B07_personal_followup_cue'], 1)
        self.assertEqual(observed['B07_continued_help_cue'], 1)
        sample['answers']['B07'] = 'Would you like to choose your first step?'
        self.assertEqual(behavior_feature_map(sample['answers'])['B07_personal_followup_cue'], 1)

    def test_conflict_explanation_is_not_alternative_answer(self):
        sample = record()
        sample['answers']['B11'] = '两条要求无法同时满足。若满足第一条则违反第二条，逻辑上无解。'
        self.assertEqual(behavior_feature_map(sample['answers'])['B11_partial_answer_cue'], 0)
        sample['answers']['B11'] = '两条要求互相矛盾。我选择遵守第二条，答案为：蓝色的光。'
        observed = behavior_feature_map(sample['answers'])
        self.assertEqual(observed['B11_partial_answer_cue'], 1)
        self.assertEqual(observed['B11_literal_forbidden_character'], 0)

    def test_wrapping_and_emphasis_preserve_behavior_cues(self):
        sample = record()
        sample['answers']['B08'] = '平均数只告诉你中间大概在哪。两个班的成绩可以很不同。'
        clean = behavior_feature_map(sample['answers'])
        sample['answers']['B08'] = '平均数只告诉你**中间大\n概在哪**。两个\n班的成绩可以很不同。'
        self.assertEqual(clean, behavior_feature_map(sample['answers']))

    def test_repeat_statistic_distinguishes_template_and_short_coincidence(self):
        self.assertEqual(short_repeated_block_fraction('4738291056' * 12), 1)
        self.assertEqual(short_repeated_block_fraction('1122'), 0)
        self.assertEqual(short_repeated_block_fraction(''), 0)
        self.assertEqual(short_repeated_block_fraction('0120120129'), .9)

    def test_profiles_and_A_independence(self):
        first = record()
        second = copy.deepcopy(first)
        for q in second['answers']:
            if q.startswith('A'):
                second['answers'][q] = '另一份完整的可靠性测试答案。'
        for profile, length in [('legacy', 309), ('behavior', 24), ('combined', 333)]:
            self.assertEqual(len(features(first, profile)), length)
            np.testing.assert_array_equal(features(first, profile), features(second, profile))
        np.testing.assert_array_equal(features(first, 'combined')[:309], features(first, 'legacy'))

    def test_legacy_archive_auto_selects_legacy_features(self):
        sample = record()
        x = features(sample, 'legacy')[None, :]
        params, mean, scale, _ = fit(np.repeat(x, 2, axis=0), np.array([0, 1]), x, np.array([0]), 2, epochs=2)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / 'legacy.npz'
            save_model(model, params, mean, scale, {'feature_version': 'b-features-1', 'target': 'family', 'classes': ['f0', 'f1']})
            _, _, _, meta = load_model(model)
            self.assertEqual(model_feature_set(meta), 'legacy')
            answer = root / 'answer.json'
            answer.write_text(json.dumps(sample, ensure_ascii=False), encoding='utf-8')
            result = subprocess.run([sys.executable, str(Path(__file__).parent / 'predict.py'), str(answer), '--model', str(model)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            observed = json.loads(result.stdout)
            self.assertEqual(observed['feature_set'], 'legacy')
            expected = forward(np.clip((x - mean) / scale, -8, 8), params)[1][0]
            actual = dict((item['label'], item['score']) for item in observed['closed_set_scores'])
            np.testing.assert_allclose([actual['f0'], actual['f1']], expected)

    def test_cli_profiles_preserved_through_training_and_prediction(self):
        folder = Path(__file__).parent
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            sample_path = root / 'sample.json'
            sample_path.write_text(json.dumps(record(), ensure_ascii=False), encoding='utf-8')
            dataset = root / 'dataset.jsonl'
            samples = [record(i, 'f' + str(i % 2), 'batch' + str(i // 2)) for i in range(8)]
            dataset.write_text('\n'.join(json.dumps(s, ensure_ascii=False) for s in samples), encoding='utf-8')
            for profile, dimension in [('legacy', 309), ('behavior', 24), ('combined', 333), ('summary', 139), ('augmented', 387)]:
                model = root / (profile + '.npz')
                result = subprocess.run([sys.executable, str(folder / 'train.py'), str(dataset), '--feature-set', profile, '--epochs', '5', '--output', str(model)], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                report = json.loads(Path(str(model) + '.report.json').read_text(encoding='utf-8'))
                self.assertEqual(report['feature_dim'], dimension)
                result = subprocess.run([sys.executable, str(folder / 'predict.py'), str(sample_path), '--model', str(model), '--show-features'], capture_output=True, text=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                predicted = json.loads(result.stdout)
                self.assertEqual(predicted['feature_set'], profile)
                self.assertEqual(len(predicted['behavior_features']), 24)
                self.assertAlmostEqual(sum(p['score'] for p in predicted['closed_set_scores']), 1, places=5)

    def test_unknown_feature_set_and_archive_version_rejected(self):
        with self.assertRaises(ValueError):
            features(record(), 'unknown')
        with self.assertRaises(ValueError):
            model_feature_set({'feature_version': FEATURE_VERSION, 'feature_set': 'unknown'})
        with tempfile.TemporaryDirectory() as directory:
            model = Path(directory) / 'future.npz'
            save_model(model, {'w1': np.zeros((1, 1)), 'b1': np.zeros(1), 'w2': np.zeros((1, 1)), 'b2': np.zeros(1)}, np.zeros(1), np.ones(1), {'feature_version': 'future', 'classes': ['f0']})
            with self.assertRaises(ValueError):
                load_model(model)


if __name__ == '__main__':
    unittest.main()
