import copy
import itertools
import json
import tempfile
import unittest
import uuid
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
import numpy as np
from classifier import identify, load_references, read_answer, fingerprint
from ability import score_ability, legacy_absolute_tier as tier
from evaluate import evaluate


@contextmanager
def test_directory():
    # Default inherited Windows ACLs; tempfile's private ACL conflicts with sandbox.
    path = Path(__file__).resolve().parent / ('test-data-' + uuid.uuid4().hex)
    path.mkdir()
    try:
        yield path
    finally:
        for file in path.iterdir():
            file.unlink()
        path.rmdir()


class CurrentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, cls.skipped = load_references()

    def test_current_inventory(self):
        self.assertEqual(len(self.refs), 33)
        self.assertEqual(self.skipped, [])
        self.assertEqual({r['family'] for r in self.refs}, {'GPT', 'GLM', 'DeepSeek', 'Claude'})
        models = {r['model'] for r in self.refs}
        self.assertIn('deepseekv4.1flash-strong', models)
        self.assertIn('deepseekv4.1flash-weak', models)
        self.assertNotIn('deepseekv4.1flash', models)
        self.assertEqual(sum(r['family'] == 'DeepSeek' for r in self.refs), 11)
        self.assertEqual(sum(r['family'] == 'GPT' for r in self.refs), 10)
        self.assertEqual(sum(r['model'] == 'gpt5.6luna' for r in self.refs), 4)
        self.assertEqual(sum(r['model'] == 'claudeopus5.5' for r in self.refs), 3)
        self.assertEqual(sum(r['model'] == 'claudehaiku5.5' for r in self.refs), 2)
        self.assertEqual(sum(r['model'] == 'gpt6astra' for r in self.refs), 2)
        self.assertEqual(sum(r['model'] == 'deepseekv4.1flash-strong' for r in self.refs), 3)
        self.assertEqual(sum(r['model'] == 'deepseekv4.1flash-weak' for r in self.refs), 6)
        self.assertTrue(any(r['source'] == 'deepseekv4.1flash-strong-03.txt' for r in self.refs))
        self.assertTrue(any(r['source'] == 'claudeopus5.5-03.txt' for r in self.refs))
        self.assertTrue(any(r['source'] == 'deepseekv4.1flash-weak-06.txt' for r in self.refs))
        self.assertTrue(any(r['source'] == 'gpt5.6luna-01.txt' for r in self.refs))
        self.assertTrue(any(r['source'] == 'gpt5.6luna-02.txt' for r in self.refs))
        self.assertTrue(any(r['source'] == 'gpt5.6luna-03.txt' for r in self.refs))
        self.assertTrue(any(r['source'] == 'gpt5.6luna-04.txt' for r in self.refs))
        self.assertTrue(any(r['source'] == 'claudehaiku5.5-02.txt' for r in self.refs))
        self.assertTrue(any(r['source'] == 'gpt6astra-02.txt' for r in self.refs))

    def test_family_independent_of_a_answers_and_metadata(self):
        original = self.refs[0]
        changed = copy.deepcopy(original)
        changed.update(family='fake', model='fake', source='fake.txt')
        for q in changed['answers']:
            if q.startswith('A'):
                changed['answers'][q] = '完全不同的错误答案'
        self.assertEqual(identify(original, self.refs), identify(changed, self.refs))

    def test_ability_independent_of_b_and_identity(self):
        original = self.refs[0]
        changed = copy.deepcopy(original)
        for q in changed['answers']:
            if q.startswith('B'):
                changed['answers'][q] = '另一个家族'
        changed.update(family='GPT', model='other')
        self.assertEqual(score_ability(original), score_ability(changed))

    def test_strict_json_and_output(self):
        record = copy.deepcopy(self.refs[0])
        record['answers'].update(A07='[2, 2, 2]\n[0, 1, 2]', A08='{"items":["pear","apple","banana","kiwi"]}', A09='[1,3]')
        result = score_ability(record)
        self.assertEqual([r['score'] for r in result['questions'][6:9]], [1, 1, 1])
        self.assertIsNone(result['tier'])
        self.assertEqual(result['pending_review'], 7)
        record['answers'].update(A07='```\n[2,2,2]\n[0,1,2]\n```', A09='[true,3]')
        result = score_ability(record)
        self.assertEqual(result['questions'][6]['score'], .8)
        self.assertEqual(result['questions'][8]['score'], 0)

    def test_manual_review_validation(self):
        review = {f'A{i:02d}': {'score': 1, 'reason': '人工验证'} for i in range(1, 13)}
        result = score_ability(self.refs[0], review)
        self.assertEqual(result['tier'], 'S 领先')
        self.assertEqual(result['score'], 100)
        for invalid in ({'A01': {'score': True, 'reason': 'x'}}, {'A01': {'score': 2, 'reason': 'x'}}, {'B01': {}}, {'A01': {'score': 1, 'reason': ''}}):
            with self.assertRaises(ValueError):
                score_ability(self.refs[0], invalid)

    def test_thresholds(self):
        self.assertTrue(tier(90).startswith('S'))
        self.assertTrue(tier(75).startswith('A'))
        self.assertTrue(tier(60).startswith('B'))
        self.assertTrue(tier(59.9).startswith('C'))
        with self.assertRaises(ValueError):
            identify(self.refs[0], self.refs, -1)

    def test_empty_duplicate_and_conflicting_references(self):
        with test_directory() as directory:
            p = Path(directory)
            a, b = self.refs[0], next(r for r in self.refs if r['family'] != self.refs[0]['family'])
            def write(name, record):
                (p/name).write_text('\n'.join('['+q+']\n'+v for q, v in record['answers'].items()), encoding='utf-8')
            write(a['source'], a)
            write(b['source'], b)
            write('claude-copy-99.txt', a)
            (p/'gpt-empty-01.txt').write_text('', encoding='utf-8')
            refs, skipped = load_references(p)
            self.assertEqual(len(refs), 2)
            self.assertEqual(len(skipped), 2)
            write('glm-conflict-01.txt', a)
            with self.assertRaises(RuntimeError):
                load_references(p)

    def test_invalid_questionnaire(self):
        with test_directory() as directory:
            p = Path(directory)/'answer.txt'
            p.write_text('[A01]\n5\n[A01]\n6', encoding='utf-8')
            with self.assertRaises(ValueError):
                read_answer(p)

    def test_model_holdout_has_no_model_leak(self):
        def fake(query, refs, **kwargs):
            self.assertNotIn(query['model'], {r['model'] for r in refs})
            self.assertNotIn(fingerprint(query), {fingerprint(r) for r in refs})
            return {'candidate': None, 'nearest_family': refs[0]['family'], 'margin': 0}
        with patch('evaluate.identify', side_effect=fake):
            report = evaluate(self.refs, 'model')
        self.assertEqual(report['tested'], 33)
        self.assertEqual(report['coverage'], 0)

    def test_numeric_answer_key_independently(self):
        self.assertEqual(pow(7, 123, 13), 5)
        self.assertEqual(sum(sum(bits) == 4 and not any(a and b for a, b in zip(bits, bits[1:])) for bits in itertools.product((0, 1), repeat=10)), 35)
        self.assertEqual(sum(sum(bits) == 4 and '111' not in ''.join(map(str, bits)) and bits[0] != bits[-1] for bits in itertools.product((0, 1), repeat=8)), 26)
        wins = [False]
        for n in range(1, 16):
            wins.append(any(not wins[n-m] for m in (1, 3, 4) if n >= m))
        self.assertEqual([m for m in (1, 3, 4) if not wins[15-m]], [1])
        dp = {(0, 0): 1}
        for x in range(9):
            for y in range(9):
                if (x, y) == (0, 0):
                    continue
                dp[x, y] = 0 if (x, y) in ((2, 2), (4, 5), (6, 3)) else dp.get((x-1, y), 0)+dp.get((x, y-1), 0)
        self.assertEqual(dp[8, 8], 3882)


if __name__ == '__main__':
    unittest.main()

