import copy
import unittest
from pathlib import Path
from classifier import load_references
from result_scoring import score_results, build_result_ranking


class ResultTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, skipped = load_references()
        assert not skipped
        cls.by_source = {r['source']:r for r in cls.refs}

    def test_partial_proof_and_format_do_not_reduce_results(self):
        for source in ('claudeopus5.5-04.txt', 'claudesonnet5.5-01.txt', 'gpt5.6luna-02.txt'):
            r = score_results(self.by_source[source])
            self.assertEqual(r['score'], 100)
            self.assertEqual(r['correct_questions'], 12)

    def test_wrong_extra_first_move_fails_entire_question(self):
        r = score_results(self.by_source['gpt6luna-01.txt'])
        self.assertEqual(r['score'], 88)
        self.assertFalse(r['questions'][9]['correct'])

    def test_explicit_final_correction_is_accepted(self):
        r = score_results(self.by_source['claudehaiku5.5-03.txt'])
        self.assertTrue(r['questions'][8]['correct'])

    def test_unwithdrawn_result_conflict_is_not_full_credit(self):
        r = score_results(self.by_source['deepseekv4.1flash-weak-04.txt'])
        self.assertFalse(r['questions'][5]['correct'])

    def test_review_invalidated_by_changed_answer(self):
        r = copy.deepcopy(self.by_source['claudeopus5.5-04.txt'])
        r['answers']['A12'] += '\n最终答案为27。'
        with self.assertRaises(ValueError):
            score_results(r)

    def test_unknown_conflicts_remain_pending(self):
        r = copy.deepcopy(self.by_source['claudeopus5.5-04.txt'])
        missing = Path(__file__).with_name('missing-result-review-for-test.json')
        self.assertFalse(missing.exists())
        result = score_results(r, missing)
        self.assertGreater(result['pending_review'], 0)
        self.assertIsNone(result['score'])

    def test_all_current_samples_and_marked_comparison(self):
        ranking = build_result_ranking(self.refs)
        self.assertEqual(ranking['sample_count'], len(self.refs))
        self.assertEqual(ranking['comparison_without_marked']['sample_count'], len(self.refs)-1)
        self.assertTrue(all(s['pending_review'] == 0 for s in ranking['samples']))
        self.assertTrue(all(s['score'] == sum(q['points'] for q in s['result']['questions']) for s in ranking['samples']))


if __name__ == '__main__':
    unittest.main()
