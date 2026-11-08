import copy
import unittest
from classifier import load_references
from ability import score_points
from ability_rules import numeric_result, culprit_result, game_components
from dual_scoring import format_score
from strict_consistency import inspect_consistency
from scoring_text import answer_body
from neural_core import vector


class BoundaryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, _ = load_references()
        cls.good = next(r for r in cls.refs if r['source'] == 'gpt6.1sol-03.txt')

    def test_boundary_rules_do_not_reduce_content_or_format(self):
        r = copy.deepcopy(self.good)
        r.pop('raw_response', None)
        r['answers'] = {q: v+'\n──────' for q, v in r['answers'].items()}
        self.assertEqual(score_points(r, {})['provisional_score'], score_points(self.good, {})['provisional_score'])
        simple = next(x for x in format_score(r)['items'] if x['name'] == 'B01/B05/B06单项输出')
        self.assertEqual(simple['points'], 10)

    def test_only_trailing_rules_are_removed(self):
        self.assertEqual(answer_body('[1,3]\n────'), '[1,3]')
        self.assertEqual(answer_body('[1,3]\n解释\n────'), '[1,3]\n解释')
        self.assertIn('────', answer_body('────\n解释'))
        self.assertEqual(answer_body('12 34'), '12 34')

    def test_explicit_culprit_recheck_keeps_error_penalty(self):
        text = '丙打破了杯子。重新核查：若甲打破，只有乙真。结论：甲打破了杯子。'
        self.assertEqual(culprit_result(text)['score'], 1)
        self.assertEqual(inspect_consistency('A02', text)['cap_fraction'], .8)

    def test_recomputed_game_retains_initial_error(self):
        text = '先手必败。重来：0:P；2:P；7:P；9:P；14:P。先手必胜，第一步取1枚。'
        winner, moves = game_components(text)
        self.assertEqual((winner['score'], moves['score']), (1, 1))
        self.assertEqual(inspect_consistency('A10', text)['cap_fraction'], .8)

    def test_dp_states_and_unconstrained_total_are_not_final_claims(self):
        text = ('dp[8][4][0]=19\ndp[8][4][1]=16\ndp[8][4][2]=10\n总计=19+16+10=45\n'
                '条件2：首位与末位不同。结果：[1][0]=1,[2][0]=1\n答：共26个。')
        self.assertEqual(numeric_result('A12', text, 26)['score'], 1)
        self.assertEqual(inspect_consistency('A12', text)['cap_fraction'], 1)

    def test_invalid_dice_and_explanations_still_fail(self):
        r = copy.deepcopy(self.good)
        r['answers']['B04'] = '3,1,4,1,5,9,2,6,5,3,5,8,9,7,9,3,2,3,8,4,6,2,6,4\n────'
        dice = next(x for x in format_score(r)['items'] if x['name'] == 'B04骰子格式')
        self.assertEqual(dice['points'], 0)

    def test_grading_does_not_mutate_identity_features(self):
        r = copy.deepcopy(self.good)
        r['answers']['B01'] += '\n────'
        before = vector(r).copy()
        score_points(r, {})
        format_score(r)
        self.assertTrue((vector(r) == before).all())


if __name__ == '__main__':
    unittest.main()
