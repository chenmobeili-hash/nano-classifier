import copy
import unittest

from ability import score_points
from classifier import load_references
from strict_consistency import inspect_consistency


class StrictGradingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, _ = load_references()

    def question(self, q, text, review=None):
        record = copy.deepcopy(self.refs[0])
        record.pop('source', None)
        record['answers'][q] = text
        return next(r for r in score_points(record, review or {})['questions'] if r['question'] == q)

    def test_corrected_wrong_answer_cannot_receive_full_credit(self):
        text = '6。更正：余数为5。费马小定理7^12≡1，123=12*10+3，7^3≡5。'
        row = self.question('A01', text)
        self.assertEqual(row['content_provisional_points'], 6)
        self.assertEqual(row['provisional_points'], 4.8)
        self.assertEqual(row['points_interval'][1], 4.8)
        self.assertEqual(row['consistency']['cap_fraction'], .8)

    def test_merely_saying_correction_does_not_cost_points(self):
        text = '5。重新核对：余数为5。费马小定理7^12≡1，123=12*10+3，7^3≡5。'
        self.assertEqual(self.question('A01', text)['provisional_points'], 6)
        self.assertEqual(inspect_consistency('A01', text)['events'], [])

    def test_wrong_calculation_then_correct_method_keeps_a_history_loss(self):
        text = '3882。2+2=5。更正：3882。动态规划N(x,y)=N(x-1,y)+N(x,y-1)，N(0,0)=1，禁点(2,2)、(4,5)、(6,3)置0。'
        row = self.question('A11', text)
        self.assertEqual(row['content_provisional_points'], 16)
        self.assertEqual(row['provisional_points'], 14.4)
        self.assertEqual(row['consistency']['cap_fraction'], .9)

    def test_corrected_wrong_congruence_still_costs_process_consistency(self):
        text = '5。7^6≡1。更正：余数为5。费马小定理7^12≡1，123=12*10+3，7^3≡5。'
        row = self.question('A01', text)
        self.assertEqual(row['content_provisional_points'], 6)
        self.assertEqual(row['provisional_points'], 5.4)

    def test_unresolved_assertions_are_stricter_than_a_clear_correction(self):
        conflicting = self.question('A02', '乙打破了杯子。答案：甲打破了杯子。甲假乙真丙假。')
        corrected = self.question('A02', '乙打破了杯子。更正：答案：甲打破了杯子。甲假乙真丙假。')
        self.assertEqual(conflicting['provisional_points'], 3.6)
        self.assertEqual(corrected['provisional_points'], 4.8)

    def test_cap_does_not_double_charge_existing_losses(self):
        text = '8232。更正：答案：3882。'
        row = self.question('A11', text)
        self.assertEqual(row['provisional_points'], 9.6)
        self.assertEqual(row['consistency']['applied_deduction'], 0)

    def test_legacy_review_cannot_erase_a_wrong_delivered_assertion(self):
        text = '8232。修正：答案：3882。'
        row = self.question('A11', text, {'A11': {'score': 1, 'reason': '最终计算正确'}})
        self.assertEqual(row['content_provisional_points'], 16)
        self.assertEqual(row['provisional_points'], 12.8)

    def test_conflicting_wrong_answers_still_have_a_consistency_cap(self):
        row = self.question('A11', '8232。答案：999。', {'A11': {'score': 1, 'reason': '内容复核分'}})
        self.assertEqual(row['consistency']['cap_fraction'], .6)
        self.assertEqual(row['provisional_points'], 9.6)

    def test_repeating_same_error_does_not_multiply_loss(self):
        a = '6。更正：余数为5。费马小定理7^12≡1，123=12*10+3，7^3≡5。'
        b = '6。答案：6。答案：6。更正：余数为5。费马小定理7^12≡1，123=12*10+3，7^3≡5。'
        self.assertEqual(self.question('A01', a)['provisional_points'], self.question('A01', b)['provisional_points'])

    def test_hypothetical_wrong_answer_and_case_analysis_are_not_assertions(self):
        text = '假设答案为6。这个假设不成立。最终答案为5。费马小定理7^12≡1，123=12*10+3，7^3≡5。'
        self.assertEqual(self.question('A01', text)['provisional_points'], 6)
        text = '甲打破。若乙打破则甲真乙假丙真，不符合；若丙打破则甲假乙真丙真，不符合。若甲打破则甲假乙真丙假。'
        self.assertEqual(inspect_consistency('A02', text)['events'], [])

    def test_corrected_extra_winning_move_is_recorded(self):
        text = '先手必胜，第一步取1枚或取4枚。更正：第一步取1枚，留下14，必败态为0、2、7、9、14。'
        row = self.question('A10', text)
        self.assertEqual(row['consistency']['cap_fraction'], .8)
        self.assertLessEqual(row['provisional_points'], 9.6)

    def test_corrected_wrong_winner_is_scored_but_not_as_perfect(self):
        text = '先手必败。更正：先手必胜，第一步取1枚，留下14，必败态为0、2、7、9、14。'
        row = self.question('A10', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['consistency']['cap_fraction'], .8)
        self.assertEqual(row['provisional_points'], 9.6)

    def test_wrong_subclass_count_is_work_not_the_main_answer(self):
        record = next(r for r in self.refs if r['source'] == 'deepseekv4.1flash-strong-02.txt')
        row = next(q for q in score_points(record)['questions'] if q['question'] == 'A12')
        self.assertEqual(row['content_provisional_points'], 14)
        self.assertEqual(row['provisional_points'], 12.6)
        self.assertEqual(row['consistency']['cap_fraction'], .9)

    def test_unsubstantiated_subcounts_do_not_get_full_method_credit(self):
        text = '26。总数为45，是(1+x+x²)⁵中x⁴的系数。首末相同有19，首末都是1有13，首末都是0有6，45-19=26。'
        row = self.question('A12', text)
        self.assertLess(row['components'][-1]['fraction'], 1)

    def test_all_scores_and_deductions_remain_consistent(self):
        for record in self.refs:
            result = score_points(record)
            self.assertAlmostEqual(result['content_provisional_score']-result['consistency_deduction'], result['provisional_score'])
            for row in result['questions']:
                self.assertAlmostEqual(row['provisional_points']+sum(d['points'] for d in row['deductions']), row['weight'])
                self.assertLessEqual(row['provisional_points'], row['weight']*row['consistency']['cap_fraction']+1e-8)


if __name__ == '__main__':
    unittest.main()
