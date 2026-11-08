import copy
import unittest
import numpy as np
from ability import score_ability, WEIGHTS
from ability_rules import arithmetic, numeric_result, recurrence_components, clean
from classifier import load_references


class AbilityRubricTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, _ = load_references()

    def question(self, q, text):
        record = copy.deepcopy(self.refs[0])
        record['answers'][q] = text
        return next(r for r in score_ability(record)['questions'] if r['question'] == q)

    def test_weights_and_all_current_bounds(self):
        self.assertEqual(sum(WEIGHTS.values()), 100)
        self.assertEqual(sum(WEIGHTS[q] for q in ('A10', 'A11', 'A12')), 42)
        for record in self.refs:
            result = score_ability(record)
            self.assertLessEqual(result['score_interval'][0], result['provisional_score'])
            self.assertLessEqual(result['provisional_score'], result['score_interval'][1])
            self.assertLessEqual(result['score_interval'][1], 100)
            for row in result['questions']:
                self.assertAlmostEqual(row['provisional_points']+sum(d['points'] for d in row['deductions']), row['weight'], places=3)

    def test_correct_number_in_work_cannot_pass_wrong_conclusion(self):
        result = self.question('A04', '15。推导中C(7,4)=35。数值结果：15。')
        self.assertEqual(result['components'][0]['fraction'], 0)
        result = self.question('A11', '答案：999。中间记录3882，但最终答案：999。')
        self.assertEqual(result['components'][0]['fraction'], 0)

    def test_bold_culprit_and_alternative_wording(self):
        for text in ('是**甲**打破的杯子。只有乙的话为真。', '打破杯子的是甲。甲假乙真丙假。'):
            row = self.question('A02', text)
            self.assertEqual(row['components'][0]['fraction'], 1)

    def test_explicit_correction_and_unresolved_conflict(self):
        corrected = self.question('A02', '打破杯子的是丙。更正：答案：甲打破了杯子。只有乙的话为真。')
        conflicting = self.question('A02', '打破杯子的是丙。答案：甲打破了杯子。')
        self.assertEqual(corrected['components'][0]['fraction'], 1)
        self.assertEqual(conflicting['components'][0]['state'], 'conflict')
        self.assertLess(conflicting['provisional_fraction'], 1)

    def test_path_answer_not_any_enumerated_path(self):
        text = '最短路径A→C→B→D，总权重8。对比路线A→C→B→E→D=7。'
        self.assertEqual(self.question('A05', text)['provisional_fraction'], 0)
        text += '更正：最短路径A→C→B→E→D，总权重7。'
        self.assertEqual(self.question('A05', text)['provisional_fraction'], .8)

    def test_recurrence_expressions_and_no_eight_number_substring(self):
        text = 'f(1)=1，f(2)=f(1)+f(0)=2，f(3)=2+1+1=4，f(4)=7，f(5)=13，f(6)=24，f(7)=44，f(8)=44+24+13=81。'
        self.assertEqual(self.question('A06', text)['provisional_fraction'], 1)
        final, seq = recurrence_components('f(8)=81。依次为1,1,2,4,7,13,24,44,81。')
        self.assertEqual(final['score'], 1)
        self.assertIsNone(seq['score'])

    def test_wrong_intermediate_output_gets_partial_not_full(self):
        row = self.question('A07', '[0,1,2]\n[0,1,2]')
        self.assertEqual(row['components'][0]['fraction'], .5)
        self.assertEqual(row['provisional_fraction'], .6)

    def test_partial_json_content_and_extra_keys(self):
        row = self.question('A08', '{"items":["pear","wrong","banana","kiwi"]}')
        self.assertEqual(row['provisional_fraction'], .8)
        row = self.question('A08', '{"items":["pear","apple","banana","kiwi"],"extra":1}')
        self.assertEqual(row['provisional_fraction'], .8)

    def test_missing_explanation_is_a_specific_omission(self):
        row = self.question('A01', '5。')
        self.assertEqual(row['components'][1]['state'], 'missing_method')
        self.assertEqual(row['provisional_fraction'], .6)
        self.assertEqual(row['points_interval'], [3.6, 3.6])

    def test_incorrect_congruence_is_not_just_method_keyword_credit(self):
        row = self.question('A01', '5。费马小定理7^12≡1，123=12*10+3，但7^3≡4。')
        self.assertEqual(row['components'][1]['fraction'], 0)
        self.assertTrue(row['components'][1]['issues'])

    def test_correct_sequence_with_false_arithmetic_needs_review(self):
        row = self.question('A06', 'f(1)=1,f(2)=2,f(3)=4,f(4)=7,f(5)=13,f(6)=24,f(7)=44,f(8)=80+80=81。')
        self.assertEqual(row['components'][1]['state'], 'conflict')
        self.assertLess(row['provisional_fraction'], 1)

    def test_fence_only_costs_format_points(self):
        row = self.question('A07', '```python\n[2,2,2]\n[0,1,2]\n```')
        self.assertEqual(row['provisional_fraction'], .8)
        self.assertAlmostEqual(row['deductions'][0]['points'], 1.2)
        row = self.question('A08', '```json\n{"items":["pear","apple","banana","kiwi"]}\n```')
        self.assertEqual(row['provisional_fraction'], .8)

    def test_json_duplicate_keys_and_bool_or_float_ids(self):
        row = self.question('A08', '{"items":[],"items":["pear","apple","banana","kiwi"]}')
        self.assertEqual(row['provisional_fraction'], 0)
        for text in ('[true,3]', '[1.0,3]', '[NaN,3]'):
            self.assertEqual(self.question('A09', text)['provisional_fraction'], 0)

    def test_correct_number_bad_period_loses_method_points(self):
        row = self.question('A01', '5。模13余数每6次循环一次，所以7^123≡5。')
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['components'][1]['fraction'], 0)
        self.assertTrue(row['components'][1]['issues'])
        self.assertEqual(row['provisional_fraction'], .6)

    def test_calculation_error_retains_method_cue_credit_without_stacking(self):
        text = '3882。动态规划dp[x][y]=dp[x-1][y]+dp[x][y-1]，禁点置0。2+2=5。'
        row = self.question('A11', text)
        repeated = self.question('A11', text+'2+2=5。2+2=5。')
        self.assertEqual(row['components'][1]['state'], 'arithmetic_error')
        self.assertEqual(row['components'][1]['fraction'], .5)
        self.assertEqual(row['provisional_points'], repeated['provisional_points'])

    def test_corrected_calculation_not_penalized_for_discarded_work(self):
        text = '3882。2+2=5。更正：3882。动态规划dp[x][y]=dp[x-1][y]+dp[x][y-1]，禁点置0。'
        row = self.question('A11', text)
        self.assertNotEqual(row['components'][1]['state'], 'arithmetic_error')

    def test_extra_losing_first_moves_are_not_fully_correct(self):
        row = self.question('A10', '先手必胜，第一步取1枚或3枚。必败位为0,5,7,12,14。')
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['components'][1]['fraction'], .5)
        self.assertEqual(row['components'][2]['fraction'], 0)

    def test_unrecognized_answer_is_pending_not_declared_wrong(self):
        row = self.question('A11', '我用另一种记法给出证明，无法由规则提取。')
        self.assertIsNone(row['score'])
        self.assertEqual(row['components'][0]['state'], 'unresolved')
        self.assertEqual(row['points_interval'], [0, 16])
        self.assertTrue(all(d['type'] == 'uncredited_pending_review' for d in row['deductions']))

    def test_math_normalization_and_bounded_parser(self):
        self.assertIn('7^12≡1', clean(r'7^{12}\equiv1'))
        self.assertEqual(arithmetic('C(7,4)'), 35)
        self.assertEqual(arithmetic('0.018/0.067'), arithmetic('18/67'))
        for expression in ('__import__("os").system("echo bad")', '2**10000000', '[1]*100', 'x+1'):
            with self.assertRaises((ValueError, SyntaxError)):
                arithmetic(expression)

    def test_mod_form_and_composed_power_are_full_a01(self):
        text = '7^123 mod 13 = 5。7^6=(7^3)^2≡25≡12≡-1，123=6×20+3，故7^123≡5。'
        row = self.question('A01', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['components'][1]['fraction'], 1)

    def test_power_chain_343_is_not_an_a01_claim(self):
        text = '余数为 5。7^12≡1，123=12*10+3，故 7^123≡7^3=343≡5（mod 13）。343=13×26+5，余数为 5。'
        row = self.question('A01', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertNotIn('343', row['components'][0]['claims'])

    def test_period_12_without_fermat_substring_is_full_a01(self):
        text = '7^123 mod 13 = 5。7^6≡12≡-1（mod 13），故周期为12。123 = 12×10 + 3，所以 7^123 ≡ 7^3 ≡ 5。'
        row = self.question('A01', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['components'][1]['fraction'], 1)

    def test_later_culprit_restatement_without_discarding_conflict_example(self):
        row = self.question('A02', '丙打破了杯子。若甲打破则甲假、乙真、丙假。故是甲打破的杯子。')
        self.assertEqual(row['components'][0]['state'], 'conflict')
        self.assertEqual(row['components'][0]['fraction'], .5)
        self.assertEqual(row['components'][1]['fraction'], 1)
        row = self.question('A02', '丙打破了杯子。若甲打破则甲假、乙真、丙假。修正说明：唯一可行者是甲。')
        self.assertEqual(row['components'][0]['fraction'], 1)

    def test_exact_fraction_supersedes_leading_wrong_ratio(self):
        text = '1/3。P=(0.9×0.02)/(0.9×0.02+0.05×0.98)=0.018/0.067=18/67。精确分数为18/67。'
        row = self.question('A03', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['components'][1]['fraction'], 1)

    def test_self_check_42_does_not_override_corrected_a12(self):
        text = ('26 个。校验：45−16=29，即总数为 16+26=42。以逐项枚举的 13+13=26 为答案。'
                '[A12 更正] 上段末尾出现自相矛盾的核对，正确答案以分类枚举为准：26 个。'
                '两类各 13，合计 26。')
        row = self.question('A12', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(set(row['components'][0]['claims']), {'26'})

    def test_subclass_heji_does_not_conflict_a12_total(self):
        text = '共 26 个。首位为1、末位为0合计 13 种。对称也是 13 种。排列(2,2,0)合计 6。总计 26 个。'
        row = self.question('A12', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(set(row['components'][0]['claims']), {'26'})
        text = '先分类。违规合计 4+3 = 7。合法 20−7 = 13。另一类同为 13。合计 26。答案：26。'
        row = self.question('A12', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertNotIn('7', row['components'][0]['claims'])

    def test_totient_equation_is_not_parenthesized_arithmetic(self):
        from ability_rules import arithmetic_checks
        issues = [c['expression'] for c in arithmetic_checks('φ(13)=12。故 7^12 ≡ 1。') if not c['correct']]
        self.assertEqual(issues, [])
        verified = {c['value'] for c in arithmetic_checks('合法路径数为 (12870-(5544+4410+1764)+(2100+630)=3882)。') if c['correct']}
        self.assertIn('3882', verified)

    def test_congruence_mod_form_is_an_a01_claim(self):
        text = '7^123 ≡ 5 (mod 13)。φ(13)=12，故 7^12 ≡ 1 (mod 13)；123 = 12×10 + 3，所以 7^123 ≡ 7^3 = 343 ≡ 5 (mod 13)。'
        row = self.question('A01', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['components'][1]['fraction'], 1)
        self.assertNotIn('343', row['components'][0]['claims'])

    def test_recheck_and_culprit_restatement(self):
        text = '乙或丙都可能。重新核对：打破者为甲时甲假乙真丙假。故打破者是甲。'
        row = self.question('A02', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['components'][1]['fraction'], 1)

    def test_bare_path_label_is_an_a05_claim(self):
        text = '路径 A→C→B→E→D，总权重 7。比较 A→C→B→D=8。'
        row = self.question('A05', text)
        self.assertEqual(row['provisional_fraction'], 1)

    def test_a10_mod7_and_wrong_losing_states(self):
        complete = '先手必胜，第一步取1枚。剩余数除以 7 余 0 或 2 时是必败局面，15 取1剩14。'
        row = self.question('A10', complete)
        self.assertEqual(row['components'][2]['fraction'], 1)
        wrong = '先手必胜，第一步取1枚。必败态（mod 5 余 2）为 2、7、12；取3→12（必败态），剩14。'
        row = self.question('A10', wrong)
        self.assertEqual(row['components'][2]['fraction'], 0)
        self.assertTrue(row['components'][2]['issues'])

    def test_a12_two_classes_asserted_without_counting_steps_are_partial(self):
        text = '共 26 个。首位是1当且仅当 g0>0，末位是1当且仅当 g4>0。空位放 0、1 或 2 个 1。小计 13。对称情况也是 13，合计 26。'
        row = self.question('A12', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['components'][1]['fraction'], .5)

    def test_mutual_negation_with_he_is_full_a02(self):
        text = '甲打破了杯子。甲和乙的话互为否定，因此其中恰有一句真话。要使三句话中只有一句是真的，丙的话必须为假，即杯子是甲打破的。'
        row = self.question('A02', text)
        self.assertEqual(row['components'][0]['fraction'], 1)
        self.assertEqual(row['components'][1]['fraction'], 1)

    def test_latex_control_space_list_is_full_a06(self):
        text = 'f(1)至f(8)依次为：\n\\[\n1,\\ 2,\\ 4,\\ 7,\\ 13,\\ 24,\\ 44,\\ 81.\n\\]\n所以 f(8)=81。'
        row = self.question('A06', text)
        self.assertEqual(row['provisional_fraction'], 1)

    def test_manual_fraction_and_final_weighted_score(self):
        review = {q: {'score': 1, 'reason': '已逐项复核'} for q in WEIGHTS}
        review['A11'] = {'score': .25, 'reason': '复核部分得分'}
        result = score_ability(self.refs[0], review)
        self.assertEqual(result['score'], 88)
        self.assertEqual(result['score_interval'], [88, 88])
        self.assertEqual(result['pending_review'], 0)
        for value in ([], {'A01': {'score': float('nan'), 'reason': 'x'}}, {'A01': {'score': True, 'reason': 'x'}}):
            with self.assertRaises(ValueError): score_ability(self.refs[0], value)


if __name__ == '__main__':
    unittest.main()
