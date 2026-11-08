import unittest

from ability import WEIGHTS
from ability_rules import arithmetic_checks, method_evidence, numeric_result, game_components


class EvidenceTests(unittest.TestCase):
    def method(self, q, text):
        result = game_components(text)[1] if q == 'A10' else numeric_result(q, text, {'A11': 3882, 'A12': 26}[q])
        return method_evidence(q, text, result)

    def test_weights_are_frozen(self):
        self.assertEqual(list(WEIGHTS.values()), [6, 6, 8, 6, 8, 8, 6, 5, 5, 12, 16, 14])

    def test_equivalent_inclusion_exclusion_and_latex(self):
        variants = [
            '3882。容斥：12870-(5544+4410+1764)+(2100+630)=3882。',
            '3882。容斥：C(16,8)-6*C(12,6)-126*35-84*21+6*10*35+6*5*21=3882。',
            r'3882。容斥：\begin{aligned}N={}&12870-6\times924-126\times35-84\times21\\&+6\times10\times35+6\times5\times21\\={}&3882.\end{aligned}',
            '3882。容斥：12870-5544-4410-1764+2100+630=│3882│。',
        ]
        for text in variants:
            with self.subTest(text=text):
                self.assertEqual(self.method('A11', text)['score'], 1)

    def test_dynamic_program_requires_recurrence_boundary_and_obstacles(self):
        text = '3882。动态规划N(x,y)=N(x-1,y)+N(x,y-1)，N(0,0)=1，禁点(2,2)、(4,5)、(6,3)置0。'
        self.assertEqual(self.method('A11', text)['score'], 1)
        self.assertLess(self.method('A11', text.replace('N(0,0)=1，', ''))['score'], 1)
        self.assertEqual(self.method('A11', text.replace('置0', '置1'))['score'], 0)

    def test_generic_forbidden_points_need_not_repeat_question(self):
        text = '3882。动态规划dp[x][y]=dp[x-1][y]+dp[x][y-1]，dp[0][0]=1，被禁止的点记为0。'
        self.assertEqual(self.method('A11', text)['score'], 1)

    def test_wrong_computation_not_fixed_by_a_correct_result_or_keyword(self):
        text = '3882。容斥：12870-(5544+4410+1764)+(2100+631)=3882。'
        result = self.method('A11', text)
        self.assertEqual(result['state'], 'arithmetic_error')
        self.assertEqual(result['score'], .5)
        self.assertLess(self.method('A11', '3882。容斥。12870、5544、4410、1764、2100、630。2*1941=3882。')['score'], 1)

    def test_subscripts_and_variable_assignments_are_not_false_equations(self):
        self.assertFalse([c for c in arithmetic_checks('g5=0，g1=2，x_3=1。φ(13)=12。') if not c['correct']])
        self.assertIn('3882', {c['value'] for c in arithmetic_checks('N=12870-5544-4410-1764+2100+630=3882。') if c['correct']})

    def test_game_checks_explicit_state_assertions(self):
        prefix = '先手必胜，第一步取1枚，留下14枚。'
        for proof in ('L={0,2,6}。按模7判断。', '剩下14枚或12枚；这两个数都是必败态。',
                      '必败态为0、5、7、12、14。', 'n=13取1、3、4留12、10、9，故P位。',
                      'n=9留8/6/5全败点，必败。'):
            with self.subTest(proof=proof):
                result = self.method('A10', prefix+proof)
                self.assertEqual(result['score'], 0)
                self.assertTrue(result['issues'])

    def test_valid_game_proofs_and_moves_do_not_label_15_losing(self):
        for proof in ('必败位置依次是0、2、7、9、14。', '败态为0,\\ 2,\\ 7,\\ 9,\\ 14。',
                      'P位置集合为{0,2,7,9,14}。', '剩余数除以7余0或2时为必败态。'):
            text = '先手必胜，第一步取1枚，留下14枚。'+proof+'15取1到14是P位。'
            self.assertEqual(self.method('A10', text)['score'], 1)

    def test_corrected_game_discards_replaced_claims(self):
        text = '先手必胜，第一步取1枚。必败态为0、5、7。修正后的结论：第一步取1枚，剩14。复核必败态序列：0、2、7、9、14。'
        self.assertEqual(self.method('A10', text)['score'], 1)

    def test_vague_enumeration_cannot_claim_full_process_credit(self):
        text = '26。首末不同分两类，通过枚举排除连续三个1，每类13，共26。'
        self.assertEqual(self.method('A12', text)['score'], .5)

    def test_explicit_middle_position_count_is_supported(self):
        text = '26。首位0末位1，中间6位需有3个1，共C(6,3)=20种；排除内部连续三个1的4种，再排除末尾两个中间位为1的未重复3种，剩13。反转对称，另一类同为13。'
        self.assertEqual(self.method('A12', text)['score'], 1)

    def test_unsupported_complement_mapping_is_not_a_proof(self):
        text = '26。首末不同，中间6位3个1，C(6,3)-4-3=13。按位取反给出另一类，合计26。'
        self.assertEqual(self.method('A12', text)['score'], 0)

    def test_enumeration_must_be_complete_and_nonduplicated(self):
        import itertools
        triples = [p for p in itertools.combinations(range(1, 7), 3)
                   if '111' not in '1'+''.join('1' if i in p else '0' for i in range(1, 7))+'0']
        listed = ','.join('{'+','.join(map(str, p))+'}' for p in triples)
        text = '26。首1末0，枚举位置集合：'+listed+'。另一类由反转得到，合计26。'
        self.assertEqual(self.method('A12', text)['score'], 1)
        duplicate = text.replace('{1,3,4}', '{1,3,5}')
        self.assertLess(self.method('A12', duplicate)['score'], 1)


if __name__ == '__main__':
    unittest.main()
