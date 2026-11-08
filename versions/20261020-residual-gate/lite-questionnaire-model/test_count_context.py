import unittest
from ability_rules import numeric_result
class CountContextTests(unittest.TestCase):
 def test_task_quantity_is_not_answer(self):
  self.assertEqual(numeric_result('A04','10个位置中选4个互不相邻的位置，数量为 C(10-4+1,4)=C(7,4)=35。答案：35 个。',35)['state'],'correct')
 def test_real_conflict_remains(self):
  self.assertEqual(numeric_result('A04','10个。答案：35个。',35)['state'],'conflict')
if __name__=='__main__': unittest.main()
