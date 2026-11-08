import unittest
from classifier import read_answer
from ability import score_points
class ConsistencyReviewTests(unittest.TestCase):
 def test_audited_corrections_preserve_penalties(self):
  a=score_points(read_answer('../nano-now/claudesonnet5.5-04.txt'))
  self.assertEqual(a['score'],97.6)
  q={r['question']:r for r in a['questions']}
  self.assertEqual(q['A02']['provisional_points'],4.8)
  self.assertEqual(q['A10']['provisional_points'],10.8)
  self.assertIn('automatic_assessment',q['A01']['consistency'])
 def test_content_review_alone_does_not_erase_caps(self):
  r=read_answer('../nano-now/claudesonnet5.5-04.txt')
  r['answers']['A06']='f(8)=149。更正：f(8)=81。'
  a=score_points(r,overrides={'A06':{'score':1,'reason':'final value correct'}})
  q=next(x for x in a['questions'] if x['question']=='A06')
  self.assertLess(q['provisional_points'],8)
 def test_invalid_consistency_review_rejected(self):
  with self.assertRaises(ValueError):
   score_points(read_answer('../nano-now/claudesonnet5.5-04.txt'),overrides={'A01':{'score':1,'reason':'checked','consistency_review':{'cap_fraction':1,'reason':''}}})
if __name__=='__main__': unittest.main()
