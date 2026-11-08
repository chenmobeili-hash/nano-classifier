import copy, unittest
import numpy as np
from consensus_guard import apply_consensus_guard

class DiagnosticTests(unittest.TestCase):
 def setUp(self):
  self.rank=[dict(label=k,score=v,distance=.8) for k,v in [('DeepSeek',.663),('Qwen',.194),('GPT',.113),('GLM',.018),('Claude',.012)]]
  self.refs=[dict(family=f) for f in ['GPT','GPT','GLM','DeepSeek','Qwen']]
  self.dist=np.array([.730,.740,.765,.85,.9])
 def check_unchanged(self,refs,dist):
  before=copy.deepcopy(self.rank)
  ranking,info=apply_consensus_guard(self.rank,dist,refs)
  self.assertEqual([(r['label'],r['score'],r['distance']) for r in ranking],[(r['label'],r['score'],r['distance']) for r in before])
  self.assertEqual(self.rank,before);self.assertFalse(info['applied']);self.assertEqual(info['neighbor_weight'],0.)
  return ranking,info
 def test_conflict_is_diagnostic_only(self):
  _,info=self.check_unchanged(self.refs,self.dist);self.assertTrue(info['conflict_detected'])
 def test_mixed_neighbors(self):
  refs=copy.deepcopy(self.refs);refs[1]['family']='Claude';_,info=self.check_unchanged(refs,self.dist);self.assertFalse(info['conflict_detected'])
 def test_repeated_claude_cannot_override_neural(self):
  refs=[dict(family='Claude') for _ in self.refs];_,info=self.check_unchanged(refs,self.dist);self.assertTrue(info['conflict_detected'])
 def test_too_few_neighbors(self):
  _,info=self.check_unchanged(self.refs[:2],self.dist[:2]);self.assertFalse(info['conflict_detected'])
 def test_empty_neighbors(self):
  ranking,info=self.check_unchanged([],np.array([]));self.assertTrue(all(r['neighbor_score']==0 for r in ranking))
 def test_zero_distance_finite(self):
  ranking,_=self.check_unchanged(self.refs,np.array([0,0,.7,.8,.9]));self.assertTrue(all(np.isfinite(r['neighbor_score']) for r in ranking))
if __name__=='__main__':unittest.main()
