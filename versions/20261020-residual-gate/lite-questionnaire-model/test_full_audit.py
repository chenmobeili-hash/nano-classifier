import json
import unittest
from pathlib import Path
from classifier import load_references, read_answer
from sample_reviews import review_for, answer_digest
from ability import score_points
from dual_scoring import score_dual
from neural_core import load_network, reference_signature


class FullAuditTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs,cls.skipped=load_references()
        cls.by={r['source']:r for r in cls.refs}

    def test_all_A_items_have_digest_bound_review(self):
        self.assertEqual(len(self.refs),48)
        self.assertFalse(self.skipped)
        for r in self.refs:
            review=review_for(r)
            self.assertEqual(set(review),{f'A{i:02d}' for i in range(1,13)})
            self.assertTrue(all(x['reason'] and x['consistency_review']['reason'] for x in review.values()))
            a=score_points(r)
            self.assertEqual(a['pending_review'],0)
            self.assertEqual(a['score_interval'],[a['score']]*2)

    def test_corrected_array_retains_new_stricter_cap(self):
        r=self.by['claudehaiku5.5-03.txt']
        a=score_points(r);d=score_dual(r,a)
        old=next(q for q in a['questions'] if q['question']=='A09')
        new=next(q for q in d['strict']['questions'] if q['question']=='A09')
        self.assertEqual(old['provisional_points'],4)
        self.assertEqual(new['provisional_points'],3.5)

    def test_recomputed_path_is_corrected_not_unrecognized(self):
        r=self.by['deepseekv4.1flash-weak-02.txt']
        q=score_points(r)['questions'][4]
        self.assertEqual(q['provisional_points'],6.4)
        self.assertEqual(q['consistency']['cap_fraction'],.8)

    def test_auxiliary_wrong_edge_cost_is_penalized(self):
        q=score_points(self.by['deepseekv4.1flash-weak-05.txt'])['questions'][4]
        self.assertEqual(q['provisional_points'],7.2)

    def test_complete_DP_method_gets_full_method_even_with_wrong_result(self):
        q=score_points(self.by['gpt5.6luna-01.txt'])['questions'][10]
        self.assertEqual(q['provisional_points'],6.4)

    def test_new_haiku_is_exact_11_and_configuration_is_known(self):
        r=self.by['claudehaiku5.5-09.txt'];d=score_dual(r,score_points(r))
        self.assertEqual((d['legacy']['provisional_score'],d['strict']['provisional_score'],d['combined']['score']),(100,98,198))
        data=json.loads(Path('sample_provenance.json').read_text(encoding='utf-8'))
        m=next(s for s in data['samples'] if s['source']==r['source'])
        self.assertEqual(m['reasoning_effort'],'medium')
        self.assertTrue(m['thinking'])

    def test_all_new_published_scores_reproduce(self):
        rank=json.loads(Path('model-ranking.json').read_text(encoding='utf-8'))
        self.assertEqual(rank['sample_count'],48)
        for s in rank['samples']:
            r=self.by[s['source']];d=score_dual(r,score_points(r))
            self.assertEqual(s['score'],d['combined']['score'])
            self.assertEqual(d['combined']['pending_review'],0)
        for m in rank['models']:
            samples=[s['score'] for s in rank['samples'] if s['model']==m['model']]
            self.assertAlmostEqual(m['mean_score'],sum(samples)/len(samples))

    def test_neural_weights_match_current_reference_cohort(self):
        self.assertEqual(reference_signature(self.refs),load_network()['meta']['reference_signature'])


if __name__=='__main__':unittest.main()
