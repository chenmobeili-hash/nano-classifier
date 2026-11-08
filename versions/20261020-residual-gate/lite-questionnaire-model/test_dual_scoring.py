import copy
import json
import unittest
from pathlib import Path
from classifier import load_references, read_answer
from ability import score_points, score_ability
from ability_ranking import build_model_ranking, calibrate_models
from dual_scoring import score_dual, format_score
from dual_ranking import build_dual_ranking, grade_dual
from neural_core import reference_signature, load_network


class DualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs,_=load_references()
        cls.good=next(r for r in cls.refs if r['source']=='gpt6.1sol-03.txt')
        cls.ranking=build_model_ranking(cls.refs)

    def test_perfect_and_sum(self):
        d=score_dual(self.good,score_points(self.good))
        self.assertEqual(d['strict']['format_score']['score'],100)
        self.assertEqual(d['combined']['provisional_score'],200)

    def test_inline_tags_penalized_without_changing_content(self):
        r=copy.deepcopy(self.good)
        import re
        r['raw_response']=re.sub(r'(\[[AB]\d{2}\])\s*\n',r'\1 ',r['raw_response'])
        d=score_dual(r,score_points(r))
        self.assertEqual(d['legacy']['provisional_score'],100)
        self.assertEqual(d['strict']['format_score']['score'],80)
        self.assertEqual(d['combined']['provisional_score'],198)

    def test_sequences_are_exact_including_internal_newlines(self):
        r=copy.deepcopy(self.good);r['answers']['B02']='1'*119;r['answers']['B03']='H'*40+'\n'+'T'*40
        self.assertEqual(format_score(r)['score'],80)

    def test_json_comment_loses_format_only_in_new_rubric(self):
        r=copy.deepcopy(self.good);r['answers']['A09']='[1,3]\n说明：正确。'
        old=score_points(r,overrides={});d=score_dual(r,old)
        self.assertEqual(next(q for q in d['strict']['questions'] if q['question']=='A09')['provisional_points'],5)
        self.assertAlmostEqual(d['strict']['format_score']['score'],93.33,places=2)

    def test_wrong_content_is_not_recovered_as_correct(self):
        r=copy.deepcopy(self.good);r['answers']['A09']='[1,8]\n说明'
        d=score_dual(r,score_points(r,overrides={}))
        self.assertLess(next(q for q in d['strict']['questions'] if q['question']=='A09')['provisional_points'],5)

    def test_stricter_corrected_errors(self):
        r=next(r for r in self.refs if r['source']=='claudesonnet5.5-04.txt')
        d=score_dual(r,score_points(r));q={q['question']:q for q in d['strict']['questions']}
        self.assertEqual(q['A02']['provisional_points'],4.2)
        self.assertEqual(q['A10']['provisional_points'],10.2)

    def test_missing_raw_cannot_get_full_format(self):
        r={'answers':self.good['answers']};f=format_score(r)
        self.assertIsNone(f['score']);self.assertEqual(f['score_interval'],[55,100])
        self.assertGreater(score_dual(r,score_points(r))['combined']['pending_review'],0)

    def test_old_scores_match_published_baseline(self):
        baseline=json.loads(Path('model-ranking.json').read_text(encoding='utf-8'))
        baseline=baseline.get('legacy_ranking',baseline)
        by={r['source']:r for r in self.refs}
        for s in baseline['samples']:
            self.assertEqual(score_points(by[s['source']])['provisional_score'],s['score'])

    def test_cohort_not_changed_and_nn_still_valid(self):
        self.assertEqual(reference_signature(self.refs),load_network()['meta']['reference_signature'])

    def test_200_point_max_and_ties(self):
        r=calibrate_models([dict(source='a',model='a',score=200),dict(source='b',model='b',score=200),dict(source='c',model='c',score=190)],max_points=200)
        self.assertEqual(r['models'][0]['tier_code'],'MAX')
        self.assertEqual(r['models'][1]['rank'],1)

    def test_layout_change_cannot_claim_reference_max(self):
        r=copy.deepcopy(self.good);r['raw_response']='开场白\n'+r['raw_response']
        d=grade_dual(score_dual(r,score_points(r)),self.ranking['dual_ranking'],r)
        self.assertIsNone(d['combined']['relative_model'])

    def test_cli_scoring_exposes_both_versions(self):
        a=score_ability(self.good,ranking=self.ranking)
        self.assertEqual(a['score_versions']['combined']['max_points'],200)
        self.assertEqual(a['provisional_score'],100)

    def test_exclusion_preserves_only_marked_sample(self):
        r=self.ranking['dual_ranking']
        self.assertEqual(r['excluded_comparison_sources'],['deepseekv4.1flash-weak-01.txt'])
        self.assertEqual(r['comparison_without_marked']['sample_count'],len(self.refs)-1)


if __name__=='__main__':unittest.main()
