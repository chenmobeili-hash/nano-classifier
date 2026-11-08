import copy
import unittest
from pathlib import Path
from classifier import load_references, read_answer, DEFAULT_REFERENCES
from sample_annotations import annotation_for
from ability import score_points
from ability_ranking import build_model_ranking
from result_scoring import build_result_ranking


class ReferenceExclusionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, cls.skipped = load_references()
        cls.name = 'deepseekv4.1flash1017-03.txt'

    def test_excluded_raw_sample_preserved_but_not_loaded(self):
        self.assertTrue((DEFAULT_REFERENCES/self.name).is_file())
        self.assertFalse(any(r['source'] == self.name for r in self.refs))
        self.assertEqual(self.skipped, [])

    def test_exclusion_is_bound_to_original_content(self):
        r = read_answer(DEFAULT_REFERENCES/self.name)
        r['source'] = self.name
        self.assertTrue(annotation_for(r)['exclude_from_training_and_ranking'])
        r['answers']['A01'] += ' changed'
        with self.assertRaises(ValueError):
            annotation_for(r)

    def test_rankings_exclude_same_sample(self):
        ranking = build_model_ranking(self.refs)
        dual = ranking['dual_ranking']
        group = next(m for m in dual['models'] if m['model']=='deepseekv4.1flash1017')
        expected_sources = {r['source'] for r in self.refs if r['model']=='deepseekv4.1flash1017'}
        self.assertEqual(group['sample_count'], len(expected_sources))
        self.assertEqual(set(group['sources']), expected_sources)
        self.assertNotIn(self.name, group['sources'])
        for r in (ranking, dual, build_result_ranking(self.refs)):
            self.assertFalse(any(s['source']==self.name for s in r['samples']))

    def test_archived_sample_can_still_be_scored(self):
        self.assertEqual(score_points(read_answer(DEFAULT_REFERENCES/self.name))['provisional_score'], 82.4)


if __name__ == '__main__':
    unittest.main()
