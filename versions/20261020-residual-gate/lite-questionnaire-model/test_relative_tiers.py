import copy
import unittest
from collections import Counter
from classifier import load_references
from ability import score_ability, score_points
from ability_ranking import build_model_ranking, calibrate_models, score_band, apply_relative_grade
from sample_annotations import annotation_for
from sample_reviews import review_for


class RelativeTierTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.refs, _ = load_references()
        cls.ranking = build_model_ranking(cls.refs)

    def test_current_model_balance_and_unique_splus(self):
        self.assertEqual(self.ranking['model_count'], 12)
        self.assertEqual(self.ranking['sample_count'], 33)
        self.assertEqual(self.ranking['distribution_by_model'], {'MAX': 3, 'S': 2, 'A': 2, 'B': 3, 'C': 2})
        self.assertFalse(self.ranking['unique_leader'])
        self.assertIsNone(self.ranking['winner'])
        self.assertEqual(sum(r['tier_code'] == 'S+' for r in self.ranking['models']), 0)
        tied = [r for r in self.ranking['models'] if r['mean_score'] == 100]
        self.assertEqual({r['model'] for r in tied}, {'gpt6astra', 'claudeopus5.5', 'gpt6.1sol'})
        self.assertEqual({r['tier_code'] for r in tied}, {'MAX'})

    def test_weighted_points_and_deductions_unchanged(self):
        for record in self.refs:
            raw = score_points(record)
            result = score_ability(record, ranking=self.ranking)
            for key in ('provisional_score', 'questions', 'score_interval', 'pending_review', 'weights', 'answer_sha256'):
                self.assertEqual(raw[key], result[key])

    def test_same_model_uses_mean_not_single_perfect_answer(self):
        records = [r for r in self.refs if r['model'] == 'glm5.3']
        results = [score_ability(r, ranking=self.ranking) for r in records]
        self.assertEqual({r['tier_code'] for r in results}, {'B'})
        self.assertEqual({r['tier_score'] for r in results}, {93.6})
        self.assertEqual(max(r['provisional_score'] for r in results), 100)

    def test_exact_current_reference_links_but_unknown_answer_cannot_claim_splus(self):
        champion = next(r for r in self.refs if r['model'] == 'gpt6astra')
        result = score_ability(champion, ranking=self.ranking)
        self.assertEqual(result['tier_code'], 'MAX')
        unknown = copy.deepcopy(champion)
        unknown['answers']['A01'] += '\n额外排版备注。'
        result = score_ability(unknown, ranking=self.ranking)
        self.assertEqual(result['provisional_score'], 100)
        self.assertEqual(result['tier_code'], 'S')
        self.assertIsNone(result['relative_model'])

    def test_manual_score_change_uses_frozen_score_ruler(self):
        record = next(r for r in self.refs if r['model'] == 'glm5.3')
        review = {q['question']: {'score': 1, 'reason': '人工确认'} for q in score_points(record)['questions']}
        # If this answer already scored100, use a non-perfect one from this model.
        record = next(r for r in self.refs if r['model'] == 'glm5.3' and score_points(r)['provisional_score'] < 100)
        result = score_ability(record, review, ranking=self.ranking)
        self.assertEqual(result['provisional_score'], 100)
        self.assertIsNone(result['relative_model'])
        self.assertEqual(result['tier_code'], 'S')

    def test_equal_means_are_not_split_by_name_or_sample_count(self):
        samples = []
        for i, score in enumerate((100, 99, 98, 98, 98, 96, 95, 94, 93, 92)):
            samples.append({'source': str(i), 'model': 'model'+str(i), 'score': score})
        table = calibrate_models(samples)
        tied = [r for r in table['models'] if r['mean_score'] == 98]
        self.assertEqual(len({r['tier_code'] for r in tied}), 1)
        self.assertEqual({r['rank'] for r in tied}, {3})

    def test_tied_first_does_not_arbitrarily_award_single_splus(self):
        samples = [{'source': str(i), 'model': 'm'+str(i), 'score': score} for i, score in enumerate((100, 100, 90, 80))]
        table = calibrate_models(samples)
        self.assertFalse(table['unique_leader'])
        self.assertIsNone(table['winner'])
        self.assertNotIn('S+', table['distribution_by_model'])
        self.assertEqual({r['tier_code'] for r in table['models'] if r['rank'] == 1}, {'MAX'})

    def test_model_sampling_count_does_not_become_extra_ranking_positions(self):
        samples = [{'source': 'a', 'model': 'a', 'score': 90}, {'source': 'b1', 'model': 'b', 'score': 80}, {'source': 'b2', 'model': 'b', 'score': 100}]
        table = calibrate_models(samples)
        self.assertEqual(table['model_count'], 2)
        self.assertEqual({r['mean_score'] for r in table['models']}, {90})
        self.assertEqual({r['rank'] for r in table['models']}, {1})

    def test_boundaries_and_bad_input(self):
        cutoffs = {b['upper_band']+'/'+b['lower_band']: b['cutoff'] for b in self.ranking['boundaries']}
        self.assertAlmostEqual(cutoffs['S/A'], 98.9)
        self.assertAlmostEqual(cutoffs['A/B'], 96.1)
        self.assertAlmostEqual(cutoffs['B/C'], 80.775)
        self.assertEqual(score_band(cutoffs['S/A'], self.ranking), 'S')
        self.assertEqual(score_band(cutoffs['A/B'], self.ranking), 'A')
        self.assertEqual(score_band(cutoffs['B/C'], self.ranking), 'B')
        self.assertEqual(score_band(cutoffs['B/C']-0.01, self.ranking), 'C')
        for value in (float('nan'), float('inf'), -1, 101, True):
            with self.assertRaises(ValueError):
                calibrate_models([{'source': 'x', 'model': 'x', 'score': value}, {'source': 'y', 'model': 'y', 'score': 80}])

    def test_including_and_excluding_suspected_sample_have_separate_means(self):
        row = next(r for r in self.ranking['models'] if r['model'] == 'deepseekv4.1flash-weak')
        self.assertAlmostEqual(row['mean_including_marked'], 62.4667, places=4)
        self.assertAlmostEqual(row['mean_excluding_marked'], 64.24, places=4)
        self.assertEqual(row['sample_count'], 6)
        self.assertEqual(row['sample_count_excluding_marked'], 5)
        self.assertEqual(self.ranking['excluded_comparison_sources'], ['deepseekv4.1flash-weak-01.txt'])
        self.assertEqual(self.ranking['comparison_without_marked']['sample_count'], 32)

    def test_flash_strong_and_weak_are_separate_models(self):
        strong = next(r for r in self.ranking['models'] if r['model'] == 'deepseekv4.1flash-strong')
        weak = next(r for r in self.ranking['models'] if r['model'] == 'deepseekv4.1flash-weak')
        self.assertEqual(strong['sample_count'], 3)
        self.assertEqual(weak['sample_count'], 6)
        self.assertAlmostEqual(strong['mean_score'], 99.5333333)
        self.assertAlmostEqual(weak['mean_score'], 62.4667, places=4)
        self.assertGreater(strong['mean_score'], weak['mean_score'])
        self.assertEqual(strong['tier_code'], 'S')
        self.assertEqual(weak['tier_code'], 'C')
        scores = {s['source']: s['score'] for s in self.ranking['samples']}
        self.assertEqual(scores['deepseekv4.1flash-strong-02.txt'], 98.6)
        self.assertEqual(scores['deepseekv4.1flash-strong-03.txt'], 100)
        self.assertEqual(scores['deepseekv4.1flash-weak-04.txt'], 61.6)
        self.assertEqual(scores['deepseekv4.1flash-weak-05.txt'], 69.6)
        self.assertEqual(scores['deepseekv4.1flash-weak-06.txt'], 62)
        self.assertEqual(scores['claudeopus5.5-01.txt'], 100)
        self.assertEqual(scores['claudeopus5.5-02.txt'], 100)
        self.assertEqual(scores['claudeopus5.5-03.txt'], 100)
        opus = next(r for r in self.ranking['models'] if r['model'] == 'claudeopus5.5')
        self.assertEqual(opus['sample_count'], 3)
        self.assertAlmostEqual(opus['mean_score'], 100)
        luna56 = next(r for r in self.ranking['models'] if r['model'] == 'gpt5.6luna')
        luna6 = next(r for r in self.ranking['models'] if r['model'] == 'gpt6luna')
        self.assertEqual(luna56['sample_count'], 4)
        self.assertAlmostEqual(luna56['mean_score'], 75.95)
        self.assertEqual(luna56['tier_code'], 'C')
        self.assertGreater(luna6['mean_score'], luna56['mean_score'])
        self.assertEqual(scores['gpt5.6luna-01.txt'], 57.8)
        self.assertEqual(scores['gpt5.6luna-02.txt'], 89.6)
        self.assertEqual(scores['gpt5.6luna-03.txt'], 81.6)
        self.assertEqual(scores['gpt5.6luna-04.txt'], 74.8)
        haiku = next(r for r in self.ranking['models'] if r['model'] == 'claudehaiku5.5')
        self.assertEqual(haiku['sample_count'], 2)
        self.assertAlmostEqual(haiku['mean_score'], 98.6)
        self.assertEqual(haiku['tier_code'], 'A')
        self.assertEqual(scores['claudehaiku5.5-01.txt'], 97.2)
        self.assertEqual(scores['claudehaiku5.5-02.txt'], 100)
        astra = next(r for r in self.ranking['models'] if r['model'] == 'gpt6astra')
        self.assertEqual(astra['sample_count'], 2)
        self.assertEqual(astra['mean_score'], 100)
        self.assertEqual(scores['gpt6astra-01.txt'], 100)
        self.assertEqual(scores['gpt6astra-02.txt'], 100)

    def test_only_exact_01_answer_gets_the_unconfirmed_annotation(self):
        for record in self.refs:
            result = score_ability(record, ranking=self.ranking)
            expected = record['source'] == 'deepseekv4.1flash-weak-01.txt'
            self.assertEqual(bool(result['sample_annotations']), expected)
            if expected:
                self.assertEqual(result['sample_annotations'][0]['status'], 'user_suspected_unconfirmed')

    def test_annotation_is_bound_to_content_and_not_new_same_filename(self):
        marked = copy.deepcopy(next(r for r in self.refs if r['source'] == 'deepseekv4.1flash-weak-01.txt'))
        marked['answers']['B01'] += ' changed'
        with self.assertRaises(ValueError): annotation_for(marked)
        result = score_ability(marked, ranking=self.ranking)
        self.assertEqual(result['sample_annotations'], [])

    def test_human_review_is_bound_to_a_digest(self):
        record = next(r for r in self.refs if r['source'] == 'deepseekv4.1flash-strong-02.txt')
        self.assertEqual(review_for(record)['A12']['score'], 1)
        changed = copy.deepcopy(record)
        changed['answers']['A12'] += '\n改写'
        with self.assertRaises(ValueError):
            review_for(changed)
        ingested = next(r for r in self.refs if r['source'] == 'deepseekv4.1flash-weak-05.txt')
        self.assertIsNone(review_for(ingested))
        self.assertAlmostEqual(score_points(ingested)['provisional_score'], 69.6)
        opus2 = next(r for r in self.refs if r['source'] == 'claudeopus5.5-02.txt')
        self.assertIsNone(review_for(opus2))
        self.assertAlmostEqual(score_points(opus2)['provisional_score'], 100)
        opus3 = next(r for r in self.refs if r['source'] == 'claudeopus5.5-03.txt')
        self.assertIsNone(review_for(opus3))
        self.assertAlmostEqual(score_points(opus3)['provisional_score'], 100)
        weak6 = next(r for r in self.refs if r['source'] == 'deepseekv4.1flash-weak-06.txt')
        self.assertIsNone(review_for(weak6))
        self.assertAlmostEqual(score_points(weak6)['provisional_score'], 62)

    def test_model_with_only_marked_sample_does_not_get_false_zero_mean(self):
        records = [r for r in self.refs if r['source'] in ('deepseekv4.1flash-weak-01.txt', 'gpt6astra-01.txt', 'claudeopus5.5-01.txt')]
        ranking = build_model_ranking(records)
        row = next(r for r in ranking['models'] if r['model'] == 'deepseekv4.1flash-weak')
        self.assertIsNone(row['mean_excluding_marked'])
        self.assertEqual(row['sample_count_excluding_marked'], 0)


if __name__ == '__main__':
    unittest.main()
