import unittest
from split_answers import parse_answers, LABEL
from classifier import load_references
from dual_scoring import format_score


class AdjacentLabelTests(unittest.TestCase):
    def test_inline_bracket_labels_parse_without_format_exemption(self):
        refs, _ = load_references()
        answers = next(r for r in refs if r['source'] == 'gpt6.1sol-03.txt')['answers']
        raw = '\n\n'.join('['+q+']'+v for q, v in answers.items())
        parsed = parse_answers(raw)
        self.assertEqual(parsed['parse_report']['status'], 'complete')
        self.assertEqual(parsed['answers'], answers)
        fmt = format_score({'raw_response':raw, 'answers':parsed['answers']})
        self.assertEqual(fmt['items'][0]['points'], 0)
        self.assertEqual(fmt['items'][1]['points'], 15)
        self.assertEqual(fmt['items'][2]['points'], 5)

    def test_unbracketed_label_still_requires_boundary(self):
        self.assertIsNone(LABEL.match('A01ordinaryword'))
        self.assertIsNotNone(LABEL.match('[A01]5'))
        self.assertIsNotNone(LABEL.match('【A01】5'))


if __name__ == '__main__':
    unittest.main()
