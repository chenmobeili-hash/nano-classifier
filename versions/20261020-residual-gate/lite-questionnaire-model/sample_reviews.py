"""Human A-question reviews bound to the A-side digest, not the filename."""
import hashlib
import json
from pathlib import Path
from questionnaire_features import canonical_answers

REVIEWS_PATH = Path(__file__).resolve().with_name('sample_reviews.json')
A_QUESTIONS = tuple('A%02d' % i for i in range(1, 13))


def answer_digest(record):
    answers = canonical_answers(record)
    payload = {q: answers[q] for q in A_QUESTIONS}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def review_for(record):
    if not REVIEWS_PATH.exists():
        return None
    data = json.loads(REVIEWS_PATH.read_text(encoding='utf-8-sig'))
    digest = answer_digest(record)
    source = record.get('source')
    matched = None
    for item in data.get('reviews', []):
        sources = item.get('sources') or []
        if item.get('answer_sha256') == digest:
            matched = item
            break
        if source and source in sources:
            raise ValueError('人工评分对应的A题原文已改变，请先复核：'+source)
    if matched is None:
        return None
    scores = matched.get('scores')
    if not isinstance(scores, dict):
        raise ValueError('人工评分缺少scores对象。')
    return scores
