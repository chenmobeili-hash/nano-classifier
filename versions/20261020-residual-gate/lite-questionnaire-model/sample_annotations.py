"""User-supplied uncertain annotations, bound to the exact full questionnaire."""
import hashlib
import json
from pathlib import Path
from questionnaire_features import canonical_answers

ANNOTATIONS_PATH = Path(__file__).resolve().with_name('sample_annotations.json')


def response_digest(record):
    answers = canonical_answers(record)
    return hashlib.sha256(json.dumps(answers, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def annotation_for(record):
    if not ANNOTATIONS_PATH.exists():
        return None
    data = json.loads(ANNOTATIONS_PATH.read_text(encoding='utf-8-sig'))
    annotation = data.get('samples', {}).get(record.get('source'))
    if annotation is None:
        return None
    if annotation.get('response_sha256') != response_digest(record):
        raise ValueError('样本备注对应的回答已改变，请先复核标记：'+record['source'])
    return {k: v for k, v in annotation.items() if k != 'response_sha256'}
