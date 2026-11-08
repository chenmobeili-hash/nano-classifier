"""Known external identities, matched by complete answers, never training labels."""
import hashlib
import json
from pathlib import Path
from questionnaire_features import canonical_answers

REGISTRY = Path(__file__).with_name('known_external_references.json')


def full_answer_digest(record):
    return hashlib.sha256(json.dumps(canonical_answers(record),ensure_ascii=False,sort_keys=True).encode()).hexdigest()


def known_reference_matches(record):
    if not REGISTRY.exists():
        return []
    data=json.loads(REGISTRY.read_text(encoding='utf-8-sig'))
    digest=full_answer_digest(record)
    return [dict(model=x['model'],family=x['family'],source=x['source'],
                 match_kind='exact_full_answer_content',label_source=x['label_source'],
                 training_included=False,ranking_included=False,
                 note='仅表示与用户已确认身份的库外参考答案内容一致；不替代神经网络预测，也不证明其他新回答的具体型号。')
            for x in data.get('samples',[]) if x['answer_sha256']==digest]
