"""Independent final-result score; process and delivery scores stay separate."""
import json
from pathlib import Path
from ability import WEIGHTS, score_points
from sample_reviews import answer_digest
from sample_annotations import annotation_for
from ability_ranking import calibrate_models

VERSION = 'final-result-1'
REVIEWS = Path(__file__).with_name('result_reviews.json')
POLICY = ('只核对A题最终结果，沿用原权重100分。每题所要求结果须全部正确才得分；'
          '不计过程、格式或更正扣分，明确更正采用最终答案；未撤回的冲突、缺答、错误结果不得分。'
          'B题不计分。本榜独立展示，不加入原200分总榜。')


def score_results(record, review_path=REVIEWS):
    digest = answer_digest(record)
    entries = json.loads(Path(review_path).read_text(encoding='utf-8-sig'))['reviews'] if Path(review_path).exists() else []
    review = next((r for r in entries if r['answer_sha256'] == digest), None)
    if review is None and any(record.get('source') in r.get('sources', []) for r in entries):
        raise ValueError('结果评分对应A题已改变，请重新复核：'+record['source'])
    auto = score_points(record, {})
    rows = []
    for q in auto['questions']:
        key = q['question']
        if review is not None:
            item = review['questions'][key]
            if type(item['correct']) is not bool or not item.get('reason'):
                raise ValueError('结果复核必须含correct布尔值与reason。')
            correct, pending, reason = item['correct'], False, item['reason']
        else:
            parts = [p for p in q['components'] if p['name'] not in ('方法与推导', '格式')]
            pending = any(p['fraction'] is None or p['state'] in ('conflict', 'unresolved') for p in parts)
            correct = not pending and all(p['fraction'] == 1 for p in parts)
            reason = '有限规则暂估最终结果；冲突或无法识别时须人工复核。'
        points = WEIGHTS[key] if correct else 0
        rows.append(dict(question=key, weight=WEIGHTS[key], correct=None if pending else correct,
                         points=points, pending_review=pending, reason=reason))
    total = sum(r['points'] for r in rows)
    pending = sum(r['pending_review'] for r in rows)
    return dict(version=VERSION, policy=POLICY, max_points=100, answer_sha256=digest,
                score=None if pending else total, provisional_score=total,
                score_interval=[total, total+sum(r['weight'] for r in rows if r['pending_review'])],
                correct_questions=sum(r['correct'] is True for r in rows), question_count=12,
                pending_review=pending, questions=rows)


def build_result_ranking(records):
    samples = []
    for record in records:
        result = score_results(record)
        samples.append(dict(source=record['source'], model=record['model'], score=result['provisional_score'],
                            score_interval=result['score_interval'], pending_review=result['pending_review'],
                            annotation=annotation_for(record), result=result))
    ranking = calibrate_models(samples)
    clean = [s for s in samples if not (s['annotation'] or {}).get('exclude_in_comparison')]
    ranking.update(version=VERSION, policy=POLICY,
                   comparison_without_marked=calibrate_models(clean))
    return ranking
