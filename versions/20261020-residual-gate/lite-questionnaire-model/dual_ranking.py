"""Model ranking on the sum of two independently displayed rubrics."""
import hashlib
import json
from questionnaire_features import canonical_answers
from ability import score_points
from dual_scoring import score_dual, VERSION
from ability_ranking import calibrate_models, score_band, LABELS
from sample_annotations import annotation_for


def delivery_digest(record):
    raw=record.get('raw_response')
    return hashlib.sha256(json.dumps([raw.replace('\r\n','\n').replace('\r','\n'),canonical_answers(record)],ensure_ascii=False,sort_keys=True).encode()).hexdigest() if isinstance(raw,str) else None


def build_dual_ranking(records):
    samples=[]
    for record in records:
        old=score_points(record);dual=score_dual(record,old)
        samples.append(dict(source=record['source'],model=record['model'],
            score=dual['combined']['provisional_score'],legacy_score=old['provisional_score'],
            strict_score=dual['strict']['provisional_score'],format_score=dual['strict']['format_score']['provisional_score'],
            interval=dual['combined']['score_interval'],pending_review=dual['combined']['pending_review'],
            answer_sha256=old['answer_sha256'],delivery_sha256=delivery_digest(record),annotation=annotation_for(record),
            dual=dual))
    def make(entries):
        ranking=calibrate_models(entries,max_points=200)
        ranking.update(version=VERSION,scoring_version=VERSION,max_points=200,scoring_policy='legacy_plus_strict_format')
        for row in ranking['models']:
            group=[s for s in entries if s['model']==row['model']]
            for key in ('legacy_score','strict_score','format_score'):
                row['mean_'+key]=sum(s[key] for s in group)/len(group)
        return ranking
    ranking=make(samples)
    clean=[s for s in samples if not (s.get('annotation') or {}).get('exclude_in_comparison')]
    comparison=make(clean) if len({s['model'] for s in clean})>=2 else None
    rows={} if comparison is None else {r['model']:r for r in comparison['models']}
    for row in ranking['models']:
        other=rows.get(row['model'])
        row.update(mean_including_marked=row['mean_score'],mean_excluding_marked=other['mean_score'] if other else None,
            sample_count_excluding_marked=other['sample_count'] if other else 0,
            rank_excluding_marked=other['rank'] if other else None,tier_excluding_marked=other['tier'] if other else None)
    ranking['comparison_without_marked']=comparison
    ranking['excluded_comparison_sources']=[s['source'] for s in samples if s not in clean]
    ranking['note']='每份回答旧版100分+严格格式版100分=合计200分，按型号合计均分排名。MAX仅授予全部样本合计200分的型号；唯一非满分总榜首授S+；其他按型号中位秩分S/A/B/C。保留含标记和排除标记两种统计。所有分数继承内容推导待复核状态。'
    return ranking


def grade_dual(dual, ranking, record):
    total=dual['combined']
    digest=delivery_digest(record)
    matches=[s for s in ranking['samples'] if digest and s['delivery_sha256']==digest
             and abs(s['score']-total['provisional_score'])<1e-8]
    models={s['model'] for s in matches}
    matched=next((r for r in ranking['models'] if len(models)==1 and r['model'] in models),None)
    code=matched['tier_code'] if matched else score_band(total['provisional_score'],ranking)
    total.update(provisional_tier=LABELS[code],tier_code=code,
        tier=None if total['pending_review'] else LABELS[code],
        relative_model=matched,ranking_basis='model_combined_mean' if matched else 'single_combined_score',
        ranking_reference=dict(model_count=ranking['model_count'],sample_count=ranking['sample_count'],
                               max_points=200,boundaries=ranking['boundaries'],cohort_sha256=ranking['cohort_sha256']))
    return dual
