"""Equal-model relative tiers. Always-perfect models share MAX; S+ is one unique non-perfect leader."""
import hashlib
import json
import math
from collections import Counter, defaultdict
from fractions import Fraction
from sample_annotations import annotation_for, response_digest

VERSION = 'relative-model-tiers-2'
LABELS = {'MAX': 'MAX 满分', 'S+': 'S+ 当前榜首', 'S': 'S 领先', 'A': 'A 较高', 'B': 'B 中段', 'C': 'C 相对靠后'}
ORDER = ('S', 'A', 'B', 'C')


def calibrate_models(samples, max_points=100):
    """Aggregate by model mean; tied scores get the same rank and tier."""
    grouped = defaultdict(list)
    for sample in samples:
        value = sample['score']
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= max_points:
            raise ValueError('参考分数必须为0至100的有限数。')
        if not isinstance(sample['model'], str) or not sample['model'].strip():
            raise ValueError('参考样本需要有效型号标签。')
        grouped[sample['model']].append(sample)
    if len(grouped) < 2:
        raise ValueError('相对型号分档至少需要两个型号。')
    rows = []
    for model, entries in grouped.items():
        # Input scores have cent precision; use exact fractions for ties.
        mean = sum(Fraction(round(e['score']*100), 100) for e in entries)/len(entries)
        rows.append({'model': model, '_mean': mean, 'mean_score': float(mean),
                     'sample_count': len(entries), 'sample_scores': [e['score'] for e in entries],
                     'min_score': min(e['score'] for e in entries), 'max_score': max(e['score'] for e in entries),
                     'score_interval': [round(sum(e.get('interval', [e['score']]*2)[i] for e in entries)/len(entries), 2) for i in (0, 1)],
                     'pending_review_samples': sum(e.get('pending_review', 0) > 0 for e in entries),
                     'annotations': [dict(source=e['source'], **e['annotation']) for e in entries if e.get('annotation')],
                     'sources': sorted(e['source'] for e in entries)})
    rows.sort(key=lambda r: (-r['_mean'], r['model']))
    for row in rows:
        row['always_full'] = row['min_score'] == max_points and row['max_score'] == max_points
    unique_leader = rows[0]['_mean'] > rows[1]['_mean'] and not rows[0]['always_full']
    winner = rows[0]['model'] if unique_leader else None
    for i, row in enumerate(rows):
        row['rank'] = 1+sum(other['_mean'] > row['_mean'] for other in rows)
        if row['always_full']:
            row['tier_code'] = 'MAX'
    # MAX sits above the quartile. A single non-perfect leader can still be S+.
    if unique_leader:
        rows[0]['tier_code'] = 'S+'
        remaining = rows[1:]
    else:
        remaining = [row for row in rows if not row['always_full']]
    start = 0
    while start < len(remaining):
        stop = start+1
        while stop < len(remaining) and remaining[stop]['_mean'] == remaining[start]['_mean']:
            stop += 1
        # Midrank quartiles: rounding falls toward the middle for a small cohort.
        bucket = min(3, ((start+stop)*2)//len(remaining))
        if start == 0: bucket = 0
        elif stop == len(remaining): bucket = 3
        for row in remaining[start:stop]:
            row['tier_code'] = ORDER[bucket]
        start = stop
    if unique_leader: rows[0]['tier_code'] = 'S+'
    for row in rows:
        row['tier'] = LABELS[row['tier_code']]
        row['gap_to_full_score'] = round(max_points-row['mean_score'], 4)
        del row['_mean']
    # Midpoints between adjacent occupied bands provide a frozen score ruler for
    # an unknown answer. A new answer cannot claim the cohort leader's S+ badge.
    band_ranges = []
    for code in ORDER:
        values = [r['mean_score'] for r in rows if r['tier_code'] == code]
        if values:
            band_ranges.append({'code': code, 'minimum': min(values), 'maximum': max(values)})
    boundaries = [{'upper_band': high['code'], 'lower_band': low['code'],
                   'cutoff': round((high['minimum']+low['maximum'])/2, 10)}
                  for high, low in zip(band_ranges, band_ranges[1:])]
    payload = sorted((s['model'], s['source'], s.get('answer_sha256'), s['score']) for s in samples)
    digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()
    return {'version': VERSION, 'status': 'provisional', 'model_count': len(rows), 'sample_count': len(samples),
            'winner': winner, 'unique_leader': unique_leader, 'cohort_sha256': digest,
            'distribution_by_model': dict(Counter(r['tier_code'] for r in rows)),
            'band_ranges': band_ranges, 'boundaries': boundaries, 'models': rows, 'samples': samples,
            'note': '按型号均分排名，每个型号占一个位置。每一份都是满分的型号同列为MAX，因为分数不能再高；MAX可以并列。唯一的非满分榜首才是S+。其余型号按中位秩四分位分为S/A/B/C；并列同分不拆档。参考集变化会改变档位。'}


def build_model_ranking(records):
    from ability import score_points, VERSION as scoring_version
    samples = []
    for record in records:
        ability = score_points(record)
        samples.append({'source': record['source'], 'model': record['model'],
                        'answer_sha256': ability['answer_sha256'], 'score': ability['provisional_score'],
                        'response_sha256': response_digest(record), 'annotation': annotation_for(record),
                        'interval': ability['score_interval'], 'pending_review': ability['pending_review']})
    ranking = calibrate_models(samples)
    ranking['scoring_version'] = scoring_version
    ranking['scoring_policy'] = 'strict_delivered_answer'
    clean = [s for s in samples if not (s.get('annotation') or {}).get('exclude_in_comparison')]
    comparison = calibrate_models(clean) if len({s['model'] for s in clean}) >= 2 else None
    comparison_rows = {} if comparison is None else {r['model']: r for r in comparison['models']}
    for row in ranking['models']:
        other = comparison_rows.get(row['model'])
        row.update(mean_including_marked=row['mean_score'],
                   mean_excluding_marked=other['mean_score'] if other else None,
                   sample_count_excluding_marked=other['sample_count'] if other else 0,
                   rank_excluding_marked=other['rank'] if other else None,
                   tier_excluding_marked=other['tier'] if other else None)
    ranking['comparison_without_marked'] = comparison
    ranking['excluded_comparison_sources'] = sorted(s['source'] for s in samples if s not in clean)
    ranking['note'] += ' 主榜包含全部样本，同时提供排除疑似标记样本的独立均分与排名对照；标记不是身份确认。'
    ranking['note'] += ' 均分采用严格交付评分，包含更正前错误的作答一致性限制。'
    from dual_ranking import build_dual_ranking
    ranking['dual_ranking'] = build_dual_ranking(records)
    return ranking


def score_band(score, ranking):
    for boundary in ranking['boundaries']:
        if score >= boundary['cutoff']-1e-9:
            return boundary['upper_band']
    ranges = ranking['band_ranges']
    return ranges[-1]['code'] if ranges else 'S'


def apply_relative_grade(ability, ranking, record=None):
    matches = [s for s in ranking['samples'] if s.get('answer_sha256') == ability['answer_sha256']]
    models = sorted({s['model'] for s in matches})
    consistent = matches and all(abs(s['score']-ability['provisional_score']) < 1e-8 for s in matches)
    matched = next((r for r in ranking['models'] if len(models) == 1 and r['model'] == models[0]), None) if consistent else None
    if matched:
        code = matched['tier_code']
        grade_score = matched['mean_score']
        interval = matched['score_interval']
        basis = 'matched_reference_model_mean'
    else:
        code = score_band(ability['provisional_score'], ranking)
        grade_score = ability['provisional_score']
        interval = ability['score_interval']
        basis = 'single_answer_against_reference_models'
    possible = []
    for value in (interval[1], *(b['cutoff'] for b in ranking['boundaries']), interval[0]):
        if interval[0] <= value <= interval[1]:
            label = LABELS[score_band(value, ranking)]
            if label not in possible: possible.append(label)
    if code in ('MAX', 'S+'): possible.insert(0, LABELS[code])
    result = dict(ability)
    exact_matches = [] if record is None else [s for s in matches if s.get('response_sha256') == response_digest(record)]
    annotations = [dict(source=s['source'], **s['annotation']) for s in exact_matches if s.get('annotation')]
    result.update(legacy_absolute_tier=ability['provisional_tier'],
                  tier_version=VERSION, tier_basis=basis, tier_code=code,
                  tier_score=round(grade_score, 4), tier_score_interval=interval,
                  provisional_tier=LABELS[code], tier=LABELS[code] if ability['pending_review'] == 0 else None,
                  possible_tiers=possible,
                  relative_model=None if matched is None else {
                      key: matched[key] for key in ('model', 'rank', 'mean_score', 'sample_count', 'pending_review_samples',
                                                   'mean_including_marked', 'mean_excluding_marked', 'sample_count_excluding_marked',
                                                   'rank_excluding_marked', 'tier_excluding_marked') if key in matched},
                  sample_annotations=annotations,
                  reference_match_models=models,
                  ranking_reference={'cohort_sha256': ranking['cohort_sha256'], 'model_count': ranking['model_count'],
                                     'sample_count': ranking['sample_count'], 'winner': ranking['winner'],
                                     'boundaries': ranking['boundaries'], 'status': 'provisional'})
    result['note'] = ability['note'].split('S≥90')[0]+ranking['note']+' 原文A题命中只用于关联参考型号评分，不证明真实型号身份；未命中或改分后的单份回答仅对照S/A/B/C标尺，不授MAX或S+。'
    return result
