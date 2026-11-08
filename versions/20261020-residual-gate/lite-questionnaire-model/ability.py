"""Weighted, component-based provisional assessment for mq-full-0.2."""
import hashlib
import json
import math
from questionnaire_features import canonical_answers
from ability_rules import (numeric_result, culprit_result, probability_result, method_evidence,
                           path_components, recurrence_components, game_components, strict_output)
from sample_reviews import review_for
from scoring_text import answer_body
from strict_consistency import inspect_consistency, CAPS, VERSION as CONSISTENCY_VERSION

VERSION = 'ability-strict-4'
WEIGHTS = {'A01': 6, 'A02': 6, 'A03': 8, 'A04': 6, 'A05': 8, 'A06': 8,
           'A07': 6, 'A08': 5, 'A09': 5, 'A10': 12, 'A11': 16, 'A12': 14}
EXPECTED = {'A01': '5；模13周期为12，7^6≡-1', 'A02': '甲；仅乙说真话',
            'A03': '18/67；贝叶斯公式', 'A04': '35；C(7,4)',
            'A05': 'A→C→B→E→D；总权重7', 'A06': '1,2,4,7,13,24,44,81',
            'A07': '[2, 2, 2]\n[0, 1, 2]', 'A08': '{"items":["pear","apple","banana","kiwi"]}',
            'A09': '[1,3]', 'A10': '先手必胜；唯一获胜第一步取1枚；必败态0、2、7、9、14',
            'A11': '3882；容斥或动态规划', 'A12': '26；反转对称的两类各13个'}


def legacy_absolute_tier(score):
    return 'S 优秀' if score >= 90 else 'A 较强' if score >= 75 else 'B 中等' if score >= 60 else 'C 待提升'


def answer_digest(answers):
    return hashlib.sha256(json.dumps({q: answers[q] for q in WEIGHTS}, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def component(name, share, check, method=False):
    value = check['score']
    state = check['state']
    # Recognized method cues remain estimates even when rule-supported.
    uncertain = (method and state in ('rule_supported', 'method_cue', 'unresolved', 'arithmetic_error')) or value is None or state == 'conflict'
    low = 0 if uncertain else value
    high = 1 if uncertain else value
    return {'name': name, 'share': share, 'fraction': value,
            'lower_fraction': low, 'upper_fraction': high,
            'state': state, 'evidence': check['evidence'], 'issues': check.get('issues', []),
            'claims': check.get('claims', [])}


def score_points(record, overrides=None):
    """Weighted content/process credit capped by strict answer consistency."""
    answers = canonical_answers(record)
    if overrides is None:
        loaded = review_for(record)
        overrides = loaded if loaded is not None else {}
    if not isinstance(overrides, dict) or set(overrides)-set(EXPECTED):
        raise ValueError('人工评分必须是以A01至A12为键的JSON对象。')
    rows = []
    for q, weight in WEIGHTS.items():
        raw = answer_body(answers[q]).strip()
        needs_review = q in ('A01', 'A02', 'A03', 'A04', 'A10', 'A11', 'A12')
        if q in ('A07', 'A08', 'A09'):
            correct, format_ok = strict_output(q, raw)
            parts = [component('内容', .8, {'score': float(correct), 'state': 'correct' if correct == 1 else 'partial' if correct > 0 else 'incorrect', 'evidence': '按输出行或数组元素的位置给内容部分分；JSON检查重复键与数据类型。'}),
                     component('格式', .2, {'score': float(format_ok), 'state': 'correct' if format_ok else 'format_error', 'evidence': '按只输出指定内容核对；代码围栏只扣格式分。'})]
        elif q == 'A05':
            path, total = path_components(raw)
            parts = [component('最短路径', .5, path), component('总权重', .5, total)]
        elif q == 'A06':
            final, sequence = recurrence_components(raw)
            parts = [component('f(8)', .4, final), component('完整序列', .6, sequence)]
        else:
            if q == 'A02': result = culprit_result(raw)
            elif q == 'A03': result = probability_result(raw)
            elif q == 'A10':
                winner, moves = game_components(raw)
                result = moves
            else: result = numeric_result(q, raw, {'A01': 5, 'A04': 35, 'A11': 3882, 'A12': 26}[q])
            parts = ([component('先手胜负', .2, winner), component('所有必胜第一步', .4, moves)] if q == 'A10'
                     else [component('最终结论', .6, result)])
            parts.append(component('方法与推导', .4, method_evidence(q, raw, result), method=True))
        fraction = sum(p['share']*(p['fraction'] if p['fraction'] is not None else 0) for p in parts)
        low = sum(p['share']*p['lower_fraction'] for p in parts)
        high = sum(p['share']*p['upper_fraction'] for p in parts)
        unresolved = any(p['state'] in ('unresolved', 'conflict') for p in parts)
        needs_review |= unresolved
        status = 'needs_review' if needs_review else 'auto_checked'
        score = None if needs_review else fraction
        reason = '按分项计分；开放推导为有限规则暂估，保留人工复核。' if needs_review else '按确定的内容、格式或数值逐项核对。'
        if q in overrides:
            item = overrides[q]
            if (not isinstance(item, dict) or type(item.get('score')) not in (int, float)
                    or not math.isfinite(item['score']) or not 0 <= item['score'] <= 1
                    or not isinstance(item.get('reason'), str) or not item['reason'].strip()):
                raise ValueError(q+'人工评分需要0至1的score及非空reason；score是本题得分比例。')
            fraction = low = high = score = float(item['score'])
            status, reason = 'human_reviewed', item['reason']
            parts = [component('人工复核', 1, {'score': score, 'state': 'human_reviewed', 'evidence': reason})]
        # Prior reviews establish content credit; a final correction cannot
        # exempt a delivered answer from the same strict consistency standard.
        consistency = inspect_consistency(q, raw)
        # Digest-bound human review can correct a falsely extracted consistency event.
        checked = overrides.get(q, {}).get('consistency_review')
        if checked is not None:
            if (not isinstance(checked, dict)
                    or type(checked.get('cap_fraction')) not in (int, float)
                    or checked['cap_fraction'] not in (0.6, 0.8, 0.9, 1.0)
                    or not isinstance(checked.get('reason'), str)
                    or not checked['reason'].strip()):
                raise ValueError(q+' consistency_review requires an allowed cap_fraction and reason.')
            consistency['automatic_assessment'] = dict(consistency)
            consistency['cap_fraction'] = checked['cap_fraction']
            consistency['review_status'] = 'human_reviewed'
            consistency['review_reason'] = checked['reason']
            consistency['events'] = ([] if checked['cap_fraction'] == 1 else [
                {'kind': 'human_reviewed_consistency', 'cap_fraction': checked['cap_fraction'],
                 'evidence': checked['reason']}])
        base_fraction = fraction
        cap = consistency['cap_fraction']
        fraction, low, high = min(fraction, cap), min(low, cap), min(high, cap)
        if score is not None:
            score = min(score, cap)
        consistency['applied_deduction'] = round(weight*(base_fraction-fraction), 4)
        deductions = []
        for p in parts:
            estimated = p['fraction'] if p['fraction'] is not None else 0
            if estimated < 1:
                deductions.append({'component': p['name'], 'points': round(weight*p['share']*(1-estimated), 4),
                                   'type': 'uncredited_pending_review' if p['fraction'] is None else p['state'],
                                   'reason': p['evidence']})
        if consistency['applied_deduction']:
            deductions.append({'component': '作答一致性', 'points': consistency['applied_deduction'],
                               'type': 'strict_consistency',
                               'reason': '本题内容与过程原分受严格一致性上限限制；同一问题不重复叠加扣分。'})
        rows.append({'question': q, 'weight': weight, 'score': score, 'status': status,
                     'content_provisional_points': round(weight*base_fraction, 4),
                     'consistency': consistency,
                     'provisional_fraction': round(fraction, 6), 'provisional_points': round(weight*fraction, 4),
                     'points_interval': [round(weight*low, 4), round(weight*high, 4)],
                     'answer_cue_found': None, 'expected': EXPECTED[q], 'reason': reason,
                     'components': parts, 'deductions': deductions})
    provisional = round(sum(r['provisional_points'] for r in rows), 2)
    low = round(sum(r['points_interval'][0] for r in rows), 2)
    high = round(sum(r['points_interval'][1] for r in rows), 2)
    pending = sum(r['status'] == 'needs_review' for r in rows)
    return {'scope': '本问卷能力分档，不代表通用智能或供应商排名', 'rubric_version': VERSION,
            'consistency_version': CONSISTENCY_VERSION, 'consistency_caps': CAPS.copy(),
            'content_provisional_score': round(sum(r['content_provisional_points'] for r in rows), 2),
            'consistency_deduction': round(sum(r['consistency']['applied_deduction'] for r in rows), 2),
            'answer_sha256': answer_digest(answers), 'weights': WEIGHTS.copy(), 'max_points': 100,
            'tier': None if pending else legacy_absolute_tier(provisional), 'score': None if pending else provisional,
            'status': 'review_required' if pending else 'scored', 'score_interval': [low, high],
            'provisional_score': provisional, 'provisional_tier': legacy_absolute_tier(provisional),
            'possible_tiers': list(dict.fromkeys(legacy_absolute_tier(x) for x in (high, 90, 75, 60, low) if low <= x <= high)),
            'pending_review': pending,
            'unresolved_components': sum(p['state'] in ('unresolved', 'conflict') for r in rows for p in r['components']),
            'questions': rows,
            'note': '加权满分100。结论/方法分别计分；A07–A09内容80%、格式20%。未识别项暂不计分但列入区间上限；不是直接判错。开放推导有限规则支持仍待复核。严格版按整份作答核对：错误定论后改对本题至多80%，错误推导后更正至多90%，未撤回的冲突定论至多60%；与既有失分取较低得分，不重复叠扣。人工score为内容与过程的0至1复核比例，仍接受一致性上限约束。S≥90、A≥75、B≥60、C<60，门槛未调整。'}


def score_ability(record, overrides=None, ranking=None):
    """Score A answers, then grade against the current equal-model cohort."""
    from ability_ranking import build_model_ranking, apply_relative_grade
    if ranking is None:
        from classifier import load_references
        records, _ = load_references()
        ranking = build_model_ranking(records)
    legacy = score_points(record, overrides)
    result = apply_relative_grade(legacy, ranking, record=record)
    from dual_scoring import score_dual
    from dual_ranking import grade_dual
    # Custom reference cohorts are passed through the legacy ranking samples.
    # Callers may attach their records to avoid ever falling back to another cohort.
    dual_ranking = ranking.get('dual_ranking')
    if dual_ranking is None:
        raise ValueError('当前排名缺少双版本标尺，请通过build_model_ranking构建。')
    result['score_versions'] = grade_dual(score_dual(record, legacy), dual_ranking, record)
    return result
