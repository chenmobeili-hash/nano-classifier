"""Conservative, reviewable questionnaire rubric. Never executes submitted code."""
import json
import re
import unicodedata
from questionnaire_features import canonical_answers

EXPECTED = {'A01': '5；模13周期为12（不是6）', 'A02': '甲；仅乙说真话',
            'A03': '18/67；贝叶斯公式', 'A04': '35；C(7,4)',
            'A05': 'A→C→B→E→D；7', 'A06': '1,2,4,7,13,24,44,81',
            'A07': '[2, 2, 2]\n[0, 1, 2]', 'A08': '{"items":["pear","apple","banana","kiwi"]}',
            'A09': '[1,3]', 'A10': '先手必胜；唯一获胜第一步取1枚',
            'A11': '3882', 'A12': '26'}


def tier(score):
    return 'S 优秀' if score >= 90 else 'A 较强' if score >= 75 else 'B 中等' if score >= 60 else 'C 待提升'


def normalize(text):
    text = unicodedata.normalize('NFKC', text).replace('**', '')
    text = re.sub(r'\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'\1/\2', text)
    return re.sub(r'\s+', '', text)


def score_ability(record, overrides=None):
    answers = canonical_answers(record)
    overrides = overrides or {}
    if not isinstance(overrides, dict) or set(overrides)-set(EXPECTED):
        raise ValueError('人工评分必须是以A01至A12为键的JSON对象。')
    rows = []
    for q, expected in EXPECTED.items():
        raw = answers[q].strip()
        text = normalize(raw)
        score, status, reason = None, 'needs_review', '开放推导需人工复核；命中数值不代表推导正确。'
        if q == 'A07':
            score = int(bool(re.fullmatch(r'\s*\[2,\s*2,\s*2\]\s*\n\s*\[0,\s*1,\s*2\]\s*', raw)))
            status, reason = 'auto_checked', '严格核对两行输出，不允许额外解释或代码围栏。'
        elif q in ('A08', 'A09'):
            try:
                value = json.loads(raw)
                target = {'items': ['pear', 'apple', 'banana', 'kiwi']} if q == 'A08' else [1, 3]
                score = int(value == target and (q != 'A09' or all(type(v) is int for v in value)))
            except (ValueError, TypeError):
                score = 0
            status, reason = 'auto_checked', '严格核对JSON结构、值及顺序。'
        hint = None
        if q in ('A01', 'A04', 'A11', 'A12'):
            number = {'A01': '5', 'A04': '35', 'A11': '3882', 'A12': '26'}[q]
            hint = bool(re.search(r'(?<!\d)' + number + r'(?!\d)', text))
        elif q == 'A02':
            hint = bool(re.search(r'甲(?:打破|是)|是甲打破', re.split(r'[。\n]', raw)[0]))
        elif q == 'A03':
            hint = '18/67' in text
        elif q == 'A05':
            hint = bool(re.search(r'A(?:→|->)C(?:→|->)B(?:→|->)E(?:→|->)D', text)) and bool(re.search(r'(?<!\d)7(?!\d)', text))
        elif q == 'A06':
            hint = ('1,2,4,7,13,24,44,81' in text.replace('，', ',') or
                    all(re.search(r'f\(' + str(i) + r'\)=' + str(n) + r'(?!\d)', text)
                        for i, n in enumerate((1, 2, 4, 7, 13, 24, 44, 81), 1)))
        elif q == 'A10':
            conclusion = re.split(r'理由|证明|分析|因为|按必败|设剩余', text)[0][:180]
            hint = bool(re.search(r'取(?:走)?\s*[1一]\s*枚', conclusion)) and '必胜' in conclusion
        if q in overrides:
            item = overrides[q]
            if (not isinstance(item, dict) or type(item.get('score')) not in (int, float)
                    or item['score'] not in (0, .5, 1) or not isinstance(item.get('reason'), str)
                    or not item['reason'].strip()):
                raise ValueError(q + '人工评分需要score=0/0.5/1及非空reason。')
            score, status, reason = item['score'], 'human_reviewed', item['reason']
        rows.append({'question': q, 'score': score, 'status': status,
                     'answer_cue_found': hint, 'expected': expected, 'reason': reason})
    earned = sum(r['score'] for r in rows if r['score'] is not None)
    pending = sum(r['score'] is None for r in rows)
    low, high = earned/12*100, (earned+pending)/12*100
    # Cue-based provisional estimate is explicitly separate from verified scores.
    provisional = sum(r['score'] if r['score'] is not None else 1 if r['answer_cue_found'] else 0 for r in rows)/12*100
    return {'scope': '本问卷能力分档，不代表通用智能或供应商排名',
            'tier': tier(low) if not pending else None,
            'status': 'review_required' if pending else 'scored',
            'score': round(low, 2) if not pending else None,
            'score_interval': [round(low, 2), round(high, 2)],
            'provisional_score': round(provisional, 2), 'provisional_tier': tier(provisional),
            'pending_review': pending, 'questions': rows,
            'note': '每题1分；人工可给0、0.5、1分。未复核开放题仅按答案线索暂计1或0分，可能误判；暂估档不能代替最终档。S≥90、A≥75、B≥60、C<60；门槛为项目约定。'}
