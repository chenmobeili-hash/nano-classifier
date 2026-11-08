"""Two visible /100 rubrics, summed /200. Legacy scoring stays untouched."""
import json
import re
from questionnaire_features import canonical_answers, QUESTIONS
from split_answers import LABEL, FENCE, parse_answers
from ability_rules import strict_output
from scoring_text import answer_body

VERSION = 'dual-delivery-1'
STRICT_VERSION = 'ability-format-strict-5'
STRICT_CAPS = {'corrected_wrong_conclusion': .7, 'corrected_wrong_work': .85,
               'unresolved_conclusions': .5}


def format_score(record):
    answers = {q: answer_body(v) for q, v in canonical_answers(record).items()}
    rows = []
    def add(name, maximum, fraction, reason):
        rows.append(dict(name=name, maximum=maximum, points=None if fraction is None else round(maximum*fraction, 4),
                         status='unavailable' if fraction is None else 'checked', reason=reason))
    raw = record.get('raw_response')
    if isinstance(raw, str):
        parsed = parse_answers(raw)
        labels = []
        fence = None
        lines = raw.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff').splitlines()
        for line in lines:
            fm = FENCE.match(line)
            if fm:
                if fence is None: fence = fm[1]
                elif fm[1][0] == fence[0] and len(fm[1]) >= len(fence) and not fm[2].strip(): fence = None
                continue
            match = LABEL.match(line) if fence is None else None
            if match:
                q = (match[1] or match[3] or match[5]).upper()+f'{int(match[2] or match[4] or match[6]):02d}'
                labels.append((q, line.strip()))
        standalone = sum(line == '['+q+']' for q, line in labels if q in QUESTIONS)
        add('标签独占一行且使用[A01]形式', 20, min(standalone,24)/24, f'{standalone}/24个标签符合')
        report = parsed['parse_report']
        valid = sum(q in parsed['answer_occurrences'] and len(parsed['answer_occurrences'][q]) == 1 and bool(answers[q].strip()) for q in QUESTIONS)
        add('24题齐全且不重复', 15, valid/24, f'{valid}/24题；异常题号：'+','.join(report['unknown']))
        add('题号顺序', 5, float([q for q,_ in labels] == QUESTIONS), '须完整按A01至B12排列')
        # Preamble and text after the B12 answer are not semantically separable;
        # only objectively identifiable global wrappers/preamble are penalized.
        wrapper = any('外层代码围栏' in w for w in report['warnings'])
        add('无额外开场及整卷围栏', 5, float(not parsed['preamble'] and not wrapper), '检查题号前文本及整卷代码围栏；不猜测B12结尾是否为结束语')
    else:
        for name, weight in [('标签独占一行且使用[A01]形式',20),('24题齐全且不重复',15),('题号顺序',5),('无额外开场及整卷围栏',5)]:
            add(name,weight,None,'无原始全文，无法核对；不假定满分')
    for q in ('A07','A08','A09'):
        _, ok = strict_output(q,answers[q])
        # Existing parser accepts code fences for content but deducts format.
        add(q+'仅输出指定内容',20/3,float(ok),'不允许说明、代码围栏或额外内容')
    simple = [bool(re.fullmatch(r'[0-9]',answers['B01'].strip())),
              bool(re.fullmatch(r'[\u4e00-\u9fff]+',answers['B05'].strip())),
              bool(re.fullmatch(r'(?:[1-9][0-9]?|100)',answers['B06'].strip()))]
    add('B01/B05/B06单项输出',10,sum(simple)/3,'分别只输出范围内整数、中文名称、范围内整数；不验证动物语义')
    for q, pattern, size in [('B02',r'[0-9]{120}',120),('B03',r'[HT]{80}',80)]:
        text=answers[q].strip()
        add(q+'长度和字符',10,float(bool(re.fullmatch(pattern,text))),f'须恰好{size}个指定字符；实有{len(text)}字符；内部空白同样违规')
    add('B04骰子格式',5,float(bool(re.fullmatch(r'[1-6](?:\s*,\s*[1-6]){23}',answers['B04'].strip()))),'24个1至6整数，以英文逗号分隔')
    points=round(sum(r['points'] or 0 for r in rows),2)
    missing=sum(r['maximum'] for r in rows if r['points'] is None)
    return dict(version='format-instruction-2',max_points=100,provisional_score=points,
                score=None if missing else points,score_interval=[points,round(points+missing,2)],
                pending_review=int(bool(missing)),items=rows,
                note='只自动核查可验证的交付格式；忽略每题末尾独占一行的横向分隔线，不忽略解释、围栏、序列内部空白。矛盾题B11如实说明不处罚；随机性、工具使用及开放题语义不靠关键词自动定罪。')


def output_content(q, raw):
    """Read a leading valid output independently of trailing format violations."""
    raw=answer_body(raw)
    direct,_=strict_output(q,raw)
    text=raw.strip()
    if text.startswith('```'):
        text=re.sub(r'^```[^\n]*\n','',text)
        text=text.split('```',1)[0].strip()
    candidate=0
    if q=='A07':
        arrays=re.findall(r'\[[^\[\]\n]*\]',text)
        if len(arrays)>=2:
            candidate=strict_output(q,'\n'.join(arrays[:2]))[0]
    else:
        try:
            _,end=json.JSONDecoder().raw_decode(text)
            candidate=strict_output(q,text[:end])[0]
        except (ValueError,TypeError): pass
    # Corrections are not silently rescued here; final-output interpretation
    # still uses a human review when a leading wrong answer was delivered.
    return max(float(direct),float(candidate))


def score_dual(record, legacy):
    answers={q: answer_body(v) for q, v in canonical_answers(record).items()}
    fmt=format_score(record)
    rows=[]
    for row in legacy['questions']:
        q=row['question'];weight=row['weight']
        consistency=row['consistency']
        old_cap=consistency['cap_fraction']
        cap={.8:.7,.9:.85,.6:.5}.get(old_cap,old_cap)
        base=row['content_provisional_points']/weight
        low,high=row['points_interval']
        if q in ('A07','A08','A09'):
            base=output_content(q,answers[q])
            if row['status']=='human_reviewed':
                # Legacy human scores include its 20% format portion. Remove
                # that portion before applying the new independent format score.
                _,ok=strict_output(q,answers[q])
                base=max(base,min(1,max(0,(row['content_provisional_points']/weight-.2*float(ok))/.8)))
            low=high=weight*base
        points=round(weight*min(base,cap),4)
        rows.append(dict(question=q,weight=weight,provisional_points=points,
                         cap_fraction=cap,legacy_points=row['provisional_points'],
                         points_interval=[round(min(low,weight*cap),4),round(min(high,weight*cap),4)]))
    content=round(sum(r['provisional_points'] for r in rows),2)
    interval=[round(.9*sum(r['points_interval'][i] for r in rows)+.1*fmt['score_interval'][i],2) for i in (0,1)]
    strict=round(.9*content+.1*fmt['provisional_score'],2)
    pending=legacy['pending_review']+fmt['pending_review']
    combined=round(legacy['provisional_score']+strict,2)
    return dict(version=VERSION,legacy=dict(version=legacy['rubric_version'],provisional_score=legacy['provisional_score'],
                    max_points=100,note='旧版A题分；包含原有A07–A09局部格式扣分，不检查整卷格式。'),
                strict=dict(version=STRICT_VERSION,provisional_score=strict,max_points=100,
                    content_score=content,format_score=fmt,questions=rows,weights={'content':.9,'format':.1},
                    consistency_caps=STRICT_CAPS,pending_review=pending,score_interval=interval,
                    note='内容90%+格式10%；更正错误结论/过程/未撤回冲突上限70%/85%/50%；A07–A09格式分迁至格式项，本版不重复扣。开放推导继承复核状态。'),
                combined=dict(provisional_score=combined,max_points=200,score=None if pending else combined,
                    normalized_score=round(combined/2,4),pending_review=pending,
                    score_interval=[round(legacy['score_interval'][i]+interval[i],2) for i in (0,1)],
                    note='旧版100分与严格格式版100分相加；型号按合计均分排名。'))
