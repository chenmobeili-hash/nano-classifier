"""Bounded, explainable checks for this fixed questionnaire. Never runs answer code."""
import ast
import json
import math
import re
import unicodedata
from fractions import Fraction


def clean(text):
    # Preserve superscript exponent meaning before NFKC turns 7¹² into 712.
    supers = '⁰¹²³⁴⁵⁶⁷⁸⁹'
    text = re.sub('(['+supers+']+)', lambda m: '^'+m[0].translate(str.maketrans(supers, '0123456789')), text)
    text = unicodedata.normalize('NFKC', text).replace('**', '').replace('__', '')
    text = text.translate(str.maketrans({'−': '-', '–': '-', '×': '*', '·': '*', '÷': '/', '，': ',', '：': ':'}))
    for command, value in (('times', '*'), ('cdot', '*'), ('equiv', '≡'), ('rightarrow', '→'), ('to', '→')):
        text = re.sub(r'\\'+command+r'(?![A-Za-z])', lambda _: value, text)
    for _ in range(5):
        text = re.sub(r'\\(?:d?frac)\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'(\1)/(\2)', text)
        text = re.sub(r'\\binom\s*\{([^{}]+)\}\s*\{([^{}]+)\}', r'C(\1,\2)', text)
    text = re.sub(r'\\binom\s*(\d)\s*(\d)', r'C(\1,\2)', text)
    text = re.sub(r'\\binom\s*(?:\{([^{}]+)\}|(\d))\s*(?:\{([^{}]+)\}|(\d))',
                  lambda m: 'C('+(m[1] or m[2])+','+(m[3] or m[4])+')', text)
    text = re.sub(r'\\(?:boxed|mathbf|mathrm|text)\{([^{}]*)\}', r'\1', text)
    text = re.sub(r'\\(?:boxed|mathbf|mathrm)(\d+)', r'\1', text)
    text = re.sub(r'\^\{([^{}]+)\}', r'^\1', text)
    text = re.sub(r'\\[()[\]]|\\[,;!]|\$', '', text)
    text = re.sub(r'\\(?:left|right)\b', '', text)
    text = re.sub(r'\)\s*(?=C\()', ')*', text)
    return text


def compact(text):
    return re.sub(r'\s+', '', clean(text))


def arithmetic(expression):
    """Parse a tiny arithmetic grammar into exact rationals, without eval/exec."""
    if len(expression) > 250:
        raise ValueError('expression too long')
    tree = ast.parse(expression.replace('^', '**'), mode='eval')
    if len(list(ast.walk(tree))) > 100:
        raise ValueError('too many operations')
    def visit(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            value = Fraction(str(node.value))
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = visit(node.operand)*(1 if isinstance(node.op, ast.UAdd) else -1)
        elif isinstance(node, ast.BinOp):
            a, b = visit(node.left), visit(node.right)
            if isinstance(node.op, ast.Add): value = a+b
            elif isinstance(node.op, ast.Sub): value = a-b
            elif isinstance(node.op, ast.Mult): value = a*b
            elif isinstance(node.op, ast.Div): value = a/b
            elif isinstance(node.op, ast.Pow) and b.denominator == 1 and 0 <= b <= 12: value = a**int(b)
            else: raise ValueError('operator not supported')
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'C' and len(node.args) == 2 and not node.keywords:
            n, k = (visit(arg) for arg in node.args)
            if n.denominator != 1 or k.denominator != 1 or not 0 <= k <= n <= 100:
                raise ValueError('combination out of range')
            value = Fraction(math.comb(int(n), int(k)))
        else:
            raise ValueError('not literal arithmetic')
        if abs(value.numerator) > 10**15 or value.denominator > 10**15:
            raise ValueError('arithmetic out of range')
        return value
    return visit(tree.body)


def number_after(fragment):
    fragment = fragment.lstrip(' \n:;=,[]{}')
    match = re.match(r'[0-9+\-().*/^=C,\s]+', fragment)
    if not match:
        return None
    expression = match[0].strip().rstrip('.,')
    annotated_number = re.match(r'^(\d+)\(', expression)
    if annotated_number and '=' not in expression:
        return Fraction(annotated_number[1])
    # A conclusion with a chain gives its final asserted value, not a recomputed
    # replacement for what the respondent actually wrote.
    expression = expression.split('=')[-1].strip()
    for _ in range(4):
        try:
            return arithmetic(expression)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            if expression.endswith(')') and expression.count(')') > expression.count('('):
                expression = expression[:-1].strip()
            else:
                return None
    return None


CORRECTION = re.compile(
    r'更正|改正|修正(?:说明|后)?|重新核对|结论应为|正确(?:答案|结果)'
    r'|(?:正确答案|最终(?:答案|结果|结论)|答案|结论).{0,12}以.{0,16}为准'
    r'|以.{0,16}为准\s*[:：]'
    r'|精确分数(?:为|是)'
    r'|(?:上面|此前|前面|上段).{0,20}(?:错|有误|矛盾)'
    r'|重新(?:核查|逐一计算|整理)|重来|重做'
    r'|等等[，,:：]?\s*(?:正确|重新|需要)'
    r'|(?:^|\n)\s*答\s*[:：]'
)


def hypothetical_claim(text, position):
    before = text[:position]
    boundary = max(before.rfind(c) for c in ('。', '\n', ';', '；'))
    return bool(re.search(r'假设|假如|反证|错误示例', before[boundary+1:]))


def decision(claims, expected, text):
    claims = [(pos, value) for pos, value in claims if not hypothetical_claim(text, pos)]
    def output(data, selected=(), corrected=False):
        active = {(pos, str(value)) for pos, value in selected}
        data.update(corrected=corrected, claim_history=[
            {'position': pos, 'value': str(value), 'correct': value == expected,
             'active': (pos, str(value)) in active}
            for pos, value in sorted(claims, key=lambda item: item[0])])
        return data
    if not claims:
        return output({'score': None, 'state': 'unresolved', 'evidence': '未识别出明确结论；不按全文数字命中判对。', 'claims': []})
    ordered = sorted(claims, key=lambda item: item[0])
    corrections = list(CORRECTION.finditer(text))
    corrected = False
    if corrections:
        last = corrections[-1]
        after = [(pos, value) for pos, value in ordered if pos >= last.start()]
        if after:
            ordered, corrected = after, True
        elif ordered[-1][0] >= corrections[0].start():
            # A later cue discarded earlier work, but no replacement claim was
            # parsed. Keep the last claim only when it sits in the last cue span.
            span_end = min(len(text), last.end()+80)
            nearby = [(pos, value) for pos, value in ordered if last.start() <= pos <= span_end]
            ordered, corrected = (nearby or ordered[-1:]), True
    values = [value for _, value in ordered]
    distinct = set(values)
    if len(distinct) > 1:
        return output({'score': .5 if expected in distinct else 0, 'state': 'conflict',
                'evidence': '识别到相互冲突的结论，未见明确更正，需复核。', 'claims': [str(v) for v in values]}, ordered, corrected)
    value = values[-1]
    return output({'score': float(value == expected), 'state': 'correct' if value == expected else 'incorrect',
            'evidence': ('按明确更正后的结论核对。' if corrected else '按明确结论核对。'), 'claims': [str(v) for v in values]}, ordered, corrected)


def numeric_result(q, raw, expected):
    text = clean(raw)
    flat = re.sub(r'\s+', '', text)
    claims = []
    # Leading bare number is an answer only when not an operand of a formula.
    lead = re.match(r'^\s*([+-]?\d+(?:\.\d+)?)(?=\s*(?:。|[个条种](?=\s*(?:[。；，,.!！]|$))|$))', text)
    if lead:
        claims.append((lead.start(), Fraction(lead[1])))
    anchors = [r'(?:最终(?:答案|结果)?|数值结果|答案|结果)(?:应为|为|是|[:=])+']
    if q == 'A01':
        # Remainder cues only. Do not take 7^123=… from a power chain such as
        # 7^123≡7^3=343≡5, which asserts 343 as an intermediate, not the answer.
        anchors += [r'余数(?:是|为|[:=])+', r'(?:mod|modulo)\s*13\s*[=≡]\s*']
    if q in ('A04', 'A12'):
        anchors += [r'^\s*(?:共|共有|数值为)\s*', r'总计\s*', r'(?:^|\n)\s*答\s*[:：]\s*(?:共|共有)?\s*']
    if q == 'A11': anchors += [r'(?:合法路径(?:数)?|合法数)(?:为|是|共|[:=])+', r'^\s*共\s*']
    for pattern in anchors:
        for m in re.finditer(pattern, text, re.M):
            prefix = text[max(0, m.start()-10):m.start()]
            if re.search(r'违规|违法|排除|不合法|中间|子情况|第一类|第二类', prefix):
                continue
            # "结果：[j][k]=..." is a DP state table, not the requested count.
            if re.match(r'\s*(?:[A-Za-z_]+)?\[', text[m.end():]):
                continue
            if q == 'A12' and m[0].strip().startswith('总计'):
                context = re.sub(r'\s+', '', text[max(0, m.start()-400):m.start()])
                if re.search(r'dp\[8\]\[4\]\[[012]\]', context) and not re.search(r'首末不同|首位与末位不同', context):
                    continue
            value = number_after(text[m.end():m.end()+240])
            if value is not None and not (q == 'A01' and (value.denominator != 1 or not 0 <= value < 13)):
                claims.append((m.start(), value))
    if q == 'A01':
        # Direct remainder assertion, not a power-chain intermediate such as 7^123≡7^3=343.
        for m in re.finditer(r'7\^123\s*≡\s*(\d+)\s*\(?\s*mod(?:ulo)?\s*13', text):
            value = Fraction(m[1])
            if value.denominator == 1 and 0 <= value < 13:
                claims.append((m.start(), value))
    if q == 'A12':
        # 合计 26 after 另一类/13+13 is a total; 合计 13 种 / 合计 6 is a subclass count.
        # 违规合计 4+3=7 is a discarded subtotal, not the answer.
        for m in re.finditer(r'合计\s*', text):
            prefix = text[max(0, m.start()-10):m.start()]
            if re.search(r'违规|违法|排除|不合法|中间|子情况|第一类|第二类', prefix):
                continue
            window = text[max(0, m.start()-48):m.end()+48]
            if not re.search(r'两类|另一类|总共|一共|两\s*类|\d+\s*\+\s*\d+', window):
                continue
            value = number_after(text[m.end():m.end()+240])
            if value is not None:
                claims.append((m.start(), value))
    for m in CORRECTION.finditer(text):
        value = number_after(text[m.end():m.end()+120])
        if value is None:
            continue
        if q == 'A01' and (value.denominator != 1 or not 0 <= value < 13):
            continue
        claims.append((m.start(), value))
    # Last therefore/formula conclusion is accepted only with a terminal numeric
    # equality, not an occurrence of the expected number somewhere in the text.
    for m in re.finditer(r'(?:因此|所以|故|容斥得|容斥后|合法路径|合法数)[^。\n]{0,220}', text):
        if re.search(r'排除|违规|违法|不合法', m[0]):
            continue
        if q in ('A04', 'A12') and not re.search(r'总数|总计|合计|最终|结果|答案', m[0]):
            continue
        terminal = re.search(r'=\s*([+-]?\d+)\s*[).。个条种]*\s*$', m[0])
        if terminal:
            claims.append((m.start(), Fraction(terminal[1])))
    if not claims and q in ('A04', 'A12'):
        terminal = re.search(r'=\s*(\d+)\s*[。个条种.)\s]*$', text)
        if terminal:
            claims.append((terminal.start(), Fraction(terminal[1])))
    return decision(claims, Fraction(expected), text)


def culprit_result(raw):
    text = compact(raw)
    claims = []
    uncertain = re.match(r'^(?:[甲乙丙](?:或[甲乙丙])+都可能|(?:题目)?条件不足|不能唯一确定)', text)
    if uncertain:
        claims.append((uncertain.start(), '不能唯一确定'))
    lead = re.match(r'^\s*(?:是|杯子是|打破杯子的是)?([甲乙丙])(?:打破|[。.]|$)', text)
    if lead: claims.append((lead.start(), lead[1]))
    for m in re.finditer(
            r'(?:答案[:：]?|结论[:：]|结论应为|故只能是|故是|故打破(?:杯子的)?者?是|故唯一符合的是|因此杯子是|所以是'
            r'|唯一(?:可行者?|满足.{0,20}|符合.{0,12})是|剩下唯一.{0,24}是)([甲乙丙])', text):
        claims.append((m.start()+1, m[1]))
    # A final direct statement after an explicit recheck is a replacement claim.
    for m in re.finditer(r'(?:因此|所以)([甲乙丙])打破', text):
        claims.append((m.start(), m[1]))
    return decision(claims, '甲', text)


def probability_result(raw):
    text = compact(raw)
    claims = []
    ratio = r'\(?([0-9]+(?:\.[0-9]+)?)\)?/\(?([0-9]+(?:\.[0-9]+)?)\)?'
    patterns = [r'^'+ratio, r'(?:条件概率(?:为|是)|精确分数(?:为|是|[:：])?|答案[:：]?)'+ratio,
                r'='+ratio+r'(?=[。()≈,]|$)']
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            if text[m.end():].lstrip(')').startswith(('/', '*', '+', '-')):
                continue
            if pattern.startswith('='):
                # A named prior/likelihood/marginal is not the requested posterior.
                clause = re.split(r'[。；;，,]', text[:m.start()])[-1]
                named = list(re.finditer(r'P\(([^()]*)\)', clause, re.I))
                if named and clause.startswith(named[0][0]):
                    event = named[-1][1].replace(r'\mid', '|')
                    if '|' not in event or event.split('|')[0] in ('+', '阳性'):
                        continue
            if Fraction(m[2]):
                claims.append((m.start(), Fraction(m[1])/Fraction(m[2])))
    for m in CORRECTION.finditer(text):
        value = number_after(text[m.end():m.end()+120])
        if value is not None:
            claims.append((m.start(), value))
    return decision(claims, Fraction(18, 67), text)


def arithmetic_checks(raw):
    from ability_methods import math_text
    text = re.sub(r'\s+', '', math_text(clean(raw)))
    checks = []
    for m in re.finditer(r'[0-9C()+*/^.,=\-]+', text):
        fragment = m[0].strip('.,')
        if '=' not in fragment:
            continue
        if fragment.startswith('(') and m.start() and re.match(r'[A-Za-zφΦϕ]', text[m.start()-1]):
            continue
        if (m.start() and text[m.start()-1] in '}\\') or (m.end() < len(text) and text[m.end()] == '\\'):
            continue
        for pair_index, (left, right) in enumerate(zip(fragment.split('='), fragment.split('=')[1:])):
            # g5=0 and p3=1 assign indexed variables; N=<arithmetic>=3882
            # still contains a verifiable numeric equality after its first '='.
            if pair_index == 0 and m.start() and re.match(r'[A-Za-z_]', text[m.start()-1]):
                continue
            for _ in range(4):
                if left.startswith('(') and left.count('(') > left.count(')'): left = left[1:]
                if right.endswith(')') and right.count(')') > right.count('('): right = right[:-1]
            try:
                a, b = arithmetic(left), arithmetic(right)
            except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
                continue
            checks.append({'expression': left+'='+right, 'correct': a == b, 'value': str(b)})
    return checks


def method_evidence(q, raw, result):
    from ability_methods import game_evidence, path_evidence, binary_evidence
    original = raw
    corrections = list(re.finditer(r'(?:更正|改正|修正(?:说明|后(?:的结论)?)?|正确解法|重新计算|重新推导|重新核对|重来|重新逐一计算)(?:如下|为|是|[:：])', raw))
    if corrections and raw[corrections[-1].end():].strip():
        raw = raw[corrections[-1].end():]
    text = compact(raw)
    checks = arithmetic_checks(raw)
    errors = [item['expression'] for item in checks if not item['correct']]
    flags = []
    if q == 'A01' and re.search(r'每6次循环|周期(?:为|是)6|7\^6≡1(?!\d)', text):
        errors.append('模13周期误写为6；实际7^6≡-1、周期为12。')
    if q == 'A01':
        for power, residue in re.findall(r'7\^(\d+)≡([+-]?\d+)(?![\d^*/])', text):
            if int(power) <= 1000 and pow(7, int(power), 13) != int(residue) % 13:
                errors.append('模13同余结论有误：7^'+power+'≡'+residue)
    verified = {c['value'] for c in checks if c['correct']}
    full = False
    partial = False
    if q == 'A01':
        full = (bool(re.search(r'7\^12≡1|费马小定理|周期(?:为|是)12', text)) and
                bool(re.search(r'123=12\*10\+3|123≡3', text)) and '7^3' in text)
        full |= bool(re.search(r'7\^6(?:\s*=[^\n≡]{0,24})?≡', text)) and '-1' in text and bool(re.search(r'123=6\*20\+3', text))
        partial = bool(re.search(r'推导|模13|费马|≡', text))
    elif q == 'A02':
        # Correcting the culprit need not repeat a still-valid case table.
        proof = compact(original) if result['state'] == 'correct' and re.search(r'上述|如上|前述|上面|前面', text) else text
        full = bool(re.search(r'甲.{0,12}假.{0,20}乙.{0,12}真.{0,20}丙.{0,12}假|只有乙.{0,10}(?:真|为真)', proof))
        full |= bool(re.search(r'甲(?:和|与|、)?乙.{0,20}(?:否定|矛盾)', proof)) and bool(re.search(r'丙.{0,16}假', proof))
        partial = bool(re.search(r'真|假|互为否定', text))
    elif q == 'A03':
        full = all(re.search(pattern, text) for pattern in (r'0\.02\*0\.90?|0\.90?\*0\.02', r'0\.98\*0\.05|0\.05\*0\.98')) and '+' in text
        partial = bool(re.search(r'贝叶斯|0\.018|P\(', text))
    elif q == 'A04':
        full = bool(re.search(r'C\(7,[34]\)|C\(10-4\+1,4\)', text)) and bool(re.search(r'间隙|空[位隙档]|隔板|位置|相邻', text))
        partial = bool(re.search(r'间隙|空[位隙档]|隔板|位置|相邻', text))
    elif q == 'A10':
        full, partial, flags = game_evidence(text)
    elif q == 'A11':
        full, partial, flags = path_evidence(text, checks, arithmetic)
        errors.extend(f for f in flags if 'DP第' in f)
        flags = [f for f in flags if 'DP第' not in f]
    elif q == 'A12':
        full, partial, flags = binary_evidence(text, checks)
    if flags or any('周期' in error or '同余' in error for error in errors):
        return {'score': 0, 'state': 'contradiction', 'evidence': '方法中存在可核对的矛盾。', 'issues': flags+errors[:6]}
    if errors:
        return {'score': .5 if partial else 0, 'state': 'arithmetic_error',
                'evidence': '发现错误等式；有方法线索时保留一半过程分，不另叠加额外罚分。', 'issues': errors[:6]}
    if not partial and re.fullmatch(r'(?:余数(?:为|是)|答案[:：]?|结果[:：]?)?\(?\d+(?:/\d+)?\)?[个条种。.!]*|(?:是|杯子是|打破杯子的是)?甲打破了?杯子[。.!]*', text):
        return {'score': 0, 'state': 'missing_method', 'evidence': '只给出结论，未提供题目要求的推导；不计过程分。', 'issues': []}
    # Isolated cues cannot overturn a wrong conclusion; a complete method can
    # still get process credit when the correct conclusion is among the claims.
    estimate = 1 if full and result['state'] == 'correct' else (
        1 if full and result['state'] == 'conflict' and result['score'] == .5 else .5 if partial else None)
    return {'score': estimate, 'state': 'rule_supported' if estimate == 1 else 'method_cue' if estimate == .5 else 'unresolved',
            'evidence': '有限规则支持该方法，完整推导仍需复核。' if estimate == 1 else '只识别到方法线索，方法暂给一半；完整推导待复核。' if estimate == .5 else '未验证该方法，不凭正确数字给方法分。', 'issues': []}


def path_components(raw):
    text = compact(raw)
    paths = []
    for m in re.finditer(r'(?:最短路径(?:为|是|:)?|路径为|路径(?=A→)|答案[:：]?(?=A→)|^)(A(?:→[A-E])+)', text):
        paths.append((m.start(), m[1]))
    path = decision(paths, 'A→C→B→E→D', text)
    values = []
    for m in re.finditer(r'(?:最短总权重|总权重)(?:为|是|:)?', text):
        value = number_after(text[m.end():m.end()+100])
        if value is not None:
            values.append((m.start(), value))
    return path, decision(values, Fraction(7), text)


def recurrence_components(raw):
    text = compact(raw)
    values = {}
    errors = []
    matches = list(re.finditer(r'(?<![+\-*/])f\s*\(\s*([1-8])\s*\)\s*=', text))
    for index, m in enumerate(matches):
        end = matches[index+1].start() if index+1 < len(matches) else len(text)
        fragment = re.split(r'[\n;；。]', text[m.end():end], maxsplit=1)[0]
        errors.extend(item['expression'] for item in arithmetic_checks(fragment) if not item['correct'])
        value = number_after(fragment.split('=')[-1])
        if value is not None:
            values.setdefault(int(m[1]), []).append(value)
    sequence = None
    # LaTeX writes the commas as ",\ " ; after compacting that is ",\d", not ",d".
    seq_text = text.replace('\\', '')
    for m in re.finditer(r'(?<![\d,、])(\d+(?:\s*[,、]\s*\d+){7})(?![\d,、])', seq_text):
        sequence = [Fraction(v) for v in re.findall(r'\d+', m[1])]
    expected = list(map(Fraction, (1, 2, 4, 7, 13, 24, 44, 81)))
    if sequence is not None:
        checks = [a == b for a, b in zip(sequence, expected)]
    elif all(i in values for i in range(1, 9)):
        checks = [set(values[i]) == {expected[i-1]} for i in range(1, 9)]
    else:
        checks = None
    final_claims = [(i, v) for i, v in enumerate(values.get(8, []))]
    if sequence is not None: final_claims.append((len(final_claims), sequence[-1]))
    final = decision(final_claims, Fraction(81), text)
    conflict = any(len(set(v)) > 1 for v in values.values())
    estimate = sum(checks)/8 if checks is not None else None
    if errors and estimate is not None: estimate = min(estimate, .5)
    seq = {'score': estimate, 'state': 'conflict' if conflict or errors else 'checked' if checks is not None else 'unresolved',
           'evidence': '递推等式存在算术矛盾；保留部分序列分并要求复核。' if errors else '重复列出的序列相互矛盾，暂按一致正确项给分并要求复核。' if conflict else '逐项核对f(1)至f(8)。' if checks is not None else '没有识别到完整8项序列；不按出现13、24等数字判对。', 'issues': errors[:6]}
    return final, seq


def game_components(raw):
    text = compact(raw)
    win_claims = [(m.start(), m[1] == '必胜') for m in re.finditer(r'先手.{0,6}?(不胜|必败|不能必胜|非必胜|必胜)', text)]
    winner = decision(win_claims, True, text)
    move_claims = []
    for m in re.finditer(r'(?:第一步(?:是|应|必须|只能|为|先|必)?|先手取|唯一.{0,8}取法(?:是|为)?)[^。；\n]{0,65}', text):
        segment = re.split(r'留下|使剩|取后|把|理由|因为|按规则|[。；]', m[0])[0]
        number = re.search(r'取([1一3三4四])(?:枚|颗)?((?:或|和|、)(?:取)?[1一3三4四](?:枚|颗)?)?', segment)
        if number:
            chars = re.findall(r'[1一3三4四]', number[0])
            mapping = {'1': 1, '一': 1, '3': 3, '三': 3, '4': 4, '四': 4}
            move_claims.append((m.start(), frozenset(mapping[v] for v in chars)))
    move = decision(move_claims, frozenset({1}), text)
    if not move_claims and winner['state'] == 'incorrect' and re.search(r'不存在|没有|无(?:法|任何)', text):
        move = {'score': 0, 'state': 'incorrect', 'evidence': '声明不存在获胜第一步，与实际取1必胜不符。', 'claims': ['无获胜取法']}
    if move['state'] == 'incorrect' and any('1' in v for v in move['claims']):
        move['score'] = .5  # Correct winning move included, but losing moves also included.
        move['evidence'] = '列出了取1，也包含不能保证获胜的取法，只给部分分。'
    return winner, move


def strict_output(q, raw):
    stripped = raw.strip()
    fence = re.fullmatch(r'```(?:json|text|python)?\s*\n(.*?)\n```', stripped, re.S)
    body = fence[1].strip() if fence else stripped
    if q == 'A07':
        arrays = re.findall(r'\[[^\[\]]*\]', body)
        content = 0
        if len(arrays) <= 2 and not re.sub(r'\[[^\[\]]*\]', '', body).strip():
            for index, (value, target) in enumerate(zip(arrays, ([2, 2, 2], [0, 1, 2]))):
                try:
                    parsed = json.loads(value)
                    content += .5*int(parsed == target and all(type(v) is int for v in parsed))
                except (ValueError, TypeError):
                    pass
        formatting = not fence and bool(re.fullmatch(r'\[\s*\d\s*,\s*\d\s*,\s*\d\s*\]\s*\n\s*\[\s*\d\s*,\s*\d\s*,\s*\d\s*\]', stripped))
    else:
        def unique(pairs):
            result = {}
            for k, v in pairs:
                if k in result: raise ValueError('重复JSON键')
                result[k] = v
            return result
        try:
            value = json.loads(body, object_pairs_hook=unique, parse_constant=lambda _: (_ for _ in ()).throw(ValueError('非标准JSON')))
            target = {'items': ['pear', 'apple', 'banana', 'kiwi']} if q == 'A08' else [1, 3]
            schema = isinstance(value, dict) and 'items' in value and isinstance(value['items'], list) and all(type(v) is str for v in value['items']) if q == 'A08' else isinstance(value, list) and all(type(v) is int for v in value)
            actual = value.get('items', []) if q == 'A08' and isinstance(value, dict) else value
            expected = target['items'] if q == 'A08' else target
            content = sum(a == b for a, b in zip(actual, expected))/len(expected) if schema and len(actual) == len(expected) else 0
            formatting = not fence and schema and (q != 'A08' or set(value) == {'items'})
        except (ValueError, TypeError):
            content, formatting = False, False
    return float(content), bool(formatting)
