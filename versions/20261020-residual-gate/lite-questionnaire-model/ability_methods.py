"""Task-specific evidence checks, independent of model names and reference scores.

These checks validate bounded arithmetic and explicit assertions; they do not
execute submitted code or claim to verify arbitrary natural-language proofs.
"""
import ast
import itertools
import re
from collections import Counter


def game_states(limit=100):
    wins = [False]
    for n in range(1, limit+1):
        wins.append(any(not wins[n-m] for m in (1, 3, 4) if m <= n))
    return wins


def game_evidence(text):
    text = text.replace('\\', '')
    text = re.sub(r'(?:假设|假如|错误示例)[^。;；]*', '', text)
    wins = game_states()
    claims = []
    sets = []
    losing = r'(?:必败(?:局面|位置|状态|态|位|点)?|败[态点]|P(?:[位态点]|位置|局面)?)'
    winning = r'(?:必胜(?:局面|位置|状态|态|位|点)?|胜[态点]|N(?:[位态点]|位置|局面)?)'
    state = '('+losing+'|'+winning+')'
    numbers = r'\{?([0-9]+(?:(?:枚)?[,、或][0-9]+)+)(?:枚)?\}?'
    patterns = [losing+r'(?:集合|序列)?(?:[（(][^）)]{0,20}[）)])?(?:依次)?(?:为|有|是|[:=])+\{?([0-9]+(?:[,、][0-9]+)+)',
                r'(?:P|L)[:=]\{([0-9]+(?:[,、][0-9]+)+)\}',
                r'集合[A-Z]=\{([0-9]+(?:[,、][0-9]+)+)\}为[^。]{0,20}必败',
                numbers+r'(?:为|是|属于)'+losing]
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            values = {int(v) for v in re.findall(r'\d+', m[1]) if len(v) <= 3}
            sets.append(values)
            claims.extend((n, False) for n in values if n <= 100)

    for m in re.finditer(r'(?<![\d^])([0-9]{1,3})(?:枚)?(?:是|为|属于|判为|[（(])(?:对手的)?'+state, text):
        n = int(m[1])
        if n <= 100:
            claims.append((n, not bool(re.fullmatch(losing, m[2]))))
    for m in re.finditer(r'(?<!\d)([0-9]{1,3})[:：]([PN])(?=[,、。;；]|$)', text):
        if int(m[1]) <= 100:
            claims.append((int(m[1]), m[2] == 'N'))
    for m in re.finditer(r'([0-9]{1,3}(?:[/、,][0-9]{1,3})+)[（(,]?(?:全|均|都)(?:为|是)?'+state, text):
        claims.extend((int(n), not bool(re.fullmatch(losing, m[2])))
                      for n in re.findall(r'\d+', m[1]) if int(n) <= 100)

    # e.g. "leave 14 or 12; both are losing", not a detached nearby number.
    for m in re.finditer(r'(?:留下|剩下|剩余)'+numbers+r'[;；,]?这(?:两个|些)(?:数)?(?:都|均)(?:是|为)'+state, text):
        claims.extend((int(v), not bool(re.fullmatch(losing, m[2])))
                      for v in re.findall(r'\d+', m[1]) if len(v) <= 3 and int(v) <= 100)

    # The subject is n only for an explicit "therefore P/N" in that n-clause.
    # "n=15 take 1 to 14 (P)" is a statement about 14, not a P label on 15.
    for m in re.finditer(r'n=([0-9]{1,3})(.*?)(?=n=\d|[;；。]|$)', text):
        n = int(m[1])
        conclusion = re.search(r'(?:故|所以|因此)(?:为|是)?'+state, m[2])
        if conclusion and n <= 100:
            claims.append((n, not bool(re.fullmatch(losing, conclusion[1]))))

    issues = sorted({str(n)+'枚实际为'+('必胜' if wins[n] else '必败')+'，与过程中的状态断言矛盾。'
                     for n, won in claims if wins[n] != won})
    if re.search(r'(?:mod|modulo|模)5', text) and not re.search(r'(?:mod|modulo|模)7', text):
        issues.append('取1、3、4的必败周期为7，不能使用模5败态。')
    residues = bool(re.search(r'(?:mod(?:ulo)?|模|除以)7', text)) and bool(
        re.search(r'(?:余|≡)0(?:或|,|、)2|(?:余|≡)0[^。;；]{0,10}(?:余|≡)2', text))
    complete_set = any({0, 2, 7, 9, 14} <= values for values in sets)
    full = (residues or complete_set) and '14' in text
    return full, bool(re.search(r'必败|败态|P[位态点]|递推', text)), issues


def math_text(text):
    """Remove display-only LaTeX after fractions/binomials were normalized."""
    text = re.sub(r'\\(?:begin|end)\{[^{}]*\}', '', text)
    text = re.sub(r'\\q?quad', ';', text)
    text = re.sub(r'\\(?:pmod|mod)\b', '', text)
    return text.translate(str.maketrans('', '', '┌─┐│└┘')).replace('\\', '').replace('&', '').replace('{', '').replace('}', '')


def signed_terms(expression, arithmetic):
    """Expand sums, while evaluating only bounded literal arithmetic terms."""
    tree = ast.parse(expression.replace('^', '**'), mode='eval')
    if len(expression) > 250 or len(list(ast.walk(tree))) > 100:
        raise ValueError('expression limit')
    def walk(node, sign=1):
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
            return walk(node.left, sign)+walk(node.right, sign if isinstance(node.op, ast.Add) else -sign)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return walk(node.operand, -sign)
        value = arithmetic(ast.unparse(node))
        return [sign*int(value)] if value.denominator == 1 else []
    return walk(tree.body)


def path_evidence(text, checks, arithmetic):
    formula = math_text(text)
    target = Counter((12870, -5544, -4410, -1764, 2100, 630))
    full_ie = False
    for check in checks:
        if not check['correct'] or check['value'] != '3882':
            continue
        try:
            terms = signed_terms(check['expression'].split('=')[0], arithmetic)
        except (ValueError, SyntaxError, ZeroDivisionError, OverflowError):
            continue
        # Readable subtotals are accepted only with their individual contributions.
        for subtotal, parts in ((8988, (5544, 4410, 1764, -2100, -630)),
                                (11718, (5544, 4410, 1764)), (2730, (2100, 630))):
            expanded = []
            for value in terms:
                if abs(value) == subtotal and all(str(abs(p)) in text for p in parts):
                    expanded.extend((1 if value >= 0 else -1)*p for p in parts)
                else:
                    expanded.append(value)
            terms = expanded
        full_ie |= Counter(terms) == target

    recurrence = re.search(r'([A-Za-z][A-Za-z_]*)(?:\(x,y\)|\[x\]\[y\])='+
                           r'\1(?:\(x-1,y\)|\[x-1\]\[y\])\+\1(?:\(x,y-1\)|\[x\]\[y-1\])', formula)
    recurrence = recurrence or re.search(
        r'([A-Za-z][A-Za-z_]*)\[i\]\[j\]=\1\[i-1\]\[j\]\+\1\[i\]\[j-1\]', formula)
    origin = bool(re.search(r'(?:\(0,0\)|\[0\]\[0\])=1', formula))
    edge = bool(re.search(r'边界.{0,50}(?:均为1|为1|=1)', formula))
    obstacles = all(p in formula for p in ('(2,2)', '(4,5)', '(6,3)'))
    explicit_points = set(re.findall(r'\([0-8],[0-8]\)', formula)) - {'(0,0)', '(8,8)'}
    obstacles |= not explicit_points and bool(re.search(r'被禁止的点|禁点|障碍点', formula))
    zeroed = bool(re.search(r'(?:障碍|禁[止用点]|阻碍|三点|三处|其中).{0,80}(?:置|记|设|为)0', formula))
    # The global "f(0,0)=1" must not make "obstacles all set to 1" acceptable.
    wrong_boundary = bool(re.search(r'(?:障碍|禁点|三点).{0,30}(?:置|记为|设为)1', formula))
    wrong_origin = bool(re.search(r'(?:\(0,0\)|\[0\]\[0\])=(?!1(?:\D|$))\d+', formula))
    full_dp = bool(recurrence) and (origin or edge) and obstacles and zeroed and not (wrong_boundary or wrong_origin)
    issues = []
    if wrong_boundary:
        issues.append('障碍点必须置0，不能置1。')
    if wrong_origin:
        issues.append('起点到自身的路径数必须为1。')
    # Verify provided numerical rows, without executing submitted programs.
    grid = {}
    for x in range(9):
        for y in range(9):
            grid[x, y] = 0 if (x, y) in {(2, 2), (4, 5), (6, 3)} else 1 if (x, y) == (0, 0) else grid.get((x-1, y), 0)+grid.get((x, y-1), 0)
    for m in re.finditer(r'y=([0-8])[:：]([0-9]+(?:,[0-9]+){8})(?![\d,])', formula):
        if [int(n) for n in m[2].split(',')] != [grid[x, int(m[1])] for x in range(9)]:
            issues.append('提供的DP第'+m[1]+'行与题目递推不一致。')
    partial = bool(re.search(r'容斥|动态规划|dp\[|[fN]\(x-1,y\)', text))
    return full_ie or full_dp, partial, issues


def binary_fact_issues(text):
    """Only check explicit counts of the fixed, named endpoint subclasses."""
    issues = []
    for label, expected in ((r'相同', 19), (r'(?:同为|都为|都是)1', 13), (r'(?:同为|都为|都是)0', 6)):
        pattern = r'首末(?:位)?'+label+r'(?:的)?(?:共有|共|有|为)?([0-9]{1,3})(?:个|种)'
        for m in re.finditer(pattern, text):
            context = text[max(0, m.start()-16):m.start()]
            if re.search(r'假设|错误示例|反证', context):
                continue
            if int(m[1]) != expected:
                issues.append(m[0]+'有误，该类实际为'+str(expected)+'个。')
    return sorted(set(issues))


def binary_evidence(text, checks):
    formula = math_text(text)
    issues = binary_fact_issues(text)
    if re.search(r'按位(?:互补|取反)|0(?:和|与)1互换|交换0(?:和|与)1', text):
        issues.append('按位互补不保持无111约束，不能用作两类等数的映射。')
    values = {c['value'] for c in checks if c['correct']}
    symmetry = bool(re.search(r'反转|倒转|镜像|对称|两类|首末位反过来', text))
    constraints = bool(re.search(r'111|连续三|每.{0,8}[≤至多]|不超过2|≤2|0[~～、,]2', text))
    has_count = '13' in values and any(c['correct'] and c['value'] == '13' and
                                     re.search(r'[+*\-/]|C\(', c['expression'].split('=')[0]) for c in checks)

    # A fixed endpoint contributes one of the four ones; the six middle positions
    # contain three ones, with four internal triples and three additional cases.
    middle = ('6' in text and '3个1' in text and bool(re.search(r'中间|内部', text)))
    middle |= bool(re.search(r'6.{0,12}放3个1|6位.{0,15}3个1', text))
    exclusions = 'C(6,3)' in formula and '4' in text and '3' in text and '13' in text and bool(
        re.search(r'开头|结尾|前两|末尾两|第2.{0,5}3|第6.{0,5}7', text))
    explicit_middle = middle and constraints and (has_count or exclusions)

    # Gap counts must show the subcases, rather than assert "enumerate -> 13".
    gap = bool(re.search(r'空[位隙档]|间隙|间隔', text))
    split_gap = gap and 'C(5,2)' in formula and 'C(4,2)' in formula and all(n in text for n in ('7', '6', '13')) and constraints
    coefficient_gap = gap and '(1+x+x^2)^4' in formula and '(1+x+x^2)^3' in formula and all(n in text for n in ('19', '6', '13'))

    runs = bool(re.search(r'段', text)) and all(n in text for n in ('2', '3', '4')) and bool(
        re.search(r'C\(3,|C\(r,|C\(k,|长度为?\(?2[,、]1[,、]1', formula)) and (has_count or
        ('C(3,k-1)' in formula and '1*3=3' in formula and '3*3=9' in formula and 'k=4' in formula and '13' in formula))
    # Complement of the endpoint constraint, with a justified total of 45.
    total45 = ('45' in values and bool(re.search(r'C\(8,4\)|70', formula))) or '(1+x+x^2)^5' in formula
    # Both subcounts need an actual calculation or complete gap classification.
    same_zero = ('6' in values and ('C(6,4)' in formula or '15-12+3=6' in formula)) or (
        '(2,2,0)' in formula and '(2,1,1)' in formula and '3种' in text and '合计6' in text)
    same_one = 'C(6,2)-2' in formula and '13' in values and bool(re.search(r'前两|后两|位置2|位置6', text))
    complement_count = total45 and same_zero and same_one and '26' in values
    run_total = ('C(r,4-r)' in formula and 'C(5,r)' in formula and '2*C(3,r-1)' in formula
                 and '26' in values and bool(re.search(r'内部间隔≥1', text)))

    triples = re.findall(r'\{([1-6],[1-6],[1-6])\}', text)
    enumerated = False
    if triples:
        claimed = [tuple(map(int, v.split(','))) for v in triples]
        expected = {p for p in itertools.combinations(range(1, 7), 3)
                    if '111' not in '1'+''.join('1' if i in p else '0' for i in range(1, 7))+'0'}
        enumerated = len(claimed) == len(expected) and set(claimed) == expected
    full = complement_count or run_total or symmetry and (explicit_middle or split_gap or coefficient_gap or runs or enumerated)
    partial = bool(re.search(r'计数|枚举|对称|反转|倒转|分段|空位|间隙|方法|首末', text))
    if '互补对称' in text and not re.search(r'反转|倒转|镜像', text):
        full = False  # Ambiguous mapping; do not silently equate complement with reversal.
    return full, partial, issues
