"""GLM 语言特征候选挖掘（不预设家族，全部按同一规则提取）。

候选 24 项，覆盖标点习惯、人称、语气词、连词类别、列举方式、简繁混用、中英混写。
对每项做 GLM vs 其它家族的 Cohen's d + 4000 次标签置换 p 值。
多重检验提示：24 项，Bonferroni 阈值 0.05/24 ≈ 0.0021。
"""
import sys, re, random, math, statistics
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classifier import load_references, DEFAULT_REFERENCES
from feature_profiles import normalized_answers

random.seed(7)

BODY = tuple('B%02d' % i for i in range(7, 13))
TRAD = '個這們麼說對時會來為開關體當經還進點後來從'
CANDIDATES = {
    'em_dash': lambda t: t.count('—'),
    'paren_cn': lambda t: t.count('（') + t.count('）'),
    'colon': lambda t: t.count('：') + t.count(':'),
    'dunhao': lambda t: t.count('、'),
    'semicolon': lambda t: t.count('；') + t.count(';'),
    'quote_cn': lambda t: t.count('“') + t.count('”') + t.count('「'),
    'ellipsis': lambda t: t.count('……') + t.count('…'),
    'modal_end': lambda t: sum(t.count(w) for w in '吧呢啊嘛哦啦'),
    'second_person': lambda t: sum(t.count(w) for w in ('你', '您')),
    'first_person': lambda t: t.count('我'),
    'contrast': lambda t: sum(t.count(w) for w in ('不过', '但是', '然而', '虽然', '其实')),
    'causal': lambda t: sum(t.count(w) for w in ('因为', '所以', '因此', '由于')),
    'condition': lambda t: sum(t.count(w) for w in ('如果', '要是', '假如', '若')),
    'colloquial': lambda t: sum(t.count(w) for w in ('就行', '而已', '反正', '呗', '挺')),
    'example': lambda t: sum(t.count(w) for w in ('比如', '例如', '举例', '举个例子')),
    'bullets': lambda t: len(re.findall(r'(?m)^\s*(?:[-*]|\d+[.、])', t)),
    'latin_words': lambda t: len(re.findall(r'[A-Za-z]{2,}', t)),
    'traditional': lambda t: sum(t.count(ch) for ch in TRAD),
    'halfwidth_in_cjk': lambda t: len(re.findall(r'[\u4e00-\u9fff][,\.\?!]|[,\.\?!][\u4e00-\u9fff]', t)),
    'repeated_filler': lambda t: len(re.findall(r'(就是|然后|这个|那个)', t)),
    'digits': lambda t: len(re.findall(r'\d', t)),
    'emoji': lambda t: len(re.findall(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]', t)),
    'four_char_idiom': lambda t: len(re.findall(r'[\u4e00-\u9fff]{4}(?=[，。、；：])', t)),
    'anaphora_this': lambda t: sum(t.count(w) for w in ('这', '那')),
}


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / max(len(a)+len(b)-2, 1))
    return (statistics.mean(a) - statistics.mean(b)) / pooled if pooled > 1e-12 else 0.0


def perm_p(vals, mask, observed, trials=4000):
    n = sum(mask)
    idx = list(range(len(vals)))
    hits = 0
    for _ in range(trials):
        random.shuffle(idx)
        a = [vals[i] for i in idx[:n]]
        b = [vals[i] for i in idx[n:]]
        if abs(cohens_d(a, b)) >= abs(observed) - 1e-12:
            hits += 1
    return hits / trials


def main():
    refs, _ = load_references(DEFAULT_REFERENCES)
    rows = []
    for r in refs:
        answers = normalized_answers(r)
        text = '\n'.join(answers.get(q, '') for q in BODY)
        n = max(len(text), 1)
        vals = {}
        for name, fn in CANDIDATES.items():
            raw = fn(text)
            vals[name] = raw * 100.0 / n if name not in ('emoji', 'bullets') else float(raw)
        vals['__family'] = r['family']
        rows.append(vals)
    is_glm = [r['__family'] == 'GLM' for r in rows]
    print('参考 %d 份 | GLM %d 份 | 候选 %d 项 | Bonferroni 阈值 p<%.4f'
          % (len(rows), sum(is_glm), len(CANDIDATES), 0.05/len(CANDIDATES)))

    out = []
    for name in CANDIDATES:
        vals = [float(r[name]) for r in rows]
        g = [v for v, m in zip(vals, is_glm) if m]
        o = [v for v, m in zip(vals, is_glm) if not m]
        d = cohens_d(g, o)
        out.append((abs(d), d, name, perm_p(vals, is_glm, d), statistics.mean(g), statistics.mean(o)))
    out.sort(reverse=True)

    print('\n  %-18s %6s %8s %10s %10s' % ('候选', 'd', 'p', 'GLM/百字', '其它/百字'))
    for ad, d, name, p, mg, mo in out:
        flag = ' *' if p <= 0.05/len(CANDIDATES) else ('' if p > 0.05 else ' (未校正)')
        print('  %-18s %6.2f %8.4f %10.3f %10.3f%s' % (name, d, p, mg, mo, flag))

    print('\n=== 前 4 项的逐家族均值 ===')
    for ad, d, name, p, mg, mo in out[:4]:
        line = []
        for fam in ('Claude', 'DeepSeek', 'GLM', 'GPT', 'Qwen'):
            v = [r[name] for r in rows if r['__family'] == fam]
            line.append('%s %.2f' % (fam, statistics.mean(v)))
        print('  %-16s %s' % (name, ' | '.join(line)))


if __name__ == '__main__':
    main()
