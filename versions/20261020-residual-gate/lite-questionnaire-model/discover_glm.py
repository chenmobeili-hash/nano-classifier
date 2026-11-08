"""GLM 家族特征发现：在参考库上找 GLM 与其它家族的差异信号。

方法：
1. 直接文本统计（长度/标点/中英比例/问号/列表符等）+ 已命名的 22 维行为变量
2. 逐项 Cohen's d（GLM vs 非GLM），并用标签置换检验给出 p 值（n_GLM=7，必须看显著性）
3. B07–B12 的 2/3-gram log-odds，找 GLM 独有措辞
"""
import sys, random, statistics, math, re
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classifier import load_references, DEFAULT_REFERENCES
from feature_profiles import normalized_answers
from behavior_features import behavior_features
from behavior_features import *  # noqa: F401,F403
import behavior_features as bf

random.seed(42)


def text_stats(answers):
    out = {}
    for q in ('A01', 'A11', 'A12', 'B07', 'B08', 'B09', 'B10', 'B11', 'B12'):
        t = answers.get(q, '')
        out[q + '_len'] = len(t)
        out[q + '_cjk'] = len(re.findall(r'[\u4e00-\u9fff]', t))
        out[q + '_latin'] = len(re.findall(r'[A-Za-z]', t))
        out[q + '_punct'] = len(re.findall(r'[，。！？；：,.!?;:]', t))
        out[q + '_qmark'] = t.count('?') + t.count('？')
        out[q + '_colon'] = t.count('：') + t.count(':')
        out[q + '_bullet'] = len(re.findall(r'(?m)^\s*(?:[-*]|\d+[.、])', t))
        out[q + '_em'] = t.count('—') + t.count('----')
        out[q + '_paren'] = t.count('（') + t.count('(')
    t = ' '.join(answers.get(q, '') for q in ('B07', 'B08', 'B09', 'B10', 'B11', 'B12'))
    out['B67_12_len'] = len(t)
    out['B67_12_english_ratio'] = len(re.findall(r'[A-Za-z]', t)) / max(len(t), 1)
    out['B67_12_emoji'] = float(bool(re.search(r'[\U0001F300-\U0001FAFF\u2600-\u27BF]', t)))
    out['B67_12_quote'] = t.count('“') + t.count('"')
    return out


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    ma, mb = statistics.mean(a), statistics.mean(b)
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) / max(len(a) + len(b) - 2, 1))
    return (ma - mb) / pooled if pooled > 1e-12 else 0.0


def perm_p(values, is_glm, observed, trials=4000):
    n = sum(is_glm)
    hits = 0
    idx = list(range(len(values)))
    for _ in range(trials):
        random.shuffle(idx)
        a = [values[i] for i in idx[:n]]
        b = [values[i] for i in idx[n:]]
        if abs(cohens_d(a, b)) >= abs(observed) - 1e-12:
            hits += 1
    return hits / trials


def main():
    refs, _ = load_references(DEFAULT_REFERENCES)
    rows = []
    for r in refs:
        a = normalized_answers(r)
        d = text_stats(a)
        beh = behavior_features(a)
        names = getattr(bf, 'NAMES', None) or getattr(bf, 'BEHAVIOR_NAMES', None) or []
        for i, v in enumerate(beh):
            d['beh%02d' % i] = float(v)
        d['__family'] = r['family']
        d['__source'] = r['source']
        rows.append(d)
    print('references:', len(rows), '| GLM:', sum(1 for r in rows if r['__family'] == 'GLM'))
    print('behavior feature names available:', len(names), (names[:3] if names else '(none exported)'))

    keys = [k for k in rows[0] if not k.startswith('__')]
    is_glm = [r['__family'] == 'GLM' for r in rows]
    results = []
    for k in keys:
        vals = [float(r[k]) for r in rows]
        d = cohens_d([v for v, g in zip(vals, is_glm) if g], [v for v, g in zip(vals, is_glm) if not g])
        results.append((abs(d), d, k, vals))
    results.sort(reverse=True)

    print('\n=== GLM vs 其它家族：效应量前 18（|d| 大 = 差异大）===')
    print('  %-22s %6s %6s %7s %7s' % ('feature', 'd', 'p', 'GLM均值', '其它均值'))
    shown = 0
    for ad, d, k, vals in results:
        if ad < 0.9:
            break
        p = perm_p(vals, is_glm, d)
        g = [v for v, flag in zip(vals, is_glm) if flag]
        o = [v for v, flag in zip(vals, is_glm) if not flag]
        print('  %-22s %6.2f %6.3f %7.2f %7.2f' % (k, d, p, statistics.mean(g), statistics.mean(o)))
        shown += 1
        if shown >= 18:
            break

    print('\n=== 逐份核对：破折号（—— / —）计数 ===')
    for name in ('B07_em', 'B12_em', 'B08_cjk'):
        k = name
        g = [(r['__source'], float(r[k])) for r in rows if r['__family'] == 'GLM']
        o = [float(r[k]) for r in rows if r['__family'] != 'GLM']
        print('  %s | GLM: %s' % (k, ', '.join('%s=%.0f' % (s.split('.')[0], v) for s, v in g)))
        print('     其它家族 min/中位/max = %.0f / %.0f / %.0f' % (min(o), statistics.median(o), max(o)))

    # n-gram log-odds（GLM vs 其它）
    print('\n=== B07–B12 措辞：GLM 独有 2/3-gram（出现于 ≥3 份 GLM）===')
    grams = {}
    for r in refs:
        a = normalized_answers(r)
        text = ' '.join(a.get(q, '') for q in ('B07', 'B08', 'B09', 'B10', 'B11', 'B12'))
        found = set()
        for chunk in re.findall(r'[\u4e00-\u9fff]+', text):
            for n in (2, 3):
                for i in range(len(chunk) - n + 1):
                    found.add(chunk[i:i + n])
        grams.setdefault(('glm' if r['family'] == 'GLM' else 'other'), []).append(found)
    glm_sets, other_sets = grams.get('glm', []), grams.get('other', [])
    counter = Counter()
    for s in glm_sets:
        for g in s:
            counter[g] += 1
    scored = []
    for g, c in counter.items():
        if c < 3:
            continue
        oc = sum(1 for s in other_sets if g in s)
        logodds = math.log((c + 1) / (len(glm_sets) - c + 1)) - math.log((oc + 1) / (len(other_sets) - oc + 1))
        scored.append((logodds, g, c, oc))
    scored.sort(reverse=True)
    for lo, g, c, oc in scored[:20]:
        print('   %-8s  GLM %d/7   其它 %d/71    log-odds %+.2f' % (g, c, oc, lo))


if __name__ == '__main__':
    main()
