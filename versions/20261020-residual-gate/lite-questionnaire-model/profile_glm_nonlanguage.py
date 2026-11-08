"""GLM 的非语言特征画像：行为线索、序列习惯、单次选择。

输出三类：
  A. 24 个命名行为变量的 GLM vs 其它（效应量 + 置换 p）
  B. 33 个命名序列变量的同上
  C. 单次选择（B01/B05/B06）的逐份取值与家族分布
"""
import sys, re, random, math, statistics
from pathlib import Path
from collections import Counter

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classifier import load_references, DEFAULT_REFERENCES
from feature_profiles import normalized_answers
from behavior_features import behavior_features, BEHAVIOR_FEATURE_NAMES
from sequence_features import sequence_feature_map

random.seed(11)


def cohens_d(a, b):
    if len(a) < 2 or len(b) < 2:
        return 0.0
    va, vb = statistics.variance(a), statistics.variance(b)
    pooled = math.sqrt(((len(a)-1)*va + (len(b)-1)*vb) / max(len(a)+len(b)-2, 1))
    return (statistics.mean(a) - statistics.mean(b)) / pooled if pooled > 1e-12 else 0.0


def perm_p(vals, mask, observed, trials=3000):
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


def rank_features(rows, names, getter, is_glm, label, top=10):
    out = []
    for name in names:
        vals = [float(getter(r, name)) for r in rows]
        d = cohens_d([v for v, m in zip(vals, is_glm) if m], [v for v, m in zip(vals, is_glm) if not m])
        out.append((abs(d), d, name, perm_p(vals, is_glm, d),
                    statistics.mean([v for v, m in zip(vals, is_glm) if m]),
                    statistics.mean([v for v, m in zip(vals, is_glm) if not m])))
    out.sort(reverse=True)
    print('\n=== %s（按 |d| 排，前 %d）===' % (label, top))
    print('  %-34s %6s %8s %8s %8s' % ('feature', 'd', 'p', 'GLM', '其它'))
    for ad, d, name, p, mg, mo in out[:top]:
        mark = ' *' if p < 0.05 else ''
        print('  %-34s %6.2f %8.4f %8.3f %8.3f%s' % (name, d, p, mg, mo, mark))
    return out


def main():
    refs, _ = load_references(DEFAULT_REFERENCES)
    rows, fams = [], []
    for r in refs:
        a = normalized_answers(r)
        rows.append((a, behavior_features(a), sequence_feature_map(a)))
        fams.append(r['family'])
    is_glm = [f == 'GLM' for f in fams]
    print('参考 %d 份 | GLM %d 份' % (len(rows), sum(is_glm)))

    rank_features(rows, BEHAVIOR_FEATURE_NAMES, lambda r, n: r[1][BEHAVIOR_FEATURE_NAMES.index(n)],
                  is_glm, 'A. 行为变量（24）', top=10)

    seq_names = list(rows[0][2].keys())
    rank_features(rows, seq_names, lambda r, n: r[2][n], is_glm, 'B. 序列变量（%d）' % len(seq_names), top=10)

    print('\n=== C. 单次选择 ===')
    print('  %-10s %-6s %-6s %-6s' % ('family', 'B01', 'B06', 'B05'))
    for fam in ('Claude', 'DeepSeek', 'GLM', 'GPT', 'Qwen'):
        picks = [(r[0]['B01'].strip()[:4], r[0]['B06'].strip()[:4], r[0]['B05'].strip()[:6])
                 for r, f in zip(rows, fams) if f == fam]
        print('  %-10s %s' % (fam, '  '.join('%s/%s/%s' % p for p in picks)))

    print('\n=== GLM 逐份明细（B05 动物 / B01 / B06 / B03 是否合规 / B02 是否合规）===')
    for r, f in zip(rows, fams):
        if f != 'GLM':
            continue
        a = r[0]
        b02_ok = bool(re.fullmatch(r'[0-9]{120}', a['B02'].strip()))
        b03_ok = bool(re.fullmatch(r'[HT]{80}', a['B03'].strip().upper()))
        b04_ok = bool(re.fullmatch(r'[1-6](?:\s*,\s*[1-6]){23}', a['B04'].strip()))
        print('  B05=%-8s B01=%-4s B06=%-4s | B02合规=%s B03合规=%s B04合规=%s'
              % (a['B05'].strip()[:8], a['B01'].strip()[:4], a['B06'].strip()[:4], b02_ok, b03_ok, b04_ok))


if __name__ == '__main__':
    main()
