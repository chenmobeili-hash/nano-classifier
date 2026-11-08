"""块权重快速扫描：不训练，用距离引擎评估 LOO 家族准确率与 margin。

复刻 feature_profiles.distances() 的逐块标准化（std 下限 0.03、平方截断 9），
一次算出 4 个块的平方距离矩阵，再对不同权重组合做线性组合。
家族距离按 identify() 的口径：min over 型号的 该型号均值距离。
"""
import sys, itertools, statistics
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classifier import load_references, DEFAULT_REFERENCES, read_answer, read_answer as _ra
from feature_profiles import blocks

BLOCKS = ['style', 'behavior', 'sequence', 'choices']
CAP = 9.0
FLOOR = 0.03


def block_matrices(refs):
    rows = [blocks(r) for r in refs]
    mats = {}
    for name in BLOCKS:
        x = np.stack([row[name] for row in rows]).astype(float)
        scale = np.maximum(x.std(axis=0), FLOOR)
        z = x / scale
        d = np.minimum((z[:, None, :] - z[None, :, :]) ** 2, CAP).mean(axis=2)
        mats[name] = d
    return mats


def family_labels(refs):
    return [(r['family'], r['model']) for r in refs]


def classify(drow, family_model, exclude):
    """drow: 距离向量; 返回 (家族, 家族距离字典)"""
    per_model = {}
    for i, (fam, model) in enumerate(family_model):
        if i == exclude:
            continue
        per_model.setdefault((fam, model), []).append(drow[i])
    fam_dist = {}
    for (fam, model), vals in per_model.items():
        v = float(np.mean(vals))
        fam_dist[fam] = min(fam_dist.get(fam, float('inf')), v)
    best = min(fam_dist, key=fam_dist.get)
    return best, fam_dist


def evaluate(mats, family_model, weights):
    total = np.zeros_like(mats['style'])
    for name, w in weights.items():
        total += w * mats[name]
    d = np.sqrt(total)
    hits, margins, rows = 0, [], []
    for i, (fam, model) in enumerate(family_model):
        best, fam_dist = classify(d[i], family_model, exclude=i)
        ordered = sorted(fam_dist.values())
        margin = ordered[1] - ordered[0]
        margins.append(margin)
        ok = best == fam
        hits += ok
        rows.append((fam, best, ok, margin))
    return hits / len(family_model), statistics.median(margins), rows


def main():
    refs, skipped = load_references(DEFAULT_REFERENCES)
    fm = family_labels(refs)
    mats = block_matrices(refs)
    print('references:', len(refs), '| skipped:', len(skipped))

    grid = []
    for style in (.20, .25, .30, .35, .40, .45, .50):
        for behavior in (.15, .20, .25, .30, .35, .40):
            for sequence in (.05, .10, .15, .20, .25, .30, .35, .40):
                for choices in (.01, .02, .05, .10):
                    s = style + behavior + sequence + choices
                    grid.append({'style': style / s, 'behavior': behavior / s,
                                 'sequence': sequence / s, 'choices': choices / s})
    print('configs:', len(grid))

    results = []
    for w in grid:
        acc, med, rows = evaluate(mats, fm, w)
        results.append((acc, med, w, rows))
    results.sort(key=lambda x: (-x[0], -x[1]))

    baseline = {'style': .35, 'behavior': .25, 'sequence': .35, 'choices': .05}
    bacc, bmed, _ = evaluate(mats, fm, baseline)
    print('\n== 基线权重 %s ==' % {k: round(v, 3) for k, v in baseline.items()})
    print('   距离引擎 LOO 准确率 %.3f | margin 中位 %.3f' % (bacc, bmed))

    print('\n== 前 12 名 ==')
    for acc, med, w, _ in results[:12]:
        print('   acc %.3f | margin %.3f | style %.2f behavior %.2f sequence %.2f choices %.2f'
              % (acc, med, w['style'], w['behavior'], w['sequence'], w['choices']))

    # GLM 家族准确率单独看（前 40 名里选 GLM 最好的）
    def glm_acc(rows):
        g = [r for r in rows if r[0] == 'GLM']
        return sum(r[2] for r in g) / max(len(g), 1)
    top = [r for r in results[:200]]
    top.sort(key=lambda x: (-glm_acc(x[3]), -x[0], -x[1]))
    print('\n== GLM 召回优先（前 8）==')
    for acc, med, w, rows in top[:8]:
        print('   GLM %.3f | acc %.3f | margin %.3f | style %.2f behavior %.2f sequence %.2f choices %.2f'
              % (glm_acc(rows), acc, med, w['style'], w['behavior'], w['sequence'], w['choices']))

    print('\n基线 GLM 召回: %.3f' % glm_acc(evaluate(mats, fm, baseline)[2]))


if __name__ == '__main__':
    main()
