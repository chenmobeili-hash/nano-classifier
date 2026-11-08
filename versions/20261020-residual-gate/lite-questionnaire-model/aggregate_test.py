"""查询侧聚合实验。

对 4 份免费车道 GLM 样本的所有子集：
  A. 单份判定（基线，已知）
  B. 特征平均后判定（centroid）
  C. 逐份判定后多数票（voting）
参考库固定为 78 份基础库，不含任何免费车道样本，因此不存在泄漏。
"""
import sys, itertools
from pathlib import Path
import numpy as np

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classfier\versions\20261019-glm-tuning\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import load_references, read_answer, DEFAULT_REFERENCES
from feature_profiles import blocks
from neural_core import load_network, vector as nn_vector, transform, forward, GROUP_WEIGHTS

SAMPLES = {
    'A': r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型\ofm-glm53flash-01.txt',
    'B': r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型\ofm-glm53flash-02.txt',
    'C': r'C:\Users\betraysilence\Documents\CRYSTALiA\.codex\新增 文字文件 (2).txt',
    'D': r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型\codex-tol-02.txt',
}
WEIGHTS = {'style': .342, 'behavior': .207, 'sequence': .207, 'choices': .144, 'extra': .10}


def block_of(rec):
    return blocks(rec)


def ref_matrix(refs):
    rows = [block_of(r) for r in refs]
    mats = {}
    for name in WEIGHTS:
        x = np.stack([row[name] for row in rows]).astype(float)
        scale = np.maximum(x.std(axis=0), .03)
        mats[name] = (x / scale, scale)
    return mats


def centroid_blocks(recs):
    rows = [block_of(r) for r in recs]
    return {name: np.mean([row[name] for row in rows], axis=0) for name in WEIGHTS}


def distance_to(refmats, qblocks, refs):
    total = np.zeros(len(refs))
    for name, w in WEIGHTS.items():
        xz, scale = refmats[name]
        q = qblocks[name] / scale
        delta = np.minimum(((xz - q) ** 2).mean(axis=1), 9.0)
        total += w * delta
    return np.sqrt(total)


def family_view(d, refs):
    nearest = {}
    for i, r in enumerate(refs):
        f = r['family']
        nearest[f] = min(nearest.get(f, 1e9), float(d[i]))
    order = sorted(nearest.items(), key=lambda kv: kv[1])
    return order[0][0], dict(order)


def neural_view(x, model):
    scores = forward(transform(x[None, :], model), model)[1][0]
    classes = model['meta']['classes']
    d = {c: float(s) for c, s in zip(classes, scores)}
    order = sorted(d.items(), key=lambda kv: -kv[1])
    return order[0][0], order[0][1] - order[1][1], d


def main():
    refs, skipped = load_references(DEFAULT_REFERENCES)
    refmats = ref_matrix(refs)
    model = load_network(TUNING / 'neural-tuned-d-glm.npz')
    recs = {k: read_answer(p) for k, p in SAMPLES.items()}
    print('库:', len(refs), '份 | 权重:', WEIGHTS)

    print('\n=== A. 单份（基线）===')
    singles = {}
    for k, rec in recs.items():
        x = nn_vector(rec)
        top, marg, dist = neural_view(x, model)
        singles[k] = (top, dist['GLM'], marg)
        print('  %s | 神经 %-9s (GLM %.3f, margin %.3f)' % (k, top, dist['GLM'], marg))

    print('\n=== B. 特征平均后判定 ===')
    for size in (2, 3, 4):
        print('  -- %d 份聚合 --' % size)
        hits = 0
        for combo in itertools.combinations(SAMPLES, size):
            qb = centroid_blocks([recs[k] for k in combo])
            d = distance_to(refmats, qb, refs)
            fam, famd = family_view(d, refs)
            x = np.concatenate([qb[name] for name in GROUP_WEIGHTS])
            top, marg, dist = neural_view(x, model)
            hit = top == 'GLM'
            hits += hit
            print('     %-10s | 神经 %-9s (GLM %.3f, margin %.3f) | 距离最近 %-9s (GLM %.3f/其它 %.3f) %s'
                  % ('+'.join(combo), top, dist['GLM'], marg, fam, famd.get('GLM'),
                     min(v for f, v in famd.items() if f != 'GLM'), 'HIT' if hit else 'miss'))
        print('     小计: %d/%d' % (hits, len(list(itertools.combinations(SAMPLES, size)))))

    print('\n=== C. 逐份判定 + 多数票（同一子集）===')
    for size in (2, 3, 4):
        hits = 0
        for combo in itertools.combinations(SAMPLES, size):
            votes = [singles[k][0] for k in combo]
            glm_votes = votes.count('GLM')
            winner = max(set(votes), key=votes.count)
            hit = winner == 'GLM'
            hits += hit
            print('     %-10s | 票: %-32s -> %-9s %s' % ('+'.join(combo), ','.join(votes), winner, 'HIT' if hit else 'miss'))
        print('     小计: %d/%d' % (hits, len(list(itertools.combinations(SAMPLES, size)))))


if __name__ == '__main__':
    main()
