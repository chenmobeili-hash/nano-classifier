"""对比：旧空间(131维/基线权重) vs 新空间(135维/D权重) 下，
4 份免费车道 GLM 样本与参考库的距离结构是否改善。

旧空间数值来自上一轮在 20261017 主版本目录的实测，此处作为常量对照。
"""
import sys, itertools
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from classifier import load_references, DEFAULT_REFERENCES, read_answer
from feature_profiles import distances

SAMPLES = [
    ('A ofm high', r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型\ofm-glm53flash-01.txt'),
    ('B ofm low ', r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型\ofm-glm53flash-02.txt'),
    ('C codex(2)', r'C:\Users\betraysilence\Documents\CRYSTALiA\.codex\新增 文字文件 (2).txt'),
    ('D codex-tol', r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型\codex-tol-02.txt'),
]

# 旧 131 维基线空间实测值：(最近GLM, 最近非GLM)
OLD_GAP = {'A ofm high': (1.352, 1.294), 'B ofm low ': (1.435, 1.315),
           'C codex(2)': (1.294, 1.350), 'D codex-tol': (1.337, 1.255)}
# 旧空间的样本两两距离
OLD_PAIR = {('A ofm high', 'B ofm low '): 1.655, ('A ofm high', 'C codex(2)'): 1.481,
            ('A ofm high', 'D codex-tol'): 1.426, ('B ofm low ', 'C codex(2)'): 1.620,
            ('B ofm low ', 'D codex-tol'): 1.568, ('C codex(2)', 'D codex-tol'): 1.559}


def gaps(d, refs):
    nearest = {}
    for i, r in enumerate(refs):
        f = r['family']
        nearest[f] = min(nearest.get(f, 1e9), float(d[i]))
    return nearest['GLM'], min(v for f, v in nearest.items() if f != 'GLM')


def main():
    refs, skipped = load_references(DEFAULT_REFERENCES)
    weights = __import__('feature_profiles').distances.__defaults__
    print('references:', len(refs), '| skipped:', len(skipped))

    print('\n=== 每份样本：到 GLM / 到其它家族 的最近距离 ===')
    print('  %-11s %-20s %-20s %s' % ('sample', '旧 131维', '新 135维', '间隔(GLm-其它) 旧→新'))
    recs = []
    for label, path in SAMPLES:
        rec = read_answer(path)
        recs.append((label, rec))
        d, _, info = distances(rec, refs, 'balanced')
        g, o = gaps(d, refs)
        bg, bo = OLD_GAP[label]
        print('  %-11s %-20s %-20s %+.3f → %+.3f'
              % (label, '%.3f / %.3f' % (bg, bo), '%.3f / %.3f' % (g, o), bo - bg, o - g))
    print('  (间隔 = 最近非GLM − 最近GLM，越大越像 GLM)')

    print('\n=== 4 份样本互相距离 ===')
    print('  %-24s %-10s %-10s %s' % ('pair', '旧', '新', '变化'))
    new_pairs = []
    for (l1, r1), (l2, r2) in itertools.combinations(recs, 2):
        dn, _, _ = distances(r1, [r2], 'balanced')
        old = OLD_PAIR.get((l1, l2)) or OLD_PAIR.get((l2, l1))
        new_pairs.append(dn[0])
        print('  %-24s %-10.3f %-10.3f %+.3f' % (l1 + ' × ' + l2, old, dn[0], dn[0] - old))
    print('  新空间中位 %.3f' % np.median(new_pairs))

    print('\n=== 参照：参考库内 GLM 7 份彼此的距离 ===')
    glm_refs = [(r['source'], r) for r in refs if r['family'] == 'GLM']
    vals = []
    for (s1, r1), (s2, r2) in itertools.combinations(glm_refs, 2):
        d, _, _ = distances(r1, [r2], 'balanced')
        vals.append(d[0])
    print('  中位 %.3f  范围 %.3f-%.3f  (n=%d 对)' % (np.median(vals), min(vals), max(vals), len(vals)))

    print('\n=== 参照：每份样本最近 3 个参考样本（新空间）===')
    for label, rec in recs:
        d, _, _ = distances(rec, refs, 'balanced')
        order = np.argsort(d)[:3]
        print('  %-11s %s' % (label, ', '.join('%s[%s] %.3f' % (refs[i]['source'], refs[i]['family'], d[i]) for i in order)))


if __name__ == '__main__':
    main()
