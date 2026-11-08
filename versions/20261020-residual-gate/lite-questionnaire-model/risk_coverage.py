"""风险-覆盖率分析：现有几种"不确定性分数"是否真的把错误挑出来。

对每个型号做整型号留出（21 折），记录：
  1) softmax 最大概率
  2) softmax 首二差（当前 uncertain 门槛用的就是这个）
  3) 距离首二差（家族最近距离的差）
  4) 神经与距离是否一致
然后按不同阈值算：接受了多少、接受里的错误率。
不做任何拟合、不调参，纯描述。
"""
import sys, json
from pathlib import Path
import numpy as np

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classfier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import load_references, DEFAULT_REFERENCES
from neural_core import train_network, predict_scores
from feature_profiles import distances

refs, _ = load_references(DEFAULT_REFERENCES)
models = sorted({r['model'] for r in refs})
rows = []
print('整型号留出：%d 折' % len(models))
for m in models:
    train = [r for r in refs if r['model'] != m]
    test = [r for r in refs if r['model'] == m]
    if not train or not test:
        continue
    net = train_network(train, seed=42, epochs=400)
    classes = net['meta']['classes']
    for r in test:
        probs = predict_scores(r, net)
        order = np.argsort(probs)[::-1]
        d, _, _ = distances(r, train, 'balanced')
        fam_min = {}
        for i, ref in enumerate(train):
            fam_min[ref['family']] = min(fam_min.get(ref['family'], 1e9), float(d[i]))
        fd = sorted(fam_min.items(), key=lambda kv: kv[1])
        rows.append({
            'source': r['source'], 'family': r['family'],
            'pred': classes[int(order[0])],
            'top1': float(probs[order[0]]),
            'top2_gap': float(probs[order[0]] - probs[order[1]]),
            'dist_family': fd[0][0],
            'dist_gap': float(fd[1][1] - fd[0][1]),
        })
    print('  %-28s %d 份' % (m, len(test)))

correct = np.array([r['pred'] == r['family'] for r in rows])
agree = np.array([r['pred'] == r['dist_family'] for r in rows])
print('\n总样本 %d | 神经准确率 %.3f | 距离准确率 %.3f | 两者一致率 %.3f | 一致时准确率 %.3f | 不一致时准确率 %.3f'
      % (len(rows), correct.mean(), np.mean([r['dist_family'] == r['family'] for r in rows]),
         agree.mean(), correct[agree].mean() if agree.any() else float('nan'),
         correct[~agree].mean() if (~agree).any() else float('nan')))


def sweep(name, score, thresholds, keep_high=True):
    print('\n--- %s（%s 更大更可信）---' % (name, '' if keep_high else '注意'))
    print('   %-8s %-8s %-8s %-8s' % ('阈值', '接受数', '覆盖率', '接受里错误率'))
    for t in thresholds:
        keep = score >= t if keep_high else score <= t
        if keep.sum() == 0:
            continue
        err = 1 - correct[keep].mean()
        print('   %-8.3f %-8d %-8.3f %-8.3f' % (t, keep.sum(), keep.mean(), err))


sweep('softmax 最大概率', np.array([r['top1'] for r in rows]), [0, .3, .4, .5, .6, .7, .8, .9])
sweep('softmax 首二差（当前门槛 0.1）', np.array([r['top2_gap'] for r in rows]), [0, .05, .1, .2, .3, .5])
sweep('距离首二差', np.array([r['dist_gap'] for r in rows]), [0, .01, .02, .05, .1, .2])
print('\n--- 神经/距离一致 vs 不一致 ---')
for label, mask in (('一致', agree), ('不一致', ~agree)):
    if mask.any():
        print('   %-6s 份数 %2d  错误率 %.3f' % (label, mask.sum(), 1 - correct[mask].mean()))

Path(TUNING.parent / 'risk_coverage.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding='utf-8')
print('\n明细已存 risk_coverage.json')
