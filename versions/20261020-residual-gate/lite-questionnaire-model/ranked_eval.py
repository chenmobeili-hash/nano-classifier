"""留出的分级评估：不把"弃权"当成错或不算，而是看真值排在第几。

指标：
  top-1 准确率（强制给一个答案）
  top-k 覆盖率（真值落在前 k 名里的比例）与平均候选集大小
  弃权样本里真值的实际排名
  平均排名（越小越好，随机基线 = 3.0，5 分类）
"""
import sys
import numpy as np

TUNING = r'C:\Users\betraysilence\Music\nano-classifier\versions\20261019-glm-tuned\lite-questionnaire-model'
sys.path.insert(0, TUNING)
from classifier import load_references, DEFAULT_REFERENCES
from neural_core import train_network, predict_scores

refs, _ = load_references(DEFAULT_REFERENCES)
models = sorted({r['model'] for r in refs})
rows = []


def collect(train, test, tag):
    net = train_network(train, seed=42, epochs=400)
    classes = net['meta']['classes']
    for r in test:
        p = predict_scores(r, net)
        order = np.argsort(p)[::-1]
        truth_rank = list(order).index(classes.index(r['family'])) + 1
        rows.append({'tag': tag, 'source': r['source'], 'expected': r['family'],
                     'pred': classes[int(order[0])], 'rank': truth_rank,
                     'margin': float(p[order[0]] - p[order[1]])})


print('逐份留出...')
for i in range(len(refs)):
    collect(refs[:i] + refs[i + 1:], [refs[i]], 'sample')
print('整型号留出...')
for m in models:
    train = [r for r in refs if r['model'] != m]
    test = [r for r in refs if r['model'] == m]
    if train and test:
        collect(train, test, 'model')

for tag in ('sample', 'model'):
    sub = [r for r in rows if r['tag'] == tag]
    n = len(sub)
    print('\n=== %s（n=%d）===' % ('逐份留出' if tag == 'sample' else '整型号留出', n))
    for k in (1, 2, 3, 5):
        hit = sum(1 for r in sub if r['rank'] <= k)
        print('  top-%d 覆盖 %d/%d = %5.2f%%   （平均候选集大小 %.2f）'
              % (k, hit, n, 100 * hit / n, min(k, 5)))
    print('  平均真值排名 %.2f（随机基线 3.00）' % float(np.mean([r['rank'] for r in sub])))
    abstain = [r for r in sub if r['margin'] < 0.1]
    print('  弃权 %d 份，其中真值排名分布：%s'
          % (len(abstain), {k: sum(1 for r in abstain if r['rank'] == k) for k in (1, 2, 3, 4, 5)}))
    for r in abstain:
        print('     %-30s 真值 %-9s 排名 %d' % (r['source'], r['expected'], r['rank']))
