"""逐份留出：单种子 vs 多种子集成。

集成方法：同一折用多个随机种子各训一次，把 softmax 概率取平均再判定。
这是纯方差缩减，不需要在测试数据上选任何参数（种子不是超参搜索，全部计入平均）。
"""
import sys
from pathlib import Path
import numpy as np

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classifier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import load_references, DEFAULT_REFERENCES
from neural_core import train_network, predict_scores

SEEDS = (7, 42, 123)


def main():
    refs, _ = load_references(DEFAULT_REFERENCES)
    classes = None
    single_hit = 0
    ens_hit = 0
    single_margin = []
    ens_margin = []
    for i in range(len(refs)):
        train = refs[:i] + refs[i + 1:]
        probs = []
        for seed in SEEDS:
            net = train_network(train, seed=seed, epochs=400)
            classes = net['meta']['classes']
            probs.append(predict_scores(refs[i], net))
        p = np.stack(probs)
        truth = classes.index(refs[i]['family'])
        # 单种子（取 seed=42，即发布配置）
        s = p[1]
        order = np.argsort(s)[::-1]
        single_hit += int(order[0] == truth)
        single_margin.append(float(s[order[0]] - s[order[1]]))
        # 集成
        m = p.mean(axis=0)
        order = np.argsort(m)[::-1]
        ens_hit += int(order[0] == truth)
        ens_margin.append(float(m[order[0]] - m[order[1]]))
    n = len(refs)
    print('逐份留出（%d 份）' % n)
    print('  单种子 seed=42 : %d/%d = %.2f%%' % (single_hit, n, 100*single_hit/n))
    print('  三种子集成     : %d/%d = %.2f%%' % (ens_hit, n, 100*ens_hit/n))
    sm = np.array(single_margin); em = np.array(ens_margin)
    print('\n首二差（margin）对比：')
    print('  单种子  中位 %.3f | 低于 0.1 的份数 %d' % (np.median(sm), int((sm < .1).sum())))
    print('  集成    中位 %.3f | 低于 0.1 的份数 %d' % (np.median(em), int((em < .1).sum())))


if __name__ == '__main__':
    main()
