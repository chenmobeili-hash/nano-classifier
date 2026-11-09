"""网络侧调参：隐藏单元数 × GLM 家族权重。

指标：
  LOO-sample  逐份留出准确率（78 折，每折重训）
  LOO-model   整型号留出准确率（每折重训，排除同型号全部样本）
  外部实测    4 份免费车道 GLM 样本的命中数（不入库，完全无泄漏）
  GLM 分      外部样本的 GLM 概率均值
"""
import sys, itertools, statistics
from pathlib import Path
import numpy as np

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classifier\versions\20261019-glm-tuning\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import load_references, read_answer, DEFAULT_REFERENCES
from neural_core import train_network, predict_scores

SAMPLES = [
    ('A', r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型\ofm-glm53flash-01.txt'),
    ('B', r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型\ofm-glm53flash-02.txt'),
    ('C', r'C:\Users\betraysilence\Documents\CRYSTALiA\.codex\新增 文字文件 (2).txt'),
    ('D', r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型\codex-tol-02.txt'),
]

CONFIGS = [
    ('hidden=8  (基线)', 8, None),
    ('hidden=16', 16, None),
    ('hidden=24', 24, None),
    ('hidden=8  GLM×1.5', 8, {'GLM': 1.5}),
    ('hidden=16 GLM×1.5', 16, {'GLM': 1.5}),
    ('hidden=16 GLM×2.0', 16, {'GLM': 2.0}),
]


def loo_sample(refs, hidden, boost):
    hits = 0
    for i in range(len(refs)):
        train = refs[:i] + refs[i + 1:]
        model = train_network(train, seed=42, epochs=400, hidden=hidden, family_boost=boost)
        classes = model['meta']['classes']
        scores = predict_scores(refs[i], model)
        hits += classes[int(np.argmax(scores))] == refs[i]['family']
    return hits / len(refs)


def loo_model(refs, hidden, boost):
    models = sorted({r['model'] for r in refs})
    hits = total = 0
    for m in models:
        train = [r for r in refs if r['model'] != m]
        test = [r for r in refs if r['model'] == m]
        if not train or not test:
            continue
        model = train_network(train, seed=42, epochs=400, hidden=hidden, family_boost=boost)
        classes = model['meta']['classes']
        for r in test:
            scores = predict_scores(r, model)
            hits += classes[int(np.argmax(scores))] == r['family']
            total += 1
    return hits / max(total, 1)


def external(refs, hidden, boost):
    model = train_network(refs, seed=42, epochs=400, hidden=hidden, family_boost=boost)
    classes = model['meta']['classes']
    glm_index = classes.index('GLM')
    hits, glm_scores, detail = 0, [], []
    for label, path in SAMPLES:
        scores = predict_scores(read_answer(path), model)
        top = classes[int(np.argmax(scores))]
        hits += top == 'GLM'
        glm_scores.append(float(scores[glm_index]))
        detail.append('%s:%s(%.2f)' % (label, top, scores[glm_index]))
    return hits, statistics.mean(glm_scores), ' '.join(detail)


def main():
    refs, _ = load_references(DEFAULT_REFERENCES)
    print('库 %d 份 | 外部样本 %d 份' % (len(refs), len(SAMPLES)))
    print('\n%-20s %-10s %-10s %-8s %-8s %s' % ('config', 'LOO-sample', 'LOO-model', '外部命中', 'GLM均值', '外部明细'))
    for name, hidden, boost in CONFIGS:
        s_acc = loo_sample(refs, hidden, boost)
        m_acc = loo_model(refs, hidden, boost)
        hits, glm_mean, detail = external(refs, hidden, boost)
        print('%-20s %-10.4f %-10.4f %-8s %-8.3f %s' % (name, s_acc, m_acc, '%d/4' % hits, glm_mean, detail))


if __name__ == '__main__':
    main()
