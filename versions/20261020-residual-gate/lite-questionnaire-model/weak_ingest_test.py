"""假设检验：加入弱样本(ofm-11) 是否能提升其它未入库样本的判定。

固定测试集：06 / 07 / 08 / 09 / cmdc-01（任何配置都不入库）
对比配置：
  A 当前 81 份
  B 81 + ofm-11
  C 81 + ofm-10 + ofm-11
  D 81 + ofm-06 + ofm-11   （把最弱的两份一起加）
每个配置跑 3 个种子，看稳定性。
"""
import sys, shutil
from pathlib import Path
import numpy as np

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classfier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references, DEFAULT_REFERENCES
from neural_core import train_network, predict_scores

BASE = TUNING.parent / 'nano-now'          # 当前 81 份
WORK = TUNING.parent / '_weaktest'
STAGE = Path(r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型')
ADD = {
    '10': STAGE / 'ofm-glm53flash-10.txt',
    '11': STAGE / 'ofm-glm53flash-11.txt',
    '06': STAGE / 'ofm-glm53flash-06.txt',
}
TEST = {
    '06': STAGE / 'ofm-glm53flash-06.txt',
    '07': STAGE / 'ofm-glm53flash-07.txt',
    '08': STAGE / 'ofm-glm53flash-08.txt',
    '09': STAGE / 'ofm-glm53flash-09.txt',
    'cmdc-01': STAGE / 'cmdc-glm53flash-01.txt',
}
CONFIGS = {
    'A 当前 81': [],
    'B +11': ['11'],
    'C +10+11': ['10', '11'],
    'D +06+11': ['06', '11'],
}


def build(tag, keys):
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in BASE.glob('*.txt'):
        shutil.copy2(f, d / f.name)
    for i, k in enumerate(keys, 1):
        shutil.copy2(ADD[k], d / ('GLM5.3FLASH-FREE-9%d.txt' % i))
    return d


def evaluate(tag, keys):
    d = build(tag, keys)
    refs, _ = load_references(d)
    held = {k: v for k, v in TEST.items() if k not in keys}
    out = {}
    for seed in (7, 42, 123):
        net = train_network(refs, seed=seed, epochs=400)
        classes = net['meta']['classes']
        for name, path in held.items():
            p = predict_scores(read_answer(path), net)
            top = classes[int(np.argmax(p))]
            glm = float(p[classes.index('GLM')])
            out.setdefault(name, []).append((top, glm))
    return out


def main():
    print('测试集（各配置均不入库的除外）: ' + ', '.join(TEST))
    for label, keys in CONFIGS.items():
        res = evaluate(label.replace(' ', '_').replace('+', '_'), keys)
        print('\n=== %s ===' % label)
        for name, vals in res.items():
            tops = [v[0] for v in vals]
            glms = [v[1] for v in vals]
            ok = sum(1 for t in tops if t == 'GLM')
            print('  %-8s 判成 %-22s GLM 均值 %.3f | 命中 %d/3'
                  % (name, '/'.join(t[:8] for t in tops), float(np.mean(glms)), ok))


if __name__ == '__main__':
    main()
