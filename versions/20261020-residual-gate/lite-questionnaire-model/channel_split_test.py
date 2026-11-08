"""渠道归档方式对比：同一个 83 份库，两种型号划分。

A  现状：FREE-01..05 全部算作一个型号 glm5.3flash-free（3 份普通免费车道 + 2 份 cmdc 混在一起）
B  按渠道拆：FREE-01..03 为 glm5.3flash-free，另设 glm5.3-cmdc 装 2 份 cmdc

型号内均衡（家族内每型号等权）会让"样本数少的型号"每份权重更高，所以两种划分的实际影响不同。
测试集：06 / 07 / 08 / 09 / cmdc-01 / zcode-01（均不入库）。
"""
import sys, shutil
from pathlib import Path
import numpy as np

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classfier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references
from neural_core import train_network, predict_scores

SRC = TUNING.parent / 'nano-now'
WORK = TUNING.parent / '_channel_split'
STAGE = Path(r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型')
TEST = {
    '06': STAGE / 'ofm-glm53flash-06.txt',
    '07': STAGE / 'ofm-glm53flash-07.txt',
    '08': STAGE / 'ofm-glm53flash-08.txt',
    '09': STAGE / 'ofm-glm53flash-09.txt',
    'cmdc-01': STAGE / 'cmdc-glm53flash-01.txt',
    'zcode-01': STAGE / 'zcode-glm53flash-01.txt',
}


def build(tag, split):
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in SRC.glob('*.txt'):
        name = f.name
        if split and name in ('GLM5.3FLASH-FREE-04.txt', 'GLM5.3FLASH-FREE-05.txt'):
            new = 'GLM5.3-CMDC-%02d.txt' % (1 if name.endswith('04.txt') else 2)
            shutil.copy2(f, d / new)
        else:
            shutil.copy2(f, d / name)
    return d


def run(split, seeds=(7, 42, 123)):
    d = build('split' if split else 'single', split)
    refs, _ = load_references(d)
    models = sorted({r['model'] for r in refs if r['family'] == 'GLM'})
    print('\n=== %s ===' % ('B 按渠道拆' if split else 'A 现状（混在一起）'))
    print('  GLM 型号: %s' % ', '.join('%s(%d)' % (m, sum(1 for r in refs if r['model'] == m)) for m in models))
    rows = {k: [] for k in TEST}
    for seed in seeds:
        net = train_network(refs, seed=seed, epochs=400)
        classes = net['meta']['classes']
        for name, path in TEST.items():
            p = predict_scores(read_answer(path), net)
            rows[name].append((classes[int(np.argmax(p))], float(p[classes.index('GLM')])))
    for name, vals in rows.items():
        tops = [v[0] for v in vals]
        glms = [v[1] for v in vals]
        print('  %-8s %-24s GLM均值 %.3f  命中 %d/%d'
              % (name, '/'.join(t[:8] for t in tops), float(np.mean(glms)),
                 sum(1 for t in tops if t == 'GLM'), len(tops)))
    return rows


run(False)
run(True)
