"""入库两份候选（ofm-13 → GLM-free，qwen-free-02 → Qwen-free），三种子对照。

价值判据：当前判错 + 所在家族最弱 + 与现有参考不冗余。
测试集：06 / 07 / 08 / 09 / cmdc-01 / zcode-01 / qwen-free-01 + 两个对照（全部不入库）。
"""
import sys, shutil
from pathlib import Path
import numpy as np

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classfier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references
from neural_core import train_network, predict_scores

BASE = TUNING.parent / 'nano-now'
WORK = TUNING.parent / '_value_ingest'
STAGE = Path(r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型')
ADD = {
    'ofm-13': ('ofm-glm53flash-13.txt', 'GLM5.3FLASH-FREE-04.txt'),
    'qwen-free-02': ('qwen-free-02.txt', 'QWEN-FREE-01.txt'),
}
TEST = {
    '06': STAGE / 'ofm-glm53flash-06.txt',
    '07': STAGE / 'ofm-glm53flash-07.txt',
    '08': STAGE / 'ofm-glm53flash-08.txt',
    '09': STAGE / 'ofm-glm53flash-09.txt',
    'cmdc-01': STAGE / 'cmdc-glm53flash-01.txt',
    'zcode-01': STAGE / 'zcode-glm53flash-01.txt',
    'qwen-free-01': STAGE / 'qwen-free-01.txt',
    'deepseek': STAGE / 'ofm-deepseek41flash-01.txt',
    'claude': STAGE / 'codex-sonnet55-01.txt',
}
TRUTH = {'06': 'GLM', '07': 'GLM', '08': 'GLM', '09': 'GLM', 'cmdc-01': 'GLM',
         'zcode-01': 'GLM', 'qwen-free-01': 'Qwen', 'deepseek': 'DeepSeek', 'claude': 'Claude'}


def build(tag, keys):
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in BASE.glob('*.txt'):
        shutil.copy2(f, d / f.name)
    for k in keys:
        src, dst = ADD[k]
        shutil.copy2(STAGE / src, d / dst)
    return d


def run(tag, keys):
    d = build(tag, keys)
    refs, _ = load_references(d)
    held = {k: v for k, v in TEST.items() if k not in keys}
    rows = {k: [] for k in held}
    for seed in (7, 42, 123):
        net = train_network(refs, seed=seed, epochs=400)
        classes = net['meta']['classes']
        for name, path in held.items():
            p = predict_scores(read_answer(path), net)
            rows[name].append((classes[int(np.argmax(p))], float(p[classes.index(TRUTH[name])])))
    return rows


def show(label, rows):
    print('\n=== %s ===' % label)
    hit = 0
    for name, vals in rows.items():
        tops = [v[0] for v in vals]
        ok = sum(1 for t in tops if t == TRUTH[name])
        hit += ok
        print('  %-13s 真值 %-9s 判成 %-22s 命中 %d/3 | 真值家族分 %.3f'
              % (name, TRUTH[name], '/'.join(t[:8] for t in tops), ok, float(np.mean([v[1] for v in vals]))))
    print('  合计命中 %d/%d' % (hit, sum(len(v) for v in rows.values()) // 3))


show('A 当前 83 份', run('base', []))
show('B +ofm-13 +qwen-free-02', run('plus2', ['ofm-13', 'qwen-free-02']))
