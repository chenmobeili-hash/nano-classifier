"""方案：从新的两份里二选一，加入库 + 把偏掉的 qwen-probe-02 一起入库。

  A: 入库 03 + probe-02  → 用 04 当留出测试（未入库的另一份 Flash）
  B: 入库 04 + probe-02  → 用 03 当留出测试
基线：85 份（未加任何新 Flash）。
观测：留出的另一半 Flash、cmdc-01、DeepSeek 对照、zcode GLM、Max、qwen-free、两个 Claude。
"""
import sys, shutil
from pathlib import Path
import numpy as np

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classifier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references
from neural_core import train_network, predict_scores

BASE = TUNING.parent / 'backup-before-qwenflash2' / 'nano-now'   # 85 份
WORK = TUNING.parent / '_choose_one'
STAGE = Path(r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型')
POOL = {
    '03': STAGE / 'qwen-probe-03.txt',
    '04': STAGE / 'qwen-probe-04.txt',
    'probe-02': STAGE / 'qwen-probe-02.txt',
}
WATCH = {
    'cmdc-01 (GLM边界)': (STAGE / 'cmdc-glm53flash-01.txt', 'GLM'),
    'deepseek 对照': (STAGE / 'ofm-deepseek41flash-01.txt', 'DeepSeek'),
    'zcode GLM': (STAGE / 'zcode-glm53flash-01.txt', 'GLM'),
    'qwen-probe-01 (Max)': (STAGE / 'qwen-probe-01.txt', 'Qwen'),
    'qwen-free-01': (STAGE / 'qwen-free-01.txt', 'Qwen'),
    'probe-01 (Haiku)': (STAGE / 'probe-01.txt', 'Claude'),
    'sonnet 5.5': (STAGE / 'codex-sonnet55-01.txt', 'Claude'),
}


def build(tag, keys):
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in BASE.glob('*.txt'):
        shutil.copy2(f, d / f.name)
    for i, k in enumerate(keys, start=3):
        shutil.copy2(POOL[k], d / ('qwen3.8flash-%02d.txt' % i))
    return d


def run(tag, keys, extra_watch):
    refs, _ = load_references(build(tag, keys))
    watch = dict(WATCH)
    watch.update(extra_watch)
    out = {}
    for seed in (7, 42, 123):
        net = train_network(refs, seed=seed, epochs=400)
        classes = net['meta']['classes']
        for name, (path, truth) in watch.items():
            p = predict_scores(read_answer(path), net)
            out.setdefault(name, []).append((classes[int(np.argmax(p))], float(p[classes.index(truth)])))
    return out


def show(label, out, truths):
    print('\n=== %s ===' % label)
    for name, vals in out.items():
        tops = [v[0] for v in vals]
        truth = truths[name]
        ok = sum(1 for t in tops if t == truth)
        print('  %-22s %-26s 真值分 %.3f  命中 %d/3'
              % (name, '/'.join(t[:8] for t in tops), float(np.mean([v[1] for v in vals])), ok))


extra_a = {'04 (留出 Flash)': (POOL['04'], 'Qwen')}
extra_b = {'03 (留出 Flash)': (POOL['03'], 'Qwen')}
show('A 入库 03 + probe-02（留出 04）', run('A', ['03', 'probe-02'], extra_a),
     {**{k: v[1] for k, v in WATCH.items()}, **{k: v[1] for k, v in extra_a.items()}})
show('B 入库 04 + probe-02（留出 03）', run('B', ['04', 'probe-02'], extra_b),
     {**{k: v[1] for k, v in WATCH.items()}, **{k: v[1] for k, v in extra_b.items()}})
