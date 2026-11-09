"""评估：把新增两份 Qwen Flash 样本入库，能否提升 Qwen 的判定概率。

基线：85 份库（qwen3.8flash 2 份）
对照：85 + 新增两份（qwen3.8flash 4 份）
观测点：
  qwen-probe-02（Flash，当前被判 DeepSeek）—— 真正要看的目标
  qwen-probe-01（Max，当前判对）
  回归集里的非 Qwen 样本（防止误吸）
三种子，避免单次运气。
"""
import sys, shutil, re
from pathlib import Path
import numpy as np

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classifier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references
from neural_core import train_network, predict_scores

BASE = TUNING.parent / 'nano-now'
WORK = TUNING.parent / '_qwen_ingest_test'
SRC = Path(r'C:\Users\betraysilence\Documents\CRYSTALiA\.codex')
STAGE = Path(r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型')
NEW = [('qwen/新增 文字文件 (3).txt', 'qwen-probe-03.txt'), ('qwen/新增 文字文件 (4).txt', 'qwen-probe-04.txt')]

WATCH = {
    'qwen-probe-02 (Flash)': (STAGE / 'qwen-probe-02.txt', 'Qwen'),
    'qwen-probe-01 (Max)': (STAGE / 'qwen-probe-01.txt', 'Qwen'),
    'qwen-free-01': (STAGE / 'qwen-free-01.txt', 'Qwen'),
    'probe-01 (Haiku)': (STAGE / 'probe-01.txt', 'Claude'),
    'sonnet 5.5': (STAGE / 'codex-sonnet55-01.txt', 'Claude'),
    'deepseek 对照': (STAGE / 'ofm-deepseek41flash-01.txt', 'DeepSeek'),
    'zcode GLM': (STAGE / 'zcode-glm53flash-01.txt', 'GLM'),
}


def build(tag, keys):
    d = WORK / tag
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in BASE.glob('*.txt'):
        shutil.copy2(f, d / f.name)
    for i, k in enumerate(keys, start=3):
        shutil.copy2(STAGE / k, d / ('qwen3.8flash-%02d.txt' % i))
    return d


def run(tag, keys, seeds=(7, 42, 123)):
    d = build(tag, keys)
    refs, _ = load_references(d)
    out = {}
    for seed in seeds:
        net = train_network(refs, seed=seed, epochs=400)
        classes = net['meta']['classes']
        for name, (path, truth) in WATCH.items():
            p = predict_scores(read_answer(path), net)
            out.setdefault(name, []).append((classes[int(np.argmax(p))], float(p[classes.index(truth)]), truth))
    return out


def show(label, out):
    print('\n=== %s ===' % label)
    print('  %-24s %-24s %-10s' % ('样本', '判成', '真值家族分'))
    for name, vals in out.items():
        tops = [v[0] for v in vals]
        scores = [v[1] for v in vals]
        print('  %-24s %-24s %.3f' % (name, '/'.join(t[:8] for t in tops), float(np.mean(scores))))


def main():
    # 归档
    for rel, target in NEW:
        raw = (SRC / rel).read_text(encoding='utf-8')
        uniq = sorted({m.group(1) for m in re.finditer(r'\[([AB]\d{2})\]', raw)})
        (STAGE / target).write_bytes((SRC / rel).read_bytes())
        print('归档 %-20s <- %-24s 标签 %d 个%s' % (target, rel, len(uniq), '' if len(uniq) == 24 else ' ⚠'))
    show('A 基线 85 份', run('base', []))
    show('B 入库两份后（qwen3.8flash 4 份）', run('plus2', ['qwen-probe-03.txt', 'qwen-probe-04.txt']))


if __name__ == '__main__':
    main()
