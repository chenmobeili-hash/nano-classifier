"""四折实测：4 份免费车道 GLM 样本，每折 3 份入库（作为 glm5.3flash-free）、1 份留出。

每折都在独立参考目录里重新加载、重新训练，留出的那一份从不参与训练。
"""
import sys, shutil, json
from pathlib import Path

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classifier\versions\20261019-glm-tuning\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import load_references, identify, read_answer, DEFAULT_REFERENCES
from neural_core import train_network

FAMILY_OF = {'GLM': 'GLM'}
SAMPLES = {
    'A ofm-high': r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型\ofm-glm53flash-01.txt',
    'B ofm-low': r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型\ofm-glm53flash-02.txt',
    'C codex2': r'C:\Users\betraysilence\Documents\CRYSTALiA\.codex\新增 文字文件 (2).txt',
    'D codex-tol': r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型\codex-tol-02.txt',
}
WORK = TUNING.parent / '_folds'


def prepare(fold_name, included):
    d = WORK / fold_name
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in Path(DEFAULT_REFERENCES).glob('*.txt'):
        shutil.copy2(f, d / f.name)
    for i, label in enumerate(included, 1):
        shutil.copy2(SAMPLES[label], d / ('GLM5.3FLASH-FREE-%02d.txt' % i))
    return d


def main():
    print('基准参考库:', DEFAULT_REFERENCES, '| 折目录:', WORK)
    rows = []
    for held in SAMPLES:
        included = [k for k in SAMPLES if k != held]
        d = prepare('hold-' + held.split()[0], included)
        refs, skipped = load_references(d)
        model = train_network(refs, seed=42, epochs=400)
        query = read_answer(SAMPLES[held])
        fam = identify(query, refs, model=model)
        glm = next((r['score'] for r in fam['ranking'] if r['label'] == 'GLM'), 0.0)
        top = fam['candidate']
        hit = top == 'GLM'
        rows.append((held, len(refs), top, fam['ranking'][0]['score'], glm, fam['margin'], fam['status'], hit))
        print('  held %-11s | 训练 %2d 份 | 判成 %-9s %.3f | GLM %.3f | margin %.3f | %s | %s'
              % (held, len(refs), top, fam['ranking'][0]['score'], glm, fam['margin'], fam['status'], 'HIT' if hit else 'miss'))
    print('\n四折命中: %d/4' % sum(1 for r in rows if r[7]))
    print('GLM 分数均值 %.3f | 最高 %.3f | 最低 %.3f'
          % (sum(r[4] for r in rows) / 4, max(r[4] for r in rows), min(r[4] for r in rows)))


if __name__ == '__main__':
    main()
