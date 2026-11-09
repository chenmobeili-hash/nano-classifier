"""五折实测：5 份免费车道 GLM 样本，每折 4 份入库（作为 glm5.3flash-free）、1 份留出。"""
import sys, shutil
from pathlib import Path

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classifier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import load_references, identify, read_answer, DEFAULT_REFERENCES
from neural_core import train_network

STAGE = Path(r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型')
SAMPLES = {('ofm-glm53flash-0%d' % i): STAGE / ('ofm-glm53flash-0%d.txt' % i) for i in (1, 2, 3, 4, 5)}
WORK = TUNING.parent / '_folds5'


def main():
    hits = 0
    print('每折 4 份入库 + 1 份留出，参考库 78 份 + 4 份')
    for held in SAMPLES:
        d = WORK / held
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        for f in Path(DEFAULT_REFERENCES).glob('*.txt'):
            shutil.copy2(f, d / f.name)
        n = 1
        for label in SAMPLES:
            if label == held:
                continue
            shutil.copy2(SAMPLES[label], d / ('GLM5.3FLASH-FREE-%02d.txt' % n))
            n += 1
        refs, _ = load_references(d)
        model = train_network(refs, seed=42, epochs=400)
        fam = identify(read_answer(SAMPLES[held]), refs, model=model)
        glm = next((r['score'] for r in fam['ranking'] if r['label'] == 'GLM'), 0.0)
        hit = fam['candidate'] == 'GLM'
        hits += hit
        print('  held %-18s 判成 %-9s %.3f | GLM %.3f | margin %.3f | %s'
              % (held, fam['candidate'], fam['ranking'][0]['score'], glm, fam['margin'], 'HIT' if hit else 'miss'))
    print('\n五折命中: %d/5' % hits)


if __name__ == '__main__':
    main()
