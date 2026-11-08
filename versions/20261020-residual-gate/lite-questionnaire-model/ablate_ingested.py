"""消融：4 份入库样本各自是否"不可或缺"。

对每一份入库样本：拿掉它 -> 重训 -> 看 4 份留出样本的命中变化。
若拿掉某份后留出命中下降，则该份对"识别同车道新样本"有实质贡献。
"""
import sys, shutil, itertools
from pathlib import Path

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classfier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references, identify
from neural_core import train_network

BASE = TUNING.parent / 'backup-before-worstfit-ingest' / 'nano-now'   # 78 份原始库
WORK = TUNING.parent / '_ablate'
STAGE = Path(r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型')
INGESTED = {
    'cmdc-01': STAGE / 'cmdc-glm53flash-01.txt',
    'ofm-01': STAGE / 'ofm-glm53flash-01.txt',
    'ofm-02': STAGE / 'ofm-glm53flash-02.txt',
    'ofm-03': STAGE / 'ofm-glm53flash-03.txt',
}
HELD = {
    '04': STAGE / 'ofm-glm53flash-04.txt',
    '05': STAGE / 'ofm-glm53flash-05.txt',
    '06': STAGE / 'ofm-glm53flash-06.txt',
    '07': STAGE / 'ofm-glm53flash-07.txt',
}


def build(keep):
    d = WORK / ('keep-' + ('+'.join(sorted(keep)) if keep else 'none'))
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    for f in BASE.glob('*.txt'):
        shutil.copy2(f, d / f.name)
    for i, key in enumerate(sorted(keep), 1):
        shutil.copy2(INGESTED[key], d / ('GLM5.3FLASH-FREE-%02d.txt' % i))
    return d


def evaluate(keep):
    refs, _ = load_references(build(keep))
    model = train_network(refs, seed=42, epochs=400)
    detail = {}
    for key, path in HELD.items():
        fam = identify(read_answer(path), refs, model=model)
        glm = next((r['score'] for r in fam['ranking'] if r['label'] == 'GLM'), 0.0)
        detail[key] = (fam['candidate'], glm)
    hits = sum(1 for v in detail.values() if v[0] == 'GLM')
    return hits, detail


def fmt(detail):
    return ' '.join('%s:%s(%.2f)' % (k, (v[0] or 'uncertain')[:8], v[1]) for k, v in detail.items())


def main():
    all_keys = sorted(INGESTED)
    print('留出样本: 04 / 05 / 06 / 07（都不入库）')
    print('%-28s %-8s %s' % ('入库组合', '命中', '明细(判成/GLM概率)'))

    hits, detail = evaluate([])
    print('%-28s %-8s %s' % ('(不入库) 78 份', '%d/4' % hits,
          fmt(detail)))

    rows = []
    for drop in all_keys:
        keep = [k for k in all_keys if k != drop]
        hits, detail = evaluate(keep)
        rows.append((drop, hits, detail))
        print('%-28s %-8s %s' % ('去掉 ' + drop, '%d/4' % hits,
              fmt(detail)))

    hits, detail = evaluate(all_keys)
    print('%-28s %-8s %s' % ('全部 4 份入库 (82 份)', '%d/4' % hits,
          fmt(detail)))

    print('\n=== 边际贡献（相对「全部入库」的命中差）===')
    full, _ = evaluate(all_keys)
    for drop, h, _ in rows:
        print('  去掉 %-10s 命中 %d/4  %+d' % (drop, h, h - full))


if __name__ == '__main__':
    main()
