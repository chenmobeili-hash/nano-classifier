"""实验：把「自信地判错」的样本入库，再测没入库的那几份。

入库挑选规则：错误家族的概率最高者优先（最严重的误判）。
留出测试：不参与训练的免费车道样本。
"""
import sys, shutil, json
from pathlib import Path

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classifier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references, identify, DEFAULT_REFERENCES
from neural_core import train_network

STAGE = Path(r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型')
ALL = {
    '01-ofm': ('ofm-glm53flash-01.txt', 'high'),
    '02-ofm': ('ofm-glm53flash-02.txt', 'low'),
    '03-codex': ('ofm-glm53flash-03.txt', 'low'),
    '04-codex': ('ofm-glm53flash-04.txt', 'low'),
    '05-codex': ('ofm-glm53flash-05.txt', 'low'),
    '06-codex': ('ofm-glm53flash-06.txt', 'low?'),
    '07-codex': ('ofm-glm53flash-07.txt', 'low?'),
    'cmdc-01': ('cmdc-glm53flash-01.txt', 'unknown'),
}
INGEST = ['cmdc-01', '01-ofm', '03-codex', '02-ofm']      # 自信地判错的四份
HELD = ['04-codex', '05-codex', '06-codex', '07-codex']   # 留出测试


def measure(refs, model):
    out = {}
    for key, (name, _) in ALL.items():
        fam = identify(read_answer(STAGE / name), refs, model=model)
        glm = next((r['score'] for r in fam['ranking'] if r['label'] == 'GLM'), 0.0)
        out[key] = (fam['candidate'], fam['ranking'][0]['score'], glm, fam['margin'])
    return out


def main():
    backup = TUNING.parent / 'backup-before-worstfit-ingest'
    if not backup.exists():
        shutil.copytree(TUNING.parent / 'nano-now', backup / 'nano-now')
        shutil.copy2(TUNING / 'neural-balanced.npz', backup)
        print('备份 ->', backup)

    base_refs, _ = load_references(DEFAULT_REFERENCES)
    base_model = train_network(base_refs, seed=42, epochs=400)
    before = measure(base_refs, base_model)

    # 入库：作为 GLM 家族下的独立型号 glm5.3flash-free
    ref_dir = TUNING.parent / 'nano-now'
    for i, key in enumerate(INGEST, 1):
        name = ALL[key][0]
        shutil.copy2(STAGE / name, ref_dir / ('GLM5.3FLASH-FREE-%02d.txt' % i))
    print('入库: %s' % ', '.join(ALL[k][0] for k in INGEST))

    new_refs, _ = load_references(DEFAULT_REFERENCES)
    new_model = train_network(new_refs, seed=42, epochs=400)
    after = measure(new_refs, new_model)

    print('\n参考库: %d -> %d 份' % (len(base_refs), len(new_refs)))
    print('\n%-10s %-6s %-28s %-28s' % ('样本', '分组', '入库前', '入库后'))
    for key in ALL:
        b = before[key]; a = after[key]
        f = lambda t: '%s %.2f (GLM %.2f)' % (str(t[0])[:8].ljust(8), t[1], t[2])
        print('%-10s %-6s %-28s %-28s' % (key, '入库' if key in INGEST else '留出', f(b), f(a)))

    hb = sum(1 for k in HELD if before[k][0] == 'GLM')
    ha = sum(1 for k in HELD if after[k][0] == 'GLM')
    print('\n留出 %d 份命中: 入库前 %d/%d -> 入库后 %d/%d' % (len(HELD), hb, len(HELD), ha, len(HELD)))
    ib = sum(1 for k in INGEST if before[k][0] == 'GLM')
    ia = sum(1 for k in INGEST if after[k][0] == 'GLM')
    print('入库 %d 份命中（训练集，仅作参考）: 入库前 %d/%d -> 入库后 %d/%d'
          % (len(INGEST), ib, len(INGEST), ia, len(INGEST)))


if __name__ == '__main__':
    main()
