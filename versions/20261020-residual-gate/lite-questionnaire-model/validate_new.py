"""新样本验收：预注册的 4 条 GLM 假设 + 家族判定。

预注册（在 78 份参考库上确立，此处只看是否复现，不新增候选）：
  H1 B05 动物落在 {狐狸, 企鹅}
  H2 B04 骰子相邻重复比例 ≈ 0.15（其它家族 ≈ 0.056）
  H3 B01 选 7
  H4 B07 不反问用户（personal_followup = 0）
"""
import sys, re, statistics
from pathlib import Path

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classfier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references, identify, DEFAULT_REFERENCES
from feature_profiles import normalized_answers
from behavior_features import behavior_feature_map
from sequence_features import sequence_feature_map
from glm_evidence import glm_evidence
from neural_core import load_network

STAGE = Path(r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型')
CODEX = Path(r'C:\Users\betraysilence\Documents\CRYSTALiA\.codex')

NEW = [
    ('glm-5.3 flash free/新增 文字文件.txt', 'ofm-glm53flash-03.txt', 'GLM5.3-flash-free'),
    ('glm-5.3 flash free/新增 文字文件 (2).txt', 'ofm-glm53flash-04.txt', 'GLM5.3-flash-free'),
    ('glm-5.3 flash free/新增 文字文件 (3).txt', 'ofm-glm53flash-05.txt', 'GLM5.3-flash-free'),
    ('claude sonnet 5.5/新增 文字文件.txt', 'codex-sonnet55-01.txt', 'Claude Sonnet 5.5'),
]
GLM_KNOWN = [
    ('A ofm high (旧)', STAGE / 'ofm-glm53flash-01.txt'),
    ('B ofm low (旧)', STAGE / 'ofm-glm53flash-02.txt'),
]


def main():
    # 1) 归档到 staging（保留原文件不动）
    staged = []
    for rel, target, label in NEW:
        src = CODEX / rel
        dst = STAGE / target
        dst.write_bytes(src.read_bytes())
        staged.append((target, dst, label))
        print('归档: %-24s <- %s' % (target, rel))

    # 2) 预注册验证（3 份新 GLM + 2 份旧 GLM）
    print('\n=== 预注册 4 条假设验证 ===')
    print('  %-22s %-10s %-10s %-6s %-6s %-8s' % ('sample', 'B05', 'B04邻重', 'B01', 'B07反问', '判定'))
    rows = []
    for label, path in GLM_KNOWN:
        rows.append((label, path))
    for target, dst, label in staged:
        if label.startswith('GLM'):
            rows.append((target, dst))
    for label, path in rows:
        rec = read_answer(path)
        a = normalized_answers(rec)
        beh = behavior_feature_map(a)
        seq = sequence_feature_map(a)
        animal = a['B05'].strip()[:6]
        h1 = ('狐狸' in animal) or ('企鹅' in animal)
        h2 = seq.get('B04_adjacent_same_fraction', 0.0)
        h3 = a['B01'].strip()
        h4 = beh.get('B07_personal_followup_cue', 0.0)
        print('  %-22s %-10s %-10.3f %-6s %-6.0f %s%s%s%s'
              % (label, animal, h2, h3, h4,
                 'H1✓' if h1 else 'H1✗', 'H2✓' if h2 >= 0.10 else 'H2✗',
                 'H3✓' if h3 == '7' else 'H3✗', '✓' if h4 == 0 else '✗'))

    # 3) 家族判定（139 维默认权重）
    refs, _ = load_references(DEFAULT_REFERENCES)
    model = load_network()
    print('\n=== 家族判定（139 维 / GLM 证据块）===')
    for target, dst, label in staged:
        fam = identify(read_answer(dst), refs, model=model)
        glm = next((r['score'] for r in fam['ranking'] if r['label'] == 'GLM'), 0.0)
        ok = 'HIT' if fam['candidate'] == ('GLM' if label.startswith('GLM') else 'Claude') else 'miss'
        print('  %-24s 真值 %-18s 判成 %-9s %.3f | GLM %.3f | margin %.3f | %s'
              % (target, label, fam['candidate'], fam['ranking'][0]['score'], glm, fam['margin'], ok))


if __name__ == '__main__':
    main()
