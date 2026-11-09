"""同一批新样本上比较 135 维(D) 与 139 维(GLM 证据块) 两个模型。"""
import sys
from pathlib import Path

TUNING = Path(r'C:\Users\betraysilence\Music\nano-classifier\versions\20261019-glm-tuned\lite-questionnaire-model')
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references, DEFAULT_REFERENCES
import neural_core
from neural_core import load_network, predict_scores

STAGE = Path(r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型')
CASES = [
    ('ofm-glm53flash-01', STAGE / 'ofm-glm53flash-01.txt', 'GLM', 'high'),
    ('ofm-glm53flash-02', STAGE / 'ofm-glm53flash-02.txt', 'GLM', 'low'),
    ('ofm-glm53flash-03', STAGE / 'ofm-glm53flash-03.txt', 'GLM', 'low'),
    ('ofm-glm53flash-04', STAGE / 'ofm-glm53flash-04.txt', 'GLM', 'low'),
    ('ofm-glm53flash-05', STAGE / 'ofm-glm53flash-05.txt', 'GLM', 'low'),
    ('codex-sonnet55-01', STAGE / 'codex-sonnet55-01.txt', 'Claude', '?'),
    ('ofm-deepseek-01', STAGE / 'ofm-deepseek41flash-01.txt', 'DeepSeek', 'high'),
]
CONFIG_135 = {'style': .342, 'behavior': .207, 'sequence': .207, 'choices': .144, 'extra': .10}
CONFIG_139 = {'style': .31, 'behavior': .19, 'sequence': .19, 'choices': .13, 'extra': .10, 'glm': .08}


def run(model_file, group_weights, dims):
    neural_core.GROUP_WEIGHTS = group_weights
    neural_core.DIMENSIONS = dims
    model = load_network(TUNING / model_file)
    classes = model['meta']['classes']
    out = {}
    for name, path, truth, effort in CASES:
        scores = predict_scores(read_answer(path), model)
        idx = list(scores)
        top = classes[int(max(range(len(idx)), key=lambda i: idx[i]))]
        glm = float(scores[classes.index('GLM')])
        out[name] = (top, glm, float(max(idx) - sorted(idx)[-2]))
    return out


def main():
    refs, _ = load_references(DEFAULT_REFERENCES)
    a = run('neural-tuned-d-glm.npz', CONFIG_135, 135)
    b = run('neural-glm-evidence.npz', CONFIG_139, 139)
    print('参考库 %d 份 | 样本 %d 份\n' % (len(refs), len(CASES)))
    print('  %-20s %-8s %-22s %-22s' % ('样本', '真值', '135维 D', '139维 +GLM证据'))
    hit135 = hit139 = 0
    for name, path, truth, effort in CASES:
        t1, g1, m1 = a[name]
        t2, g2, m2 = b[name]
        hit135 += t1 == truth
        hit139 += t2 == truth
        f = lambda t, g, m: '%s %.2f(GLMc%.2f)' % (t[:8].ljust(8), m, g)
        print('  %-20s %-8s %-22s %-22s' % (name, truth, f(t1, g1, m1), f(t2, g2, m2)))
    print('\n  命中: 135维 %d/%d | 139维 %d/%d' % (hit135, len(CASES), hit139, len(CASES)))
    free_glm = [c for c in CASES if c[2] == 'GLM']
    print('  免费车道 GLM 只有: 135维 %d/%d | 139维 %d/%d'
          % (sum(1 for n, _, _, _ in free_glm if a[n][0] == 'GLM'),
             len(free_glm),
             sum(1 for n, _, _, _ in free_glm if b[n][0] == 'GLM'),
             len(free_glm)))


if __name__ == '__main__':
    main()
