"""入库后回归套件：固定一组"永不入库"的外部样本，每次改动后跑一遍。

每个样本记录：判定家族、模型分、真值家族命中、以及 GLM 分数（监测误吸）。
输出 PASS / FAIL 汇总，便于在每次入库后快速自查有没有把别的家族带歪。
"""
import sys, json
from pathlib import Path
import numpy as np

# NLP 版：默认本目录，可用参数切换引擎与模型
TUNING = Path(__file__).resolve().parent
sys.path.insert(0, str(TUNING))
from classifier import read_answer, load_references, identify, DEFAULT_REFERENCES
from neural_core import load_network
from nlp_classifier import identify_nlp
from fusion import identify_fused
import nlp_core

ENGINE = sys.argv[1] if len(sys.argv) > 1 else 'nlp'

STAGE = Path(__file__).resolve().parents[3] / 'external-holdout'
# (文件, 真值家族, 备注)  —— 这些文件永不入库
CASES = [
    ('ofm-glm53flash-06.txt', 'GLM', '免费车道'),
    ('ofm-glm53flash-07.txt', 'GLM', '免费车道'),
    ('ofm-glm53flash-08.txt', 'GLM', '免费车道'),
    ('ofm-glm53flash-09.txt', 'GLM', '免费车道'),
    ('cmdc-glm53flash-01.txt', 'GLM', 'cmdc 边界样本'),
    ('cmdc-glm53flash-03.txt', 'GLM', 'cmdc'),
    ('zcode-glm53flash-01.txt', 'GLM', '官方 zcode'),
    ('probe-01.txt', 'Claude', 'Claude Haiku 5.5（已确认）'),
    ('codex-sonnet55-01.txt', 'Claude', 'Claude Sonnet 5.5'),
    ('ofm-deepseek41flash-01.txt', 'DeepSeek', 'DeepSeek 对照'),
    ('qwen-free-01.txt', 'Qwen', 'Qwen 免费车道'),
]


def main():
    refs, _ = load_references(DEFAULT_REFERENCES)
    model = nlp_core.load_model() if ENGINE in ('nlp', 'fused') else load_network()
    print('引擎 %s | 参考库 %d 份 | 回归样本 %d 份（均不在库内）\n' % (ENGINE, len(refs), len(CASES)))
    print('%-26s %-8s %-10s %-8s %-8s %s' % ('样本', '真值', '判定', '模型分', 'GLM', '结果'))
    hit = 0
    fp = 0
    for name, truth, note in CASES:
        path = STAGE / name
        if not path.exists():
            print('%-26s %s' % (name, '文件缺失')); continue
        fam = (identify_nlp(read_answer(path), refs, model=model) if ENGINE == 'nlp'
               else identify_fused(read_answer(path), refs) if ENGINE == 'fused'
               else identify(read_answer(path), refs, model=model))
        glm = next(r['score'] for r in fam['ranking'] if r['label'] == 'GLM')
        ok = fam['candidate'] == truth
        hit += ok
        false_glm = (truth != 'GLM' and glm >= 0.5)
        fp += false_glm
        flag = 'PASS' if ok and not false_glm else ('FAIL(误吸GLM)' if false_glm else 'FAIL')
        print('%-26s %-8s %-10s %-8.3f %-8.3f %s  %s'
              % (name, truth, fam['candidate'] or 'uncertain', fam['ranking'][0]['score'], glm, flag, note))
    print('\n家族命中 %d/%d | 非 GLM 被误吸到 GLM 的份数 %d' % (hit, len(CASES), fp))
    return 0 if hit == len(CASES) and fp == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
