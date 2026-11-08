"""NLP 版拒识门槛标定：按目标覆盖率反解 min_margin。

背景：0.1 这个 margin 门槛是为神经网络的 softmax 量纲定的。NLP 线性模型的分数
普遍更平（最高 0.47 vs 0.97），直接套用等于"更严"，逐份覆盖只剩 70%。

做法（不动外部样本、不按单样本调参）：
  1. 用留出评估里每一折的 margin 分布，按目标覆盖率反解门槛；
  2. 在同一个门槛下重算留出集的接受内准确率；
  3. 再用固定 11 份库外回归样本检验——**只报告，不用它选门槛**。

用法：
    python calibrate_nlp.py --evaluation evaluation-nlp.json --output calibration-nlp.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

from classifier import DEFAULT_REFERENCES, load_references, read_answer
import nlp_core
from nlp_classifier import identify_nlp

STAGE = Path(r'C:\Users\betraysilence\Music\nano-classfier\不确定新增不新增的模型')
CASES = [
    ('ofm-glm53flash-06.txt', 'GLM'), ('ofm-glm53flash-07.txt', 'GLM'),
    ('ofm-glm53flash-08.txt', 'GLM'), ('ofm-glm53flash-09.txt', 'GLM'),
    ('cmdc-glm53flash-01.txt', 'GLM'), ('cmdc-glm53flash-03.txt', 'GLM'),
    ('zcode-glm53flash-01.txt', 'GLM'), ('probe-01.txt', 'Claude'),
    ('codex-sonnet55-01.txt', 'Claude'), ('ofm-deepseek41flash-01.txt', 'DeepSeek'),
    ('qwen-free-01.txt', 'Qwen'),
]
NEURAL_COVERAGE = .968421052631579  # 神经网络正式版逐份留出覆盖率，作为对齐目标
GRID = (1.0, NEURAL_COVERAGE, .95, .90, .85, .80)


def threshold_for(rows, coverage):
    margins = np.array([r['margin'] for r in rows])
    if coverage >= 1.0:
        return 0.0
    return float(np.quantile(margins, 1.0 - coverage))


def accepted_stats(rows, threshold):
    accepted = [r for r in rows if r['margin'] >= threshold and r['distance'] <= r['distance_limit']]
    hits = sum(r['nearest_family'] == r['expected'] for r in accepted)
    return {'threshold': threshold, 'coverage': len(accepted) / max(len(rows), 1),
            'accepted': len(accepted), 'accepted_accuracy': hits / len(accepted) if accepted else None}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', default='evaluation-nlp.json')
    parser.add_argument('--output', default='calibration-nlp.json')
    args = parser.parse_args()
    evaluation = json.loads(Path(args.evaluation).read_text(encoding='utf-8'))
    refs, _ = load_references(DEFAULT_REFERENCES)
    model = nlp_core.load_model()

    result = {'neural_reference_coverage': NEURAL_COVERAGE, 'modes': {}, 'external': []}
    for mode in ('sample_holdout', 'model_holdout'):
        rows = [r for r in evaluation[mode]['rows'] if 'skipped' not in r]
        result['modes'][mode] = {'forced_accuracy': sum(r['nearest_family'] == r['expected'] for r in rows) / len(rows),
                                 'grid': [accepted_stats(rows, threshold_for(rows, c)) for c in GRID]}
        print('%s：强制单答 %.3f' % (mode, result['modes'][mode]['forced_accuracy']))
        for item, coverage in zip(result['modes'][mode]['grid'], GRID):
            print('  目标覆盖 %-6.3f → 门槛 %-6.4f  实际覆盖 %-6.3f  接受内准确 %s'
                  % (coverage, item['threshold'], item['coverage'],
                     'n/a' if item['accepted_accuracy'] is None else '%.3f' % item['accepted_accuracy']))

    # 与神经网络覆盖率对齐的门槛，用于库外回归的只读检验
    sample_rows = [r for r in evaluation['sample_holdout']['rows'] if 'skipped' not in r]
    aligned = threshold_for(sample_rows, NEURAL_COVERAGE)
    print('\n对齐神经网络覆盖率的门槛：min_margin = %.4f（仅报告，未用库外样本调整）' % aligned)
    hit = 0
    for name, truth in CASES:
        path = STAGE / name
        if not path.exists():
            continue
        fam = identify_nlp(read_answer(path), refs, model=model, min_margin=aligned)
        ok = fam['candidate'] == truth
        hit += ok
        result['external'].append({'source': name, 'expected': truth, 'candidate': fam['candidate'],
                                   'top': fam['nearest_family'], 'top_score': fam['top_score'],
                                   'margin': fam['margin'], 'distance': fam['ranking'][0]['distance'],
                                   'accepted': fam['accepted'], 'reject_reason': fam['reject_reason']})
        print('  %-26s 真值 %-8s 判定 %-10s top %.3f margin %.3f  %s'
              % (name, truth, fam['candidate'] or 'uncertain', fam['top_score'], fam['margin'], 'PASS' if ok else 'FAIL'))
    result['external_hits'] = hit
    result['external_total'] = len(result['external'])
    print('库外回归：%d/%d' % (hit, result['external_total']))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
