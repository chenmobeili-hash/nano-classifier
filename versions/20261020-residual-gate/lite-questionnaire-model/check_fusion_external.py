"""库外 11 份固定回归样本上的融合对照（不训练，只用已发布权重）。

读 evaluation-fusion.json 里按覆盖率反解的各变体门槛，套到库外样本上，
只报告命中，不用它调任何参数。
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import neural_core as nc
import nlp_core as N
from classifier import DEFAULT_REFERENCES, load_references, read_answer
from evaluate_fusion import geometric

STAGE = Path(__file__).resolve().parents[3] / 'external-holdout'
CASES = [
    ('ofm-glm53flash-06.txt', 'GLM'), ('ofm-glm53flash-07.txt', 'GLM'),
    ('ofm-glm53flash-08.txt', 'GLM'), ('ofm-glm53flash-09.txt', 'GLM'),
    ('cmdc-glm53flash-01.txt', 'GLM'), ('cmdc-glm53flash-03.txt', 'GLM'),
    ('zcode-glm53flash-01.txt', 'GLM'), ('probe-01.txt', 'Claude'),
    ('codex-sonnet55-01.txt', 'Claude'), ('ofm-deepseek41flash-01.txt', 'DeepSeek'),
    ('qwen-free-01.txt', 'Qwen'),
]


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--evaluation', default='evaluation-fusion.json')
    parser.add_argument('--output', default='evaluation-fusion-external.json')
    args = parser.parse_args()
    evaluation = json.loads(Path(args.evaluation).read_text(encoding='utf-8'))
    thresholds = {item['variant']: item['threshold'] for item in evaluation['runs']['sample']['summary']}
    refs, _ = load_references(DEFAULT_REFERENCES)
    mlp = nc.load_network()
    nlp = N.load_model()
    classes = mlp['meta']['classes']
    assert classes == nlp['meta']['classes']

    rows, hits = [], {name: 0 for name in ('mlp', 'nlp', 'fused_mean', 'fused_geo')}
    print('%-26s %-8s %-9s %-9s %-9s %-9s %s' % ('样本', '真值', 'mlp', 'nlp', 'mean', 'geo', '备注'))
    for name, truth in CASES:
        path = STAGE / name
        if not path.exists():
            continue
        record = read_answer(path)
        p_mlp = np.asarray(nc.predict_scores(record, mlp), float)
        p_nlp = np.asarray(N.predict_record(record, nlp)[0], float)
        scores = {'mlp': p_mlp, 'nlp': p_nlp,
                  'fused_mean': (p_mlp + p_nlp) / 2.0,
                  'fused_geo': geometric(p_mlp, p_nlp)}
        verdict = {}
        for variant, values in scores.items():
            order = np.argsort(-values)
            margin = float(values[order[0]] - values[order[1]])
            accepted = margin >= thresholds[variant]
            picked = classes[int(order[0])] if accepted else None
            verdict[variant] = picked
            hits[variant] += picked == truth
        print('%-26s %-8s %-9s %-9s %-9s %-9s' % (name, truth,
              verdict['mlp'] or 'uncertain', verdict['nlp'] or 'uncertain',
              verdict['fused_mean'] or 'uncertain', verdict['fused_geo'] or 'uncertain'))
        rows.append({'source': name, 'expected': truth, 'verdict': verdict})
    print()
    for variant in ('mlp', 'nlp', 'fused_mean', 'fused_geo'):
        print('%-12s 库外命中 %d/%d（门槛 %.4f）' % (variant, hits[variant], len(rows), thresholds[variant]))
    Path(args.output).write_text(json.dumps({'thresholds': thresholds, 'hits': hits, 'rows': rows},
                                            ensure_ascii=False, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
