"""自注意力（V1）同折评估：注意力开/关两个变体，外加库外 11 份回归。

协议与 evaluate.py / evaluate_nlp.py 一致：
  * 逐份留出：每次留出一份，用其余样本重训（不留任何折外信息）
  * 整型号留出：某型号全部留出，同型号只训练一次
  * 门槛按"目标覆盖率 96.84%"在各自分数分布上反解，保证同覆盖率比较

用法：
    python evaluate_attn.py --output evaluation-attn.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import attn_core as A
from classifier import DEFAULT_REFERENCES, inventory, load_references, read_answer

TARGET_COVERAGE = .968421052631579
VARIANTS = {'attn': True, 'noattn': False}
STAGE = Path(r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型')
CASES = [
    ('ofm-glm53flash-06.txt', 'GLM'), ('ofm-glm53flash-07.txt', 'GLM'),
    ('ofm-glm53flash-08.txt', 'GLM'), ('ofm-glm53flash-09.txt', 'GLM'),
    ('cmdc-glm53flash-01.txt', 'GLM'), ('cmdc-glm53flash-03.txt', 'GLM'),
    ('zcode-glm53flash-01.txt', 'GLM'), ('probe-01.txt', 'Claude'),
    ('codex-sonnet55-01.txt', 'Claude'), ('ofm-deepseek41flash-01.txt', 'DeepSeek'),
    ('qwen-free-01.txt', 'Qwen'),
]


def run_folds(records, x, classes, mode, seed, epochs, l2, lr):
    y = np.array([classes.index(r['family']) for r in records])
    cache, rows = {}, []
    started = time.perf_counter()
    for index in range(len(records)):
        if mode == 'model':
            train = [j for j in range(len(records)) if records[j]['model'] != records[index]['model']]
            key = records[index]['model']
        else:
            train = [j for j in range(len(records)) if j != index]
            key = index
        if not train or records[index]['family'] not in {records[j]['family'] for j in train}:
            rows.append({'source': records[index]['source'], 'skipped': '留出后该家族无参考样本'})
            continue
        if key not in cache:
            subset = [records[j] for j in train]
            cache[key] = {name: A.train(subset, classes, use_attention=flag, seed=seed,
                                       epochs=epochs, l2=l2, learning_rate=lr, x=x[train])
                          for name, flag in VARIANTS.items()}
        row = {'source': records[index]['source'], 'expected': records[index]['family']}
        for name, flag in VARIANTS.items():
            probs = A.predict_scores(x[[index]], cache[key][name], flag)[0]
            order = np.argsort(-probs)
            row[name] = {'top': classes[int(order[0])], 'margin': float(probs[order[0]] - probs[order[1]])}
        rows.append(row)
    return rows, time.perf_counter() - started


def summarise(rows, variant):
    tested = [r for r in rows if 'skipped' not in r]
    margins = np.array([r[variant]['margin'] for r in tested])
    threshold = float(np.quantile(margins, 1 - TARGET_COVERAGE))
    accepted = [r for r in tested if r[variant]['margin'] >= threshold]
    hits = sum(r[variant]['top'] == r['expected'] for r in accepted)
    return {'variant': variant, 'tested': len(tested),
            'forced_accuracy': sum(r[variant]['top'] == r['expected'] for r in tested) / len(tested),
            'threshold': threshold, 'coverage': len(accepted) / len(tested),
            'accepted_accuracy': hits / len(accepted) if accepted else None,
            'abstained': len(tested) - len(accepted)}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--output', default='evaluation-attn.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--l2', type=float, default=.001)
    parser.add_argument('--lr', type=float, default=.01)
    args = parser.parse_args()
    records, skipped = load_references(args.references)
    classes = sorted({r['family'] for r in records})
    x = A.block_matrices(records)
    result = {'engine': 'attn', 'seed': args.seed, 'target_coverage': TARGET_COVERAGE,
              'inventory': inventory(records, skipped), 'runs': {}}

    for mode in ('sample', 'model'):
        rows, seconds = run_folds(records, x, classes, mode, args.seed, args.epochs, args.l2, args.lr)
        result['runs'][mode] = {'seconds': seconds, 'summary': [summarise(rows, v) for v in VARIANTS], 'rows': rows}
        print('\n[%s] %.1fs' % (mode, seconds))
        print('%-8s %-10s %-9s %-9s %s' % ('变体', '强制单答', '门槛', '接受内', '覆盖'))
        for item in result['runs'][mode]['summary']:
            print('%-8s %-10.3f %-9.4f %-9s %.3f'
                  % (item['variant'], item['forced_accuracy'], item['threshold'],
                     'n/a' if item['accepted_accuracy'] is None else '%.3f' % item['accepted_accuracy'],
                     item['coverage']))

    # 库外 11 份：用全库训练的模型（与 train_attn.py 默认一致），门槛取逐份留出的反解值
    thresholds = {item['variant']: item['threshold'] for item in result['runs']['sample']['summary']}
    full = {name: A.train(records, classes, use_attention=flag, seed=args.seed, epochs=args.epochs,
                          l2=args.l2, learning_rate=args.lr, x=x) for name, flag in VARIANTS.items()}
    external = []
    print('\n[库外 11 份]')
    for name, truth in CASES:
        path = STAGE / name
        if not path.exists():
            continue
        record = read_answer(path)
        row = {'source': name, 'expected': truth}
        for variant, flag in VARIANTS.items():
            probs = A.predict_record(record, full[variant], flag)[0]
            order = np.argsort(-probs)
            margin = float(probs[order[0]] - probs[order[1]])
            accepted = margin >= thresholds[variant]
            row[variant] = {'top': classes[int(order[0])], 'margin': margin,
                            'candidate': classes[int(order[0])] if accepted else None}
            print('  %-26s %-8s %-8s top %-9s margin %.3f %s'
                  % (name, truth, variant, classes[int(order[0])], margin,
                     'PASS' if row[variant]['candidate'] == truth else 'FAIL'))
        external.append(row)
    result['external'] = external
    result['external_hits'] = {variant: sum(r[variant]['candidate'] == r['expected'] for r in external)
                               for variant in VARIANTS}
    print('库外命中：' + ' | '.join('%s %d/%d' % (v, result['external_hits'][v], len(external)) for v in VARIANTS))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入 ' + args.output)


if __name__ == '__main__':
    main()
