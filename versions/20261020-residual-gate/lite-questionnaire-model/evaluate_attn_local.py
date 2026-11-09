"""局部（带状）注意力评估：题块只能看相邻 ±w 块。

对比四个变体（同折、同覆盖率反解门槛）：
    none     关闭注意力（同架构对照）
    band1    每块只看相邻 1 块（B02/B03/B04 随机题、B10/B11 对抗题这样相邻有语义）
    band2    相邻 2 块
    global   V1 的全局注意力（复现锚点，用于确认与 evaluation-attn.json 一致）

用法：
    python evaluate_attn_local.py --output evaluation-attn-local.json
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
VARIANTS = {'none': (False, None), 'band1': (True, 1), 'band2': (True, 2), 'global': (True, None)}
STAGE = Path(r'C:\Users\betraysilence\Music\nano-classifier\不确定新增不新增的模型')
CASES = [
    ('ofm-glm53flash-06.txt', 'GLM'), ('ofm-glm53flash-07.txt', 'GLM'),
    ('ofm-glm53flash-08.txt', 'GLM'), ('ofm-glm53flash-09.txt', 'GLM'),
    ('cmdc-glm53flash-01.txt', 'GLM'), ('cmdc-glm53flash-03.txt', 'GLM'),
    ('zcode-glm53flash-01.txt', 'GLM'), ('probe-01.txt', 'Claude'),
    ('codex-sonnet55-01.txt', 'Claude'), ('ofm-deepseek41flash-01.txt', 'DeepSeek'),
    ('qwen-free-01.txt', 'Qwen'),
]


def build(records, classes, use_attention, width, **kwargs):
    mask = A.band_mask(len(A.QUESTIONS), width) if width is not None else None
    return A.train(records, classes, use_attention=use_attention, mask=mask, mask_width=width, **kwargs)


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
            cache[key] = {}
            for name, (use_attention, width) in VARIANTS.items():
                mask = A.band_mask(len(A.QUESTIONS), width) if width is not None else None
                cache[key][name] = A.train(subset, classes, use_attention=use_attention, seed=seed,
                                           epochs=epochs, l2=l2, learning_rate=lr, x=x[train], mask=mask,
                                           mask_width=width)
        row = {'source': records[index]['source'], 'expected': records[index]['family']}
        for name, (use_attention, width) in VARIANTS.items():
            mask = A.band_mask(len(A.QUESTIONS), width) if width is not None else None
            probs = A.predict_scores(x[[index]], cache[key][name], use_attention, mask)[0]
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
    return {'variant': variant, 'tested': len(tested), 'threshold': threshold,
            'forced_accuracy': sum(r[variant]['top'] == r['expected'] for r in tested) / len(tested),
            'coverage': len(accepted) / len(tested),
            'accepted_accuracy': hits / len(accepted) if accepted else None,
            'abstained': len(tested) - len(accepted)}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--output', default='evaluation-attn-local.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--l2', type=float, default=.001)
    parser.add_argument('--lr', type=float, default=.01)
    args = parser.parse_args()
    records, skipped = load_references(args.references)
    classes = sorted({r['family'] for r in records})
    x = A.block_matrices(records)
    result = {'engine': 'attn-local', 'target_coverage': TARGET_COVERAGE, 'seed': args.seed,
              'inventory': inventory(records, skipped), 'runs': {}}
    for mode in ('sample', 'model'):
        rows, seconds = run_folds(records, x, classes, mode, args.seed, args.epochs, args.l2, args.lr)
        result['runs'][mode] = {'seconds': seconds,
                                'summary': [summarise(rows, v) for v in VARIANTS], 'rows': rows}
        print('\n[%s] %.1fs' % (mode, seconds))
        print('%-8s %-10s %-9s %-9s %s' % ('变体', '强制单答', '门槛', '接受内', '覆盖'))
        for item in result['runs'][mode]['summary']:
            print('%-8s %-10.3f %-9.4f %-9s %.3f'
                  % (item['variant'], item['forced_accuracy'], item['threshold'],
                     'n/a' if item['accepted_accuracy'] is None else '%.3f' % item['accepted_accuracy'],
                     item['coverage']))
    thresholds = {item['variant']: item['threshold'] for item in result['runs']['sample']['summary']}
    full = {}
    for name, (use_attention, width) in VARIANTS.items():
        mask = A.band_mask(len(A.QUESTIONS), width) if width is not None else None
        full[name] = A.train(records, classes, use_attention=use_attention, seed=args.seed,
                             epochs=args.epochs, l2=args.l2, learning_rate=args.lr, x=x,
                             mask=mask, mask_width=width)
    external = []
    print('\n[库外 11 份]')
    for name, truth in CASES:
        path = STAGE / name
        if not path.exists():
            continue
        record = read_answer(path)
        row = {'source': name, 'expected': truth}
        for variant, (use_attention, width) in VARIANTS.items():
            mask = A.band_mask(len(A.QUESTIONS), width) if width is not None else None
            probs = A.predict_record(record, full[variant], use_attention, mask)[0]
            order = np.argsort(-probs)
            margin = float(probs[order[0]] - probs[order[1]])
            row[variant] = {'top': classes[int(order[0])], 'margin': margin,
                            'candidate': classes[int(order[0])] if margin >= thresholds[variant] else None}
        external.append(row)
    result['external'] = external
    result['external_hits'] = {v: sum(r[v]['candidate'] == r['expected'] for r in external) for v in VARIANTS}
    for variant in VARIANTS:
        print('  %-8s 库外 %d/%d' % (variant, result['external_hits'][variant], len(external)))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('已写入 ' + args.output)


if __name__ == '__main__':
    main()
