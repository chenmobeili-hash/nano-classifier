"""NLP 版留出评估：与 evaluate.py 同协议、同口径，便于直接对比。

  * 逐份留出（sample）：每次留出一份，用其余样本重算 IDF 并重训。
  * 整型号留出（model）：某个型号的全部样本一起留出，同型号只训练一次。
  * 留出折不参与 IDF、不参与梯度更新、不参与类别权重计算。
  * 拒识判定与 identify_nlp 一致：margin 门槛 + 参考距离门槛。

用法：
    python evaluate_nlp.py --output evaluation-nlp.json
    python evaluate_nlp.py --kind centroid --include-a --output eval-nlp-ab.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

import nlp_core as N
from classifier import DEFAULT_REFERENCES, inventory, load_references


def _radius(vectors):
    pairwise = N.distances(vectors, vectors)
    np.fill_diagonal(pairwise, np.inf)
    return max(.5, float(np.quantile(pairwise.min(axis=1), .95)) * 1.5)


def evaluate(records, mode, kind, include_a, seed, epochs, l2, min_margin=.1):
    classes = sorted({r['family'] for r in records})
    y = np.array([classes.index(r['family']) for r in records])
    counts = N.count_matrix(records, include_a)
    folds = {}
    for index in range(len(records)):
        if mode == 'model':
            train = [j for j in range(len(records)) if records[j]['model'] != records[index]['model']]
            key = records[index]['model']
        else:
            train = [j for j in range(len(records)) if j != index]
            key = index
        if records[index]['family'] not in {records[j]['family'] for j in train}:
            folds[index] = None
            continue
        if key not in folds:
            idf = N.idf_from(counts[train])
            x_train = N.apply_idf(counts[train], idf)
            weights = N.sample_weights([records[j] for j in train], classes)
            params = N.fit(x_train, y[train], classes, weights, kind, seed=seed, epochs=epochs, l2=l2)
            folds[key] = (idf, params, x_train, _radius(x_train))
        folds[index] = folds[key]

    rows = []
    for index in range(len(records)):
        fold = folds[index]
        if fold is None:
            rows.append({'source': records[index]['source'], 'skipped': '留出后该家族无参考样本'})
            continue
        idf, params, x_train, radius = fold
        query = N.apply_idf(counts[[index]], idf)
        scores = N.predict_scores(query, params, kind)[0]
        order = np.argsort(-scores)
        margin = float(scores[order[0]] - scores[order[1]])
        distance = float(N.distances(query, x_train)[0].min())
        accepted = margin >= min_margin and distance <= radius
        rows.append({'source': records[index]['source'], 'expected': records[index]['family'],
                     'candidate': classes[int(order[0])] if accepted else None,
                     'nearest_family': classes[int(order[0])],
                     'margin': margin, 'distance': distance, 'distance_limit': radius,
                     'accepted': accepted})

    tested = [r for r in rows if 'skipped' not in r]
    accepted = [r for r in tested if r['candidate'] is not None]
    abstained = [r for r in tested if r['candidate'] is None]
    nearest_hit = sum(r['nearest_family'] == r['expected'] for r in abstained)
    forced = sum(r['nearest_family'] == r['expected'] for r in tested)
    return {'mode': mode, 'kind': kind, 'include_a': include_a, 'tested': len(tested),
            'nearest_accuracy': forced / max(len(tested), 1),
            'coverage': len(accepted) / max(len(tested), 1),
            'accepted_accuracy': sum(r['candidate'] == r['expected'] for r in accepted) / len(accepted) if accepted else None,
            'abstained': len(abstained), 'abstained_nearest_hits': nearest_hit,
            'abstained_nearest_accuracy': nearest_hit / len(abstained) if abstained else None,
            'abstention_gain_vs_forced': ((sum(r['candidate'] == r['expected'] for r in accepted) / len(accepted)
                                           if accepted else 0) - forced / max(len(tested), 1)),
            'rows': rows}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--output', default='evaluation-nlp.json')
    parser.add_argument('--kind', choices=N.KINDS, default='lr')
    parser.add_argument('--include-a', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--l2', type=float, default=.001)
    parser.add_argument('--min-margin', type=float, default=.1)
    args = parser.parse_args()
    records, skipped = load_references(args.references)
    result = {'engine': 'nlp', 'kind': args.kind, 'include_a': args.include_a, 'seed': args.seed,
              'inventory': inventory(records, skipped),
              'sample_holdout': evaluate(records, 'sample', args.kind, args.include_a, args.seed, args.epochs, args.l2, args.min_margin),
              'model_holdout': evaluate(records, 'model', args.kind, args.include_a, args.seed, args.epochs, args.l2, args.min_margin),
              'note': ('开发数据上的探索评估。每折重新计算 IDF 并从零训练；留出答案不参与 IDF、梯度更新或轮数选择。'
                       '型号标签仍来自文件名。与小样本下的神经网络版不可直接等价比较，但协议与 evaluate.py 一致。')}
    with Path(args.output).open('x', encoding='utf-8') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: {n: v for n, v in result[k].items() if n != 'rows'}
                      for k in ('sample_holdout', 'model_holdout')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
