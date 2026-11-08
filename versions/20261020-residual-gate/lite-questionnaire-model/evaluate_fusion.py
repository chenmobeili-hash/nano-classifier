"""NLP × 神经网络 融合实验（晚期融合，固定规则，参数不调）。

同一批留出折上同时训练两个模型，再按**预先固定的规则**合成：
  mlp       仅神经网络（135 维手工特征 → 8 隐层）
  nlp       仅 NLP 线性模型（n-gram TF-IDF → 逻辑回归）
  fused_mean  两个概率向量的算术平均（各 0.5，不调权重）
  fused_geo   两个概率向量的几何平均（log 概率平均后归一化）

门槛一律不手工指定，按"目标覆盖率 = 96.84%"（正式版神经网络逐份留出覆盖率）
在各自分数分布上反解，保证四个变体的对比是同覆盖率下的诚实比较。

用法：
    python evaluate_fusion.py --output evaluation-fusion.json
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

import neural_core as nc
import nlp_core as N
from classifier import DEFAULT_REFERENCES, inventory, load_references

VARIANTS = ('mlp', 'nlp', 'fused_mean', 'fused_geo')
TARGET_COVERAGE = .968421052631579


def probabilities(mlp_scores, nlp_scores):
    return {'mlp': np.asarray(mlp_scores, float),
            'nlp': np.asarray(nlp_scores, float),
            'fused_mean': (np.asarray(mlp_scores, float) + np.asarray(nlp_scores, float)) / 2.0}


def geometric(mlp_scores, nlp_scores):
    logits = .5 * np.log(np.clip(mlp_scores, 1e-12, 1)) + .5 * np.log(np.clip(nlp_scores, 1e-12, 1))
    exp = np.exp(logits - logits.max())
    return exp / exp.sum()


def run(records, mode, seed, nlp_epochs, mlp_epochs):
    classes = sorted({r['family'] for r in records})
    y = np.array([classes.index(r['family']) for r in records])
    counts = N.count_matrix(records)
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
            training = [records[j] for j in train]
            idf = N.idf_from(counts[train])
            x_train = N.apply_idf(counts[train], idf)
            weights = N.sample_weights(training, classes)
            nlp_params = N.fit(x_train, y[train], classes, weights, 'lr', seed=seed, epochs=nlp_epochs)
            mlp_model = nc.train_network(training, seed=seed, epochs=mlp_epochs)
            cache[key] = (idf, nlp_params, mlp_model, x_train)
        idf, nlp_params, mlp_model, x_train = cache[key]
        p_nlp = N.predict_scores(N.apply_idf(counts[[index]], idf), nlp_params, 'lr')[0]
        p_mlp = nc.predict_scores(records[index], mlp_model)
        scores = probabilities(p_mlp, p_nlp)
        scores['fused_geo'] = geometric(p_mlp, p_nlp)
        row = {'source': records[index]['source'], 'expected': records[index]['family']}
        for name, values in scores.items():
            order = np.argsort(-values)
            row[name] = {'top': classes[int(order[0])],
                         'margin': float(values[order[0]] - values[order[1]]),
                         'prob': float(values[order[0]])}
        rows.append(row)
    return rows, time.perf_counter() - started


def summarise(rows, variant):
    tested = [r for r in rows if 'skipped' not in r]
    margins = np.array([r[variant]['margin'] for r in tested])
    threshold = float(np.quantile(margins, 1.0 - TARGET_COVERAGE)) if TARGET_COVERAGE < 1 else 0.0
    accepted = [r for r in tested if r[variant]['margin'] >= threshold]
    forced = sum(r[variant]['top'] == r['expected'] for r in tested) / len(tested)
    hits = sum(r[variant]['top'] == r['expected'] for r in accepted)
    return {'variant': variant, 'tested': len(tested), 'forced_accuracy': forced,
            'threshold': threshold, 'coverage': len(accepted) / len(tested),
            'accepted_accuracy': hits / len(accepted) if accepted else None,
            'abstained': len(tested) - len(accepted)}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--output', default='evaluation-fusion.json')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--nlp-epochs', type=int, default=400)
    parser.add_argument('--mlp-epochs', type=int, default=400)
    args = parser.parse_args()
    records, skipped = load_references(args.references)
    result = {'engine': 'fusion', 'seed': args.seed, 'target_coverage': TARGET_COVERAGE,
              'inventory': inventory(records, skipped), 'runs': {}}
    for mode in ('sample', 'model'):
        rows, seconds = run(records, mode, args.seed, args.nlp_epochs, args.mlp_epochs)
        result['runs'][mode] = {'seconds': seconds, 'summary': [summarise(rows, v) for v in VARIANTS], 'rows': rows}
        print('\n[%s] %.1fs  目标覆盖率 %.3f' % (mode, seconds, TARGET_COVERAGE))
        print('%-12s %-10s %-10s %-10s %-10s %s' % ('变体', '强制单答', '门槛', '覆盖率', '接受内准确', '弃权'))
        for item in result['runs'][mode]['summary']:
            print('%-12s %-10.3f %-10.4f %-10.3f %-10s %d'
                  % (item['variant'], item['forced_accuracy'], item['threshold'], item['coverage'],
                     'n/a' if item['accepted_accuracy'] is None else '%.3f' % item['accepted_accuracy'],
                     item['abstained']))
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('\n已写入 ' + args.output)


if __name__ == '__main__':
    main()
