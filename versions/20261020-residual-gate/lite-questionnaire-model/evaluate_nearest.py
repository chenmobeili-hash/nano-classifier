"""最近邻诊断的表示空间对照：TF-IDF 余弦 vs 135 维特征空间。

只读取现有参考库与权重，不训练、不改权重；结果用于说明页面为何改用特征空间。
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

import feature_profiles as F
import nlp_core as N
import residual_gate as R
from classifier import load_references

TOP_K = (1, 3)


def build_spaces(refs, models):
    _, bundle = R.load_bundle(Path(__file__).resolve().with_name('residual-gate.npz'))
    vectors = N.apply_idf(N.count_matrix(refs), bundle['nlp']['idf'])
    tfidf = N.distances(vectors, vectors)
    feat = np.stack([F.distances(record, refs, 'balanced')[0] for record in refs])
    return {'tfidf': tfidf, 'feature135': feat}


def rank_single(matrix, index):
    row = matrix[index].copy()
    row[index] = np.inf
    return list(np.argsort(row, kind='stable'))


def rank_group_mean(matrix, index, groups, drop=None):
    scores = {}
    for model, members in groups.items():
        if model == drop:
            continue
        members = [m for m in members if m != index]
        if members:
            scores[model] = float(matrix[index, members].mean())
    return sorted(scores, key=lambda model: scores[model])


def score_ranking(order, i, models, families, model_family, k):
    top = order[:k]
    if isinstance(top[0], str):
        hit_model = models[i] in top
        hit_family = any(model_family[model] == families[i] for model in top)
    else:
        hit_model = any(models[j] == models[i] for j in top)
        hit_family = any(families[j] == families[i] for j in top)
    return hit_model, hit_family


def evaluate(matrix, refs, models, families, model_family, groups, group_mean):
    n = len(refs)
    out = {}
    for k in TOP_K:
        hit_model = hit_family = 0
        for i in range(n):
            order = (rank_group_mean(matrix, i, groups) if group_mean
                     else rank_single(matrix, i))
            m, f = score_ranking(order, i, models, families, model_family, k)
            hit_model += m
            hit_family += f
        out['leave_one_answer_out_top%d' % k] = {
            'model': hit_model / n, 'family': hit_family / n, 'tested': n}
    for k in TOP_K:
        hit_family = 0
        for group, members in groups.items():
            for i in members:
                order = (rank_group_mean(matrix, i, groups, drop=group) if group_mean
                         else [j for j in rank_single(matrix, i) if models[j] != group])
                _, f = score_ranking(order, i, models, families, model_family, k)
                hit_family += f
        out['leave_one_model_out_top%d' % k] = {'family': hit_family / n, 'tested': n}
    return out


def separation(matrix, models, families):
    same, other_family, within_family = [], [], []
    n = len(models)
    for i in range(n):
        for j in range(i + 1, n):
            value = float(matrix[i, j])
            if models[i] == models[j]:
                same.append(value)
            else:
                other_family.append(value)
                if families[i] == families[j]:
                    within_family.append(value)
    return {'same_model_median': float(np.median(same)),
            'different_model_median': float(np.median(other_family)),
            'within_family_other_model_median': float(np.median(within_family)),
            'median_ratio': float(np.median(other_family) / np.median(same)),
            'same_model_pairs': len(same)}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='对照最近邻诊断在两种表示空间下的表现。')
    parser.add_argument('--output', default='nearest-space-report.json')
    args = parser.parse_args()
    target = Path(args.output)
    if target.exists():
        parser.error('%s 已存在；请换一个输出文件名，避免覆盖既有记录。' % target)

    refs, skipped = load_references()
    models = [r['model'] for r in refs]
    families = [r['family'] for r in refs]
    groups = defaultdict(list)
    for index, model in enumerate(models):
        groups[model].append(index)
    model_family = {model: families[members[0]] for model, members in groups.items()}

    report = {'reference_signature': N.reference_signature(refs), 'samples': len(refs),
              'models': len(groups), 'skipped': skipped, 'spaces': {}}
    for name, matrix in build_spaces(refs, models).items():
        report['spaces'][name] = {
            'single_nearest': evaluate(matrix, refs, models, families, model_family, groups, False),
            'group_mean': evaluate(matrix, refs, models, families, model_family, groups, True),
            'separation': separation(matrix, models, families)}
        print(name, json.dumps(report['spaces'][name]['separation'], ensure_ascii=False))

    target.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')

    print('\n| 空间 | 方法 | 型号@1 | 型号@3 | 家族@1 | 家族@3 | 家族留一型号@1 | 家族留一型号@3 |')
    print('|---|---|---:|---:|---:|---:|---:|---:|')
    for name, space in report['spaces'].items():
        for method, key in (('单份最近邻', 'single_nearest'), ('按型号组均值', 'group_mean')):
            row = space[key]
            print('| %s | %s | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% | %.1f%% |' % (
                name, method,
                100 * row['leave_one_answer_out_top1']['model'], 100 * row['leave_one_answer_out_top3']['model'],
                100 * row['leave_one_answer_out_top1']['family'], 100 * row['leave_one_answer_out_top3']['family'],
                100 * row['leave_one_model_out_top1']['family'], 100 * row['leave_one_model_out_top3']['family']))
    print('\n写入 %s' % target)


if __name__ == '__main__':
    main()
