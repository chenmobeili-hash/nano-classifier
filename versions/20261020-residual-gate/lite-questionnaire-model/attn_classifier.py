"""自注意力（V1）识别入口：输出结构与 classifier.identify / identify_nlp 一致。

证据来源：12 个题块的哈希向量 → 共享投影 → 单头自注意力跨题块 → 池化 → 5 类。
距离为题块摊平后的余弦距离，只做相似性诊断与启发式拒识。
"""
import numpy as np

import attn_core as A
from classifier import fingerprint
from consensus_guard import apply_consensus_guard
from known_references import known_reference_matches

# 逐份留出按 96.84% 覆盖率反解（见 evaluation-attn.json）；换库须重跑 evaluate_attn.py
CALIBRATED_MIN_MARGIN = 0.05

NOTE = ('候选只由题块级自注意力模型排序；近邻仅作相似性诊断与冲突提示，不改变分数、排序或拒识。'
        'score未校准，不是身份正确概率；最近参考型号不等于识别型号。'
        '距离为题块摊平后的余弦距离，只用于诊断和启发式拒识；不能可靠排除未知家族。')


def identify_attn(record, references, min_margin=None, min_prob=0.0, model=None,
                  reference_vectors=None):
    min_margin = CALIBRATED_MIN_MARGIN if min_margin is None else min_margin
    if not 0 <= min_margin <= 1:
        raise ValueError('候选间隔门槛必须在0和1之间。')
    if not 0 <= min_prob <= 1:
        raise ValueError('最大模型分门槛必须在0和1之间。')
    families = sorted(set(r['family'] for r in references))
    if len(families) < 2:
        raise ValueError('至少需要两个参考家族。')
    if model is None:
        network = A.load_model()
    elif isinstance(model, dict):
        network = model
    else:
        network = A.load_model(model)
    if network['meta']['reference_signature'] != A.reference_signature(references):
        raise ValueError('当前参考库与注意力权重训练数据不一致，请用该参考库重新训练，并通过 --model 指定新权重。')
    if set(network['meta']['classes']) != set(families):
        raise ValueError('注意力权重类别与当前参考库不一致。')

    probs = A.predict_record(record, network)[0]
    by_family = dict(zip(network['meta']['classes'], probs.tolist()))
    if reference_vectors is None:
        reference_vectors = A.flat_vectors(references)
    query = A.block_matrix(record).ravel()
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    distance = 1.0 - np.clip(reference_vectors @ query, -1.0, 1.0)

    model_distances = []
    for reference_model in sorted(set(r['model'] for r in references)):
        indices = [i for i, r in enumerate(references) if r['model'] == reference_model]
        model_distances.append((references[indices[0]]['family'], reference_model, float(distance[indices].mean())))
    ranking = sorted([{'label': family, 'distance': min(d for f, _, d in model_distances if f == family)}
                      for family in families], key=lambda item: item['distance'])
    best_distance_family = ranking[0]['label']
    neural_ranking = sorted([dict(item, score=by_family[item['label']]) for item in ranking],
                            key=lambda item: item['score'], reverse=True)
    ranking, decision_guard = apply_consensus_guard(neural_ranking, distance, references)
    margin = ranking[0]['score'] - ranking[1]['score']

    pairwise = 1.0 - np.clip(reference_vectors @ reference_vectors.T, -1.0, 1.0)
    np.fill_diagonal(pairwise, np.inf)
    radius = max(.5, float(np.quantile(pairwise.min(axis=1), .95)) * 1.5)

    digest = fingerprint(record)
    matches = [r['source'] for r in references if r['fingerprint'] == digest]
    top_score = float(ranking[0]['score'])
    reasons = []
    if margin < min_margin:
        reasons.append('margin')
    if top_score < min_prob:
        reasons.append('low_model_score')
    if ranking[0]['distance'] > radius:
        reasons.append('far_from_reference')
    certain = not reasons
    meta = network['meta']
    return {'candidate': ranking[0]['label'] if certain else None,
            'engine': 'attn',
            'network': {key: meta[key] for key in ('version', 'use_attention', 'hidden', 'blocks', 'samples', 'epochs')
                        if key in meta},
            'feature_details': {'engine': 'attn', 'blocks': meta['blocks'], 'block_dims': meta['block_dims'],
                                'char_ngrams': meta['char_ngrams'], 'use_attention': meta['use_attention'],
                                'nearest_distance': float(distance.min())},
            'nearest_family': ranking[0]['label'],
            'best_distance_family': best_distance_family,
            'status': 'candidate' if certain else 'uncertain',
            'ranking': ranking, 'neural_ranking': neural_ranking, 'decision_guard': decision_guard,
            'margin': margin, 'min_margin': min_margin,
            'top_score': top_score, 'min_prob': min_prob,
            'accepted': certain, 'reject_reason': reasons,
            'margin_kind': 'softmax_top_two_difference',
            'distance_limit': radius,
            'reference_matches': matches,
            'known_external_references': known_reference_matches(record),
            'nearest_samples': [{'source': references[i]['source'], 'family': references[i]['family'],
                                 'distance': float(distance[i])} for i in np.argsort(distance)[:3]],
            'note': NOTE}
