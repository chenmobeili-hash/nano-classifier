"""NLP 版识别入口：输出结构与 classifier.identify 保持一致，便于网页/CLI 复用。

与神经网络版的唯一区别是**证据来源**：这里是文本 n-gram 的线性模型，
不是 135 维手工特征 + 8 隐层 MLP。拒识门槛、近邻诊断、免责说明全部同构。
"""
import numpy as np

import nlp_core as N
from classifier import fingerprint
from consensus_guard import apply_consensus_guard
from known_references import known_reference_matches

NOTE = ('候选只由 NLP 线性模型（n-gram TF-IDF）排序；近邻仅作相似性诊断与冲突提示，'
        '不改变分数、排序或拒识。score未校准，不是身份正确概率；最近参考型号不等于识别型号。'
        '距离为 NLP 空间的余弦距离，只用于诊断和启发式拒识；不能可靠排除未知家族。'
        '参考库命中不属于独立验证。')


def identify_nlp(record, references, min_margin=None, min_prob=0.0, model=None, kind=None,
                 reference_vectors=None, scores=None, engine_label='nlp'):
    min_margin = N.CALIBRATED_MIN_MARGIN if min_margin is None else min_margin
    if not 0 <= min_margin <= 1:
        raise ValueError('候选间隔门槛必须在0和1之间。')
    if not 0 <= min_prob <= 1:
        raise ValueError('最大模型分门槛必须在0和1之间。')
    families = sorted(set(r['family'] for r in references))
    if len(families) < 2:
        raise ValueError('至少需要两个参考家族。')
    if model is None:
        network = N.load_model()
    elif isinstance(model, dict):
        network = model
    else:
        network = N.load_model(model)
    if network['meta']['reference_signature'] != N.reference_signature(references):
        raise ValueError('当前参考库与 NLP 权重训练数据不一致，请用该参考库重新训练，并通过 --model 指定新权重。')
    if set(network['meta']['classes']) != set(families):
        raise ValueError('NLP 权重类别与当前参考库不一致。')
    kind = kind or network['meta']['kind']

    include_a = network['meta'].get('include_a', False)
    if reference_vectors is None:
        reference_vectors = N.apply_idf(N.count_matrix(references, include_a), network['idf'])
    query_vector = N.apply_idf(N.count_vector(record, include_a)[None, :], network['idf'])
    if scores is None:
        scores = N.predict_scores(query_vector, network, kind)[0]
    else:
        scores = np.asarray(scores, float)
        if scores.shape != (len(network['meta']['classes']),):
            raise ValueError('外部传入的家族分维度与权重类别不一致。')
    by_family = dict(zip(network['meta']['classes'], np.asarray(scores).tolist()))
    distance = N.distances(query_vector, reference_vectors)[0]

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

    pairwise = N.distances(reference_vectors, reference_vectors)
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
    closest = int(np.argmin(distance))
    meta = network['meta']
    return {'candidate': ranking[0]['label'] if certain else None,
            'engine': engine_label,
            'network': {key: meta[key] for key in ('version', 'kind', 'dimensions', 'samples', 'epochs') if key in meta},
            'feature_details': {'engine': 'nlp', 'dimensions': meta['dimensions'],
                                'char_ngrams': meta['char_ngrams'], 'word_ngrams': meta['word_ngrams'],
                                'include_a': meta['include_a'], 'closest_sample_distance': float(distance[closest])},
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
