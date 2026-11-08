from consensus_guard import apply_consensus_guard
"""Tiny neural family classifier with existing distance diagnostics retained."""
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
import numpy as np
from questionnaire_features import canonical_answers, features
from split_answers import read_text
from known_references import known_reference_matches
from sample_annotations import annotation_for
from feature_profiles import distances, PROFILES
from neural_core import load_network, predict_scores, reference_signature

DEFAULT_REFERENCES = Path(__file__).resolve().parent.parent / 'nano-now'
FAMILIES = {'gpt': 'GPT', 'claude': 'Claude', 'deepseek': 'DeepSeek', 'glm': 'GLM', 'qwen': 'Qwen'}


def fingerprint(record):
    answers = canonical_answers(record)
    content = {q: re.sub(r'\s+', '', answers[q]) for q in answers if q.startswith('B')}
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_answer(path):
    path = Path(path)
    if path.suffix.lower() in ('.txt', '.md'):
        record = {'raw_response': read_text(path)}
    else:
        record = json.loads(path.read_text(encoding='utf-8-sig'))
        if not isinstance(record, dict):
            raise ValueError('输入必须为一份回答。')
    result = {'answers': canonical_answers(record)}
    if isinstance(record.get('raw_response'), str):
        result['raw_response'] = record['raw_response']
    return result


def load_references(directory=DEFAULT_REFERENCES):
    records, skipped, seen = [], [], {}
    for path in sorted(Path(directory).glob('*.txt')):
        model = re.sub(r'-\d+$', '', path.stem).lower()
        # Historical reference filenames retain their audit identities.
        if model in ('qwen-app-unknown', 'qwen-app-fast'):
            model = 'qwen-app-fast'
        family = next((v for k, v in FAMILIES.items() if model.startswith(k)), None)
        try:
            if family is None:
                raise ValueError('文件名没有可识别的家族标签，请核实后添加映射。')
            record = read_answer(path)
            record['source'] = path.name
            annotation = annotation_for(record)
            # Only explicit, digest-bound user exclusions affect this shared
            # reference loader: training, identification and all rankings agree.
            if annotation and annotation.get('exclude_from_training_and_ranking'):
                continue
            digest = fingerprint(record)
            if digest in seen:
                if seen[digest] != family:
                    raise RuntimeError('相同B题内容具有冲突标签：' + path.name)
                raise ValueError('B题内容重复，未重复计入参考库。')
            seen[digest] = family
            record.update(source=path.name, model=model, family=family, fingerprint=digest)
            records.append(record)
        except (ValueError, OSError) as exc:
            skipped.append({'source': path.name, 'reason': str(exc)})
    if len(set(r['family'] for r in records)) < 2:
        raise ValueError('有效参考库至少需要两个家族。')
    return records, skipped


def identify(record, references, min_margin=.1, profile='balanced', engine='neural', model=None, min_prob=0.0):
    if not 0 <= min_margin <= 1:
        raise ValueError('候选间隔门槛必须在0和1之间。')
    if not 0 <= min_prob <= 1:
        raise ValueError('最大模型分门槛必须在0和1之间。')
    if len(set(r['family'] for r in references)) < 2:
        raise ValueError('至少需要两个参考家族。')
    if engine not in ('neural', 'distance'):
        raise ValueError('识别引擎必须为neural或distance。')
    if engine == 'neural' and profile != 'balanced':
        raise ValueError('神经网络使用balanced的135维特征；其他方案请加 --engine distance。')
    distance, pairwise, feature_details = distances(record, references, profile)
    models = sorted(set(r['model'] for r in references))
    model_distances = []
    for reference_model in models:
        indices = [i for i, r in enumerate(references) if r['model'] == reference_model]
        model_distances.append((references[indices[0]]['family'], reference_model, float(distance[indices].mean())))
    families = sorted(set(r['family'] for r in references))
    ranking = sorted([{'label': family, 'distance': min(d for f, _, d in model_distances if f == family)}
                      for family in families], key=lambda item: item['distance'])
    margin = (ranking[1]['distance']-ranking[0]['distance'])/max(ranking[1]['distance'], 1e-9)
    best_distance_family = ranking[0]['label']
    network_meta = None
    neural_ranking = None
    decision_guard = None
    if engine == 'neural':
        network = load_network() if model is None else (load_network(model) if isinstance(model, (str, Path)) else model)
        if network['meta']['reference_signature'] != reference_signature(references):
            raise ValueError('当前参考库与权重训练数据不一致，请用该参考库重新训练，并通过 --model 指定新权重。')
        scores = predict_scores(record, network)
        by_family = dict(zip(network['meta']['classes'], scores.tolist()))
        if set(by_family) != set(families):
            raise ValueError('权重类别与当前参考库不一致。')
        neural_ranking = sorted([dict(item, score=by_family[item['label']]) for item in ranking], key=lambda item: item['score'], reverse=True)
        ranking, decision_guard = apply_consensus_guard(neural_ranking, distance, references)
        margin = ranking[0]['score']-ranking[1]['score']
        network_meta = {key: network['meta'][key] for key in ('version', 'architecture', 'parameters', 'seed', 'epochs')}
    np.fill_diagonal(pairwise, np.inf)
    radius = max(.5, float(np.quantile(pairwise.min(axis=1), .95))*1.5)
    digest = fingerprint(record)
    matches = [r['source'] for r in references if r['fingerprint'] == digest]
    top_score = float(ranking[0]['score']) if engine == 'neural' else None
    reasons = []
    if margin < min_margin:
        reasons.append('margin')
    if top_score is not None and top_score < min_prob:
        reasons.append('low_model_score')
    if ranking[0]['distance'] > radius:
        reasons.append('far_from_reference')
    certain = len(reasons) == 0
    block_values = feature_details.pop('block_squared_distances', {})
    closest = int(np.argmin(distance))
    feature_details['closest_sample_block_squared_distances'] = {name: values[closest] for name, values in block_values.items()}
    return {'candidate': ranking[0]['label'] if certain else None,
            'engine': engine, 'network': network_meta,
            'feature_details': feature_details,
            'nearest_family': ranking[0]['label'],
            'best_distance_family': best_distance_family,
            'status': 'candidate' if certain else 'uncertain',
            'ranking': ranking, 'neural_ranking': neural_ranking, 'decision_guard': decision_guard,
            'margin': margin, 'min_margin': min_margin,
            'top_score': top_score, 'min_prob': min_prob,
            'accepted': certain, 'reject_reason': reasons,
            'margin_kind': ('consensus_top_two_difference' if decision_guard['applied'] else 'softmax_top_two_difference') if engine == 'neural' else 'relative_distance_gap',
            'distance_limit': radius,
            'reference_matches': matches,
            'known_external_references': known_reference_matches(record),
            'nearest_samples': [{'source': references[i]['source'], 'family': references[i]['family'],
                                 'distance': float(distance[i])} for i in np.argsort(distance)[:3]],
            'note': ('候选只由神经网络排序；近邻仅作相似性诊断与冲突提示，不改变分数、排序或拒识。score未校准，不是身份正确概率；最近参考型号不等于识别型号。距离只用于诊断和启发式拒识；不能可靠排除未知家族。参考库命中不属于独立验证。' if engine == 'neural' else '距离越小越相似，不是身份概率。拒识门槛为启发式，不能可靠识别未知家族。参考库命中不属于独立验证。')}


def inventory(records, skipped):
    return {'valid_samples': len(records), 'families': dict(Counter(r['family'] for r in records)),
            'models': dict(Counter(r['model'] for r in records)), 'skipped': skipped,
            'labels': '家族及型号来自参考文件名，为用户提供标签；待测文件名不参与预测。'}
