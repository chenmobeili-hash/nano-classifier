"""NLP engine: hashed char/word n-gram TF-IDF + linear family classifiers.

纯 NumPy，无第三方 NLP 库、无预训练下载。与 neural_core.py 并列，互不影响。

设计要点（与 balanced 神经网络版的差异）：
  * 输入不再是 135 维手工行为/序列/选择特征，而是**文本本身的 n-gram**：
    中文按字符 2/3/4-gram（免分词），拉丁按词 1/2-gram，题号作为命名空间前缀。
  * TF 用 sublinear（1 + log tf），IDF 只用训练折计算，向量做 L2 归一化。
  * 三个分类器共享同一套向量，便于同折对照：
      lr      多项逻辑回归（默认）
      centroid 余弦质心
      mnb     多项朴素贝叶斯
  * 家族标签仍来自参考文件名；程序不能验证来源。
"""
import hashlib
import json
import re
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from feature_profiles import normalized_answers

VERSION = 'nlp-ngram-1'
DEFAULT_MODEL = Path(__file__).resolve().with_name('nlp-model.npz')
DIMENSIONS = 4096
CHAR_NS = (2, 3, 4)
WORD_NS = (1, 2)
# 神经网络版的 0.1 margin 门槛是按它自己的 softmax 量纲定的。线性模型的分数更平
# （95 份留出里 margin 中位数远低于 0.1），直接套用会把逐份覆盖率压到约 70%。
# 这里按"目标覆盖率 = 神经网络正式版的 96.84%"反解得到 0.0106（见 calibration-nlp.json），
# 不是按单个样本调出来的；需要神经网络口径时显式传 0.1。
CALIBRATED_MIN_MARGIN = 0.0106
DEFAULT_QUESTIONS = tuple('B%02d' % i for i in range(1, 13))
A_QUESTIONS = tuple('A%02d' % i for i in range(1, 13))
KINDS = ('lr', 'centroid', 'mnb')
WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")


def questions_for(include_a=False):
    return tuple(sorted(DEFAULT_QUESTIONS + (A_QUESTIONS if include_a else ())))


def _bucket(feature):
    """稳定哈希：不能直接用 hash()（每进程随机化）。"""
    return zlib.crc32(feature.encode('utf-8')) & (DIMENSIONS - 1)


def features(record, include_a=False):
    """返回 {bucket: 计数} —— 只读文本，不读标签/型号/文件名。"""
    answers = normalized_answers(record)
    bag = Counter()
    for q in questions_for(include_a):
        text = answers.get(q, '').lower()
        if not text:
            continue
        for n in CHAR_NS:
            for i in range(len(text) - n + 1):
                bag[_bucket(q + '\x1f' + text[i:i + n])] += 1
        words = WORD.findall(text)
        for n in WORD_NS:
            for i in range(len(words) - n + 1):
                bag[_bucket(q + '\x1e' + ' '.join(words[i:i + n]))] += 1
    return bag


def count_vector(record, include_a=False):
    """单份答卷 → (DIMENSIONS,) sublinear tf（未乘 IDF）。"""
    x = np.zeros(DIMENSIONS, dtype=np.float32)
    for bucket, count in features(record, include_a).items():
        x[bucket] = 1.0 + np.log(count)
    return x


def count_matrix(records, include_a=False):
    """整库 → (n, DIMENSIONS) sublinear tf 矩阵。"""
    return np.stack([count_vector(r, include_a) for r in records]) if records else np.zeros((0, DIMENSIONS), np.float32)


def idf_from(counts):
    """IDF 只用传入的这一折训练数据计算（留出折不参与）。"""
    documents = counts.shape[0]
    df = (counts > 0).sum(axis=0).astype(np.float64)
    return (np.log((1.0 + documents) / (1.0 + df)) + 1.0).astype(np.float32)


def apply_idf(counts, idf, l2=True):
    """乘 IDF 并逐行 L2 归一化；空行保持全零（后续由调用方决定是否拒识）。"""
    x = counts * idf
    if l2:
        norm = np.linalg.norm(x, axis=1, keepdims=True)
        x = np.divide(x, np.maximum(norm, 1e-12))
    return x.astype(np.float32)


def softmax(logits):
    logits = np.atleast_2d(logits)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- 分类器


def sample_weights(records, classes, boost=None):
    """与 neural_core 完全一致的权重：均衡家族 → 均衡家族内型号 → 均分同型号样本。"""
    counts = Counter((r['family'], r['model']) for r in records)
    model_counts = {f: len({m for family, m in counts if family == f}) for f in classes}
    boost = dict(boost or {})
    return np.array([boost.get(r['family'], 1.0) / (len(classes) * model_counts[r['family']] * counts[r['family'], r['model']])
                     for r in records], dtype=np.float64)


def fit_lr(x, y, classes, weights, seed=42, epochs=400, l2=.001, learning_rate=.05):
    """多项逻辑回归，全批量 Adam。固定轮数，不留早停余地。"""
    n, d = x.shape
    k = len(classes)
    rng = np.random.default_rng(seed)
    w = np.zeros((d, k), dtype=np.float64)
    b = np.zeros(k, dtype=np.float64)
    m = {'w': np.zeros_like(w), 'b': np.zeros_like(b)}
    v = {'w': np.zeros_like(w), 'b': np.zeros_like(b)}
    onehot = np.zeros((n, k))
    onehot[np.arange(n), y] = 1.0
    wts = weights / max(weights.sum(), 1e-12) * n
    for step in range(1, epochs + 1):
        p = softmax(x @ w + b)
        delta = (p - onehot) * wts[:, None] / n
        g = {'w': x.T @ delta + l2 * w, 'b': delta.sum(axis=0)}
        for key, target in (('w', w), ('b', b)):
            m[key] = .9 * m[key] + .1 * g[key]
            v[key] = .999 * v[key] + .001 * g[key] ** 2
            target -= learning_rate * (m[key] / (1 - .9 ** step)) / (np.sqrt(v[key] / (1 - .999 ** step)) + 1e-8)
    return {'w': w, 'b': b}


def fit_centroids(x, y, classes, weights):
    """余弦质心：类内加权平均后归一化。"""
    k = len(classes)
    cent = np.zeros((k, x.shape[1]))
    for index in range(k):
        mask = y == index
        if not mask.any():
            continue
        cent[index] = (x[mask] * weights[mask, None]).sum(axis=0)
    return {'centroids': cent / np.maximum(np.linalg.norm(cent, axis=1, keepdims=True), 1e-12)}


def fit_mnb(x, y, classes, weights):
    """多项朴素贝叶斯：特征权重当作计数，取对数后加平滑。"""
    k = len(classes)
    totals = np.zeros((k, x.shape[1]))
    priors = np.zeros(k)
    for index in range(k):
        mask = y == index
        totals[index] = (x[mask] * weights[mask, None]).sum(axis=0)
        priors[index] = np.log(max(weights[mask].sum(), 1e-12) / max(weights.sum(), 1e-12))
    smoothed = totals + 1e-6
    logprob = np.log(smoothed / smoothed.sum(axis=1, keepdims=True))
    return {'nb_logprob': logprob, 'nb_prior': priors}


def fit(x, y, classes, weights, kind='lr', seed=42, epochs=400, l2=.001, lr=.05):
    if kind == 'lr':
        return fit_lr(x, y, classes, weights, seed=seed, epochs=epochs, l2=l2, learning_rate=lr)
    if kind == 'centroid':
        return fit_centroids(x, y, classes, weights)
    if kind == 'mnb':
        return fit_mnb(x, y, classes, weights)
    raise ValueError('未知的 NLP 分类器：' + str(kind) + '；可选 ' + ', '.join(KINDS))


def predict_scores(x, params, kind='lr', temperature=20.0):
    x = np.atleast_2d(x)
    if kind == 'lr':
        return softmax(x @ np.asarray(params['w'], float) + np.asarray(params['b'], float))
    if kind == 'centroid':
        sim = x @ np.asarray(params['centroids'], float).T
        return softmax(sim * temperature)
    if kind == 'mnb':
        logits = x @ np.asarray(params['nb_logprob'], float).T + np.asarray(params['nb_prior'], float)
        return softmax(logits)
    raise ValueError('未知的 NLP 分类器：' + str(kind))


# ---------------------------------------------------------------- 距离与签名


def distances(x_query, x_reference):
    """NLP 空间的余弦距离（0–2），用于最近样本诊断与启发式拒识。"""
    similarity = np.asarray(x_query, float) @ np.asarray(x_reference, float).T
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def reference_signature(records):
    payload = sorted((r['family'], r['model'], r['fingerprint']) for r in records)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def train_model(records, kind='lr', include_a=False, seed=42, epochs=400, l2=.001, learning_rate=.05):
    """全流程训练：计数矩阵 → IDF → 分类器 → 可发布的模型 dict。"""
    classes = sorted({r['family'] for r in records})
    if len(classes) < 2:
        raise ValueError('训练至少需要两个家族。')
    counts = count_matrix(records, include_a)
    idf = idf_from(counts)
    x = apply_idf(counts, idf)
    y = np.array([classes.index(r['family']) for r in records])
    weights = sample_weights(records, classes)
    params = fit(x, y, classes, weights, kind, seed=seed, epochs=epochs, l2=l2, lr=learning_rate)
    # 三个分类器都存，方便同库对照；meta['kind'] 决定默认使用哪个。
    for other in KINDS:
        if other != kind and other not in params:
            params.update(fit(x, y, classes, weights, other, seed=seed, epochs=epochs, l2=l2, lr=learning_rate))
    model = dict(params, idf=idf, mean=counts.mean(axis=0), scale=np.maximum(counts.std(axis=0), .03))
    model['meta'] = {
        'version': VERSION, 'kind': kind, 'classes': classes, 'dimensions': DIMENSIONS,
        'char_ngrams': list(CHAR_NS), 'word_ngrams': list(WORD_NS),
        'include_a': bool(include_a), 'questions': list(questions_for(include_a)),
        'seed': seed, 'epochs': epochs, 'l2': l2, 'learning_rate': learning_rate,
        'reference_signature': reference_signature(records), 'samples': len(records),
        'sample_weighting': 'equal-family-then-model',
        'parameters': sum(int(np.asarray(v).size) for k, v in model.items() if k not in ('meta', 'mean', 'scale')),
        'training_accuracy': float(np.mean(predict_scores(x, model, kind).argmax(axis=1) == y)),
    }
    return model


def save_model(path, model):
    payload = {k: np.asarray(v).astype(np.float32) for k, v in model.items() if k != 'meta'}
    payload['metadata'] = np.array(json.dumps(model['meta'], ensure_ascii=False))
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **payload)


def load_model(path=DEFAULT_MODEL):
    try:
        with np.load(path, allow_pickle=False) as file:
            meta = json.loads(str(file['metadata'].item()))
            model = {k: file[k].astype(float) for k in file.files if k != 'metadata'}
    except FileNotFoundError as exc:
        raise ValueError('未找到 NLP 权重，请先运行 python train_nlp.py 生成 nlp-model.npz。') from exc
    if meta.get('version') != VERSION:
        raise ValueError('NLP 权重版本不兼容：' + str(meta.get('version')))
    classes = meta.get('classes')
    if not isinstance(classes, list) or len(classes) < 2:
        raise ValueError('NLP 权重缺少类别信息。')
    if model['idf'].shape != (DIMENSIONS,):
        raise ValueError('NLP 权重维度错误。')
    if not np.isfinite(model['idf']).all() or model['idf'].min() <= 0:
        raise ValueError('NLP 权重的 IDF 非法。')
    for key in ('w', 'b', 'centroids', 'nb_logprob'):
        if key in model and not np.isfinite(model[key]).all():
            raise ValueError('NLP 权重包含非有限值：' + key)
    model['meta'] = meta
    return model


def predict_record(record, model, kind=None):
    """单份答卷 → (1, 类数) 家族分。"""
    counts = count_vector(record, model['meta'].get('include_a', False))
    x = apply_idf(counts[None, :], model['idf'])
    return predict_scores(x, model, kind or model['meta']['kind'])


def record_distance(record, records, model):
    """单份答卷到整库的 NLP 余弦距离。"""
    counts = count_matrix(records, model['meta'].get('include_a', False))
    x = apply_idf(counts, model['idf'])
    q = apply_idf(count_vector(record, model['meta'].get('include_a', False))[None, :], model['idf'])
    return distances(q, x)[0]
