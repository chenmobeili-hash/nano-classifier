"""题块级单头自注意力（V1）：12 个题块之间做一次 self-attention 再分类。

与 nlp_core 的关系：nlp_core 是"整卷词袋 + 线性模型"（题与题之间没有交互）；
这里是"每个题块一个向量 → 共享投影 → 单头自注意力跨题块 → 池化 → 5 类"，
让模型能学"B11 的矛盾处理要参照 B10 的拒答风格"这类**题间交互**。

刻意做小：d=16、单头、一层、无位置编码（题块本身有序，但注意力对排列不变，
这里顺序信息由"第几个题块"隐式体现在各自独立的特征上）。

`use_attention=False` 时退化成"保留残差、不做注意力"的对照组，
用于把收益归因到注意力本身，而不是归因到"多了一层投影"。

纯 numpy，手写前向/反向；`python attn_core.py --gradcheck` 做有限差分校验。
"""
import argparse
import hashlib
import json
import zlib
from collections import Counter
from pathlib import Path

import numpy as np

from feature_profiles import normalized_answers

VERSION = 'attn-block-1'
DEFAULT_MODEL = Path(__file__).resolve().with_name('attn-model.npz')
QUESTIONS = tuple('B%02d' % i for i in range(1, 13))
BLOCK_DIMS = 512
HIDDEN = 16
CHAR_NS = (2, 3)
CLASSES_DEFAULT = ('Claude', 'DeepSeek', 'GLM', 'GPT', 'Qwen')


def _bucket(text):
    return zlib.crc32(text.encode('utf-8')) & (BLOCK_DIMS - 1)


def block_matrix(record):
    """单份答卷 → (12, 512) 题块矩阵；每块 sublinear TF 后各自 L2 归一化。"""
    answers = normalized_answers(record)
    matrix = np.zeros((len(QUESTIONS), BLOCK_DIMS), dtype=np.float64)
    for index, question in enumerate(QUESTIONS):
        text = answers.get(question, '').lower()
        if not text:
            continue
        bag = Counter()
        for n in CHAR_NS:
            for i in range(len(text) - n + 1):
                bag[_bucket(text[i:i + n])] += 1
        for bucket, count in bag.items():
            matrix[index, bucket] += np.log1p(count)
        norm = np.linalg.norm(matrix[index])
        if norm > 0:
            matrix[index] /= norm
    return matrix


def block_matrices(records):
    return np.stack([block_matrix(r) for r in records]) if records else np.zeros((0, len(QUESTIONS), BLOCK_DIMS))


# ---------------------------------------------------------------- 前向/反向


def softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


def init_params(classes, hidden=HIDDEN, seed=42, d_in=BLOCK_DIMS):
    rng = np.random.default_rng(seed)
    scale_p, scale_a = 1.0 / np.sqrt(d_in), 1.0 / np.sqrt(hidden)
    return {'Wp': rng.normal(size=(d_in, hidden)) * scale_p, 'bp': np.zeros(hidden),
            'Wq': rng.normal(size=(hidden, hidden)) * scale_a, 'Wk': rng.normal(size=(hidden, hidden)) * scale_a,
            'Wv': rng.normal(size=(hidden, hidden)) * scale_a,
            'Wo': rng.normal(size=(hidden, len(classes))) * scale_a, 'bo': np.zeros(len(classes))}


def band_mask(blocks, width):
    """局部（滑窗）注意力掩码：每个题块只能看 |i-j| <= width 的题块。

    width=None 时不加掩码（全局注意力，即 V1 的原始行为）。
    返回 (1, blocks, blocks) 的加法掩码：允许位置为 0，禁止位置为 -1e9。
    题块本身有序（B01…B12），所以"相邻"是有语义的：
    B02/B03/B04 都是随机性任务，B10/B11 都是对抗性指令任务。
    """
    if width is None:
        return None
    index = np.arange(blocks)
    allowed = np.abs(index[:, None] - index[None, :]) <= width
    return np.where(allowed, 0.0, -1e9)[None, :, :]


def forward(x, params, use_attention=True, mask=None):
    """x: (B, 12, 512) → 概率 (B, 类别数)；同时返回反向所需缓存。"""
    h = x @ params['Wp'] + params['bp']                     # (B, n, d)
    if use_attention:
        q = h @ params['Wq']
        k = h @ params['Wk']
        v = h @ params['Wv']
        logits = q @ np.swapaxes(k, -1, -2) / np.sqrt(h.shape[-1])
        if mask is not None:
            logits = logits + mask
        attn = softmax(logits, axis=-1)
        z = attn @ v
        r = z + h                                            # 残差
    else:
        q = attn = z = v = k = None
        r = h
    pooled = r.mean(axis=1)                                  # (B, d)
    out = pooled @ params['Wo'] + params['bo']
    return softmax(out), {'x': x, 'h': h, 'q': q, 'k': k, 'v': v, 'attn': attn, 'z': z,
                          'r': r, 'pooled': pooled, 'probs': None}


def loss_and_grad(x, y, params, weights, use_attention=True, l2=.001, mask=None):
    probs, cache = forward(x, params, use_attention, mask)
    batch, blocks = x.shape[0], x.shape[1]
    n = batch
    loss = -np.sum(weights * np.log(np.maximum(probs[np.arange(n), y], 1e-15))) / max(weights.sum(), 1e-12)
    for key in ('Wp', 'Wq', 'Wk', 'Wv', 'Wo'):
        if key in params and (use_attention or key not in ('Wq', 'Wk', 'Wv')):
            loss += l2 * .5 * np.sum(params[key] ** 2)

    delta = probs.copy()
    delta[np.arange(n), y] -= 1
    delta *= (weights / max(weights.sum(), 1e-12))[:, None]        # (B, K)
    grads = {'Wo': cache['pooled'].T @ delta + (l2 * params['Wo'] if use_attention else l2 * params['Wo']),
             'bo': delta.sum(axis=0)}
    dpool = delta @ params['Wo'].T                                # (B, d)
    dr = np.broadcast_to(dpool[:, None, :] / blocks, cache['r'].shape).copy()

    if use_attention:
        h, attn, v = cache['h'], cache['attn'], cache['v']
        dz = dr
        dattn = dz @ np.swapaxes(v, -1, -2)
        dv = np.swapaxes(attn, -1, -2) @ dz
        dlogits = attn * (dattn - (dattn * attn).sum(axis=-1, keepdims=True))
        dlogits /= np.sqrt(h.shape[-1])
        dq = dlogits @ cache['k']
        dk = np.swapaxes(dlogits, -1, -2) @ cache['q']
        grads['Wq'] = np.einsum('bnd,bne->de', h, dq) + l2 * params['Wq']
        grads['Wk'] = np.einsum('bnd,bne->de', h, dk) + l2 * params['Wk']
        grads['Wv'] = np.einsum('bnd,bne->de', h, dv) + l2 * params['Wv']
        dh = dq @ params['Wq'].T + dk @ params['Wk'].T + dv @ params['Wv'].T + dr
    else:
        dh = dr
        # 关闭注意力时这些参数不参与计算，返回零梯度以保持优化器简单。
        for key in ('Wq', 'Wk', 'Wv'):
            grads[key] = np.zeros_like(params[key])

    grads['Wp'] = np.einsum('bnd,bne->de', x, dh) + l2 * params['Wp']
    grads['bp'] = dh.sum(axis=(0, 1))
    return float(loss), grads


def train(records, classes, use_attention=True, hidden=HIDDEN, seed=42, epochs=400, l2=.001,
          learning_rate=.01, x=None, weights=None, mask=None, mask_width=None):
    from nlp_core import sample_weights
    x = block_matrices(records) if x is None else x
    y = np.array([classes.index(record['family']) for record in records])
    weights = sample_weights(records, classes) if weights is None else weights
    params = init_params(classes, hidden=hidden, seed=seed)
    moments = {key: np.zeros_like(value) for key, value in params.items()}
    variance = {key: np.zeros_like(value) for key, value in params.items()}
    for step in range(1, epochs + 1):
        _, grads = loss_and_grad(x, y, params, weights, use_attention, l2, mask)
        for key in params:
            moments[key] = .9 * moments[key] + .1 * grads[key]
            variance[key] = .999 * variance[key] + .001 * grads[key] ** 2
            params[key] -= learning_rate * (moments[key] / (1 - .9 ** step)) / (np.sqrt(variance[key] / (1 - .999 ** step)) + 1e-8)
    loss, _ = loss_and_grad(x, y, params, weights, use_attention, l2, mask)
    params['meta'] = {'version': VERSION, 'use_attention': bool(use_attention),
                      'mask_width': mask_width,  # None = 全局注意力
                      'blocks': len(QUESTIONS), 'block_dims': BLOCK_DIMS, 'hidden': hidden,
                      'char_ngrams': list(CHAR_NS), 'seed': seed, 'epochs': epochs, 'l2': l2,
                      'learning_rate': learning_rate, 'samples': len(records),
                      'parameters': int(sum(np.asarray(v).size for k, v in params.items() if k != 'meta')),
                      'training_loss': loss,
                      'training_accuracy': float(np.mean(forward(x, params, use_attention, mask)[0].argmax(axis=1) == y))}
    return params


def predict_scores(x, params, use_attention=None, mask=None):
    use = params['meta']['use_attention'] if use_attention is None else use_attention
    return forward(np.atleast_3d(x) if x.ndim == 2 else x, params, use, mask)[0]


def predict_record(record, params, use_attention=None, mask=None):
    return predict_scores(block_matrix(record)[None, :, :], params, use_attention, mask)


def flat_vectors(records):
    """把 (n, 12, 512) 题块矩阵摊平成 (n, 6144) 并逐行 L2 归一化，仅用于余弦距离诊断。"""
    matrix = block_matrices(records).reshape(len(records), -1)
    norm = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.maximum(norm, 1e-12)


def record_distance(record, records):
    reference = flat_vectors(records)
    query = block_matrix(record).ravel()
    query = query / max(float(np.linalg.norm(query)), 1e-12)
    return 1.0 - np.clip(reference @ query, -1.0, 1.0)


def reference_signature(records):
    payload = sorted((r['family'], r['model'], r['fingerprint']) for r in records)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def save_model(path, model):
    payload = {key: np.asarray(value).astype(np.float32) for key, value in model.items() if key != 'meta'}
    payload['metadata'] = np.array(json.dumps(model['meta'], ensure_ascii=False))
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **payload)


def load_model(path=DEFAULT_MODEL):
    try:
        with np.load(path, allow_pickle=False) as file:
            meta = json.loads(str(file['metadata'].item()))
            model = {key: file[key].astype(float) for key in file.files if key != 'metadata'}
    except FileNotFoundError as exc:
        raise ValueError('未找到注意力权重，请先运行 python train_attn.py。') from exc
    if meta.get('version') != VERSION:
        raise ValueError('注意力权重版本不兼容：' + str(meta.get('version')))
    for key, shape in (('Wp', (BLOCK_DIMS, meta['hidden'])), ('Wo', (meta['hidden'], len(meta['classes'])))):
        if model[key].shape != shape:
            raise ValueError('注意力权重形状错误：' + key)
    for key, value in model.items():
        if not np.isfinite(value).all():
            raise ValueError('注意力权重含非有限值：' + key)
    model['meta'] = meta
    return model


# ---------------------------------------------------------------- 梯度校验


def gradcheck(seed=0, blocks=3, dims=4, hidden=3, classes=3, l2=.01, eps=1e-5):
    """有限差分校验（含"关注意力"与"带状掩码"两种情形）。"""
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(2, blocks, dims))
    y = np.array([0, 2])
    weights = np.array([1.0, 1.0])
    params = {'Wp': rng.normal(size=(dims, hidden)), 'bp': rng.normal(size=hidden),
              'Wq': rng.normal(size=(hidden, hidden)), 'Wk': rng.normal(size=(hidden, hidden)),
              'Wv': rng.normal(size=(hidden, hidden)), 'Wo': rng.normal(size=(hidden, classes)),
              'bo': rng.normal(size=classes)}
    banded = band_mask(blocks, 1)
    per_key = {}
    for label, use_attention, mask in (('', True, None), ('(band1)', True, banded), ('(noattn)', False, None)):
        _, grads = loss_and_grad(x, y, params, weights, use_attention, l2, mask)
        for key, value in params.items():
            if not use_attention and key in ('Wq', 'Wk', 'Wv'):
                continue
            worst = 0.0
            for index in np.ndindex(value.shape):
                original = value[index]
                value[index] = original + eps
                plus, _ = loss_and_grad(x, y, params, weights, use_attention, l2, mask)
                value[index] = original - eps
                minus, _ = loss_and_grad(x, y, params, weights, use_attention, l2, mask)
                value[index] = original
                numeric = (plus - minus) / (2 * eps)
                analytic = grads[key][index]
                denom = max(abs(numeric), abs(analytic), 1e-8)
                worst = max(worst, abs(numeric - analytic) / denom)
            per_key['%s%s' % (key, label)] = worst
    return max(per_key.values()), per_key


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='题块级单头自注意力；--gradcheck 做有限差分校验。')
    parser.add_argument('--gradcheck', action='store_true')
    parser.add_argument('--eps', type=float, default=1e-5)
    args = parser.parse_args()
    if args.gradcheck:
        error, per_key = gradcheck(eps=args.eps)
        for key, value in per_key.items():
            print('  %-14s %.3e' % (key, value))
        print('最大相对误差 %.3e（阈值 1e-4）' % error)
        print('PASS' if error < 1e-4 else 'FAIL')
