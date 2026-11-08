"""131 -> 8 tanh -> family softmax; NumPy only, no historical weights."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from feature_profiles import blocks

VERSION = 'balanced-neural-1'
DEFAULT_MODEL = Path(__file__).resolve().with_name('neural-balanced.npz')
GROUP_WEIGHTS = {'style': .342, 'behavior': .207, 'sequence': .207, 'choices': .144, 'extra': .10}
DIMENSIONS = 135


def vector(record):
    values = blocks(record)
    x = np.concatenate([values[name] for name in GROUP_WEIGHTS])
    if x.shape != (DIMENSIONS,) or not np.isfinite(x).all():
        raise ValueError('神经网络需要131维有限特征。')
    return x


def reference_signature(records):
    # Metadata provides training labels, but never enters the answer feature vector.
    payload = sorted((r['family'], r['model'], r['fingerprint']) for r in records)
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False).encode()).hexdigest()


def input_gain():
    sizes = {'style': 42, 'behavior': 22, 'sequence': 33, 'choices': 34, 'extra': 4}
    return np.concatenate([np.full(sizes[name], np.sqrt(weight*DIMENSIONS/sizes[name]))
                           for name, weight in GROUP_WEIGHTS.items()])


def transform(x, model):
    return np.clip((x-model['mean'])/model['scale'], -3, 3)*model['gain']


def forward(x, params):
    h = np.tanh(x@params['w1']+params['b1'])
    logits = h@params['w2']+params['b2']
    exp = np.exp(logits-logits.max(axis=1, keepdims=True))
    return h, exp/exp.sum(axis=1, keepdims=True)


def loss_and_grad(x, y, params, weights, l2=.01):
    h, scores = forward(x, params)
    loss = -np.sum(weights*np.log(np.maximum(scores[np.arange(len(y)), y], 1e-15)))
    loss += l2*.5*(np.sum(params['w1']**2)+np.sum(params['w2']**2))
    delta = scores.copy()
    delta[np.arange(len(y)), y] -= 1
    delta *= weights[:, None]
    hidden_delta = (delta@params['w2'].T)*(1-h*h)
    grads = {'w1': x.T@hidden_delta+l2*params['w1'], 'b1': hidden_delta.sum(axis=0),
             'w2': h.T@delta+l2*params['w2'], 'b2': delta.sum(axis=0)}
    return float(loss), grads


def train_network(records, seed=42, epochs=400, hidden=8, family_boost=None):
    if type(epochs) is not int or not 1 <= epochs <= 10000:
        raise ValueError('训练轮数必须为1至10000的整数。')
    classes = sorted({r['family'] for r in records})
    if len(classes) < 2:
        raise ValueError('训练至少需要两个家族。')
    x = np.stack([vector(r) for r in records])
    y = np.array([classes.index(r['family']) for r in records])
    # Equal families, equal models within each family, equal samples per model.
    counts = Counter((r['family'], r['model']) for r in records)
    model_counts = {f: len({m for family, m in counts if family == f}) for f in classes}
    boost = dict(family_boost or {})
    weights = np.array([boost.get(r['family'], 1.0)/(len(classes)*model_counts[r['family']]*counts[r['family'], r['model']]) for r in records])
    model = {'mean': x.mean(axis=0), 'scale': np.maximum(x.std(axis=0), .03), 'gain': input_gain()}
    z = transform(x, model)
    rng = np.random.default_rng(seed)
    params = {'w1': rng.normal(size=(DIMENSIONS, hidden))/np.sqrt(DIMENSIONS), 'b1': np.zeros(hidden),
              'w2': rng.normal(size=(hidden, len(classes)))/np.sqrt(hidden), 'b2': np.zeros(len(classes))}
    m, v = {k: np.zeros_like(a) for k, a in params.items()}, {k: np.zeros_like(a) for k, a in params.items()}
    # Fixed schedule: evaluation queries never choose epochs, scale or checkpoints.
    for step in range(1, epochs+1):
        _, gradients = loss_and_grad(z, y, params, weights)
        for k in params:
            m[k] = .9*m[k]+.1*gradients[k]
            v[k] = .999*v[k]+.001*gradients[k]**2
            params[k] -= .01*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
    loss, _ = loss_and_grad(z, y, params, weights)
    model.update(params)
    model['meta'] = {'version': VERSION, 'profile': 'balanced', 'classes': classes,
                     'architecture': [DIMENSIONS, hidden, len(classes)], 'seed': seed, 'epochs': epochs,
                     'l2': .01, 'learning_rate': .01, 'reference_signature': reference_signature(records),
                     'samples': len(records), 'parameters': sum(a.size for a in params.values()),
                     'training_loss': loss, 'sample_weighting': 'equal-family-then-model', 'hidden': hidden, 'family_boost': boost,
                     'training_accuracy': float(np.mean(forward(z, params)[1].argmax(axis=1) == y))}
    return model


def predict_scores(record, model):
    return forward(transform(vector(record)[None, :], model), model)[1][0]


def save_network(path, model):
    with Path(path).open('xb') as f:
        np.savez_compressed(f, **{k: v.astype(np.float32) for k, v in model.items() if k != 'meta'},
                            metadata=np.array(json.dumps(model['meta'], ensure_ascii=False)))


def load_network(path=DEFAULT_MODEL):
    try:
        with np.load(path, allow_pickle=False) as file:
            meta = json.loads(str(file['metadata'].item()))
            model = {k: file[k].astype(float) for k in ('mean', 'scale', 'gain', 'w1', 'b1', 'w2', 'b2')}
    except FileNotFoundError as exc:
        raise ValueError('未找到新神经网络权重，请先运行 train_windows.bat 或 python train.py。') from exc
    classes = meta.get('classes')
    if meta.get('version') != VERSION or not isinstance(classes, list) or len(classes) < 2 or len(set(classes)) != len(classes):
        raise ValueError('神经网络权重版本或类别不兼容。')
    hidden = int(meta['architecture'][1]) if isinstance(meta.get('architecture'), list) and len(meta['architecture']) == 3 else 8
    shapes = {'mean': (DIMENSIONS,), 'scale': (DIMENSIONS,), 'gain': (DIMENSIONS,),
              'w1': (DIMENSIONS, hidden), 'b1': (hidden,), 'w2': (hidden, len(classes)), 'b2': (len(classes),)}
    if any(model[k].shape != shape or not np.isfinite(model[k]).all() for k, shape in shapes.items()):
        raise ValueError('神经网络权重维度错误或包含非有限值。')
    if (model['scale'] <= 0).any() or (model['gain'] <= 0).any():
        raise ValueError('神经网络缩放参数必须为正。')
    model['meta'] = meta
    return model
