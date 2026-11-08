"""A bounded neural residual on the unchanged TF-IDF x MLP geometric fusion.

Local inputs reuse the TRAINING-FOLD NLP classifier's n-gram evidence. No new
embedding dictionary, query fitting, nearest-neighbour votes, or A-score inputs.
NumPy training/inference; the neural branch has 56 parameters for five families.
"""
import json
from pathlib import Path
import numpy as np
import nlp_core as N
import neural_core as M
from feature_profiles import normalized_answers

VERSION = 'local-residual-gate-1'
CENTERS, EPOCHS, LR, L2, MAX_DELTA = 24, 200, .02, .02, .25
VARIANTS = ('gated', 'uniform', 'linear')


def buckets(records):
    """Sample ORIGINAL character positions; each has genuine +/-1 neighbours."""
    out = np.full((len(records), 12, CENTERS, 3, 3), -1, dtype=np.int32)
    for row, record in enumerate(records):
        answers = normalized_answers(record)
        for qi in range(12):
            q = 'B%02d' % (qi + 1)
            text = answers.get(q, '').lower()
            length = max(0, len(text) - 1)
            positions = np.unique(np.linspace(0, length - 1, min(CENTERS, length)).astype(int)) if length else []
            for si, pos in enumerate(positions):
                for ki, offset in enumerate((-1, 0, 1)):
                    start = pos + offset
                    for ni, size in enumerate(N.CHAR_NS):
                        if 0 <= start and start + size <= len(text):
                            out[row, qi, si, ki, ni] = N._bucket(q + '\x1f' + text[start:start+size])
    return out


def local_inputs(index, nlp):
    valid = index >= 0
    safe = np.maximum(index, 0)
    evidence = nlp['w'][safe] * nlp['idf'][safe, None] * valid[..., None]
    evidence = evidence.sum(axis=-2) / np.maximum(valid.sum(axis=-1, keepdims=True), 1)
    evidence -= evidence.mean(axis=-1, keepdims=True)
    u = np.tanh(.5 * evidence)
    mask = valid[..., 1, :].any(axis=-1)
    counts = mask.sum(axis=-1, keepdims=True)
    questions = (counts[..., 0] > 0).sum(axis=1, keepdims=True)
    pool = mask / np.maximum(counts, 1) / np.maximum(questions[:, :, None], 1)
    return u, pool


def fused_logits(p_mlp, p_nlp):
    return .5 * (np.log(np.maximum(p_mlp, 1e-12)) + np.log(np.maximum(p_nlp, 1e-12)))


def support(base):
    top = np.sort(base, axis=1)[:, -2:]
    return np.maximum(0., 1. - (top[:, 1] - top[:, 0]))[:, None]


def init(k=5, variant='gated'):
    if variant == 'linear':
        return {'W': np.zeros((2*k, k)), 'b': np.zeros(k), 'temperature': np.zeros(1)}
    return {'kernel': np.full((3, k), 1/3), 'hidden_bias': np.zeros(k),
            'gate': np.zeros(k), 'gate_bias': np.zeros(1),
            'W': np.zeros((k, k)), 'b': np.zeros(k)}


def forward(data, base, p, variant='gated'):
    u, pool = data
    if variant == 'linear':
        center = np.einsum('nqs,nqsd->nd', pool, u[:, :, :, 1])
        nearby = np.einsum('nqs,nqsd->nd', pool, .5*(u[:, :, :, 0]+u[:, :, :, 2]))
        z = np.concatenate((center, nearby), axis=1)
        centered = base-base.mean(axis=1, keepdims=True)
        raw = z@p['W']+p['b']+p['temperature'][0]*centered
        cache = (z, centered)
    else:
        h = np.tanh(np.einsum('nqskd,kd->nqsd', u, p['kernel'])+p['hidden_bias'])
        gate = (1/(1+np.exp(-np.clip(h@p['gate']+p['gate_bias'][0], -40, 40)))
                if variant == 'gated' else np.full(pool.shape, .5))
        z = np.einsum('nqs,nqsd->nd', pool*gate, h)
        raw = z@p['W']+p['b']
        cache = (z, h, gate)
    bounded = np.tanh(raw)
    delta = MAX_DELTA*support(base)*bounded
    return N.softmax(base+delta), delta, (cache, bounded)


def loss_grad(data, base, y, weights, p, variant='gated', l2=L2):
    prob, _, (cache, bounded) = forward(data, base, p, variant)
    loss = -np.sum(weights*np.log(np.maximum(prob[np.arange(len(y)), y], 1e-300)))
    dg = prob.copy(); dg[np.arange(len(y)), y] -= 1; dg *= weights[:, None]
    dr = dg*MAX_DELTA*support(base)*(1-bounded*bounded)
    z = cache[0]
    grad = {key: np.zeros_like(value) for key, value in p.items()}
    grad['W'] = z.T@dr; grad['b'] = dr.sum(axis=0)
    if variant == 'linear':
        grad['temperature'][0] = np.sum(dr*cache[1])
    else:
        _, h, gate = cache
        pool = data[1]
        dz = dr@p['W'].T
        dh = (pool*gate)[..., None]*dz[:, None, None, :]
        if variant == 'gated':
            ds = pool*np.einsum('nd,nqsd->nqs', dz, h)*gate*(1-gate)
            grad['gate'] = np.einsum('nqs,nqsd->d', ds, h)
            grad['gate_bias'][0] = ds.sum()
            dh += ds[..., None]*p['gate']
        dt = dh*(1-h*h)
        grad['kernel'] = np.einsum('nqskd,nqsd->kd', data[0], dt)
        grad['hidden_bias'] = dt.sum(axis=(0, 1, 2))
    prior = init(len(p['b']), variant)
    for key in p:
        if variant == 'uniform' and key in ('gate', 'gate_bias'):
            continue
        diff = p[key]-prior[key]
        loss += .5*l2*np.sum(diff*diff)
        grad[key] += l2*diff
    return float(loss), grad


def fit(data, base, records, classes, variant='gated'):
    if variant not in VARIANTS:
        raise ValueError('Unknown residual variant')
    y = np.array([classes.index(r['family']) for r in records])
    weights = N.sample_weights(records, classes)
    p = init(len(classes), variant)
    first = loss_grad(data, base, y, weights, p, variant)[0]
    m = {k: np.zeros_like(v) for k, v in p.items()}
    v = {k: np.zeros_like(a) for k, a in p.items()}
    for step in range(1, EPOCHS+1):
        _, grad = loss_grad(data, base, y, weights, p, variant)
        for key in p:
            m[key] = .9*m[key]+.1*grad[key]
            v[key] = .999*v[key]+.001*grad[key]**2
            p[key] -= LR*(m[key]/(1-.9**step))/(np.sqrt(v[key]/(1-.999**step))+1e-8)
    prob, delta, _ = forward(data, base, p, variant)
    return p, {'parameters_allocated': sum(a.size for a in p.values()),
               'parameters_trainable': sum(a.size for k, a in p.items() if variant != 'uniform' or k not in ('gate', 'gate_bias')),
               'initial_loss': first, 'final_loss': loss_grad(data, base, y, weights, p, variant)[0],
               'base_correct': int((base.argmax(axis=1) == y).sum()),
               'residual_correct': int((prob.argmax(axis=1) == y).sum()),
               'max_absolute_delta': float(np.abs(delta).max())}


def select_on_validation(base, corrected, records, classes):
    y = np.array([classes.index(r['family']) for r in records])
    old, new = base.argmax(axis=1) == y, corrected.argmax(axis=1) == y
    w = N.sample_weights(records, sorted({r['family'] for r in records}))
    w /= w.sum()
    pb = N.softmax(base)
    nll_base = float(-np.sum(w*np.log(np.maximum(pb[np.arange(len(y)), y], 1e-300))))
    nll_new = float(-np.sum(w*np.log(np.maximum(corrected[np.arange(len(y)), y], 1e-300))))
    fixes, breaks = int((~old & new).sum()), int((old & ~new).sum())
    enabled = len(records) >= 12 and len(set(r['family'] for r in records)) >= 3 and fixes >= 1 and breaks == 0 and nll_new <= nll_base
    return {'enabled': bool(enabled), 'validation_count': len(records), 'fixes': fixes,
            'breaks': breaks, 'base_nll': nll_base, 'residual_nll': nll_new,
            'rule': '>=12 samples, >=3 families, >=1 fix, zero breaks, non-increasing weighted NLL'}


def save_bundle(path, mlp, nlp, params, selection, classes, signature, threshold):
    arrays = {'mlp_'+k: v for k, v in mlp.items() if k != 'meta'}
    arrays.update({'nlp_'+k: v for k, v in nlp.items()})
    arrays.update({'gate_'+k: v for k, v in params.items()})
    meta = {'version': VERSION, 'classes': classes, 'reference_signature': signature,
            'selection': selection, 'min_margin': threshold, 'variant': 'gated',
            'max_delta': MAX_DELTA, 'epochs': EPOCHS, 'centers': CENTERS}
    arrays['metadata'] = np.array(json.dumps(meta, ensure_ascii=False))
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)


def load_bundle(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata'].item()))
        arrays = {k: archive[k].astype(float) for k in archive.files if k != 'metadata'}
    if meta.get('version') != VERSION or meta.get('max_delta') != MAX_DELTA or meta.get('centers') != CENTERS:
        raise ValueError('Residual model version/configuration mismatch')
    if any(not np.isfinite(a).all() for a in arrays.values()):
        raise ValueError('Residual weights contain non-finite values')
    return meta, {prefix: {k[len(prefix)+1:]: a for k, a in arrays.items() if k.startswith(prefix+'_')}
                  for prefix in ('mlp', 'nlp', 'gate')}


def gradcheck():
    rng = np.random.default_rng(17)
    u = rng.normal(size=(3, 12, 2, 3, 5))
    pool = np.full((3, 12, 2), 1/24); pool[0, 0] = 0
    base = rng.normal(0, .1, (3, 5)); base[0] = [4, 0, 0, 0, 0]
    y, weights = np.array([0, 2, 4]), np.array([.2, .3, .5])
    result = {}
    for variant in VARIANTS:
        p = init(5, variant)
        for key in p:
            p[key] += rng.normal(0, .1, p[key].shape)
        _, gradients = loss_grad((u, pool), base, y, weights, p, variant)
        absolute, relative = [], []
        for key in p:
            for index in np.ndindex(p[key].shape):
                old = p[key][index]
                p[key][index] = old+1e-5
                plus = loss_grad((u, pool), base, y, weights, p, variant)[0]
                p[key][index] = old-1e-5
                minus = loss_grad((u, pool), base, y, weights, p, variant)[0]
                p[key][index] = old
                numeric, analytic = (plus-minus)/2e-5, gradients[key][index]
                absolute.append(abs(numeric-analytic))
                relative.append(abs(numeric-analytic)/max(1e-8, abs(numeric)+abs(analytic)))
        result[variant] = {'max_absolute_error': max(absolute), 'max_relative_error': max(relative), 'coordinates': len(absolute)}
        assert max(absolute) < 1e-7
    initial = init()
    prob, delta, _ = forward((u, pool), base, initial)
    assert np.array_equal(prob, N.softmax(base)) and np.array_equal(delta, np.zeros_like(delta))
    for variant in VARIANTS:
        p = init(5, variant); p['W'] += 20; p['b'] += 10
        prob, delta, _ = forward((u, pool), base, p, variant)
        assert np.isfinite(prob).all() and np.max(np.abs(delta)) <= MAX_DELTA+1e-12
        assert np.array_equal(prob[0], N.softmax(base)[0])
        assert np.allclose(prob.sum(axis=1), 1)
    result['invariants'] = 'zero-head exact identity, bounded correction, strong-margin identity, finite probabilities: PASS'
    return result


if __name__ == '__main__':
    print(json.dumps(gradcheck(), indent=2))
