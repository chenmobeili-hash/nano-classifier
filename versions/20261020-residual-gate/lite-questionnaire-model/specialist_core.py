"""A shared neural classifier plus sparse, class-specific evidence weights.

Feature selection, scaling, and all learned weights use training rows only.
No provider names, filenames, collection modes, or A-task scores are inputs.
"""
import math
import re
import numpy as np
from behavior_features import BEHAVIOR_FEATURE_NAMES, behavior_feature_map
from sequence_features import SUMMARY_FEATURE_NAMES, summary_features, sequence_feature_map

ARCHITECTURE = 'weighted-specialists-3'
EVIDENCE_VERSION = 'b-evidence-1'
EVIDENCE_NAMES = tuple(
    BEHAVIOR_FEATURE_NAMES[int(name.split('_')[1])] if name.startswith('behavior_') else name
    for name in SUMMARY_FEATURE_NAMES
) + (
    'B08_numeric_example', 'B09_alternative_marker', 'B11_concise_conflict',
    'B11_conflict_without_literal', 'B12_log_words', 'B12_log_paragraphs',
    'B12_address_fraction', 'B12_contraction_fraction', 'B12_evenness_vocabulary',
    'cross_period_text_length', 'cross_period_text_brevity',
    'cross_period_concise_conflict', 'cross_period_algorithm',
    'cross_balanced_dice_text_brevity', 'cross_followup_priority',
    'cross_coin_format_digit_period',
)


def evidence_features(record):
    from lite_core import canonical_answers, style
    a = canonical_answers(record)
    base = summary_features(a, style)
    behavior = behavior_feature_map(a)
    seq = sequence_feature_map(a)
    text = a['B12'].lower()
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
    n = max(len(words), 1)
    period = seq['B02_long_repeated_block_fraction']
    length = min(len(a['B12']) / 1200, 2.)
    brevity = 1 / (1 + len(words) / 100)
    concise = behavior['B11_conflict_cue'] / (1 + len(a['B11']) / 40)
    extra = (
        float(bool(re.search(r'例如|比如|举个|举例|for example|for instance', a['B08'], re.I))
              and len(re.findall(r'\d+', a['B08'])) >= 2),
        float(bool(re.search(r'或者|或[：:]|。或|[／/]|可能指|可能是|两种|分别', a['B09']))),
        concise, behavior['B11_conflict_cue'] * (1 - behavior['B11_literal_forbidden_character']),
        math.log1p(len(words)), math.log1p(len(re.split(r'\n\s*\n', a['B12'].strip()))),
        sum(word in ('you', 'your', "you'll", "you're", "you'd") for word in words) / n,
        sum("'" in word for word in words) / n,
        sum(word in ('even', 'evenly', 'balanced', 'balance', 'spread', 'tidy', 'messy') for word in words) / n,
        period * length, period * brevity, period * concise,
        period * behavior['B12_pseudorandom_algorithm_cue'],
        seq['B04_exactly_balanced_counts'] * brevity,
        behavior['B07_personal_followup_cue'] * behavior['B11_priority_choice_cue'],
        seq['B03_strict_format'] * period,
    )
    result = np.concatenate((base, np.asarray(extra, dtype=np.float32)))
    if result.shape != (len(EVIDENCE_NAMES),) or not np.isfinite(result).all():
        raise ValueError('专门证据特征的维度或数值无效。')
    return result


def select_evidence(z, y, n_classes, top_k=12, per_group=3):
    """Select positive or negative evidence separately for each class."""
    mask = np.zeros((z.shape[1], n_classes), dtype=np.float32)
    varying = z.std(axis=0) > 1e-6
    for label in range(n_classes):
        positive, negative = z[y == label], z[y != label]
        if not len(positive) or not len(negative):
            raise ValueError('每个专门分支需要正例与反例。')
        effect = np.abs(positive.mean(axis=0) - negative.mean(axis=0))
        effect[~varying] = 0
        groups = {}
        selected = 0
        for index in np.argsort(-effect, kind='stable'):
            if selected >= top_k or effect[index] <= 1e-6:
                break
            group = EVIDENCE_NAMES[index].split('_')[0]
            if groups.get(group, 0) >= per_group:
                continue
            mask[index, label] = 1
            groups[group] = groups.get(group, 0) + 1
            selected += 1
    return mask


def combined_forward(x, evidence, params, strength=.5, shared_strength=1.):
    from lite_core import softmax
    hidden = np.tanh(x @ params['w1'] + params['b1'])
    shared = shared_strength * (hidden @ params['w2'] + params['b2'])
    adjustment = strength * (evidence @ (params['specialist_w'] * params['specialist_mask']) + params.get('specialist_bias', 0))
    return hidden, softmax(shared + adjustment), shared, adjustment


def combined_loss_and_grad(x, evidence, y, params, strength=.5, l2=.001, specialist_l2=.03):
    hidden, probabilities, _, _ = combined_forward(x, evidence, params, strength)
    mask = params['specialist_mask']
    weights = params['specialist_w'] * mask
    loss = -np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12)).mean()
    loss += .5*l2*(np.sum(params['w1']**2) + np.sum(params['w2']**2))
    loss += .5*specialist_l2*np.sum(weights**2)
    delta = probabilities.copy()
    delta[np.arange(len(y)), y] -= 1
    delta /= len(y)
    d_hidden = (delta @ params['w2'].T) * (1-hidden**2)
    gradients = {
        'w1': x.T @ d_hidden + l2*params['w1'], 'b1': d_hidden.sum(axis=0),
        'w2': hidden.T @ delta + l2*params['w2'], 'b2': delta.sum(axis=0),
        'specialist_w': mask * (strength * evidence.T @ delta + specialist_l2*weights),
    }
    return float(loss), gradients


def fit_joint_specialists(x_train, y_train, x_val, y_val, n_classes,
                    evidence_train, evidence_val, hidden=16, epochs=400, lr=.01,
                    l2=.001, patience=50, seed=42, strength=.5, specialist_l2=.03, top_k=12):
    if not 0 <= strength <= 2 or not 1 <= top_k <= len(EVIDENCE_NAMES):
        raise ValueError('专门分支的强度或特征数无效。')
    mean, scale = x_train.mean(axis=0), x_train.std(axis=0)
    scale[scale < 1e-4] = 1
    evidence_mean, evidence_scale = evidence_train.mean(axis=0), evidence_train.std(axis=0)
    evidence_scale[evidence_scale < 1e-4] = 1
    train, val = np.clip((x_train-mean)/scale, -8, 8), np.clip((x_val-mean)/scale, -8, 8)
    etrain = np.clip((evidence_train-evidence_mean)/evidence_scale, -8, 8)
    eval_ = np.clip((evidence_val-evidence_mean)/evidence_scale, -8, 8)
    mask = select_evidence(etrain, y_train, n_classes, top_k)
    rng = np.random.default_rng(seed)
    params = {
        'w1': (rng.normal(size=(train.shape[1], hidden))/np.sqrt(train.shape[1])).astype(np.float32),
        'b1': np.zeros(hidden, dtype=np.float32),
        'w2': (rng.normal(size=(hidden, n_classes))/np.sqrt(hidden)).astype(np.float32),
        'b2': np.zeros(n_classes, dtype=np.float32),
        'specialist_w': np.zeros(mask.shape, dtype=np.float32), 'specialist_mask': mask,
    }
    keys = ('w1', 'b1', 'w2', 'b2', 'specialist_w')
    first = {k: np.zeros_like(params[k]) for k in keys}
    second = {k: np.zeros_like(params[k]) for k in keys}
    best, best_loss, best_epoch, stale, step = None, float('inf'), 0, 0, 0
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(train))
        for start in range(0, len(order), 16):
            ids = order[start:start+16]
            _, gradients = combined_loss_and_grad(train[ids], etrain[ids], y_train[ids], params, strength, l2, specialist_l2)
            step += 1
            for key in keys:
                first[key] = .9*first[key] + .1*gradients[key]
                second[key] = .999*second[key] + .001*gradients[key]**2
                params[key] -= lr*(first[key]/(1-.9**step))/(np.sqrt(second[key]/(1-.999**step))+1e-8)
        _, probabilities, _, _ = combined_forward(val, eval_, params, strength)
        loss = float(-np.log(np.maximum(probabilities[np.arange(len(y_val)), y_val], 1e-12)).mean())
        if loss < best_loss - 1e-5:
            best, best_loss, best_epoch, stale = {k: v.copy() for k, v in params.items()}, loss, epoch, 0
        else:
            stale += 1
        if stale >= patience:
            break
    best.update(evidence_mean=evidence_mean, evidence_scale=evidence_scale)
    history = {'best_epoch': best_epoch, 'validation_cross_entropy': best_loss,
               'specialist_strength': strength, 'specialist_l2': specialist_l2, 'specialist_top_k': top_k}
    return best, mean, scale, history


def binary_loss_and_grad(evidence, targets, weights, bias, mask, sample_weights, l2=.03):
    coefficients = weights * mask
    logits = evidence @ coefficients + bias
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -30, 30)))
    loss = float(np.sum(sample_weights * (np.logaddexp(0, logits) - targets*logits)) / len(targets))
    loss += .5*l2*float(np.sum(coefficients**2))
    delta = sample_weights * (probabilities-targets) / len(targets)
    return loss, mask * (evidence.T @ delta + l2*coefficients), delta.sum(axis=0)


def fit_independent_specialists(x_train, y_train, x_val, y_val, n_classes,
                    evidence_train, evidence_val, hidden=16, epochs=400, lr=.01,
                    l2=.001, patience=50, seed=42, strength=.5, specialist_l2=.03, top_k=12):
    """Fit independent balanced one-vs-rest evidence heads and the shared NN."""
    from lite_core import fit
    if not 0 <= strength <= 1 or not 1 <= top_k <= len(EVIDENCE_NAMES):
        raise ValueError('专门分支的强度或特征数无效。')
    params, mean, scale, history = fit(x_train, y_train, x_val, y_val, n_classes,
                                     hidden=hidden, epochs=epochs, lr=lr, l2=l2, patience=patience, seed=seed)
    evidence_mean, evidence_scale = evidence_train.mean(axis=0), evidence_train.std(axis=0)
    evidence_scale[evidence_scale < 1e-4] = 1
    z = np.clip((evidence_train-evidence_mean)/evidence_scale, -8, 8)
    mask = select_evidence(z, y_train, n_classes, top_k)
    targets = (y_train[:, None] == np.arange(n_classes)[None, :]).astype(np.float32)
    positives = targets.sum(axis=0)
    if (positives == 0).any() or (positives == len(y_train)).any():
        raise ValueError('每个专门分支都需要正例和反例。')
    sample_weights = targets * (len(y_train)/(2*positives)) + (1-targets)*(len(y_train)/(2*(len(y_train)-positives)))
    weights = np.zeros(mask.shape, dtype=np.float32)
    bias = np.zeros(n_classes, dtype=np.float32)
    m_w, v_w, m_b, v_b = np.zeros_like(weights), np.zeros_like(weights), np.zeros_like(bias), np.zeros_like(bias)
    best_loss, best_weights, best_bias, stale, best_epoch = float('inf'), weights.copy(), bias.copy(), 0, 0
    for step in range(1, epochs+1):
        _, g_w, g_b = binary_loss_and_grad(z, targets, weights, bias, mask, sample_weights, specialist_l2)
        m_w, v_w = .9*m_w+.1*g_w, .999*v_w+.001*g_w*g_w
        m_b, v_b = .9*m_b+.1*g_b, .999*v_b+.001*g_b*g_b
        weights -= .03*(m_w/(1-.9**step))/(np.sqrt(v_w/(1-.999**step))+1e-8)
        bias -= .03*(m_b/(1-.9**step))/(np.sqrt(v_b/(1-.999**step))+1e-8)
        loss, _, _ = binary_loss_and_grad(z, targets, weights, bias, mask, sample_weights, specialist_l2)
        if loss < best_loss-1e-6:
            best_loss, best_weights, best_bias, stale, best_epoch = loss, weights.copy(), bias.copy(), 0, step
        else:
            stale += 1
        if stale >= patience:
            break
    params.update(specialist_w=best_weights, specialist_bias=best_bias, specialist_mask=mask,
                  evidence_mean=evidence_mean, evidence_scale=evidence_scale)
    history.update(specialist_strength=strength, shared_strength=1-strength,
                   specialist_l2=specialist_l2, specialist_top_k=top_k,
                   specialist_best_training_epoch=best_epoch, specialist_training_loss=best_loss)
    return params, mean, scale, history


def mixture_loss_and_grad(shared_log, evidence_log, y, mixture, ridge=.1):
    from lite_core import softmax
    difference = evidence_log-shared_log
    probabilities = softmax(shared_log + difference*mixture)
    loss = float(-np.log(np.maximum(probabilities[np.arange(len(y)), y], 1e-12)).mean() + ridge*np.sum(mixture**2))
    delta = probabilities.copy()
    delta[np.arange(len(y)), y] -= 1
    gradient = np.mean(delta*difference, axis=0) + 2*ridge*mixture
    return loss, gradient


def fit_mixture(shared_log, evidence_log, y, maximum=.5):
    mixture = np.zeros(shared_log.shape[1], dtype=np.float64)
    for _ in range(200):
        loss, gradient = mixture_loss_and_grad(shared_log, evidence_log, y, mixture)
        step = 1.
        for _ in range(20):
            proposed = np.clip(mixture-step*gradient, 0, maximum)
            proposed_loss, _ = mixture_loss_and_grad(shared_log, evidence_log, y, proposed)
            if proposed_loss <= loss + 1e-12:
                break
            step *= .5
        if np.max(np.abs(proposed-mixture)) < 1e-6:
            break
        mixture = proposed
    return mixture.astype(np.float32)


def fit_specialists(x_train, y_train, x_val, y_val, n_classes,
                    evidence_train, evidence_val, hidden=16, epochs=400, lr=.01,
                    l2=.001, patience=50, seed=42, strength=.5, specialist_l2=.03, top_k=12):
    """Learn class-specific mixture weights from inner out-of-fold predictions."""
    from lite_core import forward, softmax
    options = dict(hidden=hidden, epochs=epochs, lr=lr, l2=l2, patience=patience,
                   seed=seed, strength=strength, specialist_l2=specialist_l2, top_k=top_k)
    params, mean, scale, history = fit_independent_specialists(
        x_train, y_train, x_val, y_val, n_classes, evidence_train, evidence_val, **options)
    folds = min(3, int(np.bincount(y_train, minlength=n_classes).min()))
    if folds < 2:
        mixture = np.zeros(n_classes, dtype=np.float32)
    else:
        assignments = np.zeros(len(y_train), dtype=np.int64)
        rng = np.random.default_rng(seed+1009)
        for label in range(n_classes):
            ids = rng.permutation(np.flatnonzero(y_train == label))
            assignments[ids] = np.arange(len(ids)) % folds
        shared_log, evidence_log = np.zeros((len(y_train), n_classes)), np.zeros((len(y_train), n_classes))
        for fold in range(folds):
            train, heldout = np.flatnonzero(assignments != fold), np.flatnonzero(assignments == fold)
            p, m, s, _ = fit_independent_specialists(
                x_train[train], y_train[train], x_train[train], y_train[train], n_classes,
                evidence_train[train], evidence_train[train], **options)
            base = forward(np.clip((x_train[heldout]-m)/s, -8, 8), p)[1]
            z = np.clip((evidence_train[heldout]-p['evidence_mean'])/p['evidence_scale'], -8, 8)
            specialist = softmax(z @ (p['specialist_w']*p['specialist_mask']) + p['specialist_bias'])
            shared_log[heldout] = np.log(np.maximum(base, 1e-12))
            evidence_log[heldout] = np.log(np.maximum(specialist, 1e-12))
        mixture = fit_mixture(shared_log, evidence_log, y_train, maximum=strength)
    params['specialist_mix'] = mixture
    history.update(specialist_class_weights=mixture.tolist(), specialist_inner_folds=folds,
                   specialist_mixture_ridge=.1,
                   specialist_weight_selection='training_only_inner_out_of_fold')
    return params, mean, scale, history


def predict_probabilities(records, params, mean, scale, metadata):
    from lite_core import features, forward, model_feature_set, softmax
    x = np.stack([features(r, model_feature_set(metadata)) for r in records])
    x = np.clip((x-mean)/scale, -8, 8)
    if metadata.get('architecture', 'neural') == 'neural':
        return forward(x, params)[1], None
    if metadata.get('architecture') != ARCHITECTURE or metadata.get('evidence_version') != EVIDENCE_VERSION:
        raise ValueError('不支持的分类器结构或专门证据版本。')
    e = np.stack([evidence_features(r) for r in records])
    e = np.clip((e-params['evidence_mean'])/params['evidence_scale'], -8, 8)
    shared_log = np.log(np.maximum(forward(x, params)[1], 1e-12))
    specialist_logits = e @ (params['specialist_w']*params['specialist_mask']) + params['specialist_bias']
    specialist_log = np.log(np.maximum(softmax(specialist_logits), 1e-12))
    adjustment = params['specialist_mix'] * (specialist_log-shared_log)
    probabilities = softmax(shared_log+adjustment)
    normalization = specialist_logits-specialist_log
    other_adjustment = params['specialist_mix']*(params['specialist_bias']-normalization-shared_log)
    return probabilities, {'shared_scores': shared_log, 'specialist_adjustments': adjustment,
                           'standardized_evidence': e, 'other_adjustments': other_adjustment}


def selected_evidence(params, classes):
    return {label: [{'feature': EVIDENCE_NAMES[i], 'weight': float(params['specialist_w'][i, j])}
                    for i in np.flatnonzero(params['specialist_mask'][:, j])]
            for j, label in enumerate(classes)}
