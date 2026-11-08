"""Fixed questionnaire features and a small NumPy neural network."""
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
from behavior_features import behavior_features
from sequence_features import summary_features, extra_sequence_features

FEATURE_VERSION = 'b-features-3'
FEATURE_SETS = ('legacy', 'behavior', 'combined', 'summary', 'augmented')
QUESTIONNAIRE_VERSION = 'mq-full-0.2'
QUESTIONS = [f'{g}{i:02d}' for g in 'AB' for i in range(1, 13)]
IDENTITY = re.compile(r'\b(?:gpt[\w.\-]*|claude[\w.\-]*|openai|anthropic|gemini|deepseek|qwen|llama|mistral)\b', re.I)


def canonical_answers(record):
    if record.get('questionnaire_version', QUESTIONNAIRE_VERSION) != QUESTIONNAIRE_VERSION:
        raise ValueError('仅支持整卷版mq-full-0.2，不要混入逐题版。')
    answers = record.get('answers')
    if not isinstance(answers, dict):
        from split_answers import parse_answers
        raw = record.get('raw_response')
        if not isinstance(raw, str):
            raise ValueError('需要answers对象或raw_response字符串。')
        parsed = parse_answers(raw)
        if parsed['parse_report']['status'] != 'complete':
            raise ValueError('回答缺题、重复题号或代码围栏不完整，请先修正格式。')
        answers = parsed['answers']
    if record.get('parse_report', {}).get('status') == 'needs_review':
        raise ValueError('拆分报告要求人工检查，暂不用于训练或推理。')
    for q in QUESTIONS:
        if not isinstance(answers.get(q), str) or not answers[q].strip():
            raise ValueError('缺少有效答案：' + q)
    return {q: answers[q] for q in QUESTIONS}


def histogram(sequence, symbols):
    counts = np.array([sequence.count(s) for s in symbols], dtype=np.float32)
    return (counts / max(len(sequence), 1)).tolist()


def transitions(sequence, symbols):
    index = {symbol: i for i, symbol in enumerate(symbols)}
    matrix = np.zeros((len(symbols), len(symbols)), dtype=np.float32)
    for a, b in zip(sequence, sequence[1:]):
        if a in index and b in index:
            matrix[index[a], index[b]] += 1
    return (matrix.ravel() / max(len(sequence)-1, 1)).tolist()


def sequence_stats(sequence, expected_length):
    longest = run = 0
    last = None
    for char in sequence:
        run = run + 1 if char == last else 1
        longest = max(longest, run)
        last = char
    return [len(sequence)/expected_length, float(len(sequence) == expected_length),
            longest/max(len(sequence), 1), len(set(sequence))/max(len(sequence), 1)]


def style(text):
    text = IDENTITY.sub('[identity]', text)
    n = max(len(text), 1)
    return [math.log1p(len(text)), math.log1p(text.count('\n')),
            sum(c.isdigit() for c in text)/n,
            len(re.findall(r'[A-Za-z]', text))/n,
            len(re.findall(r'[\u4e00-\u9fff]', text))/n,
            len(re.findall(r'[，。！？；：,.!?;:]', text))/n,
            len(re.findall(r'(?m)^\s*(?:[-*]|\d+[.、])\s*', text))/max(text.count('\n')+1, 1),
            float('```' in text)]


def features(record, feature_set='augmented'):
    if feature_set not in FEATURE_SETS:
        raise ValueError('未知特征集：' + str(feature_set))
    answers = canonical_answers(record)
    if feature_set == 'summary':
        return summary_features(answers, style)
    if feature_set == 'behavior':
        return np.asarray(behavior_features(answers), dtype=np.float32)
    vector = []
    for i in range(1, 13):
        vector.extend(style(answers[f'B{i:02d}']))
    # Single choices preserve categorical preferences, not provider metadata.
    s = answers['B01'].strip()
    vector.extend([float(s == str(i)) for i in range(10)])
    vector.append(float(bool(re.fullmatch(r'[0-9]', s))))
    s = answers['B06'].strip()
    valid = bool(re.fullmatch(r'\d{1,3}', s)) and 1 <= int(s) <= 100
    vector.extend([int(s)/100 if valid else 0., float(valid)])
    seq = ''.join(re.findall(r'[0-9]', answers['B02']))
    vector.extend(histogram(seq, '0123456789'))
    vector.extend(transitions(seq, '0123456789'))
    vector.extend(sequence_stats(seq, 120))
    vector.append(float(bool(re.fullmatch(r'[0-9]{120}', answers['B02'].strip()))))
    seq = ''.join(re.findall(r'[HT]', answers['B03'].upper()))
    vector.extend(histogram(seq, 'HT'))
    vector.extend(transitions(seq, 'HT'))
    vector.extend(sequence_stats(seq, 80))
    seq = [int(x) for x in re.findall(r'\d+', answers['B04'])]
    seq = [x for x in seq if 1 <= x <= 6]
    vector.extend(histogram(seq, list(range(1, 7))))
    vector.extend(transitions(seq, list(range(1, 7))))
    vector.extend(sequence_stats(seq, 24))
    vector.append(float(bool(re.fullmatch(r'[1-6](?:\s*,\s*[1-6]){23}', answers['B04'].strip()))))
    for animal in ('猫', '狗', '鸟', '兔', '狐', '狮', '虎', '象', '鹿', '马', '鲸', '海豚', '鹰', '熊', '羊', '企鹅'):
        vector.append(float(animal in answers['B05']))
    for q in ('B09', 'B10', 'B11'):
        text = answers[q]
        for phrase in ('无法', '不能', '不足', '如果'):
            vector.append(float(phrase in text))
    if feature_set in ('combined', 'augmented'):
        vector.extend(behavior_features(answers))
    if feature_set == 'augmented':
        vector.extend(extra_sequence_features(answers))
    x = np.asarray(vector, dtype=np.float32)
    if not np.isfinite(x).all():
        raise ValueError('特征包含非有限值。')
    return x


def read_records(path):
    path = Path(path)
    files = sorted(list(path.glob('*.json')) + list(path.glob('*.jsonl'))) if path.is_dir() else [path]
    if not files:
        raise ValueError('未找到JSON/JSONL文件。')
    result = []
    for file in files:
        text = file.read_text(encoding='utf-8-sig')
        if file.suffix.lower() == '.jsonl':
            values = [json.loads(line) for line in text.splitlines() if line.strip()]
        else:
            value = json.loads(text)
            values = value if isinstance(value, list) else [value]
        for value in values:
            if not isinstance(value, dict):
                raise ValueError(str(file) + '中存在不是JSON对象的记录。')
            result.append(value)
    return result


def prepare_data(records, target, feature_set='augmented'):
    field = 'family_label' if target == 'family' else 'claimed_model'
    kept, xs, labels = [], [], []
    hashes = {}
    warnings = []
    for i, record in enumerate(records):
        label = record.get(field)
        if not isinstance(label, str) or not label.strip():
            warnings.append(f'跳过第{i+1}条：没有{field}标签。')
            continue
        label = label.strip()
        try:
            answers = canonical_answers(record)
            x = features(record, feature_set)
        except ValueError as exc:
            warnings.append(f'跳过第{i+1}条：{exc}')
            continue
        digest = hashlib.sha256(json.dumps(answers, ensure_ascii=False, sort_keys=True).encode()).hexdigest()
        if digest in hashes:
            if hashes[digest] != label:
                raise ValueError('完全相同的整卷答案存在不同标签；请确认来源，并单独保留这些无法区分的样本。')
            warnings.append(f'跳过第{i+1}条：重复整卷答案。')
            continue
        hashes[digest] = label
        kept.append(record)
        xs.append(x)
        labels.append(label)
    classes = sorted(set(labels))
    if len(classes) < 2:
        raise ValueError('至少需要两个有标签的类别。')
    y = np.array([classes.index(label) for label in labels], dtype=np.int64)
    if any(np.sum(y == i) < 2 for i in range(len(classes))):
        raise ValueError('每类至少需要2份不同的完整问卷，才能划分训练和验证。这仅是运行下限。')
    return np.stack(xs), y, classes, kept, warnings


def split_data(y, records, fraction=.25, mode='auto', group_by='batch_id', seed=42):
    rng = np.random.default_rng(seed)
    classes = set(y.tolist())
    groups = [str(r.get(group_by) or '') for r in records]
    can_group = all(groups) and len(set(groups)) >= 2
    grouped = mode == 'group' or (mode == 'auto' and can_group)
    if grouped:
        if not can_group:
            raise ValueError('分组划分需要每条记录都有分组字段，且至少有两个组。')
        unique = np.array(sorted(set(groups)))
        n = min(len(unique)-1, max(1, int(round(len(unique)*fraction))))
        for _ in range(1000):
            selected = set(rng.permutation(unique)[:n])
            val = np.array([i for i, g in enumerate(groups) if g in selected])
            train = np.array([i for i, g in enumerate(groups) if g not in selected])
            if set(y[train].tolist()) == classes and set(y[val].tolist()) == classes:
                return train, val, 'group:' + group_by
        raise ValueError('无法让训练/验证都覆盖所有类别。增加跨类别批次，或明确使用--split random做同条件探索。')
    train, val = [], []
    for c in sorted(classes):
        ids = rng.permutation(np.flatnonzero(y == c))
        n = min(len(ids)-1, max(1, int(round(len(ids)*fraction))))
        val.extend(ids[:n])
        train.extend(ids[n:])
    return np.array(train), np.array(val), 'random_stratified_same_condition_only'


def softmax(logits):
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z)
    return p / p.sum(axis=1, keepdims=True)


def forward(x, params):
    hidden = np.tanh(x @ params['w1'] + params['b1'])
    return hidden, softmax(hidden @ params['w2'] + params['b2'])


def loss_and_grad(x, y, params, l2=1e-3):
    hidden, p = forward(x, params)
    loss = -np.log(np.maximum(p[np.arange(len(y)), y], 1e-12)).mean()
    loss += .5*l2*(np.sum(params['w1']**2)+np.sum(params['w2']**2))
    delta = p.copy()
    delta[np.arange(len(y)), y] -= 1
    delta /= len(y)
    grads = {'w2': hidden.T @ delta + l2*params['w2'], 'b2': delta.sum(axis=0)}
    d_hidden = (delta @ params['w2'].T) * (1-hidden**2)
    grads['w1'] = x.T @ d_hidden + l2*params['w1']
    grads['b1'] = d_hidden.sum(axis=0)
    return float(loss), grads


def metrics(y, predicted, classes):
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    for a, b in zip(y, predicted):
        matrix[a, b] += 1
    f1s = [2*matrix[i, i]/max(matrix[i].sum()+matrix[:, i].sum(), 1) for i in range(len(classes))]
    return {'accuracy': float(np.mean(y == predicted)), 'macro_f1': float(np.mean(f1s)),
            'class_order': classes, 'confusion_matrix_rows_true_columns_predicted': matrix.tolist()}


def fit(x_train, y_train, x_val, y_val, n_classes, hidden=16, epochs=400, lr=.01, l2=.001, patience=50, seed=42):
    rng = np.random.default_rng(seed)
    mean = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1e-4] = 1
    train = np.clip((x_train-mean)/scale, -8, 8)
    val = np.clip((x_val-mean)/scale, -8, 8)
    params = {
        'w1': (rng.normal(size=(train.shape[1], hidden))/np.sqrt(train.shape[1])).astype(np.float32),
        'b1': np.zeros(hidden, dtype=np.float32),
        'w2': (rng.normal(size=(hidden, n_classes))/np.sqrt(hidden)).astype(np.float32),
        'b2': np.zeros(n_classes, dtype=np.float32),
    }
    m = {k: np.zeros_like(v) for k, v in params.items()}
    v = {k: np.zeros_like(p) for k, p in params.items()}
    best, best_loss, best_epoch, stale, step = None, float('inf'), 0, 0, 0
    for epoch in range(1, epochs+1):
        order = rng.permutation(len(train))
        for start in range(0, len(order), 16):
            ids = order[start:start+16]
            _, gradients = loss_and_grad(train[ids], y_train[ids], params, l2)
            step += 1
            for k in params:
                m[k] = .9*m[k] + .1*gradients[k]
                v[k] = .999*v[k] + .001*gradients[k]**2
                params[k] -= lr*(m[k]/(1-.9**step))/(np.sqrt(v[k]/(1-.999**step))+1e-8)
        _, p = forward(val, params)
        val_loss = float(-np.log(np.maximum(p[np.arange(len(y_val)), y_val], 1e-12)).mean())
        if val_loss < best_loss - 1e-5:
            best_loss, best_epoch = val_loss, epoch
            best = {k: value.copy() for k, value in params.items()}
            stale = 0
        else:
            stale += 1
        if stale >= patience:
            break
    return best, mean, scale, {'best_epoch': best_epoch, 'validation_cross_entropy': best_loss}


def save_model(path, params, mean, scale, metadata):
    with Path(path).open('xb') as file:
        np.savez_compressed(file, **params, mean=mean, scale=scale,
                            metadata=np.array(json.dumps(metadata, ensure_ascii=False)))


def load_model(path):
    with np.load(path, allow_pickle=False) as archive:
        meta = json.loads(str(archive['metadata'].item()))
        if meta['feature_version'] not in ('b-features-1', 'b-features-2', FEATURE_VERSION):
            raise ValueError('模型特征版本不兼容。')
        model_feature_set(meta)
        params = {k: archive[k].copy() for k in ('w1', 'b1', 'w2', 'b2')}
        architecture = meta.get('architecture', 'neural')
        if architecture != 'neural':
            from specialist_core import ARCHITECTURE, EVIDENCE_VERSION, EVIDENCE_NAMES
            if architecture != ARCHITECTURE or meta.get('evidence_version') != EVIDENCE_VERSION:
                raise ValueError('不支持的分类器结构或专门证据版本。')
            for key in ('specialist_w', 'specialist_bias', 'specialist_mix', 'specialist_mask', 'evidence_mean', 'evidence_scale'):
                if key not in archive:
                    raise ValueError('模型缺少专门分支参数：' + key)
                params[key] = archive[key].copy()
            expected = (len(EVIDENCE_NAMES), len(meta['classes']))
            if params['specialist_w'].shape != expected or params['specialist_mask'].shape != expected:
                raise ValueError('专门分支的维度与当前版本不一致。')
            if any(params[key].shape != (len(EVIDENCE_NAMES),) for key in ('evidence_mean', 'evidence_scale')):
                raise ValueError('专门证据标准化参数的维度无效。')
            if params['specialist_bias'].shape != (len(meta['classes']),):
                raise ValueError('专门分支偏置的维度无效。')
            if params['specialist_mix'].shape != (len(meta['classes']),) or not np.isfinite(params['specialist_mix']).all() or ((params['specialist_mix'] < 0) | (params['specialist_mix'] > .5)).any():
                raise ValueError('专门分支的类别混合权重无效。')
        return params, archive['mean'].copy(), archive['scale'].copy(), meta


def model_feature_set(metadata):
    if metadata['feature_version'] == 'b-features-1':
        return 'legacy'
    default = 'combined' if metadata['feature_version'] == 'b-features-2' else 'augmented'
    feature_set = metadata.get('feature_set', default)
    if feature_set not in FEATURE_SETS:
        raise ValueError('模型声明了未知特征集。')
    return feature_set
