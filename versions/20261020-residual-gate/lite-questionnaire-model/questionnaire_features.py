"""Fixed questionnaire features; no neural network or legacy weights."""
import hashlib
import json
import math
import re
from pathlib import Path
import numpy as np
from behavior_features import behavior_features
from sequence_features import summary_features, extra_sequence_features

FEATURE_VERSION = 'b-features-4-extra'
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



