"""Fixed lightweight feature candidates; no neural network or identity features."""
import re
import unicodedata
from collections import Counter
import numpy as np
from questionnaire_features import canonical_answers, features
from open_cues import extra_map
from extra_features import extra_features

PROFILES = ('baseline', 'balanced', 'enriched', 'lexical')
IDENTITY = re.compile(r'(?:openai|anthropic|deepseek|claude|chatgpt|gemini|grok|qwen|glm|gpt|智谱|深度求索)[\w.\-]*', re.I)


def normalized_answers(record):
    result = {}
    for q, value in canonical_answers(record).items():
        if q.startswith('A'):
            result[q] = value
            continue
        value = unicodedata.normalize('NFKC', value).replace('**', '').replace('__', '')
        value = IDENTITY.sub('[identity]', value)
        value = re.sub(r'\s+', ' ', value).strip()
        value = re.sub(r'(?<=[\u4e00-\u9fff]) (?=[\u4e00-\u9fff])', '', value)
        if q in ('B01', 'B02', 'B03', 'B04', 'B06'):
            value = re.sub(r'\s+', '', value)
        result[q] = value
    return result


def blocks(record, include_open=False):
    answers = normalized_answers(record)
    summary = features({'answers': answers}, 'summary').astype(float)
    result = {'style': summary[[i for i in range(48) if i % 8 != 1]],
            'behavior': summary[48:70], 'choices': summary[72:106],
            'sequence': summary[106:],
            'extra': np.asarray(extra_features(answers), dtype=float)}
    if include_open:
        result['open'] = np.array(list(extra_map(answers).values()))
    return result


def tokens(record):
    result = Counter()
    answers = normalized_answers(record)
    for q in ('B07', 'B08', 'B09', 'B10', 'B11', 'B12'):
        text = answers[q].lower()
        words = re.findall(r"[a-z]+(?:'[a-z]+)?", text)
        for word in words:
            if word != 'identity':
                result[q+':'+word] += 1
        for chunk in re.findall(r'[\u4e00-\u9fff]+', text):
            for n in (2, 3):
                result.update(q+':'+chunk[i:i+n] for i in range(len(chunk)-n+1))
    return result


def lexical_vectors(reference_tokens, query_tokens):
    df = Counter(token for row in reference_tokens for token in row)
    vocabulary = sorted((t for t, count in df.items() if count >= 2), key=lambda t: (-df[t], t))[:256]
    idf = np.array([np.log((1+len(reference_tokens))/(1+df[t]))+1 for t in vocabulary])
    def vector(row):
        x = np.log1p([row.get(t, 0) for t in vocabulary])*idf
        return x/max(float(np.linalg.norm(x)), 1e-12)
    return np.stack([vector(row) for row in reference_tokens]), vector(query_tokens), vocabulary


def distances(record, references, profile):
    if profile not in PROFILES:
        raise ValueError('Unknown feature profile: '+str(profile))
    if profile == 'baseline':
        x = np.stack([features(r, 'summary') for r in references]).astype(float)
        q = features(record, 'summary').astype(float)
        scale = np.maximum(x.std(axis=0), .15)
        delta = np.minimum(((x-q)/scale)**2, 25)
        pairs = np.minimum(((x[:, None]-x[None, :])/scale)**2, 25)
        return np.sqrt(delta.mean(axis=1)), np.sqrt(pairs.mean(axis=2)), {'profile': profile, 'dimensions': 139}
    rows, query = [blocks(r, profile != 'balanced') for r in references], blocks(record, profile != 'balanced')
    weights = ({'style': .342, 'behavior': .207, 'sequence': .207, 'choices': .144, 'extra': .10} if profile == 'balanced'
               else {'style': .25, 'behavior': .2, 'sequence': .25, 'choices': .05, 'open': .25})
    d, pair = np.zeros(len(rows)), np.zeros((len(rows), len(rows)))
    dimensions = 0
    parts = {}
    for name, weight in weights.items():
        x = np.stack([row[name] for row in rows])
        q = query[name]
        # Small fractional statistics should not be erased by a universal .15 floor.
        scale = np.maximum(x.std(axis=0), .03)
        delta = np.minimum(((x-q)/scale)**2, 9)
        p = np.minimum(((x[:, None]-x[None, :])/scale)**2, 9)
        parts[name] = delta.mean(axis=1).tolist()
        d += weight*delta.mean(axis=1)
        pair += weight*p.mean(axis=2)
        dimensions += x.shape[1]
    if profile == 'lexical':
        x, q, vocabulary = lexical_vectors([tokens(r) for r in references], tokens(record))
        ld = np.clip(1-x@q, 0, 1)
        lp = np.clip(1-x@x.T, 0, 1)
        d, pair = .75*d+.25*ld, .75*pair+.25*lp
        parts['lexical'] = ld.tolist()
        dimensions += len(vocabulary)
    return np.sqrt(d), np.sqrt(pair), {'profile': profile, 'dimensions': dimensions, 'block_weights': weights, 'block_squared_distances': parts}
