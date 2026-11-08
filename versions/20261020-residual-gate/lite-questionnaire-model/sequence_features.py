"""Compact sequence statistics and exact animal categories for all families."""
import collections
import math
import re
import numpy as np
from behavior_features import behavior_features, compact, short_repeated_block_fraction

ANIMALS = ('猫', '大熊猫', '狗', '鸟', '兔', '狐狸', '狮子', '老虎', '大象', '鹿',
           '马', '鲸鱼', '海豚', '鹰', '熊', '羊', '企鹅', '水獭', '水豚', '穿山甲')
ANIMAL_ALIASES = {'猫咪': '猫', '家猫': '猫', '熊猫': '大熊猫', '狗狗': '狗',
                  '小狗': '狗', '兔子': '兔', '狐': '狐狸', '狮': '狮子',
                  '虎': '老虎', '象': '大象', '鲸': '鲸鱼'}
SEQUENCE_FEATURE_NAMES = (
    'B02_length_ratio', 'B02_strict_format', 'B02_digit_entropy',
    'B02_uniform_deviation', 'B02_bigram_distinct_fraction',
    'B02_trigram_repeat_fraction', 'B02_quadgram_repeat_fraction',
    'B02_trigram_peak_fraction', 'B02_best_lag_match', 'B02_best_lag_fraction',
    'B02_adjacent_same_fraction', 'B02_adjacent_ascending_fraction',
    'B02_adjacent_descending_fraction', 'B02_longest_run_fraction',
    'B02_permutation_block_fraction', 'B02_long_repeated_block_fraction',
    'B03_length_ratio', 'B03_strict_format', 'B03_heads_fraction',
    'B03_alternation_fraction', 'B03_longest_run_fraction',
    'B03_pair_entropy', 'B03_best_lag_match', 'B03_head_imbalance',
    'B04_length_ratio', 'B04_strict_format', 'B04_face_entropy',
    'B04_uniform_deviation', 'B04_adjacent_same_fraction',
    'B04_bigram_distinct_fraction', 'B04_best_lag_match',
    'B04_longest_run_fraction', 'B04_exactly_balanced_counts',
)
STYLE_NAMES = ('log_length', 'log_newlines', 'digit_fraction', 'latin_fraction',
               'chinese_fraction', 'punctuation_fraction', 'list_fraction', 'code_fence')
SUMMARY_FEATURE_NAMES = (
    tuple(f'B{i:02d}_{name}' for i in range(7, 13) for name in STYLE_NAMES)
    + tuple(f'behavior_{i:02d}' for i in range(24))
    + tuple(f'B01_choice_{i}' for i in range(10)) + ('B01_valid', 'B06_value', 'B06_valid')
    + tuple('B05_exact_' + animal for animal in ANIMALS) + ('B05_other_animal',)
    + SEQUENCE_FEATURE_NAMES
)


def exact_animal(text):
    value = compact(text).strip('"\'“”‘’「」『』。.!！')
    return ANIMAL_ALIASES.get(value, value)


def normalized_entropy(values, possible_count):
    if not values or possible_count <= 1:
        return 0.
    counts = collections.Counter(values)
    n = len(values)
    return -sum((count / n) * math.log(count / n) for count in counts.values()) / math.log(possible_count)


def uniform_deviation(values, symbols):
    if not values:
        return 0.
    counts = collections.Counter(values)
    n, k = len(values), len(symbols)
    return sum((counts[s] / n - 1/k) ** 2 for s in symbols) * k / max(k-1, 1)


def ngrams(sequence, size):
    return [tuple(sequence[i:i+size]) for i in range(max(len(sequence)-size+1, 0))]


def repeated_ngram_fraction(sequence, size):
    windows = ngrams(sequence, size)
    return (len(windows)-len(set(windows))) / max(len(windows), 1)


def distinct_pair_fraction(sequence, alphabet_size):
    windows = ngrams(sequence, 2)
    return len(set(windows)) / max(min(len(windows), alphabet_size ** 2), 1)


def best_lag_match(sequence, max_lag=40):
    # Compare enough positions to avoid a single end match appearing periodic.
    best_match, best_lag = 0., 0
    for lag in range(1, min(max_lag, len(sequence)//2) + 1):
        match = sum(a == b for a, b in zip(sequence[lag:], sequence[:-lag])) / (len(sequence)-lag)
        # Keep the smallest equally good lag, rather than a multiple of it.
        if match > best_match:
            best_match, best_lag = match, lag
    return best_match, best_lag / max(len(sequence), 1)


def longest_run_fraction(sequence):
    longest = run = 0
    previous = None
    for item in sequence:
        run = run + 1 if item == previous else 1
        longest = max(longest, run)
        previous = item
    return longest / max(len(sequence), 1)


def adjacent_fraction(sequence, predicate):
    return sum(predicate(a, b) for a, b in zip(sequence, sequence[1:])) / max(len(sequence)-1, 1)


def sequence_feature_map(answers):
    digits = ''.join(re.findall('[0-9]', answers['B02']))
    numeric_digits = [int(c) for c in digits]
    triples = collections.Counter(ngrams(digits, 3))
    digit_lag, digit_lag_fraction = best_lag_match(digits)
    blocks = [digits[i:i+10] for i in range(0, len(digits)-9, 10)]
    coins = ''.join(re.findall('[HT]', answers['B03'].upper()))
    pairs = ngrams(coins, 2)
    heads = coins.count('H') / max(len(coins), 1)
    dice = [int(c) for c in re.findall(r'\d+', answers['B04']) if 1 <= int(c) <= 6]
    dice_counts = [dice.count(i) for i in range(1, 7)]
    values = (
        len(digits)/120, float(bool(re.fullmatch('[0-9]{120}', answers['B02'].strip()))),
        normalized_entropy(digits, 10), uniform_deviation(digits, '0123456789'),
        distinct_pair_fraction(digits, 10), repeated_ngram_fraction(digits, 3),
        repeated_ngram_fraction(digits, 4), max(triples.values(), default=0)/max(len(digits)-2, 1),
        digit_lag, digit_lag_fraction,
        adjacent_fraction(digits, lambda a, b: a == b),
        adjacent_fraction(numeric_digits, lambda a, b: (b-a) % 10 == 1),
        adjacent_fraction(numeric_digits, lambda a, b: (a-b) % 10 == 1),
        longest_run_fraction(digits),
        sum(len(set(block)) == 10 for block in blocks)/max(len(blocks), 1),
        short_repeated_block_fraction(digits, max_block=40),
        len(coins)/80, float(bool(re.fullmatch('[HT]{80}', answers['B03'].strip()))), heads,
        adjacent_fraction(coins, lambda a, b: a != b), longest_run_fraction(coins),
        normalized_entropy(pairs, 4), best_lag_match(coins, 12)[0], abs(heads-.5)*2 if coins else 0.,
        len(dice)/24, float(bool(re.fullmatch(r'[1-6](?:\s*,\s*[1-6]){23}', answers['B04'].strip()))),
        normalized_entropy(dice, 6), uniform_deviation(dice, range(1, 7)),
        adjacent_fraction(dice, lambda a, b: a == b), distinct_pair_fraction(dice, 6),
        best_lag_match(dice, 6)[0], longest_run_fraction(dice),
        float(len(dice) == 24 and len(set(dice_counts)) == 1),
    )
    return dict(zip(SEQUENCE_FEATURE_NAMES, values))


def summary_features(answers, style_function):
    values = []
    for i in range(7, 13):
        values.extend(style_function(answers[f'B{i:02d}']))
    values.extend(behavior_features(answers))
    choice = answers['B01'].strip()
    values.extend(float(choice == str(i)) for i in range(10))
    values.append(float(bool(re.fullmatch('[0-9]', choice))))
    number = answers['B06'].strip()
    valid = bool(re.fullmatch(r'\d{1,3}', number)) and 1 <= int(number) <= 100
    values.extend((int(number)/100 if valid else 0., float(valid)))
    animal = exact_animal(answers['B05'])
    values.extend(float(animal == name) for name in ANIMALS)
    values.append(float(animal not in ANIMALS))
    values.extend(sequence_feature_map(answers).values())
    result = np.asarray(values, dtype=np.float32)
    if not np.isfinite(result).all():
        raise ValueError('汇总特征含非有限值。')
    return result


def extra_sequence_features(answers):
    """New variables to append while preserving every existing v2 feature."""
    animal = exact_animal(answers['B05'])
    return ([float(animal == name) for name in ANIMALS]
            + [float(animal not in ANIMALS)]
            + list(sequence_feature_map(answers).values()))
