"""Nested leave-model-out selection, with fixed extraction cached (never fit state)."""
import json
from functools import lru_cache
from pathlib import Path
import feature_profiles as fp
from classifier import load_references
from compare_features import evaluate, predict


def cache_extraction():
    # Cache by answer content only, not labels. Reference-dependent scales and
    # vocabulary are still fitted afresh inside every fold.
    for name in ('features', 'blocks', 'tokens'):
        original = getattr(fp, name)
        def make(original):
            @lru_cache(maxsize=128)
            def cached(payload, args):
                return original(json.loads(payload), *args)
            def wrapped(record, *args):
                return cached(json.dumps({'answers': record['answers']}, ensure_ascii=False, sort_keys=True), args)
            return wrapped
        setattr(fp, name, make(original))


def main():
    cache_extraction()
    records, _ = load_references()
    folds, rows = [], []
    for model in sorted({r['model'] for r in records}):
        train = [r for r in records if r['model'] != model]
        test = [r for r in records if r['model'] == model]
        inner = {profile: evaluate(train, profile, 'model') for profile in fp.PROFILES}
        score = {profile: sum(f['correct']/f['total'] for f in result['by_family'].values())/len(result['by_family']) for profile, result in inner.items()}
        chosen = max(fp.PROFILES, key=lambda profile: score[profile])
        supported = sorted(inner[chosen]['by_family'])
        fold = {'held_out_model': model, 'chosen': chosen, 'inner_macro_accuracy': score,
                'inner_supported_families': supported,
                'inner_unsupported_families': sorted({r['family'] for r in train}-set(supported))}
        folds.append(fold)
        for query in test:
            rows.append(dict(source=query['source'], expected=query['family'], profile=chosen, **predict(query, train, chosen)))
        print(model, chosen, flush=True)
    accepted = [r for r in rows if r['candidate']]
    result = {'method': 'Outer leave-one-model-out; inner leave-one-model-out selects among four frozen profiles by macro family accuracy. Ties favor earlier/simpler profile. Families unsupported in an inner fold are excluded and listed. All 20 answers have prior feature-development exposure; this is a selection diagnostic, not independent evidence.',
              'accuracy': sum(r['nearest'] == r['expected'] for r in rows)/len(rows),
              'coverage': len(accepted)/len(rows),
              'accepted_accuracy': sum(r['candidate'] == r['expected'] for r in accepted)/len(accepted) if accepted else None,
              'folds': folds, 'rows': rows}
    Path('nested-feature-check.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print({k: v for k, v in result.items() if k not in ('folds', 'rows')}, flush=True)


if __name__ == '__main__':
    main()
