"""All fixed candidates reported; development comparisons are not blind tests."""
import json
from pathlib import Path
import numpy as np
from classifier import load_references
from feature_profiles import PROFILES, distances


def predict(query, refs, profile):
    d, pairs, detail = distances(query, refs, profile)
    family_distances = {}
    for model in sorted({r['model'] for r in refs}):
        indices = [i for i, r in enumerate(refs) if r['model'] == model]
        family = refs[indices[0]]['family']
        value = float(d[indices].mean())
        family_distances[family] = min(value, family_distances.get(family, float('inf')))
    ranking = sorted(family_distances, key=family_distances.get)
    margin = 1-family_distances[ranking[0]]/max(family_distances[ranking[1]], 1e-9)
    np.fill_diagonal(pairs, np.inf)
    radius = max(.5, float(np.quantile(pairs.min(axis=1), .95))*1.5)
    return {'nearest': ranking[0], 'candidate': ranking[0] if margin >= .1 and family_distances[ranking[0]] <= radius else None,
            'margin': margin}


def evaluate(records, profile, mode):
    rows = []
    for query in records:
        refs = [r for r in records if (r['model'] != query['model'] if mode == 'model' else r['fingerprint'] != query['fingerprint'])]
        if query['family'] not in {r['family'] for r in refs} or len({r['family'] for r in refs}) < 2:
            continue
        rows.append(dict(source=query['source'], expected=query['family'], **predict(query, refs, profile)))
    accepted = [r for r in rows if r['candidate'] is not None]
    return {'accuracy': sum(r['nearest'] == r['expected'] for r in rows)/len(rows),
            'coverage': len(accepted)/len(rows),
            'accepted_accuracy': sum(r['candidate'] == r['expected'] for r in accepted)/len(accepted) if accepted else None,
            'by_family': {f: {'correct': sum(r['nearest'] == f for r in rows if r['expected'] == f), 'total': sum(r['expected'] == f for r in rows)} for f in sorted({r['expected'] for r in rows})},
            'rows': rows}


if __name__ == '__main__':
    refs, skipped = load_references()
    results = {}
    for profile in PROFILES:
        results[profile] = {mode: evaluate(refs, profile, mode) for mode in ('sample', 'model')}
        print(profile, {mode: {k: v for k, v in data.items() if k != 'rows'} for mode, data in results[profile].items()}, flush=True)
    Path('feature-comparison.json').write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding='utf-8')
