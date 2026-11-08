"""Frozen-protocol nested residual experiment; no change to the fusion engine.

Outer: leave-one-answer-out AND leave-one-model-out, same 95-answer library.
Inner: a preassigned model-group holdout selects only gated vs exact baseline.
Gate training always uses cross-fitted base evidence, never base training scores.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
import numpy as np
import nlp_core as N
import neural_core as M
import residual_gate as R
from classifier import load_references, inventory, fingerprint, read_answer

SEED = 42
TARGET_COVERAGE = 92/95
FIXED_MARGIN = .07457089686688191
NAMES = ('baseline', 'gated', 'uniform', 'linear', 'selected')


def group_map(records):
    """Preassigned stratified GROUPS, no answer feature or metric is consulted."""
    rng = np.random.default_rng(SEED)
    out = {}
    for family in sorted({r['family'] for r in records}):
        models = sorted({r['model'] for r in records if r['family'] == family})
        rng.shuffle(models)
        start = int(rng.integers(3))
        out.update({model: (i+start) % 3 for i, model in enumerate(models)})
    return out


class Experiment:
    def __init__(self, records):
        self.records = records
        self.classes = sorted({r['family'] for r in records})
        self.y = np.array([self.classes.index(r['family']) for r in records])
        self.counts = N.count_matrix(records)
        self.handcrafted = np.stack([M.vector(r) for r in records])
        self.buckets = R.buckets(records)
        self.groups = group_map(records)
        self.fold_ids = np.array([self.groups[r['model']] for r in records])
        self.cache, self.selection_cache = {}, {}
        self.base_fit_seconds = 0.

    def training(self, indices):
        return [self.records[i] for i in indices]

    def valid_training(self, indices):
        return bool(len(indices)) and {self.records[i]['family'] for i in indices} == set(self.classes)

    def base(self, indices):
        key = tuple(sorted(indices))
        if key in self.cache:
            return self.cache[key]
        assert self.valid_training(indices)
        started = time.perf_counter()
        training = self.training(indices)
        idf = N.idf_from(self.counts[list(indices)])
        x = N.apply_idf(self.counts, idf)
        weights = N.sample_weights(training, self.classes)
        params = N.fit(x[list(indices)], self.y[list(indices)], self.classes, weights,
                       seed=SEED, epochs=400)
        nlp = dict(params, idf=idf)
        # The exact existing optimizer, with only stateless per-record vectors cached.
        original = M.vector
        by_id = {id(r): x for r, x in zip(self.records, self.handcrafted)}
        try:
            M.vector = lambda record: by_id[id(record)]
            mlp = M.train_network(training, seed=SEED, epochs=400)
        finally:
            M.vector = original
        pm = M.forward(M.transform(self.handcrafted, mlp), mlp)[1]
        pn = N.predict_scores(x, nlp)
        value = {'mlp': mlp, 'nlp': nlp, 'logits': R.fused_logits(pm, pn)}
        self.cache[key] = value
        self.base_fit_seconds += time.perf_counter()-started
        return value

    def crossfit(self, indices):
        """For each row, ALL base/IDF/MLP statistics exclude its entire model group."""
        indices = np.array(sorted(indices), dtype=int)
        u = np.zeros((len(indices), 12, R.CENTERS, 3, len(self.classes)))
        pool = np.zeros((len(indices), 12, R.CENTERS))
        logits = np.zeros((len(indices), len(self.classes)))
        written = np.zeros(len(indices), dtype=bool)
        for fold in sorted(set(self.fold_ids[indices])):
            test_positions = np.flatnonzero(self.fold_ids[indices] == fold)
            test = indices[test_positions]
            train = indices[self.fold_ids[indices] != fold]
            if not self.valid_training(train):
                return None
            assert not (set(self.records[i]['model'] for i in train) & set(self.records[i]['model'] for i in test))
            fit = self.base(train)
            here_u, here_pool = R.local_inputs(self.buckets[test], fit['nlp'])
            u[test_positions], pool[test_positions] = here_u, here_pool
            logits[test_positions] = fit['logits'][test]
            written[test_positions] = True
        assert written.all()
        return (u, pool), logits

    def selection(self, indices):
        key = tuple(sorted(indices))
        if key in self.selection_cache:
            return self.selection_cache[key]
        indices = np.array(key)
        train = indices[self.fold_ids[indices] != 0]
        val = indices[self.fold_ids[indices] == 0]
        fallback = {'enabled': False, 'reason': 'insufficient independent model groups for nested validation',
                    'training_sources': [self.records[i]['source'] for i in train],
                    'validation_sources': [self.records[i]['source'] for i in val]}
        if not self.valid_training(train) or len(val) < 12:
            return fallback
        cf = self.crossfit(train)
        if cf is None:
            return fallback
        data, base = cf
        params, diag = R.fit(data, base, self.training(train), self.classes, 'gated')
        fitted = self.base(train)
        val_base = fitted['logits'][val]
        val_data = R.local_inputs(self.buckets[val], fitted['nlp'])
        prob = R.forward(val_data, val_base, params)[0]
        result = R.select_on_validation(val_base, prob, self.training(val), self.classes)
        result.update(training_sources=fallback['training_sources'], validation_sources=fallback['validation_sources'],
                      inner_gate_training=diag)
        self.selection_cache[key] = result
        return result

    def fit_all(self, indices):
        indices = np.array(sorted(indices))
        base = self.base(indices)
        selection = self.selection(indices)
        crossfit = self.crossfit(indices)
        if crossfit is None:
            selection = {'enabled': False, 'reason': 'insufficient groups to cross-fit the gate'}
            return base, {v: R.init(len(self.classes), v) for v in R.VARIANTS}, selection, {}
        data, logits = crossfit
        params, diagnostics = {}, {}
        for variant in R.VARIANTS:
            params[variant], diagnostics[variant] = R.fit(data, logits, self.training(indices), self.classes, variant)
        return base, params, selection, diagnostics


def verdict(scores, expected, classes):
    order = np.argsort(-scores)
    return {'top': classes[int(order[0])], 'correct': classes[int(order[0])] == expected,
            'margin': float(scores[order[0]]-scores[order[1]]), 'scores': scores.tolist()}


def summary(rows, name, threshold=None):
    threshold = float(np.quantile([r[name]['margin'] for r in rows], 1-TARGET_COVERAGE)) if threshold is None else threshold
    accepted = [r for r in rows if r[name]['margin'] >= threshold]
    fixes = [r['source'] for r in rows if not r['baseline']['correct'] and r[name]['correct']]
    breaks = [r['source'] for r in rows if r['baseline']['correct'] and not r[name]['correct']]
    return {'tested': len(rows), 'correct': sum(r[name]['correct'] for r in rows),
            'accuracy': sum(r[name]['correct'] for r in rows)/len(rows),
            'threshold': threshold, 'accepted': len(accepted),
            'accepted_correct': sum(r[name]['correct'] for r in accepted),
            'accepted_accuracy': sum(r[name]['correct'] for r in accepted)/len(accepted) if accepted else None,
            'coverage': len(accepted)/len(rows), 'fixes': fixes, 'breaks': breaks,
            'changed_top': [r['source'] for r in rows if r[name]['top'] != r['baseline']['top']],
            'enabled_folds': sum(r['selection']['enabled'] for r in rows) if name == 'selected' else None}


def run(exp, mode):
    rows, diagnostics, visited = [], [], set()
    total = len(exp.records) if mode == 'sample' else len({r['model'] for r in exp.records})
    started = time.perf_counter()
    for qi, record in enumerate(exp.records):
        key = record['model'] if mode == 'model' else qi
        if key in visited:
            continue
        visited.add(key)
        test = [i for i, r in enumerate(exp.records) if r['model'] == key] if mode == 'model' else [qi]
        train = [i for i in range(len(exp.records)) if i not in test]
        assert set(train).isdisjoint(test)
        base, params, selection, diag = exp.fit_all(train)
        logits = base['logits'][test]
        data = R.local_inputs(exp.buckets[test], base['nlp'])
        scores = {'baseline': N.softmax(logits)}
        for variant in R.VARIANTS:
            scores[variant] = R.forward(data, logits, params[variant], variant)[0]
        scores['selected'] = scores['gated'] if selection['enabled'] else scores['baseline']
        diagnostics.append({'heldout': key, 'training_count': len(train), 'gate': diag, 'selection': selection})
        for j, i in enumerate(test):
            row = {'source': exp.records[i]['source'], 'expected': exp.records[i]['family'],
                   'model': exp.records[i]['model'], 'selection': selection}
            for name, prob in scores.items():
                row[name] = verdict(prob[j], row['expected'], exp.classes)
            rows.append(row)
        if len(visited) % 5 == 0 or len(visited) == total:
            print('%s %d/%d | base fits %d | %.1fs' % (mode, len(visited), total, len(exp.cache), time.perf_counter()-started), flush=True)
    return {'seconds': time.perf_counter()-started, 'rows': rows, 'diagnostics': diagnostics,
            'summary': {name: summary(rows, name) for name in NAMES},
            'fixed_baseline_threshold': {name: summary(rows, name, FIXED_MARGIN) for name in NAMES}}


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='evaluation-residual-gate.json')
    parser.add_argument('--bundle', default='residual-gate.npz')
    args = parser.parse_args()
    for p in (args.output, args.bundle):
        if Path(p).exists():
            raise FileExistsError('Use a new output filename: '+p)
    started = time.perf_counter()
    records, skipped = load_references()
    exp = Experiment(records)
    result = {'version': R.VERSION, 'inventory': inventory(records, skipped), 'classes': exp.classes,
              'reference_signature': N.reference_signature(records), 'seed': SEED,
              'preregistered': {'base_epochs': 400, 'gate_epochs': R.EPOCHS, 'max_delta': R.MAX_DELTA,
                               'inner_group_map': exp.groups, 'target_coverage': TARGET_COVERAGE,
                               'fixed_baseline_margin': FIXED_MARGIN, 'no_outer_tuning': True,
                               'code_sha256': {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in ('residual_gate.py', 'evaluate_residual_gate.py')}},
              'gradcheck': R.gradcheck(), 'runs': {}}
    for mode in ('sample', 'model'):
        result['runs'][mode] = run(exp, mode)
        Path(args.output+'.partial').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
        print(mode, {k: v['correct'] for k, v in result['runs'][mode]['summary'].items()}, flush=True)
    # External files are first read only after all train/internal/model evaluations.
    from check_fusion_external import STAGE, CASES
    full, params, selection, diag = exp.fit_all(list(range(len(records))))
    external_rows = []
    for source, truth in CASES:
        record = read_answer(STAGE/source)
        assert fingerprint(record) not in {r['fingerprint'] for r in records}
        pm = M.predict_scores(record, full['mlp'])
        pn = N.predict_scores(N.apply_idf(N.count_vector(record)[None], full['nlp']['idf']), full['nlp'])[0]
        logits = R.fused_logits(pm[None], pn[None])
        data = R.local_inputs(R.buckets([record]), full['nlp'])
        scores = {'baseline': N.softmax(logits)}
        for variant in R.VARIANTS:
            scores[variant] = R.forward(data, logits, params[variant], variant)[0]
        scores['selected'] = scores['gated'] if selection['enabled'] else scores['baseline']
        row = {'source': source, 'expected': truth, 'selection': selection}
        for name, prob in scores.items():
            row[name] = verdict(prob[0], truth, exp.classes)
        external_rows.append(row)
    result['external'] = {'role': 'frozen regression only, previously observed in development', 'rows': external_rows,
                          'summary': {name: summary(external_rows, name, result['runs']['sample']['summary'][name]['threshold']) for name in NAMES}}
    result['full_training'] = {'selection': selection, 'diagnostics': diag}
    threshold = result['runs']['sample']['summary']['selected']['threshold']
    R.save_bundle(args.bundle, full['mlp'], full['nlp'], params['gated'], selection,
                  exp.classes, N.reference_signature(records), threshold)
    result['base_fit_count'] = len(exp.cache)
    result['base_fit_seconds'] = exp.base_fit_seconds
    result['total_seconds'] = time.perf_counter()-started
    Path(args.output).write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print('external', {k: v['correct'] for k, v in result['external']['summary'].items()}, flush=True)
    print('full selection', selection['enabled'], '| %.1fs' % result['total_seconds'], flush=True)


if __name__ == '__main__':
    main()
