"""Independent CLI for the residual trial. --raw explicitly bypasses its selector."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import nlp_core as N
import neural_core as M
import residual_gate as R
from classifier import read_answer, load_references


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('answer')
    parser.add_argument('--model', default=str(Path(__file__).with_name('residual-gate.npz')))
    parser.add_argument('--raw', action='store_true', help='force the experimental residual even if internal validation rejected it')
    args = parser.parse_args()
    meta, models = R.load_bundle(args.model)
    references, _ = load_references()
    if N.reference_signature(references) != meta['reference_signature']:
        raise ValueError('Reference library changed; rerun the experiment before using these weights.')
    record = read_answer(args.answer)
    pm = M.predict_scores(record, models['mlp'])
    pn = N.predict_scores(N.apply_idf(N.count_vector(record)[None], models['nlp']['idf']), models['nlp'])[0]
    base = R.fused_logits(pm[None], pn[None])
    prob, delta, _ = R.forward(R.local_inputs(R.buckets([record]), models['nlp']), base, models['gate'])
    enabled = bool(args.raw or meta['selection']['enabled'])
    scores = prob[0] if enabled else N.softmax(base)[0]
    order = np.argsort(-scores)
    margin = float(scores[order[0]]-scores[order[1]])
    print(json.dumps({'engine': 'fused-local-residual-experiment',
                      'candidate': meta['classes'][int(order[0])] if margin >= meta['min_margin'] else None,
                      'ranking': [{'label': meta['classes'][int(i)], 'score': float(scores[i])} for i in order],
                      'baseline_scores': dict(zip(meta['classes'], N.softmax(base)[0].tolist())),
                      'residual_enabled': enabled, 'forced_raw': args.raw,
                      'selection': {k: v for k, v in meta['selection'].items()
                                    if k not in ('training_sources', 'validation_sources', 'inner_gate_training')},
                      'bounded_logit_delta': delta[0].tolist() if enabled else [0.]*len(order),
                      'margin': margin, 'min_margin': meta['min_margin'],
                      'note': 'Experimental family scores, not identity probabilities. This CLI uses a margin gate; it does not implement unknown-family detection.'},
                     ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
