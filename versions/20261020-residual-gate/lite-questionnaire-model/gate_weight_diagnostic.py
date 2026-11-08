"""Post-fit diagnosis only: no retraining, selection or threshold tuning."""
import json
from pathlib import Path
import numpy as np
import residual_gate as R
from classifier import load_references
import neural_core as M
import nlp_core as N


def main():
    meta, models = R.load_bundle('residual-gate.npz')
    records, _ = load_references()
    assert meta['reference_signature'] == N.reference_signature(records)
    pm = np.stack([M.predict_scores(r, models['mlp']) for r in records])
    pn = N.predict_scores(N.apply_idf(N.count_matrix(records), models['nlp']['idf']), models['nlp'])
    data = R.local_inputs(R.buckets(records), models['nlp'])
    _, _, (cache, _) = R.forward(data, R.fused_logits(pm, pn), models['gate'])
    z, _, gate = cache
    active = data[1] > 0
    output = {'role': 'post-fit diagnosis on full-data model; not a validation metric',
              'parameter_l2_norms': {k: float(np.linalg.norm(v)) for k, v in models['gate'].items()},
              'gate_min_valid': float(gate[active].min()), 'gate_max_valid': float(gate[active].max()),
              'max_abs_local_head_term_before_tanh': float(np.max(np.abs(z@models['gate']['W']))),
              'max_abs_output_bias_before_tanh': float(np.max(np.abs(models['gate']['b'])))}
    Path('GATE_WEIGHT_DIAGNOSTIC.json').write_text(json.dumps(output, indent=2), encoding='utf-8')
    print(json.dumps(output, indent=2))


if __name__ == '__main__':
    main()
