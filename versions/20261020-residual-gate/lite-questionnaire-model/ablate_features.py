"""Descriptive block ablations; not another parameter-selection round."""
import json
from pathlib import Path
import numpy as np
import feature_profiles as fp
from classifier import load_references
from compare_features import evaluate
from nested_check import cache_extraction


if __name__ == '__main__':
    cache_extraction()
    original = fp.blocks
    records, _ = load_references()
    report = {}
    for removed in ('style', 'behavior', 'sequence', 'choices'):
        def without(record, include_open=False):
            result = dict(original(record, include_open))
            result[removed] = np.zeros_like(result[removed])
            return result
        fp.blocks = without
        report[removed] = {mode: evaluate(records, 'balanced', mode) for mode in ('sample', 'model')}
        print(removed, {mode: value['accuracy'] for mode, value in report[removed].items()})
    fp.blocks = original
    Path('feature-ablation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
