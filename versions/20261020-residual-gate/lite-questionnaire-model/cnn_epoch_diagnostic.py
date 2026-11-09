"""Fixed 20/80/200-epoch diagnostic of the earlier CNN's FIRST sample CV fold.

No checkpoint selection. This is a training-budget diagnostic, not a new full-CV
CNN claim. Keeps the old architecture, seed, library, and fold exactly as they were.
"""
import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import json
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn-version', default=r'C:\Users\betraysilence\Music\nano-classifier\versions\20261020-cnn-experiment\lite-questionnaire-model')
    parser.add_argument('--output', default='CNN_EPOCH_DIAGNOSTIC.json')
    args = parser.parse_args()
    sys.path.insert(0, args.cnn_version)
    import numpy as np
    import torch
    import cnn_core as C
    from cnn_holdout_compare import sample_folds
    from classifier import load_references
    records, skipped = load_references()
    assert len(records) == 99 and not skipped
    held = sample_folds(records)[0]
    train = [r for i, r in enumerate(records) if i not in held]
    test = [r for i, r in enumerate(records) if i in held]
    rows = []
    for epochs in (20, 80, 200):
        started = time.perf_counter()
        model = C.train_cnn(train, epochs=epochs)
        classes, vocab = model['meta']['classes'], model['vocab']
        net = C.CharacterCNN(max(vocab.values(), default=14)+1, len(classes))
        net.load_state_dict({k: torch.as_tensor(v) for k, v in model['state'].items()})
        net.eval()
        item = {'epochs': epochs}
        for name, subset in (('train', train), ('heldout_first_fold', test)):
            y = torch.tensor([classes.index(r['family']) for r in subset])
            with torch.no_grad():
                logits = net(torch.from_numpy(C._encode(subset, vocab)))
                item[name] = {'correct': int((logits.argmax(dim=1) == y).sum()), 'count': len(subset),
                              'unweighted_ce': float(torch.nn.functional.cross_entropy(logits, y))}
        item['seconds'] = time.perf_counter()-started
        rows.append(item)
        print(json.dumps(item), flush=True)
    Path(args.output).write_text(json.dumps({'version': 'frozen-cnn-epoch-diagnostic-1',
        'scope': 'first fold only, no checkpoint selection; not complete cross-validation',
        'rows': rows, 'heldout_sources': [r['source'] for r in test],
        'unchanged': '64 chars/question, vocab512, embedding8, conv3/5/7x12, dropout .45, AdamW .003 wd .025, seed42'}, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()
