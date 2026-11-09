"""Follow-up fixed 200-epoch CNN, same old 99 records and exact 5-fold partitions.

Architecture unchanged, no checkpoint/seed search. This budget was chosen after
the training diagnostic, so this is explicitly a follow-up development experiment.
Reports forced top-family accuracy only, not old MLP-calibrated rejection rates.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cnn-version', default=r'C:\Users\betraysilence\Music\nano-classifier\versions\20261020-cnn-experiment\lite-questionnaire-model')
    parser.add_argument('--output', default='CNN_200_EPOCH_COMPARISON.json')
    args = parser.parse_args()
    sys.path.insert(0, args.cnn_version)
    import numpy as np
    import torch
    import cnn_core as C
    from cnn_holdout_compare import sample_folds, model_folds
    from classifier import load_references
    from neural_core import reference_signature
    records, skipped = load_references()
    assert len(records) == 99 and not skipped
    out = {'version': 'cnn-fixed-200-follow-up-1', 'reference_signature': reference_signature(records),
           'sample_count': 99, 'epochs': 200, 'seed': 42,
           'protocol': 'same architecture/data/folds as 20-epoch trial; fixed longer budget after training diagnostic; internal development, not an independent benchmark',
           'runs': {}}
    old_file = Path(args.cnn_version)/'CNN_HOLDOUT_COMPARISON_20261020.json'
    out['previous_20_epoch_sha256'] = hashlib.sha256(old_file.read_bytes()).hexdigest()
    old = json.loads(old_file.read_text(encoding='utf-8'))
    for mode, folds in (('sample', sample_folds(records)), ('model', model_folds(records))):
        rows, times = [], []
        for fi, held in enumerate(folds):
            started = time.perf_counter()
            train = [r for i, r in enumerate(records) if i not in held]
            test = [records[i] for i in held]
            if mode == 'model':
                assert not ({r['model'] for r in train} & {r['model'] for r in test})
            model = C.train_cnn(train, epochs=200)
            classes, vocab = model['meta']['classes'], model['vocab']
            net = C.CharacterCNN(max(vocab.values(), default=14)+1, len(classes))
            net.load_state_dict({k: torch.as_tensor(v) for k, v in model['state'].items()})
            net.eval()
            with torch.no_grad():
                prob = torch.softmax(net(torch.from_numpy(C._encode(test, vocab))), dim=1).numpy()
                train_pred = net(torch.from_numpy(C._encode(train, vocab))).argmax(dim=1).numpy()
            for record, scores in zip(test, prob):
                rows.append({'source': record['source'], 'expected': record['family'],
                             'top': classes[int(scores.argmax())], 'scores': scores.tolist(), 'fold': fi})
            times.append({'seconds': time.perf_counter()-started, 'train_correct': int((train_pred == [classes.index(r['family']) for r in train]).sum()), 'train_count': len(train)})
            print(json.dumps({'mode': mode, 'fold': fi+1, 'correct': sum(r['top'] == r['expected'] for r in rows), 'tested': len(rows), **times[-1]}), flush=True)
        out['runs'][mode] = {'correct': sum(r['top'] == r['expected'] for r in rows), 'tested': len(rows),
                             'old_cnn_20_correct': old['cnn'][mode]['preferred_correct'],
                             'old_mlp_correct': old['mlp'][mode]['preferred_correct'], 'rows': rows, 'folds': times}
        Path(args.output).write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding='utf-8')


if __name__ == '__main__':
    main()
