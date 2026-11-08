# Local residual gate: fixed protocol before final evaluation

- Date: 2026-10-20, Australia/Perth. Source: versions/20261021-nlp, 95 valid answers / 22 model groups / five families.
- Preserve the existing 400-epoch MLP and 400-epoch TF-IDF LR; geometric fusion, 0.5 + 0.5. No nearest-neighbour score.
- New inputs: genuine character +/-1 neighbours, char 2/3/4-gram buckets in the original question namespace; uniformly sample up to 24 original positions per B answer. Each token's five-dimensional evidence comes from the TRAINING-FOLD NLP weights/IDF. The main TF-IDF branch still reads its complete original input.
- New trainable path: shared diagonal convolution (3x5), tanh, scalar sigmoid gate (5+1), pooled five-dimensional state, 5x5 head + five bias. 56 parameters. Zero output head initially gives exact baseline.
- Residual per logit: 0.25 * max(0, 1 - base top-two logit gap) * tanh(head). Logit gap >=1 gives exact baseline scores; gap >1/3 guarantees the top family cannot change. This guarantee concerns forced top-family prediction, not identical margins or coverage in the remaining range.
- Frozen Adam 200 epochs, LR .02, L2 .02 around initial weights. No stopping on outer results.
- Controls: uniform gate (same convolution/input, frozen 0.5 gate, 50 trainable / 56 allocated), and a 56-parameter linear correction on center/context evidence plus baseline temperature. Same correction bound, optimizer, weighting, and folds.
- Outer evaluation: leave-one-answer-out (95 fits) and leave-one-model-out (22 fits). All outer answers excluded from IDF, scaling, base models, local evidence weights, residual training and selection.
- Residual training: three-fold cross-fitting by model group INSIDE each outer training set. No residual training row uses base models trained on that row/model.
- Safe selector: inside outer training only, group fold0 is held for validation; its other groups provide cross-fitted gate training. Enable only with >=12 validation answers from >=3 families, >=1 repair, zero newly wrong predictions and non-increasing weighted cross-entropy. If independent groups are insufficient or the condition fails, use exact baseline. Raw variants are reported regardless of selection.
- Scores report forced accuracy, repairs AND regressions, fixed baseline margin .07457089686688191, and descriptive equal-coverage quantiles (92/95). The latter are retrospective calibration and do not guarantee future coverage.
- Only after both outer evaluations: read frozen 11-file external regression set. It has already been observed repeatedly in development and is not an unbiased model-selection set. No tuning or model selection using those results.
- No requirement that adding a module must win; if raw/selected residual has no reliable gain, retain it as an optional negative experiment.
