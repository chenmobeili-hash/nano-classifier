# nano-classfier Latest Version (Consolidated Package)

[简体中文](README.md) ｜ **English**

A lightweight black-box model-family attribution tool driven by a fixed questionnaire. Paste an A01–A12 / B01–B12 answer sheet to obtain relative scores for five families (Claude / DeepSeek / GLM / GPT / Qwen), the top family, the nearest reference samples, and an independent questionnaire score.

| Item | Details |
|---|---|
| Version tag | Residual-gated fusion · 2026-10-27 GPT-6.1-Terra addendum |
| Version directory | `versions/20261020-residual-gate` (the directory date is the creation date) |
| Reference library | 102 valid answer sheets, 25 model/mode groups, 5 families |
| Architecture | 135-dim hand-crafted feature MLP + 4096-dim hashed n-gram TF-IDF logistic regression + 56-parameter local residual gate |
| Dependencies | Python 3, NumPy (`numpy>=1.26,<3`); no GPU, no network required |
| Size | Complete latest version with training and evaluation scripts; older version snapshots not included |

This package is a complete copy of `20261020-residual-gate`: inference, training, evaluation, reference library, rankings, and documentation are all included, with paths resolved relative to this package. After deleting the original `nano-classfier`, double-clicking `最新版_粘贴答案测试.bat` still works. Earlier version directories, zips, and historical backups are not included.

---

## 1. Quick Start

**Windows:** double-click `最新版_粘贴答案测试.bat` in the root directory. The browser opens `http://127.0.0.1:PORT/` automatically; paste an answer sheet or choose a file to start testing; close the command window to stop the service.

You can also launch it with an answer file:

```bat
最新版_粘贴答案测试.bat "C:\path\to\answer.txt"
```

**Command line:**

```bat
cd versions\20261020-residual-gate\lite-questionnaire-model
python -B residual_demo.py                 :: open the web page
python -B residual_demo.py answer.txt      :: open the page and preload the answer
python -B residual_demo.py --json answer.txt   :: print comparison JSON only
python -B residual_demo.py --no-browser --port 8765
```

Install dependencies:

```bat
python -m pip install -r versions\20261020-residual-gate\lite-questionnaire-model\requirements.txt
```

Verification: `--json` on `nano-now/gpt6sol-01.txt` should report GPT as the top family; after startup, `/api/status` should report `samples: 102, families: 5`.

---

## 2. Directory Layout

```
nano-classfier-最新版/
├── README.md                       ← this file (usage / file list)
├── README.en.md                    ← English version
├── CURRENT_VERSION.json            ← machine-readable current version identifier
├── VERSIONS.md                     ← version history and per-version metrics
├── external-holdout/               ← 11 regression samples that never enter the library
├── 更新的樣本/                     ← identical to gpt6.1terra-01/02 in the library
├── 识别结果/                       ← identification output for those two answer sheets
├── 最新版_粘贴答案测试.bat          ← the only entry point
├── 文档/
│   ├── 架构入门_白话版.docx          ← for first-time readers (start here)
│   ├── 当前分类器架构与原理说明.md   ← full technical documentation
│   ├── 当前分类器架构与原理说明.docx
│   └── verification data (4 JSON files)
└── versions/
    ├── README.md                   ← version index
    └── 20261020-residual-gate/
        ├── README.md               ← summary and metrics for this version
        ├── PROTOCOL.md             ← fixed protocol for the residual-gate experiment
        ├── nano-now/               ← reference answer library (103 files / 102 valid, required)
        └── lite-questionnaire-model/
            ├── residual_demo.py / residual_demo.html   ← local test page
            ├── *.py                                    ← inference and scoring programs
            ├── *.npz                                   ← trained weights
            ├── *.json                                  ← evaluation, ranking, and review data
            └── *.md                                    ← reports for this version
```

`最新版_粘贴答案测试.bat` resolves paths relative to its own location, so the whole folder can be moved anywhere, but **the internal structure must not be changed** (in particular, `nano-now` must remain a sibling of `lite-questionnaire-model` under the version directory).

---

## 3. Required Files (missing any of these prevents startup)

### 3.1 Entry Point and Page
| File | Purpose |
|---|---|
| `最新版_粘贴答案测试.bat` | Launcher; switches to the version directory and runs `residual_demo.py` |
| `residual_demo.py` | Local HTTP service, inference pipeline, AI-review routing |
| `residual_demo.html` | Single-page test UI (paste / open file / score / review) |

### 3.2 Model and Weights
| File | Purpose |
|---|---|
| `neural_core.py` + `neural-balanced.npz` | 135-dim hand-crafted feature MLP |
| `nlp_core.py` + `nlp-model.npz` | Hashed n-gram TF-IDF + logistic regression |
| `residual_gate.py` + `residual-gate.npz` | 56-parameter local residual gate and geometric fusion |
| `feature_profiles.py`, `questionnaire_features.py`, `behavior_features.py`, `sequence_features.py`, `open_cues.py`, `extra_features.py` | Feature extraction |

### 3.3 Grading and Scoring
| File | Purpose |
|---|---|
| `ability.py`, `ability_rules.py`, `ability_methods.py` | A-question content scoring and rules |
| `strict_consistency.py`, `scoring_text.py` | Consistency caps, text extraction |
| `result_scoring.py`, `dual_scoring.py`, `score_output.py` | Result score, strict-format score, output aggregation |
| `ability_ranking.py`, `dual_ranking.py` | Model rankings and relative percentiles |
| `sample_reviews.py` + `sample_reviews.json` | Per-question review records (scoring basis) |
| `result_reviews.json` | Result-score review records |
| `sample_annotations.py` + `sample_annotations.json` | Sample annotations and suspected-outlier flags |
| `sample_provenance.json` | Sample provenance |

### 3.4 Classifier and Review
| File | Purpose |
|---|---|
| `classifier.py` | Loads the reference library, fingerprints, and normalised answers |
| `consensus_guard.py` | Consistency guard |
| `known_references.py` + `known_external_references.json` | Comparison against known external references |
| `split_answers.py` | Answer-sheet splitting / file reading |
| `ai_review.py` | Optional manual / external AI-assisted review (key held in memory only) |
| `CURRENT_RELEASE.json` | Version label shown on the page |
| `evaluation-residual-gate.json` | Per-protocol thresholds loaded at startup; signature must match the weights |

### 3.5 Reference Library
`versions/20261020-residual-gate/nano-now/` contains 103 `.txt` files. Among them, `deepseekv4.1flash1017-03.txt` is excluded according to the annotation in `sample_annotations.json` (the user marked it as a suspected outlier), leaving 102 valid references. The program loads every file and compares `reference_signature`; **do not delete or modify these files**. Adding or removing samples requires retraining and updating the weights.

---

## 4. Important Files (not required for startup, but core assets)

| File | Description |
|---|---|
| `model-ranking.json`, `MODEL_RANKING.md` | 24-group composite model ranking (out of 200); consistency snapshot written when reviews are saved |
| `result-ranking.json`, `RESULT_RANKING.md` | Independent ranking based on final results only (out of 100) |
| `HAIKU45_INGESTION_REPORT_20261021.md` | Historical 98→100 addendum report |
| `HAIKU45_SCORING_AUDIT_20261023.md` | Per-question review of two Haiku 4.5 sheets |
| `GPT_OSS_INGESTION_REPORT_20261021.md`, `GPT_OSS_SCORING_AUDIT_20261021.md` | 95→98 GPT-OSS 120B addendum and review (historical) |
| `RESIDUAL_GATE_REPORT.md` | Full residual-gate experiment report (controls, repaired/new errors, external regression) |
| `AI_REVIEW_UI.md`, `AI_REVIEW_UI_VALIDATION_20261024.json` | Page review feature documentation and validation record |
| `NEAREST_SPACE_REPORT.md`, `nearest-space-report.json`, `evaluate_nearest.py` | 2026-10-27 comparison report for switching model diagnostics to the 135-dim feature space, with a rerunnable script |
| `PROTOCOL.md` | Fixed protocol for the gate experiment (prevents post-hoc tuning) |
| `VERSIONS.md`, `CURRENT_VERSION.json`, `versions/README.md` | Version history and index |
| `文档/当前分类器架构与原理说明.md` / `.docx` | **Main technical document**: task definition, features, three-model fusion, gating, evaluation protocols, and limitations |
| `文档/架构入门_白话版.docx` | **Plain-language introduction**: explains the three components by analogy, how to read the numbers, and five common misunderstandings; suitable for first-time readers |
| `文档/架构核对数据_20261024.json`, `门控权重核对_20261024.json`, `文档核对记录_20261024.json`, `门控权重核对_20261027.json` | Consistency checks between documentation, code, and weights (20261027 recomputed after the addendum) |

---

## 5. Current Metrics and Caveats

- Leave-one-answer-out: baseline 91/102, gated 91/102, selected 91/102.
- Leave-one-model-out: baseline 63/102, gated 65/102, selected 63/102.
- Fixed external regression: 10/11 (observed many times; not an unbiased selection set).
- 2026-10-27 added two GPT-6.1-Terra sheets (`gpt6.1terra`, 2 files); the reference library grew 100→102 and model groups 24→25; all three weights and the evaluation were retrained together, signature `4aa85bb1…`.
- The A questions in these two sheets have not been reviewed item by item; their ranking scores are provisional (composite 176.58/200, result 89.00/100) and the number of pending reviews is marked.
- **`gate_policy_enabled: false`**: the conservative policy did not pass the enablement check, so the default remains the original geometric fusion; the page can toggle a view of the forced gate result.
- Family scores are not probability-calibrated and must not be read as "probability that the identity is correct"; nearest samples and model candidates are diagnostics only and do not participate in voting or score fusion.
- Since 2026-10-27, model candidates and nearest samples use the 135-dim feature space (the same space as the classifier); model candidates are ranked by mean distance within each model group. Weights, reference library, and family decisions are unchanged. Under leave-one-answer-out, top-3 model hit rate rose from 42.2% (TF-IDF) to 84.3% (135-dim group mean), but this remains "lookup among known models" and cannot identify a new model outside the library.
- The `.md` files under `文档/` have been synchronised with the changes above; the `.docx` files were edited paragraph by paragraph in place, with other formatting and content untouched. In this package, `当前分类器架构与原理说明.docx` could not be overwritten because Word had it open, so the latest content is provided first as `当前分类器架构与原理说明_20261027.docx`; after closing Word you can use it to replace the original file (the two are identical except for the filename).
- The system answers "which known family does this answer sheet resemble", not "what model actually produced it".
- AI review requires configuring an API on the page (click the small dot in the footer 5 times within 2.4 seconds); the key exists only in the memory of the current service and is never written to disk; review results must be confirmed manually before saving and never enter training.
- Restarting the service is required to refresh the weights and reference-library cache.

---

## 6. Training, Identification, and What Is Not Included

The three current weights were trained on these 102 A01–B12 answer sheets, signature `4aa85bb1…`. This package recognises only these 24 questions; extra question numbers in an answer sheet make the split result `needs_review` and are not fed into the 135-dim features. The adaptive feature set for varying question counts lives in `模型识别\mini-classifier`.

Identify one answer sheet:

```bat
cd versions\20261020-residual-gate\lite-questionnaire-model
python -B residual_demo.py --json "..\..\..\更新的樣本\新增 文字文件.txt"
python -B predict_residual_gate.py "..\..\..\更新的樣本\新增 文字文件.txt"
```

Retraining writes new files and never overwrites released weights. `train.py`, `train_nlp.py`, and `evaluate_residual_gate.py` refuse to run when the output file already exists; use a new `--output` / `--bundle` instead.

The fixed external regression samples are in `external-holdout/`. `regression_suite.py` now reads from that directory.

Not included in this package: earlier version directories, zips, historical backups, and the old programs that one-off experiment scripts pointing at those directories depended on. Those scripts are not current entry points for startup, training, or identification. The two files in `更新的樣本` are identical in full text to `gpt6.1terra-01.txt` and `gpt6.1terra-02.txt` in the reference library; see `识别结果/` for the identification output.
