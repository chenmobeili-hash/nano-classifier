# nano-classfier 最新版（精简整理包）

**简体中文** ｜ [English](README.en.md)

固定问卷驱动的轻量级黑盒模型家族归因工具。粘贴一份 A01–A12／B01–B12 答卷，即可得到五家族（Claude／DeepSeek／GLM／GPT／Qwen）相对分数、首选家族、最近参考样本，以及独立的问卷评分。

| 项目 | 内容 |
|---|---|
| 版本标签 | 残差门控融合 · 2026-10-27 GPT-6.1-Terra 补样 |
| 版本目录 | `versions/20261020-residual-gate`（目录日期为创建日期） |
| 参考库 | 102 份有效答卷、25 个型号／模式组、5 个家族 |
| 架构 | 135 维手工特征 MLP ＋ 4096 维哈希 n-gram TF-IDF 逻辑回归 ＋ 56 参数局部残差门控 |
| 依赖 | Python 3、NumPy（`numpy>=1.26,<3`），不需要 GPU、不需要联网 |
| 大小 | 包含训练与评估脚本的完整最新版；旧版本快照未纳入 |

本包是 `20261020-residual-gate` 的完整副本：推理、训练、评估、参考库、榜单与说明都在包内，路径相对本包计算。删除原来的 `nano-classfier` 后，双击 `最新版_粘贴答案测试.bat` 仍可启动。更早的版本目录、zip 与历史备份没有并入。

---

## 1. 快速开始

**Windows：** 直接双击根目录的 `最新版_粘贴答案测试.bat`。浏览器会自动打开 `http://127.0.0.1:端口号/`，粘贴答卷或选择文件后开始测试；关闭命令窗口即停止服务。

也可以带一份答卷文件启动：

```bat
最新版_粘贴答案测试.bat "C:\path\to\answer.txt"
```

**命令行：**

```bat
cd versions\20261020-residual-gate\lite-questionnaire-model
python -B residual_demo.py                 :: 打开网页
python -B residual_demo.py answer.txt      :: 打开网页并预载答卷
python -B residual_demo.py --json answer.txt   :: 只输出对照 JSON
python -B residual_demo.py --no-browser --port 8765
```

安装依赖：

```bat
python -m pip install -r versions\20261020-residual-gate\lite-questionnaire-model\requirements.txt
```

验证方式：`--json` 对 `nano-now/gpt6sol-01.txt` 应输出 GPT 首选；服务启动后 `/api/status` 回报 `samples: 102, families: 5`。

---

## 2. 目录结构

```
nano-classfier-最新版/
├── README.md                       ← 本文件（使用说明／文件清单）
├── README.en.md                    ← 英文版（English version）
├── CURRENT_VERSION.json            ← 机器可读的当前版本标识
├── VERSIONS.md                     ← 版本沿革与各版指标
├── external-holdout/               ← 11 份永不入库的回归样本
├── 更新的樣本/                     ← 与库内 gpt6.1terra-01/02 全文相同
├── 识别结果/                       ← 这两份答卷的识别输出
├── 最新版_粘贴答案测试.bat          ← 唯一入口
├── 文档/
│   ├── 架构入门_白话版.docx          ← 给第一次接触的人（先读这份）
│   ├── 当前分类器架构与原理说明.md   ← 完整技术文档
│   ├── 当前分类器架构与原理说明.docx
│   └── 核对数据（4 个 JSON）
└── versions/
    ├── README.md                   ← 版本索引
    └── 20261020-residual-gate/
        ├── README.md               ← 本版摘要与指标
        ├── PROTOCOL.md             ← 残差门控实验协议
        ├── nano-now/               ← 参考答卷库（103 个文件／102 份有效，必要）
        └── lite-questionnaire-model/
            ├── residual_demo.py / residual_demo.html   ← 本机测试页
            ├── *.py                                    ← 推理与评分程序
            ├── *.npz                                   ← 已训练权重
            ├── *.json                                  ← 评估、榜单、复核数据
            └── *.md                                    ← 本版报告
```

`最新版_粘贴答案测试.bat` 以自身位置推算路径，因此整个文件夹可任意搬移，但**内部结构不可更改**（尤其 `nano-now` 必须与 `lite-questionnaire-model` 同层于版本目录下）。

---

## 3. 必要文件（缺少即无法启动）

### 3.1 入口与页面
| 文件 | 作用 |
|---|---|
| `最新版_粘贴答案测试.bat` | 启动脚本，切换到版本目录并执行 `residual_demo.py` |
| `residual_demo.py` | 本机 HTTP 服务、推理流程、AI 复核路由 |
| `residual_demo.html` | 单页测试界面（粘贴／打开文件／评分／复核） |

### 3.2 模型与权重
| 文件 | 作用 |
|---|---|
| `neural_core.py` + `neural-balanced.npz` | 135 维手工特征 MLP |
| `nlp_core.py` + `nlp-model.npz` | 哈希 n-gram TF-IDF ＋ 逻辑回归 |
| `residual_gate.py` + `residual-gate.npz` | 56 参数局部残差门控与几何融合 |
| `feature_profiles.py`、`questionnaire_features.py`、`behavior_features.py`、`sequence_features.py`、`open_cues.py`、`extra_features.py` | 特征抽取 |

### 3.3 判卷与评分
| 文件 | 作用 |
|---|---|
| `ability.py`、`ability_rules.py`、`ability_methods.py` | A 题内容评分与规则 |
| `strict_consistency.py`、`scoring_text.py` | 一致性上限、文本抽取 |
| `result_scoring.py`、`dual_scoring.py`、`score_output.py` | 结果分、严格格式版、输出汇总 |
| `ability_ranking.py`、`dual_ranking.py` | 型号榜单与相对分位 |
| `sample_reviews.py` + `sample_reviews.json` | 逐题复核记录（评分依据） |
| `result_reviews.json` | 结果分复核记录 |
| `sample_annotations.py` + `sample_annotations.json` | 样本注记与疑似标记 |
| `sample_provenance.json` | 样本来源 |

### 3.4 分类器与复核
| 文件 | 作用 |
|---|---|
| `classifier.py` | 加载参考库、指纹、标准化答案 |
| `consensus_guard.py` | 一致性保护 |
| `known_references.py` + `known_external_references.json` | 已知外部参考比对 |
| `split_answers.py` | 答卷切分／读取文件 |
| `ai_review.py` | 手动／外部 AI 辅助复核（Key 只存内存） |
| `CURRENT_RELEASE.json` | 页面显示的版本标签 |
| `evaluation-residual-gate.json` | 启动时读取各口径阈值，签名须与权重一致 |

### 3.5 参考库
`versions/20261020-residual-gate/nano-now/` 共 103 个 `.txt`：其中 `deepseekv4.1flash1017-03.txt` 依 `sample_annotations.json` 注记（用户标记为疑似离群）排除，实际计入 102 份有效参考。程序会加载全部文件并比对 `reference_signature`，**不可删改**；增减样本需重新训练并更新权重。

---

## 4. 重要文件（非启动必需，但属核心资产）

| 文件 | 说明 |
|---|---|
| `model-ranking.json`、`MODEL_RANKING.md` | 24 组型号综合榜（200 分制），页面复核写入时的一致性快照 |
| `result-ranking.json`、`RESULT_RANKING.md` | 只看最终结果的独立榜（100 分制） |
| `HAIKU45_INGESTION_REPORT_20261021.md` | 历史 98→100 份补样报告 |
| `HAIKU45_SCORING_AUDIT_20261023.md` | 两份 Haiku4.5 逐题复核 |
| `GPT_OSS_INGESTION_REPORT_20261021.md`、`GPT_OSS_SCORING_AUDIT_20261021.md` | 95→98 份 GPT-OSS 120B 补样与复核（历史） |
| `RESIDUAL_GATE_REPORT.md` | 残差门控完整实验报告（含对照、救回／新增错误、库外回归） |
| `AI_REVIEW_UI.md`、`AI_REVIEW_UI_VALIDATION_20261024.json` | 页面复核功能说明与验证记录 |
| `NEAREST_SPACE_REPORT.md`、`nearest-space-report.json`、`evaluate_nearest.py` | 2026-10-27 型号诊断改用 135 维特征空间的对照报告与可复跑脚本 |
| `PROTOCOL.md` | 门控实验的固定协议（防止事后调参） |
| `VERSIONS.md`、`CURRENT_VERSION.json`、`versions/README.md` | 版本沿革与索引 |
| `文档/当前分类器架构与原理说明.md` / `.docx` | **主技术文档**：任务定义、特征、三模型融合、门控、评估口径与限制 |
| `文档/架构入门_白话版.docx` | **白话入门**：用比喻解释三个零件、数字怎么读、五个常见误解；适合第一次接触 |
| `文档/架构核对数据_20261024.json`、`门控权重核对_20261024.json`、`文档核对记录_20261024.json`、`门控权重核对_20261027.json` | 文档与程序／权重的一致性核对数据（20261027 为补样后重算） |

---

## 5. 当前指标与注意事项

- 逐份留一：baseline 91/102、gated 91/102、selected 91/102。
- 整型号留一：baseline 63/102、gated 65/102、selected 63/102。
- 固定库外回归：10/11（已多次观察，非无偏选型集）。
- 2026-10-27 补入两份 GPT-6.1-Terra（`gpt6.1terra`，2 份），参考库 100→102、型号组 24→25；三个权重与评估同步重训，签名 `4aa85bb1…`。
- 这两份样本的 A 题尚未逐题复核，榜单分数为暂估（综合 176.58/200、结果 89.00/100），已标注待复核份数。
- **`gate_policy_enabled: false`**：保守策略未通过启用检查，默认仍使用原几何融合；页面可切换查看强制门控结果。
- 家族分数未经概率校准，不能当作「身份正确概率」；最近样本与型号候选只是诊断，不参与投票或分数融合。
- 2026-10-27 起，型号候选与最近样本改用 135 维特征空间（与分类器同一空间），型号候选按型号组取平均距离排序；权重、参考库与家族决策未变。逐份留一下三份候选的型号命中率由 42.2%（TF-IDF）提升至 84.3%（135 维组均值），但仍属「已知型号查表」，不能识别库外新型号。
- `文档/` 的 `.md` 已同步到上述变更；`.docx` 是直接改写原档的对应段落，其余格式与内容未动。本包内 `当前分类器架构与原理说明.docx` 因被 Word 占用暂未覆盖，最新内容先以 `当前分类器架构与原理说明_20261027.docx` 提供；关闭 Word 后可用它覆盖原文件（两者内容相同，差别只在文件名）。
- 系统回答的是「这份答卷更像哪个已知家族」，不是对真实模型身份的认证。
- AI 复核需自行在页面设置 API（页脚小圆点 2.4 秒内连点 5 次）；Key 只存在于本次服务内存，不写入文件；复核结果需人工确认后才写入，且不进入训练。
- 重新启动服务才会刷新权重与参考库缓存。

---

## 6. 训练、识别与未纳入内容

当前三个权重已在这 102 份、A01–B12 答卷上训练完成，签名 `4aa85bb1…`。本包只认这 24 题；答卷里多出的题号会让拆分结果变成 needs_review，不会进入 135 维特征。题目数量会变的自适应特征在 `模型识别\mini-classifier`。

识别一份答卷：

```bat
cd versions\20261020-residual-gate\lite-questionnaire-model
python -B residual_demo.py --json "..\..\..\更新的樣本\新增 文字文件.txt"
python -B predict_residual_gate.py "..\..\..\更新的樣本\新增 文字文件.txt"
```

重训会写新文件，不会覆盖已发布权重。`train.py`、`train_nlp.py` 与 `evaluate_residual_gate.py` 在输出文件已存在时会拒绝执行，需要改用新的 `--output`／`--bundle`。

固定库外回归样本在 `external-holdout/`。`regression_suite.py` 已改为读取这个目录。

没有放进本包的是更早版本目录、zip、历史备份，以及指向那些旧目录的一次性实验脚本所依赖的旧程序。这些脚本不是目前的启动、训练或识别入口。`更新的樣本` 两份文件与参考库中的 `gpt6.1terra-01.txt`、`gpt6.1terra-02.txt` 全文相同，识别结果见 `识别结果/`。
