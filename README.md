# nano-classfier 最新版（精簡整理包）

固定問卷驅動的輕量級黑盒模型家族歸因工具。貼入一份 A01–A12／B01–B12 答卷，即可得到五家族（Claude／DeepSeek／GLM／GPT／Qwen）相對分數、首選家族、最近參考樣本，以及獨立的問卷評分。

| 項目 | 內容 |
|---|---|
| 版本標籤 | 殘差門控融合 · 2026-10-27 GPT-6.1-Terra補樣 |
| 版本目錄 | `versions/20261020-residual-gate`（目錄日期為建立日期） |
| 參考庫 | 102 份有效答卷、25 個型號／模式組、5 個家族 |
| 架構 | 135 維手工特徵 MLP ＋ 4096 維哈希 n-gram TF-IDF 邏輯回歸 ＋ 56 參數局部殘差門控 |
| 依賴 | Python 3、NumPy（`numpy>=1.26,<3`），不需 GPU、不需聯網 |
| 大小 | 含訓練與評估腳本的完整最新版；舊版本快照未納入 |

本包是 `20261020-residual-gate` 的完整副本：推理、訓練、評估、參考庫、榜單與說明都在包內，路徑相對本包計算。刪除原來的 `nano-classfier` 後，雙擊 `最新版_粘贴答案测试.bat` 仍可啟動。更早的版本目錄、zip 與歷史備份沒有併入。

---

## 1. 快速開始

**Windows：** 直接雙擊根目錄的 `最新版_粘贴答案测试.bat`。瀏覽器會自動開啟 `http://127.0.0.1:埠號/`，貼上答卷或選擇檔案後開始測試；關閉命令視窗即停止服務。

也可以帶一份答卷檔案啟動：

```bat
最新版_粘贴答案测试.bat "C:\path\to\answer.txt"
```

**命令列：**

```bat
cd versions\20261020-residual-gate\lite-questionnaire-model
python -B residual_demo.py                 :: 開網頁
python -B residual_demo.py answer.txt      :: 開網頁並預載答卷
python -B residual_demo.py --json answer.txt   :: 只輸出對照 JSON
python -B residual_demo.py --no-browser --port 8765
```

安裝依賴：

```bat
python -m pip install -r versions\20261020-residual-gate\lite-questionnaire-model\requirements.txt
```

驗證方式：`--json` 對 `nano-now/gpt6sol-01.txt` 應輸出 GPT 首選；服務啟動後 `/api/status` 回報 `samples: 102, families: 5`。

---

## 2. 目錄結構

```
nano-classfier-最新版/
├── README.md                       ← 本文件（使用說明／檔案清單）
├── CURRENT_VERSION.json            ← 機器可讀的當前版本標識
├── VERSIONS.md                     ← 版本沿革與各版指標
├── external-holdout/               ← 11 份永不入庫的回歸樣本
├── 更新的樣本/                     ← 與庫內 gpt6.1terra-01/02 全文相同
├── 识别结果/                       ← 這兩份答卷的識別輸出
├── 最新版_粘贴答案测试.bat          ← 唯一入口
├── 文档/
│   ├── 架构入门_白话版.docx          ← 給第一次接觸的人（先讀這份）
│   ├── 当前分类器架构与原理说明.md   ← 完整技術文檔
│   ├── 当前分类器架构与原理说明.docx
│   └── 核对数据（4 個 JSON）
└── versions/
    ├── README.md                   ← 版本索引
    └── 20261020-residual-gate/
        ├── README.md               ← 本版摘要與指標
        ├── PROTOCOL.md             ← 殘差門控實驗協定
        ├── nano-now/               ← 參考答卷庫（103 檔／102 份有效，必要）
        └── lite-questionnaire-model/
            ├── residual_demo.py / residual_demo.html   ← 本機測試頁
            ├── *.py                                    ← 推理與評分程式
            ├── *.npz                                   ← 已訓練權重
            ├── *.json                                  ← 評估、榜單、複核資料
            └── *.md                                    ← 本版報告
```

`最新版_粘贴答案测试.bat` 以自身位置推算路徑，因此整個資料夾可任意搬移，但**內部結構不可更動**（尤其 `nano-now` 必須與 `lite-questionnaire-model` 同層於版本目錄下）。

---

## 3. 必要檔案（缺少即無法啟動）

### 3.1 入口與頁面
| 檔案 | 作用 |
|---|---|
| `最新版_粘贴答案测试.bat` | 啟動腳本，切換到版本目錄並執行 `residual_demo.py` |
| `residual_demo.py` | 本機 HTTP 服務、推理流程、AI 複核路由 |
| `residual_demo.html` | 單頁測試介面（貼上／開檔／評分／複核） |

### 3.2 模型與權重
| 檔案 | 作用 |
|---|---|
| `neural_core.py` + `neural-balanced.npz` | 135 維手工特徵 MLP |
| `nlp_core.py` + `nlp-model.npz` | 哈希 n-gram TF-IDF ＋ 邏輯回歸 |
| `residual_gate.py` + `residual-gate.npz` | 56 參數局部殘差門控與幾何融合 |
| `feature_profiles.py`、`questionnaire_features.py`、`behavior_features.py`、`sequence_features.py`、`open_cues.py`、`extra_features.py` | 特徵抽取 |

### 3.3 判卷與評分
| 檔案 | 作用 |
|---|---|
| `ability.py`、`ability_rules.py`、`ability_methods.py` | A 題內容評分與規則 |
| `strict_consistency.py`、`scoring_text.py` | 一致性上限、文字抽取 |
| `result_scoring.py`、`dual_scoring.py`、`score_output.py` | 結果分、嚴格格式版、輸出彙總 |
| `ability_ranking.py`、`dual_ranking.py` | 型號榜單與相對分位 |
| `sample_reviews.py` + `sample_reviews.json` | 逐題複核記錄（評分依據） |
| `result_reviews.json` | 結果分複核記錄 |
| `sample_annotations.py` + `sample_annotations.json` | 樣本註記與疑似標記 |
| `sample_provenance.json` | 樣本來源 |

### 3.4 分類器與複核
| 檔案 | 作用 |
|---|---|
| `classifier.py` | 載入參考庫、指紋、標準化答案 |
| `consensus_guard.py` | 一致性保護 |
| `known_references.py` + `known_external_references.json` | 已知外部參考比對 |
| `split_answers.py` | 答卷切分／讀檔 |
| `ai_review.py` | 手動／外部 AI 輔助複核（Key 只存記憶體） |
| `CURRENT_RELEASE.json` | 頁面顯示的版本標籤 |
| `evaluation-residual-gate.json` | 啟動時讀取各口徑門檻值，簽名須與權重一致 |

### 3.5 參考庫
`versions/20261020-residual-gate/nano-now/` 共 103 個 `.txt`：其中 `deepseekv4.1flash1017-03.txt` 依 `sample_annotations.json` 註記（使用者標記為疑似離群）排除，實際計入 102 份有效參考。程式會載入全部檔案並比對 `reference_signature`，**不可刪改**；增減樣本需重新訓練並更新權重。

---

## 4. 重要檔案（非啟動必需，但屬核心資產）

| 檔案 | 說明 |
|---|---|
| `model-ranking.json`、`MODEL_RANKING.md` | 24 組型號綜合榜（200 分制），頁面複核寫入時的一致性快照 |
| `result-ranking.json`、`RESULT_RANKING.md` | 只看最終結果的獨立榜（100 分制） |
| `HAIKU45_INGESTION_REPORT_20261021.md` | 歷史 98→100 份補樣報告 |
| `HAIKU45_SCORING_AUDIT_20261023.md` | 兩份 Haiku4.5 逐題複核 |
| `GPT_OSS_INGESTION_REPORT_20261021.md`、`GPT_OSS_SCORING_AUDIT_20261021.md` | 95→98 份 GPT-OSS 120B 補樣與複核（歷史） |
| `RESIDUAL_GATE_REPORT.md` | 殘差門控完整實驗報告（含對照、救回／新增錯誤、庫外回歸） |
| `AI_REVIEW_UI.md`、`AI_REVIEW_UI_VALIDATION_20261024.json` | 頁面複核功能說明與驗證記錄 |
| `NEAREST_SPACE_REPORT.md`、`nearest-space-report.json`、`evaluate_nearest.py` | 2026-10-27 型號診斷改用 135 維特徵空間的對照報告與可複跑腳本 |
| `PROTOCOL.md` | 門控實驗的固定協定（防止事後調參） |
| `VERSIONS.md`、`CURRENT_VERSION.json`、`versions/README.md` | 版本沿革與索引 |
| `文档/当前分类器架构与原理说明.md` / `.docx` | **主技術文檔**：任務定義、特徵、三模型融合、門控、評估口徑與限制 |
| `文档/架构入门_白话版.docx` | **白話入門**：用比喻解釋三個零件、數字怎麼讀、五個常見誤解；適合第一次接觸 |
| `文档/架构核对数据_20261024.json`、`门控权重核对_20261024.json`、`文档核对记录_20261024.json`、`门控权重核对_20261027.json` | 文檔與程式／權重的一致性核對資料（20261027 為補樣後重算） |

---

## 5. 當前指標與注意事項

- 逐份留一：baseline 91/102、gated 91/102、selected 91/102。
- 整型號留一：baseline 63/102、gated 65/102、selected 63/102。
- 固定庫外回歸：10/11（已多次觀察，非無偏選型集）。
- 2026-10-27 補入兩份 GPT-6.1-Terra（`gpt6.1terra`，2 份），參考庫 100→102、型號組 24→25；三個權重與評估同步重訓，簽名 `4aa85bb1…`。
- 這兩份樣本的 A 題尚未逐題複核，榜單分數為暫估（綜合 176.58/200、結果 89.00/100），已標註待複核份數。
- **`gate_policy_enabled: false`**：保守策略未通過啟用檢查，預設仍使用原幾何融合；頁面可切換檢視強制門控結果。
- 家族分數未經機率校準，不能當作「身份正確機率」；最近樣本與型號候選只是診斷，不參與投票或分數融合。
- 2026-10-27 起，型號候選與最近樣本改用 135 維特徵空間（與分類器同一空間），型號候選按型號組取平均距離排序；權重、參考庫與家族決策未變。逐份留一下三份候選的型號命中率由 42.2%（TF-IDF）提升至 84.3%（135 維組均值），但仍屬「已知型號查表」，不能識別庫外新型號。
- `文档/` 的 `.md` 已同步到上述變更；`.docx` 是直接改寫原檔的對應段落，其餘格式與內容未動。本包內 `当前分类器架构与原理说明.docx` 因被 Word 佔用暫未覆蓋，最新內容先以 `当前分类器架构与原理说明_20261027.docx` 提供；關閉 Word 後可用它覆蓋原名檔（兩者內容相同，差別只在檔名）。
- 系統回答的是「這份答卷更像哪個已知家族」，不是對真實模型身份的認證。
- AI 複核需自行在頁面設定 API（頁尾小圓點 2.4 秒內連點 5 次）；Key 只存在本次服務記憶體，不寫檔；複核結果需人工確認後才寫入，且不進入訓練。
- 重新啟動服務才會刷新權重與參考庫快取。

---

## 6. 訓練、識別與未納入內容

當前三個權重已在這 102 份、A01–B12 答卷上訓練完成，簽名 `4aa85bb1…`。本包只認這 24 題；答卷裡多出的題號會讓拆分結果變成 needs_review，不會進入 135 維特徵。題目數量會變的自適應特徵在 `模型识别\mini-classifier`。

識別一份答卷：

```bat
cd versions\20261020-residual-gate\lite-questionnaire-model
python -B residual_demo.py --json "..\..\..\更新的樣本\新增 文字文件.txt"
python -B predict_residual_gate.py "..\..\..\更新的樣本\新增 文字文件.txt"
```

重訓會寫新檔，不會覆蓋已發布權重。`train.py`、`train_nlp.py` 與 `evaluate_residual_gate.py` 在輸出檔已存在時會拒絕執行，需要改用新的 `--output`／`--bundle`。

固定庫外回歸樣本在 `external-holdout/`。`regression_suite.py` 已改為讀這個目錄。

沒有放進本包的是更早版本目錄、zip、歷史備份，以及指向那些舊目錄的一次性實驗腳本所依賴的舊程式。這些腳本不是目前的啟動、訓練或識別入口。`更新的樣本` 兩份文件與參考庫中的 `gpt6.1terra-01.txt`、`gpt6.1terra-02.txt` 全文相同，識別結果見 `识别结果/`。
