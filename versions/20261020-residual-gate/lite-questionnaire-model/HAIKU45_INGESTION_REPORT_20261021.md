# Haiku 4.5 Extended补样与验证（2026-10-21）

当前版本目录仍为 versions/20261020-residual-gate；有效库98→100份、23→24个型号/模式组、五家族。新增两份用户确认的Claude Haiku4.5，单独分组claudehaiku4.5-extended；extended来自来源文件夹名，未推测具体effort或平台。

开始读取时来源文件夹有三份物理文件，其中新增 文字文件 (2).txt与(3).txt逐字节相同。本次选择无编号及(2)两份；训练期间(3)被用户更新，未纳入本次两份训练批次。原始文件不改动。

|入库名|原文件|入库前原融合 / 门控 / 保守|重训后原融合 / 门控 / 保守|
|---|---|---|---|
|claudehaiku4.5-extended-01.txt|新增 文字文件.txt|DeepSeek / DeepSeek / DeepSeek|Claude / Claude / Claude|
|claudehaiku4.5-extended-02.txt|新增 文字文件 (2).txt|GLM / GLM / GLM|Claude / Claude / Claude|

重训后的同卷自测含训练样本，不算独立识别率。下列留出每折重新训练、只在训练子集拟合统计量；新样本没有用于改特征、选新参数或调整启用规则。

|口径|更新前原融合|更新后原融合|更新前强制门控|更新后强制门控|更新前保守策略|更新后保守策略|
|---|---:|---:|---:|---:|---:|---:|
|逐份留一|90/98（91.84%）|89/100（89.00%）|90/98（91.84%）|89/100（89.00%）|90/98（91.84%）|89/100（89.00%）|
|整型号留一|59/98（60.20%）|62/100（62.00%）|61/98（62.24%）|63/100（63.00%）|59/98（60.20%）|62/100（62.00%）|

|新答卷|逐份留出：原融合 / 强制门控 / 保守|整型号留出：原融合 / 强制门控 / 保守|
|---|---|---|
|claudehaiku4.5-extended-01.txt|DeepSeek / DeepSeek / DeepSeek|DeepSeek / DeepSeek / DeepSeek|
|claudehaiku4.5-extended-02.txt|Claude / Claude / Claude|GLM / GLM / GLM|

逐份留出可使用另一份Haiku4.5；整型号留出同时排除两份Haiku4.5。

|原98份配对比较|之前正确|更新后正确|救回|新增错误|
|---|---:|---:|---|---|
|逐份·原融合|90/98|88/98|无|deepseek-app-thinking-03.txt, deepseekv4.1flash-weak-02.txt|
|逐份·强制门控|90/98|88/98|无|deepseek-app-thinking-03.txt, deepseekv4.1flash-weak-02.txt|
|逐份·保守|90/98|88/98|无|deepseek-app-thinking-03.txt, deepseekv4.1flash-weak-02.txt|
|整型号·原融合|59/98|62/98|claudehaiku5.5-03.txt, claudehaiku5.5-05.txt, claudehaiku5.5-08.txt, claudehaiku5.5-09.txt|deepseekv4.1flash1017-05.txt|
|整型号·强制门控|61/98|63/98|claudehaiku5.5-03.txt, claudehaiku5.5-05.txt, claudehaiku5.5-08.txt, claudehaiku5.5-09.txt|deepseek-app-thinking-02.txt, deepseekv4.1flash1017-05.txt|
|整型号·保守|59/98|62/98|claudehaiku5.5-03.txt, claudehaiku5.5-05.txt, claudehaiku5.5-08.txt, claudehaiku5.5-09.txt|deepseekv4.1flash1017-05.txt|

|固定库外回归|更新前|更新后|
|---|---:|---:|
|baseline|9/11（81.82%）|10/11（90.91%）|
|gated|9/11（81.82%）|10/11（90.91%）|
|selected|9/11（81.82%）|10/11（90.91%）|

固定11份是开发期间已经观察过的回归集，未用来训练或调参，不算新盲测。具体变动见HAIKU45_VALIDATION_20261021.json。

全库保守策略：内部检查未通过，沿用原融合。

阈值沿用原协议的固定目标覆盖率，由本次OOF分差分位数计算。上表是强制首选正确率；接受内准确率和实际覆盖见完整评估JSON，内部校准不当作独立拒识验证。

两份新答卷的能力分尚待逐题复核。本次不改变判卷规则和权重，保留GPT-OSS已完成的复核记录。更新了与100份库一致的残差权重、单独MLP/NLP权重、评估、门槛及静态榜单。

原98份部署保存到历史备份/模型入库/20261021-haiku45-extended。关闭旧服务窗口后重新启动最新版_粘贴答案测试.bat；页面顶部应显示Haiku4.5补样及100份参考。

