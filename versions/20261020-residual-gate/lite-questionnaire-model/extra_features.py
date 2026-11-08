"""通用行为特征增益 (extra block, v2)。

依据参考库上的效应量筛选（GLM vs 其它家族，n_GLM=7，置换检验）：
  B12_em    d=2.21  p<0.001   7/7 份 GLM 都出现破折号；其它家族中位数 0
  B07_em    d=2.06  p<0.001
  B08_cjk   d=1.20  p=0.006   GLM 全部 ≥185 汉字
  self_correction_density  31/78 有分布（唯一在库内真正铺开的原候选）

剔除的原候选（激活 0/78 或 1/78，等于隔离探测器，会造成假新颖性）：
  script_leak / latin_word_density / emoji_present

所有变量对任何家族的样本用同一套定义提取，不读取文件名、型号或标签。
"""
import re

EM_DASH = re.compile(r'[—―]+')
CJK = re.compile(r'[\u4e00-\u9fff]')
CORRECTION_MARKERS = ('不对', '重新', '应为', '更正', '等等', '实际上', '重写', '修正', '矛盾', '再核')
PHRASE_CUES = ('此外', '规定答案', '特别理解', '答案必须')

EXTRA_NAMES = ('em_dash_density', 'b08_length_cue', 'self_correction_density', 'phrasing_cue')


def extra_features(answers):
    """返回 4 维，全部落在 [0, 1]。"""
    answers = {k: (v if isinstance(v, str) else '') for k, v in answers.items()}
    b_text = '\n'.join(answers.get('B%02d' % i, '') for i in range(1, 13))
    a_text = '\n'.join(answers.get('A%02d' % i, '') for i in range(1, 13))

    head = answers.get('B07', '') + answers.get('B12', '')
    em = len(EM_DASH.findall(head))
    em_density = min(em, 8) / 8.0

    b08_len = len(CJK.findall(answers.get('B08', '')))
    length_cue = min(max(b08_len - 150, 0), 150) / 150.0

    corrections = sum(a_text.count(m) for m in CORRECTION_MARKERS)
    correction_density = min(corrections, 10) / 10.0

    cues = sum(1 for p in PHRASE_CUES if p in b_text)
    phrasing = cues / len(PHRASE_CUES)

    return [em_density, length_cue, correction_density, phrasing]
