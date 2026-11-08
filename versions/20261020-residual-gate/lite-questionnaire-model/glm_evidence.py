"""GLM 证据块：把刚才在 78 份参考库上验证过的 4 个 GLM 独有信号，
从通用块里显式提出来单独加权（原始信号仍在各自块中，此处是显式复述）。

依据（GLM vs 其它家族，n_GLM=7）：
  B04_adjacent_same_fraction  d=1.58  p=0.0003   GLM 0.149 / 其它 0.056
  B07_personal_followup_cue   d=-0.72 p=0.098    GLM 0.000 / 其它 0.324
  B08_age_height_example_cue  d=1.12  p=0.029    GLM 0.429 / 其它 0.085
  B01 是否选 7                GLM 6/7            （单次选择，家族内高度一致）

这四个变量对任何家族都按同一规则提取，不读取文件名、型号或标签。
"""
from behavior_features import behavior_feature_map
from sequence_features import sequence_feature_map

GLM_EVIDENCE_NAMES = ('b04_adjacent_same', 'b07_no_followup', 'b08_height_age', 'b01_is_seven')


def glm_evidence(answers):
    seq = sequence_feature_map(answers)
    beh = behavior_feature_map(answers)
    b01 = answers.get('B01', '').strip()
    return [
        float(seq.get('B04_adjacent_same_fraction', 0.0)),
        1.0 - float(beh.get('B07_personal_followup_cue', 0.0)),
        float(beh.get('B08_age_height_example_cue', 0.0)),
        float(b01 == '7'),
    ]
