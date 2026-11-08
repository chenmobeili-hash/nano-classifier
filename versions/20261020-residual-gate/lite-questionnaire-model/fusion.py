"""晚期融合：NLP 线性模型 × 神经网络 MLP 的概率合成（固定规则，参数不调）。

规则只有两个，都预先固定：
  mean  两个概率向量算术平均（各 0.5）
  geo   log 概率平均后重新归一化（几何平均，各 0.5）

门槛按"目标覆盖率 = 96.84%"在留出分数上反解得到（见 evaluation-fusion.json），
不在这里按样本调整。两个模型都必须是同一参考库训练出来的（签名校验）。
"""
import numpy as np

import neural_core as nc
import nlp_core as N
from nlp_classifier import identify_nlp

RULES = ('mean', 'geo')
# evaluation-fusion.json：逐份留出按 96.84% 覆盖率反解出的门槛
CALIBRATED_MIN_MARGIN = {'mean': 0.0876, 'geo': 0.0746}


def _resolve(model, loader, label):
    if model is None:
        return loader()
    if isinstance(model, dict):
        return model
    return loader(model)


def fused_scores(record, mlp_model, nlp_model, rule='geo'):
    """返回 (融合分, mlp 分, nlp 分)；类别顺序沿用模型内部的 sorted(classes)。"""
    if rule not in RULES:
        raise ValueError('融合规则必须是 ' + ' 或 '.join(RULES))
    if mlp_model['meta']['classes'] != nlp_model['meta']['classes']:
        raise ValueError('两个模型的类别顺序不一致，无法融合。')
    p_mlp = np.asarray(nc.predict_scores(record, mlp_model), float)
    p_nlp = np.asarray(N.predict_record(record, nlp_model)[0], float)
    if rule == 'mean':
        fused = (p_mlp + p_nlp) / 2.0
    else:
        logits = .5 * np.log(np.clip(p_mlp, 1e-12, 1)) + .5 * np.log(np.clip(p_nlp, 1e-12, 1))
        exp = np.exp(logits - logits.max())
        fused = exp / exp.sum()
    return fused, p_mlp, p_nlp


def identify_fused(record, references, rule='geo', min_margin=None, min_prob=0.0,
                   mlp_model=None, nlp_model=None, reference_vectors=None):
    """与 identify_nlp 同结构，只是分数换成融合分。"""
    mlp = _resolve(mlp_model, nc.load_network, 'mlp')
    nlp = _resolve(nlp_model, N.load_model, 'nlp')
    for model, name in ((mlp, '神经网络'), (nlp, 'NLP')):
        if model['meta']['reference_signature'] != N.reference_signature(references):
            raise ValueError('当前参考库与%s权重训练数据不一致，请用该参考库重新训练。' % name)
    fused, p_mlp, p_nlp = fused_scores(record, mlp, nlp, rule)
    if min_margin is None:
        min_margin = CALIBRATED_MIN_MARGIN[rule]
    result = identify_nlp(record, references, min_margin=min_margin, min_prob=min_prob,
                          model=nlp, reference_vectors=reference_vectors,
                          scores=fused, engine_label='fused')
    result['fuse_rule'] = rule
    result['components'] = {'mlp': dict(zip(mlp['meta']['classes'], p_mlp.tolist())),
                            'nlp': dict(zip(nlp['meta']['classes'], p_nlp.tolist()))}
    result['note'] = ('候选由 NLP 线性模型与神经网络 MLP 的概率按固定规则（%s）合成，两路权重各 0.5，'
                      '未在留出集上调整；近邻仅作诊断。融合分未校准，不是身份正确概率。' % rule)
    return result
