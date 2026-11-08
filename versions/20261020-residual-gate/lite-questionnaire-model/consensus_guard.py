"""Near-neighbor diagnostics only; neural scores and decisions stay unchanged."""
import numpy as np

POLICY_VERSION = 'neural-only-diagnostic-1'
NEIGHBOR_WEIGHT = 0.0


def apply_consensus_guard(neural_ranking, distance, references):
    indices = np.argsort(distance)[:3]
    nearest = [references[i]['family'] for i in indices]
    weights = 1 / np.maximum(np.asarray(distance)[indices], .05) ** 2
    total = float(weights.sum())
    votes = {row['label']: float(sum(w for i, w in zip(indices, weights)
             if references[i]['family'] == row['label']) / total) if total else 0.
             for row in neural_ranking}
    neural_top = neural_ranking[0]['label']
    conflict = (len(nearest) == 3 and nearest[0] == nearest[1]
                and nearest[0] != neural_top and neural_top not in nearest)
    ranking = [dict(row, neural_score=row['score'], neighbor_score=votes[row['label']])
               for row in neural_ranking]
    info = dict(version=POLICY_VERSION, applied=False, conflict_detected=conflict,
                neural_top=neural_top, neighbor_family=nearest[0] if nearest else None,
                nearest_families=nearest, neural_weight=1., neighbor_weight=0., scores=votes,
                rule='最终排序只使用神经网络；近邻仅作相似性与冲突提示，不改变识别分、排序或拒识。',
                note='近邻可能同源或高度相关，不能视作多份独立证据；最近参考型号不等于识别型号。网络分未校准，不是身份概率。')
    return ranking, info
