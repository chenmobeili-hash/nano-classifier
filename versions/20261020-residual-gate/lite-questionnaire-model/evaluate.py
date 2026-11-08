import argparse
import json
from pathlib import Path
from classifier import DEFAULT_REFERENCES, identify, inventory, load_references
from feature_profiles import PROFILES
from neural_core import train_network


def evaluate(records, mode, profile='balanced', engine='neural', seed=42):
    rows = []
    fitted = {}
    for query in records:
        refs = [r for r in records if (r['model'] != query['model'] if mode == 'model' else r['fingerprint'] != query['fingerprint'])]
        if query['family'] not in {r['family'] for r in refs}:
            rows.append({'source': query['source'], 'skipped': '留出后该家族无参考样本'})
            continue
        model = None
        if engine == 'neural':
            key = query['model'] if mode == 'model' else query['fingerprint']
            if key not in fitted:
                fitted[key] = train_network(refs, seed=seed)
            model = fitted[key]
        prediction = identify(query, refs, profile=profile, engine=engine, model=model)
        rows.append({'source': query['source'], 'expected': query['family'],
                     'candidate': prediction['candidate'], 'nearest_family': prediction['nearest_family'],
                     'margin': prediction['margin']})
    tested = [r for r in rows if 'skipped' not in r]
    accepted = [r for r in tested if r['candidate'] is not None]
    abstained = [r for r in tested if r['candidate'] is None]
    # 弃权不等于没信息：记录弃权样本里"最近家族"是否仍判对。
    abstained_nearest_hit = sum(r['nearest_family'] == r['expected'] for r in abstained)
    forced_correct = sum(r['nearest_family'] == r['expected'] for r in tested)
    return {'mode': mode, 'tested': len(tested),
            'nearest_accuracy': forced_correct/max(len(tested), 1),
            'coverage': len(accepted)/max(len(tested), 1),
            'accepted_accuracy': sum(r['candidate'] == r['expected'] for r in accepted)/len(accepted) if accepted else None,
            'abstained': len(abstained),
            'abstained_nearest_hits': abstained_nearest_hit,
            'abstained_nearest_accuracy': abstained_nearest_hit/len(abstained) if abstained else None,
            'abstention_gain_vs_forced': ((sum(r['candidate'] == r['expected'] for r in accepted)/len(accepted) if accepted else 0)
                                          - forced_correct/max(len(tested), 1)),
            'rows': rows}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--references', default=str(DEFAULT_REFERENCES))
    p.add_argument('--output', default='evaluation-current.json')
    p.add_argument('--profile', choices=PROFILES, default='balanced')
    p.add_argument('--engine', choices=('neural', 'distance'), default='neural')
    p.add_argument('--seed', type=int, default=42)
    a = p.parse_args()
    records, skipped = load_references(a.references)
    if a.engine == 'neural' and a.profile != 'balanced':
        p.error('神经网络仅支持balanced；其他特征方案请使用 --engine distance。')
    result = {'engine': a.engine, 'profile': a.profile, 'seed': a.seed, 'inventory': inventory(records, skipped),
              'sample_holdout': evaluate(records, 'sample', a.profile, a.engine, a.seed),
              'model_holdout': evaluate(records, 'model', a.profile, a.engine, a.seed),
              'note': '开发数据上的探索评估。神经网络每折从随机初值重新训练400轮，不读取发布权重；留出答案不参与标准化、梯度更新或轮数选择。型号标签仍来自文件名。'}
    with Path(a.output).open('x', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(json.dumps({k: {n: v for n, v in result[k].items() if n != 'rows'} for k in ('sample_holdout', 'model_holdout')}, ensure_ascii=False, indent=2))
