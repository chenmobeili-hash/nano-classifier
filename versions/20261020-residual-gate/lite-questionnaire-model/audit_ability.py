"""Generate reviewable before/after ability scores without changing answers."""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from classifier import DEFAULT_REFERENCES, load_references
from ability import score_points as score_ability, WEIGHTS, VERSION
from ability_v1_snapshot import score_ability as legacy_score


def main():
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser()
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--output', default='ability-audit-current.json')
    parser.add_argument('--report', default='ABILITY_AUDIT_CURRENT.md')
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.report).exists():
        raise ValueError('报告已存在，请指定新的 --output 和 --report，避免覆盖。')
    records, skipped = load_references(args.references)
    rows = []
    for record in records:
        previous = legacy_score(record)
        current = score_ability(record)
        rows.append({'source': record['source'], 'claimed_model': record['model'],
                     'previous_score': previous['provisional_score'], 'previous_tier': previous['provisional_tier'],
                     'ability': current})
    result = {'version': VERSION, 'weights': WEIGHTS, 'samples': len(rows), 'skipped': skipped,
              'previous_distribution': dict(Counter(r['previous_tier'] for r in rows)),
              'new_distribution': dict(Counter(r['ability']['provisional_tier'] for r in rows)),
              'new_mean': round(sum(r['ability']['provisional_score'] for r in rows)/len(rows), 2),
              'rows': rows,
              'note': '对比历史v1与当前判题规则；本报告使用历史绝对档位，并非当前相对型号榜单。包含自动暂估及已存复核；方法规则支持不代表完整证明已验证。权重为任务设计约定，不按当前型号排名拟合。未识别项不直接判错，区间不是概率置信区间。'}
    lines = ['# 当前'+str(len(rows))+'份回答的加权暂估评分对照', '', result['note'], '',
             '| 档位 | 旧规则 | 新规则 |', '|---|---:|---:|']
    for tier in ('S 优秀', 'A 较强', 'B 中等', 'C 待提升'):
        lines.append('|'+tier+'|'+str(result['previous_distribution'].get(tier, 0))+'|'+str(result['new_distribution'].get(tier, 0))+'|')
    lines += ['', '新平均暂估分：'+str(result['new_mean'])+'。', '',
              '| 样本 | 旧分 | 新分 | 新暂估档 | 待复核区间 | 未识别/冲突分项 |', '|---|---:|---:|---|---|---:|']
    for row in rows:
        current = row['ability']
        lines.append('|'+row['source']+'|'+str(row['previous_score'])+'|'+str(current['provisional_score'])+'|'+current['provisional_tier']+'|'+str(current['score_interval'])+'|'+str(current['unresolved_components'])+'|')
    lines += ['', '开放推导题全部保留人工复核。上表“未识别/冲突分项”不包含已识别但仍需复核的推导，因此0并不表示已经人工复核完成。', '']
    for row in rows:
        lines += ['## '+row['source'], '']
        for q in row['ability']['questions']:
            if q['provisional_points'] == q['weight']: continue
            lines.append('- '+q['question']+'：暂估'+str(q['provisional_points'])+'/'+str(q['weight'])+'分。')
            for part in q['components']:
                if part['fraction'] == 1: continue
                value = '待确认，暂不计入估分' if part['fraction'] is None else str(part['fraction'])+'得分比例'
                lines.append('  - '+part['name']+'：'+value+'；'+part['evidence'])
                lines.extend('    - '+issue.replace('\n', ' ') for issue in part['issues'])
        lines.append('')
    with Path(args.output).open('x', encoding='utf-8') as file:
        file.write(json.dumps(result, ensure_ascii=False, indent=2))
    with Path(args.report).open('x', encoding='utf-8') as file:
        file.write('\n'.join(lines))
    print(json.dumps({k: v for k, v in result.items() if k != 'rows'}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
