"""Publish a provisional mean-score model leaderboard and relative grade ruler."""
import argparse
import json
import sys
from pathlib import Path
from classifier import DEFAULT_REFERENCES, load_references
from ability import score_points
from ability_ranking import build_model_ranking, apply_relative_grade


def main():
    if hasattr(sys.stdout, 'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='按型号均分分档。始终满分列为MAX，唯一的非满分榜首授予S+。')
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--output', default='model-ranking.json')
    parser.add_argument('--report', default='MODEL_RANKING.md')
    args = parser.parse_args()
    if Path(args.output).exists() or Path(args.report).exists():
        raise ValueError('报告已存在，请指定新的 --output 和 --report。')
    records, skipped = load_references(args.references)
    if skipped: raise ValueError('请先处理被排除的参考文件：'+json.dumps(skipped, ensure_ascii=False))
    ranking = build_model_ranking(records)
    ranking['answer_grades'] = [dict(source=r['source'], ability=apply_relative_grade(score_points(r), ranking, record=r)) for r in records]
    lines = ['# 当前型号的相对能力暂估排名', '', ranking['note'], '',
             '评分版本：'+ranking['scoring_version']+'。分数与MAX均为当前规则下的暂估；待复核回答数大于0表示尚未完成全部推导复核。', '',
             '| 含全部排名 | 型号标签 | 回答数 | 含标记均分 | 排除标记均分 | 排除后回答数 | 含全部档位 | 排除后排名/档位 | 待复核回答数 |', '|---:|---|---:|---:|---:|---:|---|---|---:|']
    for row in ranking['models']:
        without = row['mean_excluding_marked']
        comparison = '无可用样本' if without is None else str(row['rank_excluding_marked'])+'/'+row['tier_excluding_marked']
        label = row['model']+(' †' if row['annotations'] else '')
        lines.append('|'+str(row['rank'])+'|'+label+'|'+str(row['sample_count'])+'|'+format(row['mean_score'], '.2f')+'|'+('—' if without is None else format(without, '.2f'))+'|'+str(row['sample_count_excluding_marked'])+'|'+row['tier']+'|'+comparison+'|'+str(row['pending_review_samples'])+'|')
    lines += ['', '含标记均分保留全部回答；排除标记均分剔除明确标记的回答。其他同型号样本不自动标记。两份榜单分别重新计算排名和档位。', '']
    for row in ranking['models']:
        lines.extend('† '+a['source']+'：'+a['label']+'。' for a in row['annotations'])
    lines += ['', '按型号的档位分布：'+json.dumps(ranking['distribution_by_model'], ensure_ascii=False), '',
              '排除标记后按型号的档位分布：'+json.dumps(ranking['comparison_without_marked']['distribution_by_model'] if ranking['comparison_without_marked'] else {}, ensure_ascii=False), '',
              '每一份回答都是满分的型号同列为MAX，可以并列；分数不能再高，不再参加S/A/B/C四分。S+只给唯一的非满分榜首。型号同分不强行拆档。', '',
              '当前唯一非满分榜首回答数：'+str(next((r['sample_count'] for r in ranking['models'] if r['model'] == ranking['winner']), '无'))+'；样本数量不同，排名不是经过独立验证的通用能力排序。所有开放推导评分仍待人工复核。', '',
              '## 单份回答的显示', '',
              '已知参考回答按A题原文摘要关联型号，显示该型号均分档位，并同时保留本份原始加权分。陌生回答按当前参考型号之间的分界线对照S/A/B/C，不授予MAX或S+；原文命中不证明模型身份。', '',
              '| 分界 | 分数中点 |', '|---|---:|']
    for boundary in ranking['boundaries']:
        lines.append('|'+boundary['upper_band']+'/'+boundary['lower_band']+'|'+format(boundary['cutoff'], '.4f')+'|')
    lines += ['', '模型均分决定分档，本份回答分数和逐题扣分不随此规则改变。参考集发生变化时，相对档位与分界线会变化。', '',
              '| 回答 | 内容与过程原分 | 一致性扣分 | 本份严格分 | 型号均分 | 型号相对档位 |', '|---|---:|---:|---:|---:|---|']
    for answer in ranking['answer_grades']:
        a = answer['ability']
        label = answer['source']+(' †' if a['sample_annotations'] else '')
        lines.append('|'+label+'|'+format(a['content_provisional_score'], '.2f')+'|'+format(a['consistency_deduction'], '.2f')+'|'+format(a['provisional_score'], '.2f')+'|'+format(a['tier_score'], '.2f')+'|'+a['provisional_tier']+'|')
    with Path(args.output).open('x', encoding='utf-8') as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)
    with Path(args.report).open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(json.dumps({k: ranking[k] for k in ('winner', 'model_count', 'sample_count', 'distribution_by_model', 'boundaries')}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
