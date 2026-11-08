"""Publish the independent final-result ranking."""
import argparse
import json
from pathlib import Path
from classifier import DEFAULT_REFERENCES, load_references
from result_scoring import build_result_ranking


def main():
    p = argparse.ArgumentParser(description='仅最终结果的独立100分排行榜。')
    p.add_argument('--references', default=str(DEFAULT_REFERENCES))
    p.add_argument('--output', default='result-ranking.json')
    p.add_argument('--report', default='RESULT_RANKING.md')
    a = p.parse_args()
    if Path(a.output).exists() or Path(a.report).exists():
        raise ValueError('输出已存在，请指定新文件名。')
    records, skipped = load_references(a.references)
    if skipped:
        raise ValueError('参考库存在排除项：'+str(skipped))
    ranking = build_result_ranking(records)
    comparison = {m['model']:m for m in ranking['comparison_without_marked']['models']}
    lines = ['# 最终结果独立排名', '', ranking['policy'], '',
             '按型号均分排序；同分并列。身份标签沿用当前样本库，包含Grok Bot推断归因样本，来源见sample_provenance.json。', '',
             '| 排名 | 型号 | 份数 | 结果均分/100 | 排除疑似灰测后/100 | 待复核份数 |',
             '|---:|---|---:|---:|---:|---:|']
    for m in ranking['models']:
        lines.append(f"|{m['rank']}|{m['model']}|{m['sample_count']}|{m['mean_score']:.2f}|{comparison[m['model']]['mean_score']:.2f}|{m['pending_review_samples']}|")
    lines += ['', '| 样本 | 结果分/100 | 全对题数/12 | 未全对题目 |', '|---|---:|---:|---|']
    for s in ranking['samples']:
        r = s['result']
        wrong = '、'.join(q['question'] for q in r['questions'] if q['correct'] is not True) or '无'
        lines.append(f"|{s['source']}|{s['score']:.2f}|{r['correct_questions']}|{wrong}|")
    with Path(a.output).open('x', encoding='utf-8') as f:
        json.dump(ranking, f, ensure_ascii=False, indent=2)
    with Path(a.report).open('x', encoding='utf-8') as f:
        f.write('\n'.join(lines)+'\n')


if __name__ == '__main__':
    main()
