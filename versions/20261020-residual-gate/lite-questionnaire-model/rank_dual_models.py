"""Publish the /200 combined ranking, keeping both /100 scores visible."""
import argparse
import json
import sys
from pathlib import Path
from classifier import load_references, DEFAULT_REFERENCES
from ability_ranking import build_model_ranking


def main():
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description='旧版100+严格格式版100，按合计均分排名。')
    p.add_argument('--references',default=str(DEFAULT_REFERENCES))
    p.add_argument('--output',default='model-ranking.json')
    p.add_argument('--report',default='MODEL_RANKING.md')
    a=p.parse_args()
    if Path(a.output).exists() or Path(a.report).exists():raise ValueError('输出已存在，请先备份或指定新文件名。')
    records,skipped=load_references(a.references)
    if skipped:raise ValueError('参考库有排除文件：'+str(skipped))
    legacy=build_model_ranking(records)
    ranking=legacy.pop('dual_ranking')
    ranking['legacy_ranking']=legacy
    lines=['# 双版本合计排名','',ranking['note'],'',
        '旧版保留原A题规则（含A07–A09原有局部格式分），不检查整卷格式。严格版=新版内容×90%+格式×10%。两版求和满分200。格式对合计直接贡献最多10分，即5%。',
        '更正错误结论/错误过程/未撤回冲突的新版上限为70%/85%/50%。开放推导仍有待复核，分数不是已确认的真实能力。','',
        '| 排名 | 型号 | 份数 | 旧版均分/100 | 严格版均分/100 | 格式均分/100 | 合计均分/200 | 档位 | 排除标记合计/200 | 排除后名次 | 待复核份数 |',
        '|---:|---|---:|---:|---:|---:|---:|---|---:|---:|---:|']
    for r in ranking['models']:
        lines.append('|'+ '|'.join(map(str,[r['rank'],r['model'],r['sample_count'],f"{r['mean_legacy_score']:.2f}",f"{r['mean_strict_score']:.2f}",f"{r['mean_format_score']:.2f}",f"{r['mean_score']:.2f}",r['tier'],f"{r['mean_excluding_marked']:.2f}" if r['mean_excluding_marked'] is not None else '—',r['rank_excluding_marked'],r['pending_review_samples']]))+'|')
    lines += ['','排除标记只剔除已明确标注的那一份疑似灰测样本。weak/strong继续独立统计。来源与effort不参与格式判分。','',
        '| 样本 | 旧版/100 | 新版内容/100 | 格式/100 | 严格版/100 | 合计/200 | 格式失分项 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for s in ranking['samples']:
        d=s['dual'];f=d['strict']['format_score']
        issues='；'.join(i['name']+'：'+i['reason'] for i in f['items'] if i['points'] is None or i['points']<i['maximum']-0.0001)
        lines.append('|'+ '|'.join([s['source'],f"{s['legacy_score']:.2f}",f"{d['strict']['content_score']:.2f}",f"{s['format_score']:.2f}",f"{s['strict_score']:.2f}",f"{s['score']:.2f}",issues.replace('|','/').replace('\n',' ') or '无'])+'|')
    lines+=['','## 旧版单独排名','', '| 排名 | 型号 | 旧版均分/100 |','|---:|---|---:|']
    for r in legacy['models']:lines.append(f"|{r['rank']}|{r['model']}|{r['mean_score']:.2f}|")
    with Path(a.output).open('x',encoding='utf-8') as f:json.dump(ranking,f,ensure_ascii=False,indent=2)
    with Path(a.report).open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')
    print(json.dumps([{k:r[k] for k in ('rank','model','mean_legacy_score','mean_strict_score','mean_format_score','mean_score','tier')} for r in ranking['models']],ensure_ascii=False,indent=2))


if __name__=='__main__':main()
