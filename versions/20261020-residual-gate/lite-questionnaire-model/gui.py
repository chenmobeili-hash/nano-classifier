"""Paste or open one questionnaire; display both independent results."""
import json
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText
from classifier import canonical_answers, identify, inventory, load_references, read_answer
from ability import score_ability
from result_scoring import score_results
from score_output import score_summary, score_lines
from ability_ranking import build_model_ranking


def main():
    root = tk.Tk()
    root.title('当前模型问卷识别 · 家族与能力分档')
    root.geometry('1100x820')
    toolbar = ttk.Frame(root, padding=10)
    toolbar.pack(fill='x')
    ttk.Label(root, text='粘贴完整问卷；家族由轻量神经网络识别，能力按 A 题独立评分。').pack(anchor='w', padx=12)
    source = ScrolledText(root, height=15, wrap='word', font=('Microsoft YaHei UI', 10))
    source.pack(fill='both', expand=True, padx=12, pady=8)
    summary = tk.StringVar(value='等待输入')
    ttk.Label(root, textvariable=summary, padding=10, font=('Microsoft YaHei UI', 12)).pack(anchor='w')
    output = ScrolledText(root, height=16, wrap='word', font=('Microsoft YaHei UI', 10))
    output.pack(fill='both', expand=True, padx=12, pady=8)
    state = {'result': None, 'review': None}

    def opened():
        path = filedialog.askopenfilename(filetypes=[('回答', '*.txt *.md *.json')])
        if not path:
            return
        try:
            record = read_answer(path)
            source.delete('1.0', 'end')
            source.insert('1.0', record.get('raw_response') or '\n\n'.join('['+q+']\n'+v for q, v in record['answers'].items()))
            state.update(result=None, review=None)
            review_status.set('未加载人工评分')
        except Exception as exc:
            messagebox.showerror('读取失败', str(exc))

    def run():
        try:
            record = {'raw_response': source.get('1.0', 'end-1c')}
            record['answers'] = canonical_answers(record)
            refs, skipped = load_references()
            # A review applies only to the exact answer content loaded with it.
            review = state['review']
            if review and review['text'] != source.get('1.0', 'end'):
                raise ValueError('回答已改变，请重新加载对应的人工评分，或清除评分。')
            family = identify(record, refs)
            ability = score_ability(record, review['scores'] if review else None, ranking=build_model_ranking(refs))
            results = score_results(record)
            scores = score_summary(ability, results)
            state['result'] = {'scores': scores, 'family': family, 'ability': ability, 'result_only': results, 'reference_inventory': inventory(refs, skipped)}
            label = family['candidate'] or '无法确定（首选：'+family['nearest_family']+'）'
            grade = ability['score_versions']['combined']['provisional_tier']+'（双版本合计排名）'
            result_label = str(results['provisional_score'])+'/100'+('（暂估）' if results['pending_review'] else '')
            summary.set('家族：'+label+'    结果分：'+result_label+'    严格合计：'+str(ability['score_versions']['combined']['provisional_score'])+'/200')
            output.delete('1.0', 'end')
            model_grade = ability['relative_model']
            basis = ('参考型号：'+model_grade['model']+'；均分'+format(model_grade['mean_score'], '.2f')+'；第'+str(model_grade['rank'])+'名；'+str(model_grade['sample_count'])+'份回答（按A题原文匹配）' if model_grade
                     else '未关联唯一参考型号；按本份分数对照当前型号标尺，不授予MAX或S+。')
            dual = ability['score_versions']
            lines = score_lines(scores) + ['', '双版本合计暂估：'+str(dual['combined']['provisional_score'])+'/200；'+dual['combined']['provisional_tier'],
                     '旧版：'+str(dual['legacy']['provisional_score'])+'/100；严格格式版：'+str(dual['strict']['provisional_score'])+'/100',
                     '新版内容分：'+str(dual['strict']['content_score'])+'；格式分：'+str(dual['strict']['format_score']['provisional_score'])+'（90%内容+10%格式）',
                     '以下内容为保留的旧版逐题诊断：',
                     '本份加权暂估分：'+str(ability['provisional_score'])+'/100',
                     '内容与过程原分：'+str(ability['content_provisional_score'])+'；一致性扣分：'+str(ability['consistency_deduction']),
                     '相对档位：'+grade, basis,
                     '档位方案：'+ability['tier_version'],
                     '评分方案：'+ability['rubric_version']+'；待复核区间：'+str(ability['score_interval']),
                     '家族识别：轻量神经网络；近邻只作诊断，不参与加权',
                     '家族特征方案：'+family['feature_details']['profile']+'（'+str(family['feature_details']['dimensions'])+'维）',
                     '待复核题数：'+str(ability['pending_review']),
                     '识别分（未校准）：'+ '；'.join(r['label']+' '+format(r['score'], '.3f') for r in family['ranking']),
                     '参考距离（辅助检查）：'+ '；'.join(r['label']+' '+format(r['distance'], '.3f') for r in family['ranking']),
                     family['note'], ability['note'], '']
            if family.get('neural_ranking'):
                lines += ['网络原始分：'+'；'.join(r['label']+' '+format(r['score'], '.3f') for r in family['neural_ranking']), '近邻诊断：'+('与网络冲突，不改变排序' if family['decision_guard'].get('conflict_detected') else '仅供参考，不改变排序')]
            lines += ['格式核查：'+item['name']+' '+str(item['points'])+'/'+str(round(item['maximum'],2))+'；'+item['reason'] for item in dual['strict']['format_score']['items']]
            if model_grade:
                other_mean = model_grade['mean_excluding_marked']
                lines += ['型号双均分：包含标记 '+format(model_grade['mean_including_marked'], '.2f')+'；排除标记 '+('无可用样本' if other_mean is None else format(other_mean, '.2f')),
                          '排除标记后：'+str(model_grade['sample_count_excluding_marked'])+'份，排名'+str(model_grade['rank_excluding_marked'])+'，'+str(model_grade['tier_excluding_marked'])]
            lines.extend('样本标识：'+a['label']+'（'+a['source']+'）' for a in ability['sample_annotations'])
            lines.extend('已知库外参考：'+a['model']+'；未参与训练和主排名。' for a in family.get('known_external_references', []))
            lines += ['', '最终结果逐题核对：']
            for item in results['questions']:
                verdict = '待复核' if item['pending_review'] else '完整正确' if item['correct'] else '未完整正确'
                lines.append(item['question']+'：'+verdict+'；'+str(item['points'])+'/'+str(item['weight'])+'分')
            lines += ['', '内容与过程逐题核对：']
            for row in ability['questions']:
                score = str(row['provisional_points'])+'/'+str(row['weight'])+'分'+('（暂估，待复核）' if row['status'] == 'needs_review' else '')
                lines.extend([row['question']+'：'+score, '标准：'+row['expected'], '依据：'+row['reason']])
                if row['consistency']['events']:
                    lines.append('  严格一致性：本题最高'+str(round(row['consistency']['cap_fraction']*100))+'%；实际扣除'+str(row['consistency']['applied_deduction'])+'分（不与既有失分重复叠加）')
                    for event in row['consistency']['events']:
                        lines.append('    '+event['evidence'])
                        lines.extend('      '+issue for issue in event.get('issues', []))
                for part in row['components']:
                    earned = '待确认' if part['fraction'] is None else str(round(part['fraction']*row['weight']*part['share'], 2))+'分'
                    lines.append('  '+part['name']+'：'+earned+'；'+part['evidence'])
                    lines.extend('    '+issue for issue in part['issues'])
                for deduction in row['deductions']:
                    label = '待确认未计分' if deduction['type'] == 'uncredited_pending_review' else '失分'
                    lines.append('  '+label+' '+str(deduction['points'])+'分（'+deduction['component']+'）')
                lines.append('')
            if skipped:
                lines.append('排除的参考文件：'+json.dumps(skipped, ensure_ascii=False))
            output.insert('1.0', '\n'.join(lines))
        except Exception as exc:
            state['result'] = None
            summary.set('识别失败，请检查输入')
            output.delete('1.0', 'end')
            messagebox.showerror('识别失败', str(exc))

    def review_load():
        path = filedialog.askopenfilename(filetypes=[('人工评分', '*.json')])
        if path:
            try:
                with open(path, encoding='utf-8-sig') as f:
                    scores = json.load(f)
                score_ability({'raw_response': source.get('1.0', 'end')}, scores)
                state['review'] = {'text': source.get('1.0', 'end'), 'scores': scores}
                review_status.set('已加载人工评分')
                run()
            except Exception as exc:
                messagebox.showerror('评分读取失败', str(exc))

    def clear_review():
        state['review'] = None
        review_status.set('未加载人工评分')
        state['result'] = None

    def save():
        # Recompute to avoid exporting stale results after editing answers.
        run()
        if state['result'] is None:
            return
        path = filedialog.asksaveasfilename(defaultextension='.json', filetypes=[('JSON', '*.json')])
        if path:
            try:
                with open(path, 'w', encoding='utf-8') as f:
                    json.dump(state['result'], f, ensure_ascii=False, indent=2)
            except OSError as exc:
                messagebox.showerror('保存失败', str(exc))

    for label, callback in [('打开回答', opened), ('识别与分档', run), ('加载人工评分', review_load), ('清除评分', clear_review), ('保存结果', save)]:
        ttk.Button(toolbar, text=label, command=callback).pack(side='left', padx=4)
    review_status = tk.StringVar(value='未加载人工评分')
    ttk.Label(toolbar, textvariable=review_status).pack(side='left', padx=10)
    root.mainloop()


if __name__ == '__main__':
    main()
