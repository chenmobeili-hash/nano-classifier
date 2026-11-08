#!/usr/bin/env python3
"""Offline questionnaire splitter. Python 3.9+, standard library only."""
import argparse
import hashlib
import json
import re
import sys
import uuid
from pathlib import Path

EXPECTED = [f'{group}{i:02d}' for group in 'AB' for i in range(1, 13)]
LABEL = re.compile(
    r'^\s*(?:#{1,6}\s+)?(?:\*\*|__)?'
    r'(?:\[\s*([ABab])\s*(\d{1,2})\s*\]|【\s*([ABab])\s*(\d{1,2})\s*】|([ABab])(\d{1,2}))'
    r'(?:\*\*|__)?(?:\s*[:：]\s*(.*)|(?:[ \t]+|(?<=[\]】]))(.*)|[ \t]*)$'
)
FENCE = re.compile(r'^\s*(`{3,}|~{3,})(.*)$')


def parse_answers(raw):
    """Keep raw untouched; normalize line endings only for section parsing."""
    lines = raw.replace('\r\n', '\n').replace('\r', '\n').lstrip('\ufeff').splitlines(keepends=True)
    warnings = []
    # A model may wrap the entire answer in one text/markdown code block.
    first = next((i for i, line in enumerate(lines) if line.strip()), None)
    last = next((i for i in range(len(lines)-1, -1, -1) if lines[i].strip()), None)
    if first is not None and last > first:
        opening = FENCE.match(lines[first].rstrip('\n'))
        if opening and opening[2].strip().lower() in ('', 'text', 'markdown', 'md'):
            marker = opening[1]
            if re.fullmatch(r'\s*' + re.escape(marker[0]) + '{' + str(len(marker)) + r',}\s*', lines[last]):
                lines[first] = ''
                lines[last] = ''
                warnings.append('检测到整份回答的外层代码围栏，已仅在解析时去除。')
    sections = []
    preamble = []
    current = None
    fence = None
    for line_number, line in enumerate(lines, 1):
        fm = FENCE.match(line.rstrip('\n'))
        if fm:
            if fence is None:
                fence = fm[1]
            elif fm[1][0] == fence[0] and len(fm[1]) >= len(fence) and not fm[2].strip():
                fence = None
            (current['parts'] if current else preamble).append(line)
            continue
        match = LABEL.match(line.rstrip('\n')) if fence is None else None
        if match:
            letter = match[1] or match[3] or match[5]
            number = match[2] or match[4] or match[6]
            current = {'question_id': f'{letter.upper()}{int(number):02d}', 'line': line_number, 'parts': []}
            if match[7] is not None or match[8] is not None:
                current['parts'].append((match[7] if match[7] is not None else match[8]) + '\n')
            sections.append(current)
        else:
            (current['parts'] if current else preamble).append(line)
    occurrences = {}
    for section in sections:
        qid = section['question_id']
        occurrences.setdefault(qid, []).append({'text': ''.join(section['parts']).strip(), 'line': section['line']})
    missing = [q for q in EXPECTED if q not in occurrences]
    duplicates = [q for q, values in occurrences.items() if len(values) > 1]
    unknown = [q for q in occurrences if q not in EXPECTED]
    empty = [q for q in EXPECTED if q in occurrences and any(not v['text'] for v in occurrences[q])]
    if not sections:
        warnings.append('未识别到题号；请检查回答是否使用[A01]等独占一行的题号。')
    if missing:
        warnings.append('缺少题号：' + ', '.join(missing))
    if duplicates:
        warnings.append('重复题号：' + ', '.join(duplicates) + '；没有覆盖答案，全部出现位置已保留。')
    if unknown:
        warnings.append('未知题号：' + ', '.join(unknown))
    if empty:
        warnings.append('空答案：' + ', '.join(empty))
    if fence:
        warnings.append('存在未闭合的代码围栏，可能影响后续题号识别。')
    answers = {q: occurrences[q][0]['text'] if q in occurrences and len(occurrences[q]) == 1 else None for q in EXPECTED}
    return {
        'answers': answers,
        'answer_occurrences': occurrences,
        'preamble': ''.join(preamble).strip(),
        'parse_report': {
            'parsed_unique_count': sum(v is not None for v in answers.values()),
            'missing': missing, 'duplicates': duplicates, 'unknown': unknown,
            'empty': empty, 'warnings': warnings,
            'status': 'complete' if not (missing or duplicates or unknown or empty or fence) else 'needs_review',
        },
    }


def make_record(raw, model='', family='', batch=''):
    return {
        'schema_version': 'splitter-1.0',
        'sample_id': str(uuid.uuid4()),
        'questionnaire_version': 'mq-full-0.2',
        'delivery_mode': 'whole_questionnaire_single_turn',
        'claimed_model': model.strip() or None,
        'family_label': family.strip() or None,
        'label_source': 'user_assigned' if family.strip() else 'unknown',
        'batch_id': batch.strip() or None,
        'raw_response': raw,
        'response_sha256': hashlib.sha256(raw.encode('utf-8')).hexdigest(),
        **parse_answers(raw),
    }


def read_text(path):
    data = Path(path).read_bytes()
    if data.startswith((b'\xff\xfe', b'\xfe\xff')):
        return data.decode('utf-16')
    try:
        return data.decode('utf-8-sig')
    except UnicodeDecodeError:
        raise ValueError('文件不是UTF-8或带BOM的UTF-16，请用记事本另存为UTF-8。')


def append_record(path, record):
    """Reject duplicate answer content, validate existing lines, append one record."""
    target = Path(path)
    if target.exists():
        data = target.read_text(encoding='utf-8-sig')
        for i, line in enumerate(data.splitlines(), 1):
            if not line.strip():
                continue
            old = json.loads(line)
            if not isinstance(old, dict):
                raise ValueError(f'数据集第{i}行不是JSON对象。')
            old_hash = old.get('response_sha256')
            if old_hash is None and isinstance(old.get('raw_response'), str):
                old_hash = hashlib.sha256(old['raw_response'].encode('utf-8')).hexdigest()
            if old_hash == record['response_sha256']:
                raise ValueError('这份完整回答已经在数据集中，未重复追加。若不同采样确实得到相同原文，请另存JSON后人工确认。')
        needs_newline = bool(data) and not data.endswith('\n')
    else:
        needs_newline = False
    with target.open('a', encoding='utf-8', newline='\n') as file:
        if needs_newline:
            file.write('\n')
        file.write(json.dumps(record, ensure_ascii=False) + '\n')


def run_gui():
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk
    from tkinter.scrolledtext import ScrolledText
    app = tk.Tk()
    app.title('问卷答案一键拆分（离线）')
    app.geometry('960x720')
    ttk.Label(app, text='粘贴模型完整回答，然后点击“拆分并预览”。不记录实际采集时间。').pack(anchor='w', padx=12, pady=10)
    fields = ttk.Frame(app)
    fields.pack(fill='x', padx=12)
    entries = []
    for column, label in enumerate(('平台标称型号（选填）', '家族标签（你指定，选填）', '批次编号（选填）')):
        box = ttk.Frame(fields)
        box.grid(row=0, column=column, sticky='ew', padx=4)
        fields.columnconfigure(column, weight=1)
        ttk.Label(box, text=label).pack(anchor='w')
        entry = ttk.Entry(box)
        entry.pack(fill='x')
        entries.append(entry)
    raw_box = ScrolledText(app, height=19, wrap='word')
    raw_box.pack(fill='both', expand=True, padx=12, pady=8)
    buttons = ttk.Frame(app)
    buttons.pack(fill='x', padx=12)
    status = tk.StringVar(value='等待输入。')
    ttk.Label(app, textvariable=status).pack(anchor='w', padx=12, pady=8)
    preview = ScrolledText(app, height=10, wrap='word', state='disabled')
    preview.pack(fill='both', expand=True, padx=12, pady=(0, 12))
    cache = {'key': None, 'record': None}

    def build():
        raw = raw_box.get('1.0', 'end-1c')
        if not raw.strip():
            messagebox.showwarning('没有回答', '请先粘贴或打开完整回答。')
            return None
        key = (raw,) + tuple(entry.get() for entry in entries)
        if key != cache['key']:
            cache.update(key=key, record=make_record(*key))
        record = cache['record']
        report = record['parse_report']
        status.set(f"已拆分 {report['parsed_unique_count']}/24 题；" + ('题号完整。' if report['status'] == 'complete' else '请检查下方提示。'))
        preview.configure(state='normal')
        preview.delete('1.0', 'end')
        preview.insert('1.0', json.dumps({'parse_report': report, 'answers': record['answers']}, ensure_ascii=False, indent=2))
        preview.configure(state='disabled')
        return record

    def open_file():
        path = filedialog.askopenfilename(filetypes=[('文本', '*.txt *.md'), ('所有文件', '*.*')])
        if path:
            try:
                text = read_text(path)
                raw_box.delete('1.0', 'end')
                raw_box.insert('1.0', text)
            except (OSError, ValueError) as exc:
                messagebox.showerror('读取失败', str(exc))

    def save(append=False):
        record = build()
        if record is None:
            return
        extension = '.jsonl' if append else '.json'
        if append:
            path = filedialog.asksaveasfilename(title='选择数据集文件（已有文件会追加）', defaultextension=extension, initialfile='dataset.jsonl', confirmoverwrite=False, filetypes=[('JSONL', '*.jsonl')])
        else:
            path = filedialog.asksaveasfilename(defaultextension=extension, initialfile='sample.json', filetypes=[('JSON', '*.json')])
        if path:
            try:
                if append:
                    append_record(path, record)
                else:
                    Path(path).write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
                messagebox.showinfo('已保存', str(path))
            except (OSError, ValueError) as exc:
                messagebox.showerror('保存失败', str(exc))

    for label, action in [('打开回答文件', open_file), ('拆分并预览', build), ('保存JSON', save), ('追加到JSONL数据集', lambda: save(True))]:
        ttk.Button(buttons, text=label, command=action).pack(side='left', padx=4)
    app.mainloop()


def main():
    parser = argparse.ArgumentParser(description='拆分24题问卷答案；不传输入文件则打开图形界面。')
    parser.add_argument('input', nargs='?', help='完整回答的txt/md文件；-表示标准输入')
    parser.add_argument('-o', '--output', help='输出JSON路径；默认不覆盖已有文件')
    parser.add_argument('--append', help='追加到JSONL数据集，拒绝相同原文的重复导入')
    parser.add_argument('--model', default='')
    parser.add_argument('--family', default='')
    parser.add_argument('--batch', default='')
    args = parser.parse_args()
    if args.input is None:
        run_gui()
        return
    if args.output and args.append:
        parser.error('-o和--append请选择其中一个')
    raw = sys.stdin.read() if args.input == '-' else read_text(args.input)
    if not raw.strip():
        raise ValueError('输入回答为空。')
    record = make_record(raw, args.model, args.family, args.batch)
    if args.append:
        append_record(args.append, record)
        print('已追加：' + args.append)
    else:
        output = Path(args.output) if args.output else Path('sample-' + record['sample_id'] + '.json')
        with output.open('x', encoding='utf-8') as file:
            file.write(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
        print('已保存：' + str(output))
    print(f"拆分 {record['parse_report']['parsed_unique_count']}/24 题")
    for warning in record['parse_report']['warnings']:
        print('提示：' + warning)


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print('操作失败：' + str(exc), file=sys.stderr)
        print('图形界面需要Tkinter；也可运行 python split_answers.py 回答.txt', file=sys.stderr)
        sys.exit(1)
