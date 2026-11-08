"""Synthetic pipeline examples, never real model fingerprints."""
import argparse
import json
from pathlib import Path
import numpy as np
from split_answers import make_record


def main():
    parser = argparse.ArgumentParser(description='生成虚构样本，仅验证代码可运行。')
    parser.add_argument('--output-dir', default='demo-data')
    a = parser.parse_args()
    directory = Path(a.output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    paths = [directory/'synthetic.jsonl', directory/'synthetic-answer.txt']
    if any(p.exists() for p in paths):
        raise ValueError('演示输出已存在，请使用新的--output-dir。')
    rng = np.random.default_rng(7)
    records = []
    for group in range(2):
        for i in range(16):
            answers = {f'A{k:02d}': '虚构示例，不是客观题答案。' for k in range(1, 13)}
            answers.update({
                'B01': str(rng.choice([1, 2, 3] if group == 0 else [7, 8, 9])),
                'B02': ''.join(str(x) for x in rng.choice([0, 1, 2] if group == 0 else [7, 8, 9], 120)),
                'B03': ''.join(rng.choice(list('HT'), 80)),
                'B04': ','.join(str(x) for x in rng.integers(1, 7, 24)),
                'B05': '兔' if group == 0 else '象',
                'B06': str(rng.integers(1, 101)),
            })
            for k in range(7, 13):
                answers[f'B{k:02d}'] = ('自然段落。' * int(rng.integers(2, 8))) if group == 0 else ('1. 简单建议。\n2. 下一步。\n' * int(rng.integers(2, 8)))
            raw = '\n\n'.join(f'[{q}]\n{text}' for q, text in answers.items())
            record = make_record(raw, f'synthetic-model-{group}', f'synthetic-family-{group}', f'demo_batch_{i%4}')
            record['sample_id'] = f'synthetic_{group}_{i}'
            record['label_source'] = 'synthetic_demo_not_real'
            records.append(record)
    paths[0].write_text(''.join(json.dumps(r, ensure_ascii=False)+'\n' for r in records), encoding='utf-8')
    paths[1].write_text(records[0]['raw_response'], encoding='utf-8')
    print('已生成32份虚构整卷样本；仅用于流程演示，不能作为真实识别数据。')


if __name__ == '__main__':
    main()
