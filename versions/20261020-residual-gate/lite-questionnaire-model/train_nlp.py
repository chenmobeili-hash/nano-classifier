"""训练 NLP 版家族识别器：n-gram TF-IDF → 线性模型。

用法：
    python train_nlp.py                                   # 默认 lr，仅 B 题
    python train_nlp.py --kind centroid --output x.npz
    python train_nlp.py --include-a                        # 消融：A+B 全文本
"""
import argparse
import json
import sys
from pathlib import Path

import nlp_core as N
from classifier import DEFAULT_REFERENCES, inventory, load_references


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='训练 n-gram TF-IDF + 线性模型的家族识别器。')
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--output', default=str(N.DEFAULT_MODEL))
    parser.add_argument('--kind', choices=N.KINDS, default='lr')
    parser.add_argument('--include-a', action='store_true', help='把 A 题也纳入文本（消融用）')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--l2', type=float, default=.001)
    args = parser.parse_args()
    records, skipped = load_references(args.references)
    if skipped:
        raise ValueError('参考库有被排除的文件，请先修正：' + json.dumps(skipped, ensure_ascii=False))
    if Path(args.output).exists() or Path(args.output + '.report.json').exists():
        raise ValueError('输出或报告已存在，请指定新的 --output，或先备份并移走旧文件。')
    model = N.train_model(records, kind=args.kind, include_a=args.include_a,
                          seed=args.seed, epochs=args.epochs, l2=args.l2)
    N.save_model(args.output, model)
    report = {'model': model['meta'], 'inventory': inventory(records, skipped),
              'note': ('全部当前样本用于发布权重训练；training_accuracy 是训练集自评，'
                       '不是留出测试准确率。三个分类器（lr/centroid/mnb）都存进了同一个 npz，'
                       'meta.kind 决定默认使用哪个。')}
    Path(args.output + '.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print('训练失败：' + str(exc), file=sys.stderr)
        sys.exit(1)
