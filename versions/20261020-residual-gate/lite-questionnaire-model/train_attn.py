"""训练题块级单头自注意力（V1）。

用法：
    python train_attn.py                                  # 默认：注意力开，16 维，400 轮
    python train_attn.py --no-attention --output attn-ablation-noattn.npz
    python train_attn.py --gradcheck                      # 有限差分梯度校验
"""
import argparse
import json
import sys
from pathlib import Path

import attn_core as A
from classifier import DEFAULT_REFERENCES, inventory, load_references


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='题块级单头自注意力家族识别器。')
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--output', default=str(A.DEFAULT_MODEL))
    parser.add_argument('--no-attention', action='store_true', help='对照组：保留残差但关闭注意力')
    parser.add_argument('--hidden', type=int, default=A.HIDDEN)
    parser.add_argument('--epochs', type=int, default=400)
    parser.add_argument('--l2', type=float, default=.001)
    parser.add_argument('--lr', type=float, default=.01)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gradcheck', action='store_true')
    args = parser.parse_args()

    if args.gradcheck:
        error, per_key = A.gradcheck(eps=1e-5)
        for key, value in per_key.items():
            print('  %-14s %.3e' % (key, value))
        print('最大相对误差 %.3e' % error)
        print('PASS' if error < 1e-4 else 'FAIL')
        sys.exit(0 if error < 1e-4 else 1)

    records, skipped = load_references(args.references)
    if skipped:
        raise ValueError('参考库有被排除的文件，请先修正：' + json.dumps(skipped, ensure_ascii=False))
    if Path(args.output).exists() or Path(args.output + '.report.json').exists():
        raise ValueError('输出或报告已存在，请指定新的 --output，或先备份并移走旧文件。')
    classes = sorted({r['family'] for r in records})
    model = A.train(records, classes, use_attention=not args.no_attention, hidden=args.hidden,
                    seed=args.seed, epochs=args.epochs, l2=args.l2, learning_rate=args.lr)
    model['meta']['reference_signature'] = A.reference_signature(records)
    A.save_model(args.output, model)
    report = {'model': model['meta'], 'inventory': inventory(records, skipped),
              'note': ('全部当前样本用于发布权重训练；training_accuracy 是训练集自评，不是留出准确率。'
                       '题块 = B01–B12；每块 char 2/3-gram 哈希到 512 桶后 sublinear TF + 块内 L2。')}
    Path(args.output + '.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'model': model['meta']}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print('训练失败：' + str(exc), file=sys.stderr)
        sys.exit(1)
