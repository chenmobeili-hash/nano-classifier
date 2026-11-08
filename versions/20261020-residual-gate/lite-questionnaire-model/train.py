import argparse
import json
import sys
from pathlib import Path
from classifier import DEFAULT_REFERENCES, load_references, inventory
from neural_core import DEFAULT_MODEL, train_network, save_network


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='用当前nano-now问卷训练131→8→家族数的小型神经网络。')
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--output', default=str(DEFAULT_MODEL))
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=400)
    args = parser.parse_args()
    records, skipped = load_references(args.references)
    if skipped:
        raise ValueError('参考库有被排除的文件，请先修正：'+json.dumps(skipped, ensure_ascii=False))
    if Path(args.output).exists() or Path(args.output+'.report.json').exists():
        raise ValueError('输出或报告已存在，请指定新的 --output，或先备份并移走旧文件。')
    model = train_network(records, seed=args.seed, epochs=args.epochs)
    save_network(args.output, model)
    report = {'model': model['meta'], 'inventory': inventory(records, skipped),
              'note': '全部当前样本用于发布权重训练；训练准确率不是留出测试准确率。'}
    Path(args.output+'.report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, RuntimeError) as exc:
        print('训练失败：'+str(exc), file=sys.stderr)
        sys.exit(1)
