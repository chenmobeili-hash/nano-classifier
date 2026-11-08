import argparse
import json
import sys
from pathlib import Path
from classifier import DEFAULT_REFERENCES, identify, inventory, load_references, read_answer
from nlp_classifier import identify_nlp
from fusion import RULES as FUSE_RULES, identify_fused
import nlp_core
from ability import score_ability
from result_scoring import score_results
from score_output import score_summary
from ability_ranking import build_model_ranking
from feature_profiles import PROFILES


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='家族识别（NLP n-gram 版）与独立能力分档。')
    parser.add_argument('answers')
    parser.add_argument('--references', default=str(DEFAULT_REFERENCES))
    parser.add_argument('--min-margin', type=float, default=None,
                        help='候选间隔门槛；NLP 默认用按覆盖率反解的 0.0106，传 0.1 可复现神经网络口径')
    parser.add_argument('--min-prob', type=float, default=0.0, help='最大模型分低于此值即拒答；0 表示不启用')
    parser.add_argument('--engine', choices=('nlp', 'neural', 'distance', 'fused'), default='nlp',
                        help='nlp=文本n-gram线性模型（本版默认）；fused=与神经网络晚期融合；neural/distance=原135维方案')
    parser.add_argument('--fuse', choices=FUSE_RULES, default='geo', help='fused 引擎的融合规则：geo 几何平均 / mean 算术平均')
    parser.add_argument('--nlp-kind', choices=nlp_core.KINDS, help='NLP 分类器；默认用权重 meta 里的 kind')
    parser.add_argument('--model', help='可选新权重NPZ；nlp 默认用脚本旁 nlp-model.npz，neural/distance 默认用 neural-balanced.npz')
    parser.add_argument('--profile', choices=PROFILES, default='balanced', help='balanced为新版默认；baseline可复现旧139维方案')
    parser.add_argument('--review', help='可选人工评分JSON：{"A01":{"score":1,"reason":"推导正确"}}')
    parser.add_argument('--output')
    args = parser.parse_args()
    record = read_answer(args.answers)
    refs, skipped = load_references(args.references)
    overrides = json.loads(Path(args.review).read_text(encoding='utf-8-sig')) if args.review else None
    ability = score_ability(record, overrides, ranking=build_model_ranking(refs))
    results = score_results(record)
    if args.engine == 'nlp':
        family = identify_nlp(record, refs, min_margin=args.min_margin, min_prob=args.min_prob,
                              model=args.model, kind=args.nlp_kind)
    elif args.engine == 'fused':
        family = identify_fused(record, refs, rule=args.fuse, min_margin=args.min_margin,
                                min_prob=args.min_prob, mlp_model=args.model)
    else:
        family = identify(record, refs, args.min_margin if args.min_margin is not None else .1,
                          profile=args.profile, engine=args.engine,
                          model=args.model, min_prob=args.min_prob)
    result = {'scores': score_summary(ability, results), 'family': family, 'ability': ability,
              'result_only': results, 'reference_inventory': inventory(refs, skipped)}
    output = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        with open(args.output, 'x', encoding='utf-8') as stream:
            stream.write(output + '\n')
    print(output)


if __name__ == '__main__':
    try:
        main()
    except (ValueError, RuntimeError, OSError, KeyError) as exc:
        print('识别失败：' + str(exc), file=sys.stderr)
        sys.exit(1)
