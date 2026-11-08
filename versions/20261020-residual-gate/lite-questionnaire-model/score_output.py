"""Shared score summary for CLI JSON, GUI display and exports."""


def score_summary(ability, results):
    dual = ability['score_versions']
    def row(label, value, maximum, pending, interval):
        return dict(label=label, score=None if pending else value,
                    provisional_score=value, max_points=maximum,
                    pending_review=pending, score_interval=interval)
    pending = ability['pending_review']
    fmt = dual['strict']['format_score']
    return {
        'result_only': row('最终结果分', results['provisional_score'], 100,
                           results['pending_review'], results['score_interval']),
        'legacy': row('旧版分（含过程）', ability['provisional_score'], 100,
                      pending, ability['score_interval']),
        'strict': row('严格格式版', dual['strict']['provisional_score'], 100,
                      dual['strict']['pending_review'], dual['strict']['score_interval']),
        'format': row('格式分', fmt['provisional_score'], 100,
                      fmt['pending_review'], fmt['score_interval']),
        'combined': row('旧版＋严格版合计', dual['combined']['provisional_score'], 200,
                        dual['combined']['pending_review'], dual['combined']['score_interval'])}


def score_lines(summary):
    lines = []
    for item in summary.values():
        value = item['provisional_score']
        line = f"{item['label']}：{value:.2f}/{item['max_points']}"
        if item['pending_review']:
            low, high = item['score_interval']
            line += f"（暂估；{item['pending_review']}题待复核；区间{low:.2f}–{high:.2f}）"
        lines.append(line)
    lines.append('最终结果分独立展示，不加入200分合计。')
    return lines
