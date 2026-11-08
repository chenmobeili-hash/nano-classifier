"""文体学补充块 (rich block)。

只用通用文体学量，全部在 B07–B12 作答上计算，定义与家族无关：
  1 第二人称密度（你/您）
  2 第一人称密度（我/我们）
  3 平均句长（按中文句末标点切分，归一化）
  4 最长句长（归一化）
  5 问号密度
  6 语气助词/填充词密度（其实/就是/确实/大概/也许/不过/反正）
  7 段落数（换行）
  8 连接词密度（首先/其次/另外/此外/所以/因此/总之/然后）
与现有 style 块的 8 个粗比例**不重叠**。
"""
import re

HEAD_QUESTIONS = tuple('B%02d' % i for i in range(7, 13))
SENT_SPLIT = re.compile(r'[。！？!?；;]+')
FIRST = ('我们', '我')
SECOND = ('你们', '你', '您')
FILLER = ('其实', '就是', '确实', '大概', '也许', '不过', '反正', '基本上')
CONNECT = ('首先', '其次', '另外', '此外', '所以', '因此', '总之', '然后')

RICH_NAMES = ('second_person', 'first_person', 'avg_sentence', 'max_sentence',
              'qmark_density', 'filler_density', 'paragraphs', 'connective_density')


def rich_features(answers):
    answers = {k: (v if isinstance(v, str) else '') for k, v in answers.items()}
    text = '\n'.join(answers.get(q, '') for q in HEAD_QUESTIONS)
    n = max(len(text), 1)

    def density(words):
        return min(sum(text.count(w) for w in words) * 100.0 / n, 10.0) / 10.0

    sentences = [s for s in SENT_SPLIT.split(text) if s.strip()]
    lens = [len(s) for s in sentences] or [0]
    avg_len = min(sum(lens) / len(lens), 120.0) / 120.0
    max_len = min(max(lens), 300.0) / 300.0
    paragraphs = min(text.count('\n'), 20) / 20.0
    return [
        density(SECOND),
        density(FIRST),
        avg_len,
        max_len,
        min(text.count('?') + text.count('？'), 10) / 10.0,
        density(FILLER),
        paragraphs,
        density(CONNECT),
    ]
