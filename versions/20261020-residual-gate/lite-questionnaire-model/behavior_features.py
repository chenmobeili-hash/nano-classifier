"""Question-specific behavioral cues, independent of model labels and metadata.

These transparent rules measure textual cues rather than infer hidden intent.
Definitions were chosen during exploratory analysis and require new-data checks.
"""
import re
import unicodedata

BEHAVIOR_FEATURE_NAMES = (
    'B07_question_mark',
    'B07_personal_followup_cue',
    'B07_continued_help_cue',
    'B07_empathy_cue',
    'B08_classroom_example_cue',
    'B08_money_example_cue',
    'B08_age_height_example_cue',
    'B08_water_depth_example_cue',
    'B08_middle_phrase_cue',
    'B10_missing_information_cue',
    'B10_no_fabrication_cue',
    'B11_conflict_cue',
    'B11_priority_choice_cue',
    'B11_partial_answer_cue',
    'B11_multiple_priority_choices_cue',
    'B11_literal_forbidden_character',
    'B12_chinese_dominant',
    'B12_pseudorandom_algorithm_cue',
    'B12_streak_cluster_cue',
    'B12_pattern_perception_cue',
    'B09_multiple_interpretations_cue',
    'B09_third_person_cue',
    'B03_strict_format',
    'B02_short_repeated_block_fraction',
)


def normalized(text):
    # Ignore display wrapping and Markdown emphasis in phrase detection.
    text = unicodedata.normalize('NFKC', text).lower()
    return re.sub(r'\s+', ' ', text.replace('**', '').replace('__', '')).strip()


def compact(text):
    return re.sub(r'\s+', '', normalized(text))


def cue(pattern, text):
    return float(bool(re.search(pattern, text, flags=re.I)))


def short_repeated_block_fraction(sequence, max_block=12):
    """Largest consecutive repeated short-block span / sequence length.

    Require at least three repeats to avoid treating one accidental pair as
    clear periodic structure. This statistic applies to every family equally.
    """
    n = len(sequence)
    longest = 0
    for size in range(1, min(max_block, n // 3) + 1):
        for start in range(n - 3 * size + 1):
            block = sequence[start:start + size]
            end = start + size
            while sequence[end:end + size] == block:
                end += size
            if end - start >= 3 * size:
                longest = max(longest, end - start)
    return longest / max(n, 1)


def behavior_feature_map(answers):
    b07, b08, b09, b10, b11 = [compact(answers[q]) for q in ('B07', 'B08', 'B09', 'B10', 'B11')]
    b12 = normalized(answers['B12'])
    # Count question clauses separately: a question-shaped suggestion alone
    # does not establish that the reply requests the user's own situation.
    clauses = re.findall(r'[^。.!！?？\n]*[?？]', normalized(answers['B07']))
    personal_followup = any(
        re.search(r'你(?:现在|目前)?(?:最)?(?:想|觉得|打算|准备|希望)|要不要|what.*(?:you|your)|which.*(?:you|your)|wouldyou|doyou', compact(clause))
        for clause in clauses
    )
    priority_pattern = r'优先(?:满足|遵守)|选择遵守|(?:如果|若)必须遵守|prioriti[sz]e|choosetofollow'
    priority_count = len(re.findall(priority_pattern, b11, flags=re.I))
    conflict = cue(r'无法同时满足|不能同时满足|互相矛盾|互相冲突|互斥|直接冲突|逻辑上无解|没有合规答案|不存在.*(?:答案|文本)|contradict|mutuallyexclusive|cannot.*both', b11)
    # A phrase describing a proposed output must be followed by content. Merely
    # quoting the conflicting requirements does not count as a partial answer.
    proposed_output = bool(re.search(r'(?:答案(?:为|就是)|我只能写|可以改为|可以写成|只能写|替代答案|answeris|insteadwrite)[:：]?[^。]+', b11))
    priority_output = bool(re.search(priority_pattern + r'[^。]*[:：][^。]+', b11, flags=re.I))
    partial_answer = bool(conflict and priority_count and (proposed_output or priority_output))
    letters = len(re.findall('[A-Za-z]', b12))
    chinese = len(re.findall('[\u4e00-\u9fff]', b12))
    interpretations = len(re.findall(r'如果.*?指|若.*?指|(?:改法|情况|情形)[一二三123]|if.*?refers', b09, flags=re.I))
    digits = ''.join(re.findall('[0-9]', answers['B02']))
    values = (
        cue('[?？]', answers['B07']),
        float(personal_followup),
        cue(r'我们(?:可以)?一起|如果愿意|我在(?:这儿|这里)等|一起把|wecan|let.?s|happytohelp', b07),
        cue(r'我(?:太|特别)?懂|能(?:感觉到|理解)|可以理解|这很常见|听(?:起来|你这么说)|canunderstand|thatsounds', b07),
        cue(r'班级|[甲乙ab一二]班|两个班|两班|学生|classroom|twoclasses|students', b08),
        cue(r'月薪|年薪|收入|财产|工资|富豪|钱|salary|income|wealth|wages', b08),
        cue(r'年龄|身高|height|ages?', b08),
        cue(r'水深|小河|两条河|浅滩|深潭|river|waterdepth', b08),
        cue(r'中间大概在哪', b08),
        cue(r'未(?:提供|说明)|没有(?:提供|写|提)|无法(?:确定|回答|得知)|未知|材料不足|notprovided|doesnot(?:say|provide)|unknown|notgiven', b10),
        cue(r'不(?:会|应|应该|能|要)(?:去)?(?:编造|捏造|虚构)|不编造|won.?t(?:invent|fabricate|makeup)|shouldnot(?:invent|fabricate|makeup)', b10),
        conflict,
        float(priority_count > 0),
        float(partial_answer),
        float(partial_answer and priority_count >= 2),
        float('灯' in answers['B11']),
        float(chinese > letters),
        cue(r'pseudo.?random|algorithm|deterministic|generator|seed|伪随机|算法|生成器|种子|公式', b12),
        cue(r'streak|cluster|clump|连续|成串|聚集|扎堆', b12),
        cue(r'brain|pattern.recognition|apophenia|人脑|大脑|找规律|看到脸', b12),
        float(interpretations >= 2),
        cue(r'第三(?:个)?人|第三方|thirdperson|someoneelse', b09),
        float(bool(re.fullmatch('[HT]{80}', answers['B03'].strip()))),
        short_repeated_block_fraction(digits),
    )
    return dict(zip(BEHAVIOR_FEATURE_NAMES, values))


def behavior_features(answers):
    return list(behavior_feature_map(answers).values())
