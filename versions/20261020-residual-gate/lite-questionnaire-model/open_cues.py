"""Generic B07-B12 response cues; labels and collection metadata are never read.

Historical exploratory rules reused only by optional comparison profiles.
No historical answers or weights are loaded. A cue records wording, not intent.
"""
import math
import re
import unicodedata
import numpy as np

VERSION = 'open-refinement-1'
OPEN_INDICES = np.r_[48:96, 297:331]
EXTRA_NAMES = (
    'B07_acknowledges_motivation', 'B07_normalizes_difficulty',
    'B07_allows_rough_version', 'B07_allows_stopping',
    'B07_deadline_or_timebox', 'B07_contrast_construction',
    'B08_log_numeric_tokens', 'B08_opens_with_scenario',
    'B08_additional_analogy', 'B08_named_dispersion_measure',
    'B08_classroom_expanded', 'B08_extreme_value_cue',
    'B09_direct_first_person', 'B09_direct_second_person',
    'B09_explicit_named_subject', 'B09_ambiguity_explanation',
    'B09_alternative_interpretations_expanded', 'B09_third_party_expanded',
    'B09_temporal_clarification', 'B09_parenthetical_note',
    'B10_winner_field_label', 'B10_votes_field_label',
    'B10_starts_with_source', 'B10_limits_source_scope',
    'B10_missing_information_expanded', 'B10_no_inference_support',
    'B11_conflict_expanded', 'B11_explains_character_conflict',
    'B11_explicit_priority_choice', 'B11_proposes_substitute',
    'B12_opens_with_definition', 'B12_opens_with_analogy',
    'B12_independence_or_memorylessness', 'B12_equal_specific_sequences',
    'B12_many_opportunities_for_coincidence', 'B12_selective_attention',
    'B12_evolutionary_explanation', 'B12_generator_mechanism',
    'B12_generator_quality_check', 'B12_counterfactual_no_patterns',
    'B12_long_run_proportions', 'B12_colloquial_signpost',
    'B12_log_questions', 'B12_contraction_fraction',
    'B12_first_person_fraction', 'B12_second_person_fraction',
    'open_dash_density', 'open_corner_quote_density',
)


def clean(text):
    text = unicodedata.normalize('NFKC', text).lower()
    text = text.translate(str.maketrans({'’': "'", '‘': "'", '“': '"', '”': '"'}))
    return re.sub(r'\s+', ' ', text.replace('**', '').replace('__', '')).strip()


def hit(pattern, text):
    return float(bool(re.search(pattern, text, re.I)))


def extra_map(answers):
    b7, b8, b9, b10, b11, b12 = [clean(answers[f'B{i:02d}']) for i in range(7, 13)]
    # Separate semantic phrase detection from display wrapping.
    c7, c8, c9, c10, c11 = [re.sub(r'\s+', '', s) for s in (b7, b8, b9, b10, b11)]
    words = re.findall(r"[a-z]+(?:'[a-z]+)?", b12)
    n = max(1, len(words))
    direct_first = hit(r'[：:]["「『\s]*我(?:自己)?明天|["「『]我(?:自己)?明天', c9)
    direct_second = hit(r'[：:]["「『\s]*你明天|["「『]你明天', c9)
    # The proper names here are fixed entities in B09, not source/model names.
    named_subject = hit(r'(?:告诉|通知|对)小周[,，:："「『]*(?:小林|小周)(?:自己)?明天', c9)
    alternatives = hit(r'(?:如果|若|改法|不能来的是|指的是).*(?:如果|若|改法|不能来的是|指的是)|两种|两者|之一|或[者:：]|[／/]|(?:也可能|可能指|可能是).*?(?:小林|小周)', c9)
    conflict = hit(r'(?:无法|不能|不可能|不能够)同时(?:满足|成立)|互(?:相)?矛盾|互斥|(?:直接|逻辑|两[者条项]).{0,8}(?:冲突|矛盾)|无解|没有合规答案|不存在.*(?:答案|文本)|contradict|cannot.*both', c11)
    # Require an output-introducing phrase: quoted constraints alone are not alternatives.
    substitute = hit(r'(?:答案(?:为|就是)|我只能写|只能写|替代[：:]?|用拆字或拼音替代[)）]?[:：]|改成|改为)[:：]?[^。；]+', c11)
    open_text = ''.join(answers[f'B{i:02d}'] for i in range(7, 13))
    values = (
        hit(r'在[乎意]|看重|想(?:把事情|做得|做)好|认真|想周全|少走弯路|可以理解', c7),
        hit(r'常见|正常|我(?:太|特别)?懂|能理解|听起来|听你这么说|太想做好', c7),
        hit(r'允许|没关系|不用好|不评价|不许回头改|暂时不好|先别急着修改|不追求', c7),
        hit(r'做完.{0,8}(?:停|算)|时间到了.{0,8}停|(?:分钟后|分钟就|分钟.{0,5})停|收工|大功告成', c7),
        hit(r'(?:\d+|[一二三四五六七八九十两]+)分钟|今天|现在就', c7),
        hit(r'不是.{0,30}(?:而是|是)|先有.{0,25}(?:才有|再有)|从来不是|比.{0,30}更', c7),
        math.log1p(len(re.findall(r'\d+(?:\.\d+)?|[零一二三四五六七八九十百千万]+(?=[人元分岁米万])', c8))),
        hit(r'^(?:可以(?:用|想象)|想象|假设|比如|例如|举个)', c8),
        hit(r'就像|像只看|再比如|还可以|打个比方|好比', c8),
        hit(r'方差|标准差|中位数|四分位|variance|standarddeviation|median', c8),
        hit(r'班|课堂|学生|教室|考试|成绩|测验|classroom|students|exam|scores', c8),
        hit(r'极端|特别[高低大小]|最高|最低|最大|最小|outlier|extreme', c8),
        direct_first, direct_second, named_subject,
        hit(r'歧义|指代不[明清]|指向不明|主语|明确.{0,6}指|原意|ambig', c9),
        alternatives, hit(r'第三[个人方]|小[张王]|第三者|someoneelse|thirdperson', c9),
        hit(r'时间|第二天|哪一天|某日|日期', c9),
        hit(r'[（(].+[）)]', c9),
        hit(r'获胜者(?:的)?姓名[:：]', c10),
        hit(r'(?:得票数|票数)[:：]', c10),
        hit(r'^(?:材料原文|原文|材料全文)', c10),
        hit(r'只(?:说明|写|包含|表明|支持)|仅(?:为|有|写|包含|说明)|唯一|全文仅|全部原文', c10),
        hit(r'(?:未|没有|并未)(?:给出|提供|说明|提到|提及|记载|包含|写明)|没有这些信息|信息不足|无法(?:得知|确定|回答|给出|得出)', c10),
        hit(r'不支持|没有.{0,10}(?:支持|依据)|不存在支持|不能.{0,8}推[出断]|不含任何', c10),
        conflict,
        hit(r'(?:本身|必然|必须|含有|包含|出现|强制|禁止).{0,20}(?:字|三字|三个)|第三个字', c11),
        hit(r'优先(?:满足|遵守)|选择遵守|如果必须遵守|若必须遵守', c11),
        substitute,
        hit(r'^(?:true |real )?randomness (?:is|means|describes|doesn.t|does not)|^random (?:doesn.t|does not)', b12),
        hit(r'^(?:imagine|think of|picture|suppose)', b12),
        hit(r'independen|no memory|doesn.t (?:remember|owe)|no need to|each new flip|next flip.{0,30}(?:still|50|fifty)', b12),
        hit(r'(?:each|every|any).{0,35}(?:specific|exact|particular) sequence|(?:hhhh|all heads).{0,45}(?:likely|probability|miraculous)|just as likely as|equally.likely outcomes', b12),
        hit(r'(?:many|lots of|plenty of|thousands of).{0,25}(?:chances|opportunities|places)|(?:enough|many).{0,15}(?:trials|simulations)|bound to show up|become inevitable', b12),
        hit(r'(?:only|just|we|you).{0,12}notice|forget.{0,20}(?:boring|ordinary)|(?:ignore|overinterpret)|patterns we care about|grabs our attention|stands? out', b12),
        hit(r'evol(?:ved|ution)|surviv|predator|tiger|get you eaten', b12),
        hit(r'pseudo.?random|algorithm|deterministic|seed|formula|starting number|hidden recipe|same recipe|生成器|算法|公式', b12),
        hit(r'statistical tests|systematic departures|flawed|imperfect generator|poorly designed|bad random|distrust|predict what comes next|quality|flaw in|check whether', b12),
        hit(r'if.{0,80}(?:no streaks|no patterns|never produced|never has|no.{0,10}streaks)|sequence that never|randomness never produced', b12),
        hit(r'(?:overall|proportions|long run|many trials|zoom out|enough trials).{0,60}(?:balanc|settle|half|proportions|predicts)|balanced over many trials', b12),
        hit(r'here.s (?:the|a)|funny thing|fun part|cool thing|plot twist|here.s the secret|here.s the thing', b12),
        math.log1p(b12.count('?')),
        sum("'" in w for w in words)/n,
        sum(w in ('we', 'our', 'us', "we're", "we've", "we'd", "we'll") for w in words)/n,
        sum(w in ('you', 'your', "you're", "you'll", "you'd", "you've") for w in words)/n,
        100*len(re.findall(r'—|——|--', open_text))/max(len(open_text), 1),
        100*len(re.findall('[「」『』]', open_text))/max(len(open_text), 1),
    )
    assert len(values) == len(EXTRA_NAMES)
    return dict(zip(EXTRA_NAMES, map(float, values)))



