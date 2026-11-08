"""Strict delivery grading: errors remain visible after a final correction.

Caps apply to each question, never to the whole 100-point total. Existing
content/method losses and these caps use the lower score, without stacking.
"""
import re

from ability_rules import (CORRECTION, arithmetic_checks, clean, compact,
                           numeric_result, culprit_result, probability_result,
                           path_components, recurrence_components, game_components,
                           hypothetical_claim)
from ability_methods import game_evidence, binary_fact_issues

VERSION = 'delivery-consistency-1'
CAPS = {'corrected_wrong_conclusion': .8, 'corrected_wrong_work': .9,
        'unresolved_conclusions': .6}


def display_claim(value):
    if value in ('True', 'False'):
        return '先手必胜' if value == 'True' else '先手不胜'
    if value.startswith('frozenset'):
        return '第一步取'+'或'.join(re.findall(r'\d+', value))+'枚'
    return value


def assertion_checks(question, raw):
    if question == 'A02':
        return [culprit_result(raw)], compact(raw)
    if question == 'A03':
        return [probability_result(raw)], compact(raw)
    if question == 'A05':
        return list(path_components(raw)), compact(raw)
    if question == 'A06':
        return [recurrence_components(raw)[0]], compact(raw)
    if question == 'A10':
        return list(game_components(raw)), compact(raw)
    if question in ('A01', 'A04', 'A11', 'A12'):
        expected = {'A01': 5, 'A04': 35, 'A11': 3882, 'A12': 26}[question]
        return [numeric_result(question, raw, expected)], clean(raw)
    return [], clean(raw)


def hypothetical(text, position):
    return hypothetical_claim(text, position)


def inspect_consistency(question, raw):
    checks, text = assertion_checks(question, raw)
    events = []
    for check in checks:
        history = check.get('claim_history', [])
        wrong = [h for h in history if not h['correct'] and not hypothetical(text, h['position'])]
        if wrong and check['state'] == 'conflict':
            events.append({'kind': 'unresolved_conclusions', 'cap_fraction': CAPS['unresolved_conclusions'],
                           'evidence': '同题保留了互相冲突的定论，没有明确撤回错误定论。',
                           'claims': [h['value'] for h in history]})
        if check.get('corrected') and check['state'] == 'correct':
            for claim in wrong:
                if claim['active']:
                    continue
                position = claim['position']
                excerpt = text[max(0, position-25):position+90]
                # A12 totals of a different subclass are work, not its requested answer.
                if question == 'A12' and position > 0 and re.search(r'总数|首末相同|首末同为', excerpt):
                    kind = 'corrected_wrong_work'
                else:
                    kind = 'corrected_wrong_conclusion'
                events.append({'kind': kind, 'cap_fraction': CAPS[kind],
                               'evidence': '先明确给出错误断言“'+display_claim(claim['value'])+'”，后经更正才改对。',
                               'excerpt': excerpt})

    # Examine the whole delivered answer, including the work the final patch discarded.
    correction = bool(CORRECTION.search(raw))
    arithmetic_errors = []
    for check in arithmetic_checks(raw):
        if check['correct']:
            continue
        expression = check['expression']
        position = compact(raw).find(expression)
        if position >= 0 and hypothetical(compact(raw), position):
            continue
        arithmetic_errors.append(expression)
    method_issues = []
    if question == 'A01':
        normalized = compact(raw)
        for m in re.finditer(r'7\^([0-9]{1,3})≡([+-]?[0-9]+)(?![\d^*/])', normalized):
            if not hypothetical(normalized, m.start()) and pow(7, int(m[1]), 13) != int(m[2]) % 13:
                method_issues.append(m[0]+'不满足模13同余。')
        if re.search(r'周期(?:为|是)6|每6次循环', normalized):
            method_issues.append('7在模13下的周期为12，不是6。')
    elif question == 'A10':
        method_issues = game_evidence(compact(raw))[2]
    elif question == 'A12':
        method_issues = binary_fact_issues(compact(raw))
    if correction and (arithmetic_errors or method_issues):
        events.append({'kind': 'corrected_wrong_work', 'cap_fraction': CAPS['corrected_wrong_work'],
                       'evidence': '整份回答中仍保留可核对的错误推导或计数，更正不能抹去已交付的错误。',
                       'issues': sorted(set(arithmetic_errors+method_issues))[:8]})

    # Avoid repeating a penalty because the same error was restated several times.
    unique = []
    seen = set()
    for event in events:
        key = (event['kind'], event.get('excerpt'), tuple(event.get('issues', [])))
        if key not in seen:
            seen.add(key)
            unique.append(event)
    return {'version': VERSION, 'cap_fraction': min((e['cap_fraction'] for e in unique), default=1),
            'events': unique}
