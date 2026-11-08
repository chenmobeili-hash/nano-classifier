"""Presentation normalization for grading only; identity features retain raw text."""
import re


def answer_body(text):
    """Ignore only standalone horizontal rules at the end of an answer.

    Keep internal rules, all prose, fences, sequence whitespace and actual data.
    These boundary decorations are not evidence of an incorrect JSON result.
    """
    lines = text.rstrip().splitlines()
    while lines and (not lines[-1].strip() or re.fullmatch(r'[-─━—_]{3,}', lines[-1].strip())):
        lines.pop()
    return '\n'.join(lines)
