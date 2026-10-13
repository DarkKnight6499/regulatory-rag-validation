"""Groundedness: does every returned sentence appear in the retrieved passages?"""
from __future__ import annotations

from .chunking import Passage, sentences
from .textproc import normalize_text, tokenize


def sentence_grounded(sentence: str, passages: list[Passage], min_token_coverage: float = 1.0) -> bool:
    """True if the sentence is a verbatim span of a passage (default) or covers its tokens at the given rate."""
    norm = normalize_text(sentence)
    if not norm:
        return True
    if any(norm in normalize_text(p.text) for p in passages):
        return True
    if min_token_coverage >= 1.0:
        return False
    toks = tokenize(sentence)
    if not toks:
        return True
    best = 0.0
    for p in passages:
        ptoks = set(tokenize(p.text))
        best = max(best, sum(t in ptoks for t in toks) / len(toks))
    return best >= min_token_coverage


def groundedness_rate(answer_text: str, passages: list[Passage], min_token_coverage: float = 1.0) -> float:
    """Fraction of answer sentences grounded in the passages; an empty answer counts as fully grounded."""
    sents = sentences(answer_text)
    if not sents:
        return 1.0
    return sum(sentence_grounded(s, passages, min_token_coverage) for s in sents) / len(sents)
