"""Tokenisation shared by BM25, coverage scoring and the extractive answerer."""
from __future__ import annotations

import re

STOPWORDS = frozenset("""
a an and are as at be been by for from has have how in is it its of on or that the their this to under was were what when where
which who whom whose why will with must may shall does do did can could should would not no than then there these those such any
all each per into over also other if so we you your about between within without
""".split())

_TOKEN_RE = re.compile(r"[a-z0-9]+(?:\.[0-9]+)*", re.I)


def stem(tok: str) -> str:
    """Very light suffix stripping; enough to unify plurals and common verb endings."""
    if len(tok) <= 3 or tok[0].isdigit():
        return tok
    for suf, rep in (("ies", "y"), ("sses", "ss"), ("ings", ""), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3 and not (suf == "s" and tok.endswith("ss")):
            return tok[: len(tok) - len(suf)] + rep
    return tok


def tokenize(text: str, keep_stopwords: bool = False) -> list[str]:
    toks = [stem(t.lower()) for t in _TOKEN_RE.findall(text)]
    if keep_stopwords:
        return toks
    return [t for t in toks if t not in STOPWORDS]


def normalize_text(text: str) -> str:
    """Lowercase, collapse whitespace and strip punctuation for span and sentence matching."""
    return " ".join(re.findall(r"[a-z0-9]+(?:\.[0-9]+)*", text.lower()))
