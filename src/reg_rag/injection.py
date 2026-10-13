"""Heuristic prompt-injection detection for retrieved passages and user questions."""
from __future__ import annotations

import re

from .chunking import sentences

_PATTERNS = [
    r"\bignore (?:all |any |the |your )?(?:previous|prior|above|earlier)?\s*(?:instructions?|rules?|guidelines?|prompts?)\b",
    r"\bdisregard (?:all |any |the |your )?(?:previous|prior|above|earlier)?\s*(?:instructions?|rules?|guidelines?|context)\b",
    r"\b(?:you|assistant|system) (?:must|should|will) (?:now )?(?:answer|respond|reply|say|output|reveal)\b",
    r"\b(?:answer|respond|reply) (?:only )?with\b",
    r"\bsystem prompt\b",
    r"\bdo not (?:cite|abstain|mention|refuse)\b",
    r"\bnew instructions?\b",
    r"\bact as\b",
    r"\bpretend (?:to be|that)\b",
    r"\boverride\b.{0,30}\b(?:rules?|instructions?|policy)\b",
]
_RE = [re.compile(p, re.I) for p in _PATTERNS]


def is_injected(text: str) -> bool:
    return any(r.search(text) for r in _RE)


def strip_injected_sentences(text: str) -> tuple[str, int]:
    """Remove sentences that look like instructions; return the cleaned text and how many were removed."""
    kept, removed = [], 0
    for s in sentences(text):
        if is_injected(s):
            removed += 1
        else:
            kept.append(s)
    return " ".join(kept), removed
