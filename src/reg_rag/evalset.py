"""Evaluation set loading and validation against the corpus (gold citations and answer spans must exist)."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Passage
from .textproc import normalize_text

ANSWERABLE_TYPES = {"answerable", "adversarial"}
ABSTAIN_TYPES = {"unanswerable", "adversarial_abstain"}


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    gold_citations: tuple[str, ...]
    answer_span: str
    payload: str = ""
    clean_twin: str = ""

    @property
    def answerable(self) -> bool:
        return self.type in ANSWERABLE_TYPES


def load_questions(path: Path) -> list[Question]:
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        out.append(Question(d["id"], d["type"], d["question"], tuple(d.get("gold_citations", [])),
                            d.get("answer_span", ""), d.get("payload", ""), d.get("clean_twin", "")))
    return out


def validate_questions(questions: list[Question], passages: list[Passage]) -> list[str]:
    """Return problems: unknown gold keys, spans absent from the gold passages, or malformed entries."""
    problems: list[str] = []
    ids = [q.id for q in questions]
    if len(set(ids)) != len(ids):
        problems.append("duplicate question ids")
    by_key: dict[str, list[Passage]] = {}
    for p in passages:
        for k in p.covers:
            by_key.setdefault(k, []).append(p)
    for q in questions:
        if q.answerable:
            if not q.gold_citations or not q.answer_span:
                problems.append(f"{q.id}: answerable question needs gold citation and span")
                continue
            gold = [p for k in q.gold_citations for p in by_key.get(k, [])]
            missing = [k for k in q.gold_citations if k not in by_key]
            if len(missing) == len(q.gold_citations):
                problems.append(f"{q.id}: no gold citation exists in corpus: {missing}")
                continue
            span = normalize_text(q.answer_span)
            if not any(span in normalize_text(p.text) for p in gold):
                problems.append(f"{q.id}: span not found in gold passages")
        elif q.type in ABSTAIN_TYPES:
            if q.gold_citations or q.answer_span:
                problems.append(f"{q.id}: abstain question must not carry gold evidence")
        else:
            problems.append(f"{q.id}: unknown type {q.type}")
    return problems


def gold_hit(gold: tuple[str, ...], passage: Passage) -> bool:
    return any(k in passage.covers for k in gold)
