"""Clause topic dataset: transparent structural labelling rules and a leakage-free split by document section."""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..chunking import Passage

ID_COLUMN = "chunk_id"
GROUP_COLUMN = "group"
CITATION_COLUMN = "citation"
TEXT_COLUMN = "text"
LABEL_COLUMN = "label"
SPLIT_COLUMN = "split"
MIN_CHARS = 120

LABELS = ["model_governance", "model_validation", "liquidity_buffer", "liquidity_risk_management", "stress_testing",
          "capital", "reporting"]

# Structural rules: the label is decided by where a clause sits (document and section), not by its words.
# Classifiers see only the clause text, so a keyword baseline cannot read the rule off the label.
_SR_VALIDATION_SECTIONS = ("V. MODEL VALIDATION",)
_REPORTING_CFR_SECTIONS = {"249.90", "249.91", "249.130", "249.131", "252.45", "252.46"}


def _cfr_section(section_path: str) -> str:
    m = re.search(r"§\s*([\d.]+)", section_path)
    return m.group(1) if m else ""


def topic_for(p: Passage) -> str | None:
    """Return the topic label for a passage, or None if the passage is excluded from the dataset."""
    if len(p.text) < MIN_CHARS or p.text.startswith(("[footnote]", "[table]")):
        return None
    if p.doc_id == "sr11-7":
        return "model_validation" if p.section_path.startswith(_SR_VALIDATION_SECTIONS) else "model_governance"
    if p.doc_id == "fr2052a-instructions":
        return "reporting"
    if p.doc_id.startswith("12cfr217"):
        return "capital"
    sec = _cfr_section(p.section_path)
    if sec in _REPORTING_CFR_SECTIONS:
        return "reporting"
    if p.doc_id == "12cfr249":
        return "liquidity_buffer"
    if p.doc_id == "12cfr252-e":
        return "stress_testing"
    if p.doc_id == "12cfr252-d":
        if sec == "252.35":
            return "liquidity_buffer" if p.citation.startswith("12 CFR 252.35(b)") else "stress_testing"
        return "liquidity_risk_management"
    return None


def build_dataset(passages: list[Passage]) -> pd.DataFrame:
    rows = []
    for p in passages:
        label = topic_for(p)
        if label is None:
            continue
        rows.append({ID_COLUMN: p.chunk_id, GROUP_COLUMN: f"{p.doc_id}|{p.section_path}", CITATION_COLUMN: p.citation,
                     TEXT_COLUMN: p.text, LABEL_COLUMN: label})
    return pd.DataFrame(rows)


def group_split(df: pd.DataFrame, seed: int = 0, fractions: tuple[float, float, float] = (0.6, 0.2, 0.2)) -> pd.DataFrame:
    """Assign whole groups (sections) to train, val or test, balancing label counts greedily.

    No group appears in more than one split, so neighbouring paragraphs of one section never leak across splits.
    """
    rng = np.random.default_rng(seed)
    names = ["train", "val", "test"]
    groups = df.groupby(GROUP_COLUMN)
    order = list(groups.groups.keys())
    rng.shuffle(order)
    order.sort(key=lambda g: -len(groups.get_group(g)))  # place big groups first, ties stay shuffled
    labels = sorted(df[LABEL_COLUMN].unique())
    target = {s: {lab: fractions[i] * (df[LABEL_COLUMN] == lab).sum() for lab in labels} for i, s in enumerate(names)}
    have = {s: {lab: 0 for lab in labels} for s in names}
    n_label = {lab: int((df[LABEL_COLUMN] == lab).sum()) for lab in labels}
    assign: dict[str, str] = {}
    for g in order:
        counts = groups.get_group(g)[LABEL_COLUMN].value_counts()
        best, best_gain = None, None
        for s in names:
            gain = sum(min(counts.get(lab, 0), max(target[s][lab] - have[s][lab], 0)) / max(n_label[lab], 1) for lab in labels)
            gain += 1e-6 * rng.random()
            if best_gain is None or gain > best_gain:
                best, best_gain = s, gain
        assign[g] = best
        for lab in labels:
            have[best][lab] += counts.get(lab, 0)
    out = df.copy()
    out[SPLIT_COLUMN] = out[GROUP_COLUMN].map(assign)
    return out
