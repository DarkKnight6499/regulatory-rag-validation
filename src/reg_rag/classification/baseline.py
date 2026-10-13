"""Zero-shot keyword baseline standing in for a prompting approach, plus classification metrics."""
from __future__ import annotations

from collections import Counter
from typing import Sequence

import numpy as np

from ..textproc import tokenize

# Topic descriptions written from the topic names alone, before looking at classifier results.
TOPIC_KEYWORDS: dict[str, list[str]] = {
    "model_governance": ["model", "governance", "policies", "board", "senior management", "audit", "inventory", "risk management framework",
                         "owner", "developers", "challenge"],
    "model_validation": ["validation", "validate", "back-testing", "benchmarking", "outcomes analysis", "sensitivity analysis",
                         "conceptual soundness", "monitoring", "vendor", "testing"],
    "liquidity_buffer": ["liquidity coverage ratio", "hqla", "high-quality liquid assets", "level 1", "level 2a", "level 2b", "outflow",
                         "inflow", "net cash outflow", "stable funding", "liquidity buffer", "unencumbered", "deposit"],
    "liquidity_risk_management": ["liquidity risk", "risk committee", "chief risk officer", "contingency funding plan",
                                  "cash-flow projections", "liquidity risk tolerance", "independent review", "board of directors", "limits"],
    "stress_testing": ["stress test", "scenario", "severely adverse", "baseline", "planning horizon", "stress", "covered company",
                       "adverse"],
    "capital": ["capital", "tier 1", "common equity", "risk weight", "leverage ratio", "risk-weighted assets", "buffer", "deduction",
                "regulatory capital"],
    "reporting": ["report", "submit", "submission", "disclosure", "disclose", "field", "data element", "product", "reporting entity",
                  "publicly", "template"],
}


class KeywordClassifier:
    """Counts matches of topic keyword stems in the clause; ties and no-match fall back to the first label in `fallback`."""

    def __init__(self, keywords: dict[str, list[str]] = TOPIC_KEYWORDS, fallback: str = "model_governance") -> None:
        self.stems = {lab: [tuple(tokenize(k)) for k in kws] for lab, kws in keywords.items()}
        self.fallback = fallback

    def predict_one(self, text: str) -> str:
        toks = tokenize(text)
        grams = Counter(tuple(toks[i:i + n]) for n in (1, 2, 3) for i in range(len(toks) - n + 1))
        scores = {lab: sum(grams.get(k, 0) * len(k) for k in kws if k) for lab, kws in self.stems.items()}
        best = max(scores.values())
        if best == 0:
            return self.fallback
        winners = [lab for lab, s in scores.items() if s == best]
        return self.fallback if self.fallback in winners else sorted(winners)[0]

    def predict(self, texts: Sequence[str]) -> list[str]:
        return [self.predict_one(t) for t in texts]


def accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    return float(np.mean([a == b for a, b in zip(y_true, y_pred)])) if len(y_true) else float("nan")


def confusion_matrix(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> np.ndarray:
    """Rows are true labels, columns are predicted labels."""
    idx = {lab: i for i, lab in enumerate(labels)}
    m = np.zeros((len(labels), len(labels)), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[idx[t], idx[p]] += 1
    return m


def macro_f1(y_true: Sequence[str], y_pred: Sequence[str], labels: Sequence[str]) -> float:
    """Macro F1 over labels present in y_true (absent labels carry no support and are skipped)."""
    m = confusion_matrix(y_true, y_pred, labels)
    f1s = []
    for i in range(len(labels)):
        if m[i].sum() == 0:
            continue
        tp = m[i, i]
        prec = tp / m[:, i].sum() if m[:, i].sum() else 0.0
        rec = tp / m[i].sum()
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return float(np.mean(f1s))
