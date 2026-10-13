"""Evaluation metrics and bootstrap confidence intervals."""
from __future__ import annotations

from collections import Counter
from typing import Callable, Sequence

import numpy as np

from .textproc import normalize_text


def recall_at_k(ranked_hits: Sequence[bool], k: int) -> float:
    """1.0 if any of the top-k ranked items is a gold hit (single-answer recall)."""
    return float(any(ranked_hits[:k]))


def reciprocal_rank(ranked_hits: Sequence[bool]) -> float:
    for i, h in enumerate(ranked_hits, 1):
        if h:
            return 1.0 / i
    return 0.0


def token_f1(prediction: str, gold: str) -> float:
    p, g = normalize_text(prediction).split(), normalize_text(gold).split()
    if not p or not g:
        return float(p == g)
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0:
        return 0.0
    prec, rec = common / len(p), common / len(g)
    return 2 * prec * rec / (prec + rec)


def span_contained(prediction: str, gold_span: str) -> bool:
    """Lenient exact match: the normalised gold span occurs inside the normalised prediction."""
    return normalize_text(gold_span) in normalize_text(prediction)


def best_sentence_f1(sentences: Sequence[str], gold: str) -> float:
    """SQuAD-style: best token F1 of any returned sentence against the gold span, plus the joined text."""
    if not sentences:
        return 0.0
    cands = list(sentences) + [" ".join(sentences)]
    return max(token_f1(c, gold) for c in cands)


def abstention_scores(should_abstain: Sequence[bool], did_abstain: Sequence[bool]) -> dict[str, float]:
    tp = sum(s and d for s, d in zip(should_abstain, did_abstain))
    fp = sum((not s) and d for s, d in zip(should_abstain, did_abstain))
    fn = sum(s and (not d) for s, d in zip(should_abstain, did_abstain))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": precision, "recall": recall, "tp": tp, "fp": fp, "fn": fn}


def bootstrap_ci(values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0,
                 stat: Callable[[np.ndarray], float] = np.mean) -> tuple[float, float, float]:
    """Percentile bootstrap over items; returns (estimate, low, high)."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    boots = np.array([stat(arr[i]) for i in idx])
    return float(stat(arr)), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def bootstrap_ratio_ci(num: Sequence[float], den: Sequence[float], n_boot: int = 2000, alpha: float = 0.05,
                       seed: int = 0) -> tuple[float, float, float]:
    """Bootstrap CI for sum(num)/sum(den) (e.g. abstention precision), resampling question pairs."""
    n, d = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    if n.size == 0 or d.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n.size, size=(n_boot, n.size))
    ns, ds = n[idx].sum(axis=1), d[idx].sum(axis=1)
    ok = ds > 0
    boots = ns[ok] / ds[ok]
    return float(n.sum() / d.sum()), float(np.quantile(boots, alpha / 2)), float(np.quantile(boots, 1 - alpha / 2))


def paired_bootstrap_diff(a: Sequence[float], b: Sequence[float], n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """CI for mean(a - b) on paired per-question scores."""
    return bootstrap_ci(np.asarray(a, dtype=float) - np.asarray(b, dtype=float), n_boot=n_boot, seed=seed)
