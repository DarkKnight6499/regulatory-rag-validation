"""Okapi BM25 implemented directly so the fast tests need no model or extra dependency."""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from .textproc import tokenize


class BM25:
    def __init__(self, documents: list[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.doc_tokens = [tokenize(d) for d in documents]
        self.doc_len = np.array([len(t) for t in self.doc_tokens], dtype=float)
        self.avgdl = float(self.doc_len.mean()) if len(documents) else 0.0
        self.tf = [Counter(t) for t in self.doc_tokens]
        df: Counter = Counter()
        for c in self.tf:
            df.update(c.keys())
        n = len(documents)
        self.idf = {t: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for t, d in df.items()}
        self.postings: dict[str, list[int]] = {}
        for i, c in enumerate(self.tf):
            for t in c:
                self.postings.setdefault(t, []).append(i)

    def scores(self, query: str) -> np.ndarray:
        out = np.zeros(len(self.tf))
        for t in set(tokenize(query)):
            idf = self.idf.get(t)
            if idf is None:
                continue
            for i in self.postings[t]:
                f = self.tf[i][t]
                norm = 1.0 - self.b + self.b * self.doc_len[i] / (self.avgdl or 1.0)
                out[i] += idf * f * (self.k1 + 1.0) / (f + self.k1 * norm)
        return out

    def top(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        order = np.argsort(-s, kind="stable")[:k]
        return [int(i) for i in order if s[i] > 0]
