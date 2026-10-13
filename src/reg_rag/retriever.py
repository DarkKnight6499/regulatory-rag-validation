"""Retrieval API: BM25, dense and reciprocal-rank-fusion hybrid, returning passages with citations."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .bm25 import BM25
from .chunking import Passage, chunk_cfr_markdown, chunk_guidance_markdown
from .corpus import read_manifest
from .dense import DenseIndex, Embedder

MODES = ("bm25", "dense", "hybrid")
RRF_K = 60


def rrf(rankings: list[list[int]], k: int = RRF_K) -> list[tuple[int, float]]:
    """Reciprocal rank fusion: score(d) = sum over rankings of 1 / (k + rank), rank starting at 1."""
    score: dict[int, float] = {}
    for ranking in rankings:
        for r, d in enumerate(ranking, 1):
            score[d] = score.get(d, 0.0) + 1.0 / (k + r)
    return sorted(score.items(), key=lambda kv: (-kv[1], kv[0]))


@dataclass(frozen=True)
class Hit:
    passage: Passage
    score: float
    rank: int


class Retriever:
    def __init__(self, passages: list[Passage], embedder: Embedder, pool: int = 50) -> None:
        self.passages = passages
        self.pool = pool
        texts = [p.index_text for p in passages]
        self.bm25 = BM25(texts)
        self.dense = DenseIndex(texts, embedder)

    def search(self, query: str, k: int = 5, mode: str = "hybrid") -> list[Hit]:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        n = min(self.pool, len(self.passages))
        if mode == "bm25":
            ids = self.bm25.top(query, k)
            scores = self.bm25.scores(query)
            ranked = [(i, float(scores[i])) for i in ids]
        elif mode == "dense":
            ids = self.dense.top(query, k)
            scores = self.dense.scores(query)
            ranked = [(i, float(scores[i])) for i in ids]
        else:
            fused = rrf([self.bm25.top(query, n), self.dense.top(query, n)])
            ranked = fused[:k]
        return [Hit(self.passages[i], s, r) for r, (i, s) in enumerate(ranked, 1)]


def load_passages(corpus_dir: Path) -> list[Passage]:
    """Chunk every successfully fetched document listed in the corpus manifest."""
    corpus_dir = Path(corpus_dir)
    passages: list[Passage] = []
    for e in read_manifest(corpus_dir / "manifest.json"):
        if e["status"] != "ok":
            continue
        text = (corpus_dir / e["file"]).read_text(encoding="utf-8")
        if e["doc_id"].startswith("12cfr"):
            passages += chunk_cfr_markdown(e["doc_id"], text)
        elif e["doc_id"] == "sr11-7":
            passages += chunk_guidance_markdown(e["doc_id"], text, "SR 11-7")
        elif e["doc_id"] == "fr2052a-instructions":
            passages += chunk_guidance_markdown(e["doc_id"], text, "FR 2052a Instructions")
    return passages
