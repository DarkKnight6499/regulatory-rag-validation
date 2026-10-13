"""Dense retrieval: embedder interface, a deterministic hashing embedder (test double), and a sentence-transformers wrapper."""
from __future__ import annotations

import hashlib
import pickle
import zlib
from pathlib import Path
from typing import Protocol

import numpy as np

from .textproc import tokenize


class Embedder(Protocol):
    name: str

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        """Return an (n, d) float32 array of L2-normalised embeddings."""


def _l2(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return (x / np.maximum(n, 1e-12)).astype(np.float32)


class HashingEmbedder:
    """Signed feature hashing of unigrams and bigrams; deterministic and dependency free (not a semantic model)."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.name = f"hashing-{dim}"

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            toks = tokenize(t)
            feats = toks + [f"{a}_{b}" for a, b in zip(toks, toks[1:])]
            for f in feats:
                h = zlib.crc32(f.encode("utf-8"))
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        return _l2(out)


class SentenceTransformerEmbedder:
    """Wrapper around a small sentence-transformers model, forced onto the CPU, with a per-text disk cache."""

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", query_prefix: str = "",
                 cache_dir: Path | None = None, batch_size: int = 32) -> None:
        from sentence_transformers import SentenceTransformer  # lazy: tests never import this

        self.model = SentenceTransformer(model_name, device="cpu")
        self.name = model_name
        self.query_prefix = query_prefix
        self.batch_size = batch_size
        self._cache: dict[str, np.ndarray] = {}
        self._cache_path: Path | None = None
        if cache_dir is not None:
            Path(cache_dir).mkdir(parents=True, exist_ok=True)
            tag = hashlib.sha256(f"{model_name}|{query_prefix}".encode("utf-8")).hexdigest()[:12]
            self._cache_path = Path(cache_dir) / f"emb_{tag}.pkl"
            if self._cache_path.exists():
                with open(self._cache_path, "rb") as fh:
                    self._cache = pickle.load(fh)

    def encode(self, texts: list[str], is_query: bool = False) -> np.ndarray:
        texts = [(self.query_prefix + t) if is_query else t for t in texts]
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            emb = self.model.encode(todo, batch_size=self.batch_size, normalize_embeddings=True, show_progress_bar=False)
            for t, e in zip(todo, np.asarray(emb, dtype=np.float32)):
                self._cache[t] = e
            if self._cache_path is not None:
                with open(self._cache_path, "wb") as fh:
                    pickle.dump(self._cache, fh)
        return np.stack([self._cache[t] for t in texts]).astype(np.float32)


class DenseIndex:
    def __init__(self, texts: list[str], embedder: Embedder) -> None:
        self.embedder = embedder
        self.matrix = embedder.encode(texts, is_query=False)

    def scores(self, query: str) -> np.ndarray:
        q = self.embedder.encode([query], is_query=True)[0]
        return self.matrix @ q

    def top(self, query: str, k: int) -> list[int]:
        s = self.scores(query)
        return [int(i) for i in np.argsort(-s, kind="stable")[:k]]
