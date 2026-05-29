"""Embedding provider interface and implementations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class EmbeddingProvider(ABC):
    """Abstract embedding provider."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Vector dimension."""
        ...

    @abstractmethod
    def embed(self, text: str) -> list[float]:
        """Embed a single text."""
        ...

    @abstractmethod
    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed multiple texts."""
        ...


class NoEmbeddingProvider(EmbeddingProvider):
    """Stub when no embedding provider is configured. Tier 0: BM25 only."""

    @property
    def dimension(self) -> int:
        return 0

    def embed(self, text: str) -> list[float]:
        return []

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [[] for _ in texts]


class HashTrigramEmbeddingProvider(EmbeddingProvider):
    """Lightweight zero-dependency embedding: char-trigram feature hashing.

    Maps each character trigram to a fixed bucket via blake2b, accumulates
    signed counts (sign from a second hash), and L2-normalizes. Pure Python,
    no torch/numpy required. Won't beat MiniLM but should beat pure BM25
    on paraphrase / morphological variants because trigrams of "darkmode"
    and "darktheme" overlap on "dar", "ark", "rk_", etc.
    """

    def __init__(self, dimension: int = 256, ngram: int = 3):
        if dimension <= 0:
            raise ValueError("dimension must be positive")
        if ngram < 2:
            raise ValueError("ngram must be >= 2")
        self._dimension = dimension
        self._ngram = ngram

    @property
    def dimension(self) -> int:
        return self._dimension

    def _tokens(self, text: str) -> list[str]:
        # lowercase, collapse whitespace, pad with spaces so word boundaries
        # become trigrams (e.g. " da", "rk ").
        s = " " + " ".join(text.lower().split()) + " "
        n = self._ngram
        if len(s) < n:
            return [s]
        return [s[i:i + n] for i in range(len(s) - n + 1)]

    def embed(self, text: str) -> list[float]:
        import hashlib
        import math

        vec = [0.0] * self._dimension
        for tok in self._tokens(text):
            h = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(h[:4], "little") % self._dimension
            sign = 1.0 if (h[4] & 1) else -1.0
            vec[bucket] += sign
        norm = math.sqrt(sum(v * v for v in vec))
        if norm > 0:
            vec = [v / norm for v in vec]
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        return [self.embed(t) for t in texts]


class SentenceTransformerProvider(EmbeddingProvider):
    """Local sentence-transformers embeddings (optional dependency)."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Install sentence-transformers: pip install engram[embeddings]")
        import os
        device = os.environ.get("ENGRAM_ST_DEVICE")  # 'mps' | 'cuda' | 'cpu' | None
        if device:
            self._model = SentenceTransformer(model_name, device=device)
        else:
            self._model = SentenceTransformer(model_name)
        # sentence-transformers ≥3.x renamed get_sentence_embedding_dimension
        # → get_embedding_dimension; fall back for older pins.
        if hasattr(self._model, "get_embedding_dimension"):
            self._dimension = self._model.get_embedding_dimension()
        else:
            self._dimension = self._model.get_sentence_embedding_dimension()

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        import os
        bs = int(os.environ.get("ENGRAM_ST_BATCH", "64"))
        vecs = self._model.encode(
            texts,
            batch_size=bs,
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=False,
        )
        return [v.tolist() for v in vecs]


class LiteLLMEmbeddingProvider(EmbeddingProvider):
    """API-based embeddings via litellm (optional dependency)."""

    def __init__(self, model: str = "text-embedding-3-small", dimension: int = 1536):
        try:
            import litellm
            self._litellm = litellm
        except ImportError:
            raise ImportError("Install litellm: pip install engram[llm]")
        self.model = model
        self._dimension = dimension

    @property
    def dimension(self) -> int:
        return self._dimension

    def embed(self, text: str) -> list[float]:
        resp = self._litellm.embedding(model=self.model, input=[text])
        return resp.data[0]["embedding"]

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        resp = self._litellm.embedding(model=self.model, input=texts)
        return [d["embedding"] for d in resp.data]
