"""Caching wrapper around any EmbeddingProvider.

Why: in `evals.sweep_vector_weight` we run the same dataset under N different
`vector_weight` values. The dataset content is identical across cells, so the
sentence-transformer encoder is called N× on the same strings — pure waste.

This wrapper memoizes `embed()` and short-circuits `embed_batch()` on a per-text
basis. The cache key is the raw text (datasets are small and texts are short;
keeping a string key avoids any hash collision risk and is faster than hashing
for our sizes).

Contract:
  - `dimension` is delegated to the wrapped provider
  - `embed(text)` returns the same vector across calls (provider determinism
    is assumed; ST + hash trigram both satisfy this)
  - `embed_batch(texts)` returns vectors in the SAME order as input, with
    cache hits short-circuited and only misses forwarded to the inner batch
  - Cache is unbounded by default — datasets we care about are <= ~10k texts.
    A `max_entries` knob is provided for safety; LRU eviction when set.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Optional

from engram.providers.embeddings import EmbeddingProvider


class CachingEmbeddingProvider(EmbeddingProvider):
    """Memoize embed() / embed_batch() by raw text.

    Stats accessible via `.stats` (hits, misses, size).
    """

    def __init__(
        self,
        inner: EmbeddingProvider,
        max_entries: Optional[int] = None,
    ) -> None:
        self._inner = inner
        self._max = max_entries
        # OrderedDict so we can do cheap LRU eviction when bounded
        self._cache: OrderedDict[str, list[float]] = OrderedDict()
        self._hits = 0
        self._misses = 0

    @property
    def dimension(self) -> int:
        return self._inner.dimension

    @property
    def stats(self) -> dict:
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "hit_rate": (self._hits / total) if total else 0.0,
        }

    def _store(self, text: str, vec: list[float]) -> None:
        self._cache[text] = vec
        if self._max is not None and len(self._cache) > self._max:
            # Evict oldest (LRU since we move-to-end on hit)
            self._cache.popitem(last=False)

    def embed(self, text: str) -> list[float]:
        v = self._cache.get(text)
        if v is not None:
            self._hits += 1
            self._cache.move_to_end(text)
            return v
        self._misses += 1
        v = self._inner.embed(text)
        self._store(text, v)
        return v

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Short-circuit fully cached batches without a model call
        out: list[list[float] | None] = [None] * len(texts)
        miss_idx: list[int] = []
        miss_texts: list[str] = []
        for i, t in enumerate(texts):
            v = self._cache.get(t)
            if v is not None:
                self._hits += 1
                self._cache.move_to_end(t)
                out[i] = v
            else:
                miss_idx.append(i)
                miss_texts.append(t)
        if miss_texts:
            self._misses += len(miss_texts)
            new_vecs = self._inner.embed_batch(miss_texts)
            for j, vec in zip(miss_idx, new_vecs):
                out[j] = vec
                self._store(texts[j], vec)
        # All slots are filled now (every miss got resolved)
        assert all(v is not None for v in out)
        return out  # type: ignore[return-value]
