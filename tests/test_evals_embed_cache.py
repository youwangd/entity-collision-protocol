"""Tests for `evals._embed_cache.CachingEmbeddingProvider`.

This wrapper underpins every vector_weight sweep in `evals/`: it memoizes
embeddings across N cells of the same corpus. A regression here silently
inflates wall-clock and — worse — could return stale vectors if the
short-circuit logic confuses indices.
"""
from __future__ import annotations


from evals._embed_cache import CachingEmbeddingProvider


class _Counting:
    """Deterministic, instrumented embedding provider. embed(t) returns a
    1-d vector that uniquely identifies `t` so we can detect off-by-one
    bugs in batch slot assignment."""

    def __init__(self, dim: int = 4):
        self._dim = dim
        self.embed_calls = 0
        self.batch_calls = 0
        self.batched_texts: list[list[str]] = []

    @property
    def dimension(self) -> int:
        return self._dim

    @staticmethod
    def _vec(text: str, dim: int) -> list[float]:
        # Distinct, deterministic vector per text.
        h = hash(text) & 0xFFFFFFFF
        return [float((h >> (i * 4)) & 0xF) for i in range(dim)]

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return self._vec(text, self._dim)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        self.batched_texts.append(list(texts))
        return [self._vec(t, self._dim) for t in texts]


# --------------------------------------------------------------------------- #
# embed() — single text                                                       #
# --------------------------------------------------------------------------- #

class TestEmbedScalar:
    def test_first_call_misses_second_hits(self):
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner)
        v1 = cache.embed("hello")
        v2 = cache.embed("hello")
        assert v1 == v2
        assert inner.embed_calls == 1
        assert cache.stats["hits"] == 1
        assert cache.stats["misses"] == 1
        assert cache.stats["size"] == 1
        assert cache.stats["hit_rate"] == 0.5

    def test_distinct_texts_distinct_vectors(self):
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner)
        a = cache.embed("alpha")
        b = cache.embed("bravo")
        assert a != b
        assert inner.embed_calls == 2

    def test_dimension_delegates_to_inner(self):
        inner = _Counting(dim=7)
        cache = CachingEmbeddingProvider(inner)
        assert cache.dimension == 7

    def test_initial_stats(self):
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner)
        s = cache.stats
        assert s == {"hits": 0, "misses": 0, "size": 0, "hit_rate": 0.0}


# --------------------------------------------------------------------------- #
# embed_batch() — short-circuit + slot ordering                                #
# --------------------------------------------------------------------------- #

class TestEmbedBatch:
    def test_all_misses_calls_inner_once(self):
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner)
        out = cache.embed_batch(["a", "b", "c"])
        assert len(out) == 3
        assert inner.batch_calls == 1
        assert inner.batched_texts[0] == ["a", "b", "c"]
        assert cache.stats == {
            "hits": 0, "misses": 3, "size": 3, "hit_rate": 0.0,
        }

    def test_all_hits_skips_inner_entirely(self):
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner)
        cache.embed_batch(["a", "b"])
        inner.batch_calls = 0  # reset
        inner.batched_texts.clear()
        out = cache.embed_batch(["a", "b"])
        assert inner.batch_calls == 0  # full short-circuit
        assert cache.stats["hits"] == 2
        # And vectors match the cached ones.
        assert out == [_Counting._vec("a", 4), _Counting._vec("b", 4)]

    def test_mixed_hit_miss_preserves_input_order(self):
        """Critical correctness invariant: output[i] must correspond to
        input[i] regardless of which slots were misses. A bug in the
        miss_idx unpack would silently shuffle vectors with the wrong
        texts in retrieval."""
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner)
        cache.embed_batch(["x", "y"])  # prewarm
        out = cache.embed_batch(["new1", "x", "new2", "y", "new3"])
        # Inner only saw the misses, in the order they appeared.
        assert inner.batched_texts[-1] == ["new1", "new2", "new3"]
        # And output slot i lines up with input slot i.
        for i, t in enumerate(["new1", "x", "new2", "y", "new3"]):
            assert out[i] == _Counting._vec(t, 4), f"slot {i} ({t!r}) mismatched"

    def test_duplicate_in_single_batch(self):
        # Same text twice in one batch: with current impl, BOTH miss the
        # cache check (cache only updates after the inner call), so the
        # inner is asked for the same text twice — wasteful but correct.
        # We pin behavior here so future optimization is intentional.
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner)
        out = cache.embed_batch(["dup", "dup"])
        assert out[0] == out[1]
        assert inner.batched_texts[0] == ["dup", "dup"]


# --------------------------------------------------------------------------- #
# LRU eviction                                                                 #
# --------------------------------------------------------------------------- #

class TestLRU:
    def test_unbounded_by_default(self):
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner)
        for i in range(50):
            cache.embed(f"t{i}")
        assert cache.stats["size"] == 50

    def test_max_entries_evicts_oldest(self):
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner, max_entries=3)
        cache.embed("a")
        cache.embed("b")
        cache.embed("c")
        cache.embed("d")  # evicts "a"
        assert cache.stats["size"] == 3
        # "a" must miss again (was evicted).
        before = inner.embed_calls
        cache.embed("a")
        assert inner.embed_calls == before + 1
        # "d" must hit (most recent).
        before = inner.embed_calls
        cache.embed("d")
        assert inner.embed_calls == before

    def test_hit_promotes_to_recent(self):
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner, max_entries=3)
        cache.embed("a")
        cache.embed("b")
        cache.embed("c")
        cache.embed("a")  # hit; "a" promoted to MRU
        cache.embed("d")  # evicts LRU = "b" (not "a")
        # "b" is gone, "a" stays.
        before = inner.embed_calls
        cache.embed("a")
        assert inner.embed_calls == before, "a should still be cached"
        cache.embed("b")
        assert inner.embed_calls == before + 1, "b should have been evicted"

    def test_batch_path_also_evicts(self):
        inner = _Counting()
        cache = CachingEmbeddingProvider(inner, max_entries=2)
        cache.embed_batch(["a", "b", "c"])  # 3 misses, max=2
        assert cache.stats["size"] == 2
