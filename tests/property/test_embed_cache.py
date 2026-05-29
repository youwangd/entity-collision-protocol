"""Property-based + integration tests for evals._embed_cache.CachingEmbeddingProvider.

Contracts locked:
  - Wrapped vectors are bit-identical to the inner provider's output
  - Repeated calls hit the cache (inner is not re-invoked)
  - embed_batch returns vectors in input order even when partially cached
  - Stats accounting is correct (hits + misses == total calls; size == unique inputs)
  - LRU eviction respects max_entries
  - dimension is delegated correctly
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from engram.providers.embeddings import HashTrigramEmbeddingProvider
from evals._embed_cache import CachingEmbeddingProvider


class _CountingProvider:
    """Test double: deterministic vectors + call counters."""

    def __init__(self, dim: int = 8) -> None:
        self._dim = dim
        self.embed_calls = 0
        self.batch_calls = 0
        self.batch_total = 0  # total texts seen across batch calls

    @property
    def dimension(self) -> int:
        return self._dim

    def _vec(self, text: str) -> list[float]:
        # Stable, distinct per text
        h = abs(hash(text))
        return [float((h >> (i * 4)) & 0xF) for i in range(self._dim)]

    def embed(self, text: str) -> list[float]:
        self.embed_calls += 1
        return self._vec(text)

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        self.batch_calls += 1
        self.batch_total += len(texts)
        return [self._vec(t) for t in texts]


def test_dimension_delegated() -> None:
    inner = HashTrigramEmbeddingProvider(dimension=64)
    cache = CachingEmbeddingProvider(inner)
    assert cache.dimension == 64


def test_embed_caches_repeats() -> None:
    inner = _CountingProvider()
    cache = CachingEmbeddingProvider(inner)
    v1 = cache.embed("hello")
    v2 = cache.embed("hello")
    v3 = cache.embed("world")
    assert v1 == v2
    assert v3 != v1
    assert inner.embed_calls == 2  # one per unique
    s = cache.stats
    assert s["hits"] == 1
    assert s["misses"] == 2
    assert s["size"] == 2


def test_embed_batch_partial_hit_preserves_order() -> None:
    inner = _CountingProvider()
    cache = CachingEmbeddingProvider(inner)
    # Warm cache with two of the four texts
    cache.embed("a")
    cache.embed("c")
    inner.embed_calls = 0
    inner.batch_calls = 0
    inner.batch_total = 0

    batch = ["a", "b", "c", "d", "b"]
    vecs = cache.embed_batch(batch)
    # Order preserved: each output vector matches the corresponding input
    for txt, vec in zip(batch, vecs):
        assert vec == inner._vec(txt)
    # Inner batch was called once with only the new-unique misses
    assert inner.batch_calls == 1
    # Misses sent to inner should be the unique new ones (b, d) — but our
    # implementation forwards every miss including dupes within the batch.
    # That's fine; the contract is "no re-encode of already-cached", not
    # "dedupe within a batch". Lock the looser contract.
    assert inner.batch_total == 3  # b, d, b (b appears twice in batch, both miss until first stores)


def test_embed_batch_pure_hit_skips_inner() -> None:
    inner = _CountingProvider()
    cache = CachingEmbeddingProvider(inner)
    for t in ["x", "y", "z"]:
        cache.embed(t)
    inner.batch_calls = 0
    inner.batch_total = 0
    vecs = cache.embed_batch(["z", "y", "x"])
    assert vecs[0] == inner._vec("z")
    assert vecs[1] == inner._vec("y")
    assert vecs[2] == inner._vec("x")
    assert inner.batch_calls == 0  # all hits, no inner call


def test_lru_eviction() -> None:
    inner = _CountingProvider()
    cache = CachingEmbeddingProvider(inner, max_entries=2)
    cache.embed("a")
    cache.embed("b")
    cache.embed("c")  # evicts "a"
    assert cache.stats["size"] == 2
    # Re-querying "a" should miss again
    inner.embed_calls = 0
    cache.embed("a")
    assert inner.embed_calls == 1
    # And "b"/"c" should still be cached (b just evicted? no — c evicted a, b is fine)
    inner.embed_calls = 0
    cache.embed("c")
    assert inner.embed_calls == 0


_text = st.text(alphabet=st.characters(min_codepoint=ord("a"), max_codepoint=ord("z")),
                min_size=1, max_size=10)


@given(texts=st.lists(_text, min_size=1, max_size=20))
@settings(max_examples=100, deadline=None)
def test_cache_matches_inner_on_arbitrary_inputs(texts: list[str]) -> None:
    inner = HashTrigramEmbeddingProvider(dimension=32)
    cache = CachingEmbeddingProvider(inner)
    for t in texts:
        a = cache.embed(t)
        b = inner.embed(t)
        assert a == b


@given(
    seq=st.lists(_text, min_size=1, max_size=40),
    cap=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=100, deadline=None)
def test_lru_eviction_order_matches_oracle(seq: list[str], cap: int) -> None:
    """Cache contents under bounded max_entries match a pure-Python LRU oracle.

    Oracle: OrderedDict with move-to-end on hit, popitem(last=False) on overflow.
    Both `embed()` (mixed hit/miss) and pure `embed()` work the same way.
    """
    from collections import OrderedDict

    inner = _CountingProvider()
    cache = CachingEmbeddingProvider(inner, max_entries=cap)
    oracle: OrderedDict[str, None] = OrderedDict()

    for t in seq:
        cache.embed(t)
        if t in oracle:
            oracle.move_to_end(t)
        else:
            oracle[t] = None
            if len(oracle) > cap:
                oracle.popitem(last=False)

    # Same set of keys retained
    assert set(cache._cache.keys()) == set(oracle.keys())
    # Same eviction order (least->most recently used)
    assert list(cache._cache.keys()) == list(oracle.keys())
    assert cache.stats["size"] == len(oracle)
    assert cache.stats["size"] <= cap


@given(
    seq=st.lists(_text, min_size=1, max_size=30, unique=True),
    cap=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=80, deadline=None)
def test_lru_eviction_under_batch_matches_oracle(seq: list[str], cap: int) -> None:
    """Same invariant when texts arrive via embed_batch (single big batch).

    NOTE: locked to unique-text batches. Within a single batch, duplicate
    texts that aren't pre-cached all register as misses (queued before any
    is stored), so duplicates don't induce an LRU touch the way `embed()`
    would. That's an internal implementation detail, not a public contract,
    so we don't fuzz it here — the unique-sequence property is what matters
    for sweep workloads (datasets are deduped at ingest).
    """
    from collections import OrderedDict

    inner = _CountingProvider()
    cache = CachingEmbeddingProvider(inner, max_entries=cap)
    cache.embed_batch(seq)

    oracle: OrderedDict[str, None] = OrderedDict()
    for t in seq:
        oracle[t] = None
        if len(oracle) > cap:
            oracle.popitem(last=False)

    assert list(cache._cache.keys()) == list(oracle.keys())
    assert cache.stats["size"] <= cap


@given(
    cap=st.integers(min_value=1, max_value=5),
    fill=st.lists(_text, min_size=2, max_size=15, unique=True),
)
@settings(max_examples=60, deadline=None)
def test_recent_access_is_never_evicted_next(cap: int, fill: list[str]) -> None:
    """Touching a cached key must protect it from being the immediate next eviction.

    Insert `fill` (capped), then re-touch the oldest, then insert one fresh key.
    The just-touched key must still be present; the previously-second-oldest is gone.
    """
    inner = _CountingProvider()
    cache = CachingEmbeddingProvider(inner, max_entries=cap)
    for t in fill:
        cache.embed(t)
    # Final cache holds last `cap` distinct items from fill
    retained = list(cache._cache.keys())
    # Need cache to be FULL for the next insert to actually evict
    if len(retained) < cap or len(retained) < 2:
        return  # not enough state to exercise eviction
    oldest = retained[0]
    second_oldest = retained[1]
    # Touch oldest -> moves to MRU
    cache.embed(oldest)
    # Insert a brand-new text that isn't already cached
    fresh = "".join(fill) + "_zzz_unique"
    cache.embed(fresh)
    keys = list(cache._cache.keys())
    assert oldest in keys, "freshly-touched key was evicted"
    assert fresh in keys
    assert second_oldest not in keys, "expected second-oldest to be evicted"


@given(texts=st.lists(_text, min_size=1, max_size=15),
       repeats=st.integers(min_value=1, max_value=4))
@settings(max_examples=50, deadline=None)
def test_repeated_calls_increase_hits_not_misses(
    texts: list[str], repeats: int
) -> None:
    inner = _CountingProvider()
    cache = CachingEmbeddingProvider(inner)
    unique = list(dict.fromkeys(texts))  # preserves order, dedupes
    # First pass: all misses (one inner call per unique)
    for t in texts:
        cache.embed(t)
    s1 = cache.stats
    assert s1["misses"] == len(unique)
    assert inner.embed_calls == len(unique)
    # Subsequent passes: all hits, no new inner calls
    inner.embed_calls = 0
    for _ in range(repeats):
        for t in texts:
            cache.embed(t)
    assert inner.embed_calls == 0
    s2 = cache.stats
    assert s2["misses"] == s1["misses"]
    assert s2["hits"] == s1["hits"] + repeats * len(texts)
