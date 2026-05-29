"""Property-based dedup invariants — write-side cosine@threshold dedup.

Hypothesis strategies generate random write sequences and assert four
invariants that must hold under SEQUENTIAL writes (single-threaded). The
concurrency torture for the same dedup window lives in
`tests/concurrency/test_dedup_race.py`; here we pin the model-level
properties without thread interleaving noise.

Invariants pinned:

  (P1) Idempotence: repeated writes of the *same* content under
       threshold > 0 land exactly one memory regardless of N.
  (P2) Disjoint-cluster preservation: K clusters of identical-within /
       lexically-disjoint-across content collapse to K memories.
  (P3) Threshold monotonicity: for a fixed sequence with at least one
       repeated payload, the landed-count is monotonically non-increasing
       in the threshold across {0.0, 0.5, 0.9, 0.92, 0.99}. (0.0 disables
       dedup, so it is always the upper bound on landed count.)
  (P4) Order-independence: permuting the order of the input sequence
       does not change the final landed count under threshold > 0
       (since the deterministic embedder is content-only).

A deterministic 16-d bag-of-chars embedder is used so cosine values are
reproducible across Hypothesis examples and platforms.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

from engram import Engram, Config
from engram.providers.embeddings import EmbeddingProvider
from engram.store.vector import SQLiteVecStore


# ---------------------------------------------------------------------------
# Deterministic embedder — identical text → cosine == 1.0; lexically disjoint
# alphabets → cosine == 0.0 (within float tolerance).
# ---------------------------------------------------------------------------


class _DetEmbedder(EmbeddingProvider):
    def __init__(self, dim: int = 16):
        self._dim = dim

    @property
    def dimension(self) -> int:
        return self._dim

    def embed(self, text: str) -> list[float]:
        v = [0.0] * self._dim
        for ch in text.lower():
            v[ord(ch) % self._dim] += 1.0
        norm = sum(x * x for x in v) ** 0.5
        if norm == 0:
            return v
        return [x / norm for x in v]

    def embed_batch(self, texts):
        return [self.embed(t) for t in texts]


def _mk_engram(tmp: str, threshold: float):
    cfg = Config(path=tmp)
    cfg.storage.write_dedup_threshold = threshold
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    eng._embeddings = _DetEmbedder()
    eng._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
    return eng


# ---------------------------------------------------------------------------
# (P1) Idempotence
# ---------------------------------------------------------------------------


@given(
    text=st.text(alphabet="abcdefghij ", min_size=5, max_size=80).filter(lambda s: s.strip()),
    n=st.integers(min_value=2, max_value=15),
)
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_dedup_idempotent_same_content(text: str, n: int):
    """N sequential writes of identical content land exactly 1 memory."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _mk_engram(tmp, threshold=0.92)
        try:
            for _ in range(n):
                eng.remember(text, salience=0.4)
            assert eng.status()["total_memories"] == 1, (
                f"identical content × {n} writes should land 1, got "
                f"{eng.status()['total_memories']}"
            )
        finally:
            eng.close()


# ---------------------------------------------------------------------------
# (P2) Disjoint clusters preserved
# ---------------------------------------------------------------------------

# Use disjoint alphabets to guarantee cosine == 0 across clusters under the
# bag-of-chars embedder (the dim-16 modulo means we have to keep clusters
# in entirely different mod-classes; we use disjoint single-char prefixes
# and pad enough to dominate the bag).
_CLUSTER_PREFIXES = ["aaaaaa", "bbbbbb", "cccccc", "dddddd", "eeeeee"]


@given(
    k=st.integers(min_value=2, max_value=5),
    repeats=st.integers(min_value=1, max_value=6),
)
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_dedup_preserves_k_disjoint_clusters(k: int, repeats: int):
    """K disjoint clusters × repeats writes each → exactly K landed."""
    with tempfile.TemporaryDirectory() as tmp:
        eng = _mk_engram(tmp, threshold=0.92)
        try:
            cluster_texts = [_CLUSTER_PREFIXES[i] for i in range(k)]
            for _ in range(repeats):
                for t in cluster_texts:
                    eng.remember(t, salience=0.4)
            landed = eng.status()["total_memories"]
            assert landed == k, (
                f"{k} disjoint clusters × {repeats} repeats should land {k}, "
                f"got {landed}"
            )
        finally:
            eng.close()


# ---------------------------------------------------------------------------
# (P3) Threshold monotonicity
# ---------------------------------------------------------------------------


@given(
    base=st.text(alphabet="abcdefghij ", min_size=5, max_size=40).filter(lambda s: s.strip()),
    n=st.integers(min_value=3, max_value=8),
)
@settings(max_examples=15, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_dedup_threshold_monotonic_in_landed_count(base: str, n: int):
    """For the same sequence (N copies of identical content), landed count
    is monotonically non-increasing as threshold rises through the ladder.

    threshold=0.0 disables dedup entirely (landed == n). Threshold = 0.99
    is so strict that even bit-identical content (cosine=1.0) deduplicates.
    The full ladder must be a non-increasing sequence.
    """
    ladder = [0.0, 0.5, 0.9, 0.92, 0.99]
    counts: list[int] = []
    for thr in ladder:
        with tempfile.TemporaryDirectory() as tmp:
            eng = _mk_engram(tmp, threshold=thr)
            try:
                for _ in range(n):
                    eng.remember(base, salience=0.4)
                counts.append(eng.status()["total_memories"])
            finally:
                eng.close()

    # threshold=0.0 → no dedup → exactly n
    assert counts[0] == n, f"threshold=0 must disable dedup, got {counts[0]} != {n}"
    # Strict thresholds eventually collapse identical content to 1.
    assert counts[-1] == 1, f"threshold=0.99 should keep 1, got {counts[-1]}"
    # Monotonic non-increasing across the ladder.
    for i in range(1, len(counts)):
        assert counts[i] <= counts[i - 1], (
            f"landed count must be non-increasing in threshold; "
            f"thresholds={ladder}, counts={counts}"
        )


# ---------------------------------------------------------------------------
# (P5) Cosine-0.92 boundary decision — the actual threshold semantics
# ---------------------------------------------------------------------------
#
# (P1)–(P4) pin the *aggregate* behavior of the dedup window but never
# exercise cosine values between 0 and 1; they only cover the cos=1.0
# (identical) and cos=0.0 (disjoint-alphabet) extremes. (P5) measures
# the cosine between two payloads under the test embedder and asserts
# the dedup *decision* flips at the configured threshold:
#
#   measured cos ≥ threshold ⇒ second write deduped (landed = 1)
#   measured cos <  threshold ⇒ second write lands  (landed = 2)
#
# We sample pairs (a, b) drawn from a small, varied alphabet so that
# the bag-of-chars embedder produces a mix of cosines, then bucket
# each example by its measured cosine relative to the threshold.


def _cos(a: str, b: str) -> float:
    e = _DetEmbedder()
    va, vb = e.embed(a), e.embed(b)
    return sum(x * y for x, y in zip(va, vb))


@given(
    a=st.text(alphabet="abcdefgh ", min_size=4, max_size=20).filter(lambda s: s.strip()),
    b=st.text(alphabet="abcdefgh ", min_size=4, max_size=20).filter(lambda s: s.strip()),
    threshold=st.sampled_from([0.50, 0.80, 0.92, 0.99]),
)
@settings(max_examples=80, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_dedup_threshold_boundary_decision(a: str, b: str, threshold: float):
    """For any pair (a, b), measured-cos vs. threshold predicts landed count.

    This is the boundary version of (P3): not just 'monotonic in
    threshold' but 'the decision at *each* threshold matches the
    measured cosine'. Skips ties (within 1e-9) where float wobble
    can flip the comparison either direction.
    """
    cos = _cos(a, b)
    if abs(cos - threshold) < 1e-9:
        # Boundary tie — comparator-side rounding noise can land either
        # way, and the contract doesn't pin tie-break direction.
        return
    expected = 1 if cos >= threshold else 2
    with tempfile.TemporaryDirectory() as tmp:
        eng = _mk_engram(tmp, threshold=threshold)
        try:
            eng.remember(a, salience=0.4)
            eng.remember(b, salience=0.4)
            landed = eng.status()["total_memories"]
            assert landed == expected, (
                f"cos({a!r}, {b!r})={cos:.4f}, thr={threshold}; "
                f"expected landed={expected}, got {landed}"
            )
        finally:
            eng.close()


# ---------------------------------------------------------------------------
# (P4) Order-independence (was P6 — bumped down by P5 boundary test)
# ---------------------------------------------------------------------------


@given(
    seq=st.lists(
        st.sampled_from(_CLUSTER_PREFIXES),
        min_size=3,
        max_size=15,
    ),
    perm_seed=st.integers(min_value=0, max_value=10_000),
)
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_dedup_order_independent(seq: list[str], perm_seed: int):
    """Final landed count is invariant under permutation of the write
    sequence, given a deterministic content-only embedder."""
    import random
    rng = random.Random(perm_seed)
    permuted = seq[:]
    rng.shuffle(permuted)

    def _run(order: list[str]) -> int:
        with tempfile.TemporaryDirectory() as tmp:
            eng = _mk_engram(tmp, threshold=0.92)
            try:
                for t in order:
                    eng.remember(t, salience=0.4)
                return eng.status()["total_memories"]
            finally:
                eng.close()

    a = _run(seq)
    b = _run(permuted)
    assert a == b, (
        f"final landed count must be order-independent: "
        f"original={seq} → {a}, permuted={permuted} → {b}"
    )
    # Sanity: a equals the number of distinct items (since clusters are
    # disjoint by construction).
    assert a == len(set(seq))
