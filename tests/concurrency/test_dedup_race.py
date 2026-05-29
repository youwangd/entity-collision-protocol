"""Concurrency torture: write-side cosine dedup window race (mission item 2c).

50+ writer threads concurrently call `remember()` with content that should all
collide in the cosine@0.92 dedup window. The dedup check (vector_store.search
+ score-compare) is *not* atomic with the projection insert, so naive code
will produce a TOCTOU race where many threads pass the "no near-duplicate"
check before any of them have committed. This test pins the observed
behaviour:

  * No crashes / exceptions escape any worker.
  * The number of memories that survived dedup is bounded — not the full N
    (full N would mean the dedup is broken under contention).
  * The buffer / event log keeps every remember() event (audit invariant
    from §3.5 of the paper).
  * Distinct content always survives (control arm).

We don't require the projection count to be exactly 1: with the current
implementation a small race window is acceptable and documented. We require:
  - duplicates_landed << total_writes (≤ 25% of writers under heavy contention)
  - distinct content always passes (no false positive dedup)

Tested with the deterministic bag-of-chars embedder so cosine values are
predictable across runs.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from engram import Engram, Config
from engram.providers.embeddings import EmbeddingProvider
from engram.store.vector import SQLiteVecStore

pytestmark = pytest.mark.concurrency


class _DeterministicEmbedder(EmbeddingProvider):
    """Bag-of-chars 16-d unit vector. Identical text → cosine=1.0."""

    def __init__(self):
        self._dim = 16

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


def _mk_dedup_engram(tmp_path: Path, threshold: float = 0.92) -> Engram:
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.storage.write_dedup_threshold = threshold
    cfg.security.max_events_per_minute = 0  # disable rate-limit under torture
    eng = Engram(config=cfg)
    eng._embeddings = _DeterministicEmbedder()
    eng._vector = SQLiteVecStore(tmp_path / "engram" / "vectors.db", dimension=16)
    return eng


# --- the torture ---


def test_50_writers_identical_content_dedup_race(tmp_path: Path):
    """50 threads write the *same* string. Dedup should keep most out of projections.

    With a TOCTOU window between cosine-check and insert, *some* duplicates
    will leak through under contention. We bound that leakage at <= 25% of
    writers (12 of 50). In the steady state (no contention) it's 1.
    """
    eng = _mk_dedup_engram(tmp_path)
    try:
        N = 50
        text = "the launch is scheduled for next monday"
        errors: list[BaseException] = []
        barrier = threading.Barrier(N)

        def writer():
            try:
                barrier.wait()
                eng.remember(text, salience=0.4)
            except BaseException as ex:  # noqa: BLE001
                errors.append(ex)

        with ThreadPoolExecutor(max_workers=N) as pool:
            futs = [pool.submit(writer) for _ in range(N)]
            for f in as_completed(futs):
                f.result()

        assert not errors, f"writer errors: {errors[:3]}"

        landed = eng.status()["total_memories"]
        # Audit invariant: every remember() is in the buffer regardless of dedup.
        events = list(eng._buffer.scan())
        remember_events = [e for e in events if e.type.value == "explicit_remember"]
        assert len(remember_events) == N, \
            f"audit log must keep all {N} writes; got {len(remember_events)}"

        # Dedup invariant: under contention, at least *most* are dropped.
        assert landed <= int(N * 0.25), \
            f"dedup leaked too many duplicates: {landed} of {N} survived"
        # Sanity: at least one landed (otherwise the test would be vacuous).
        assert landed >= 1
    finally:
        eng.close()


def test_50_writers_distinct_content_no_false_positives(tmp_path: Path):
    """Control: 50 threads write distinct content. None should be deduped."""
    eng = _mk_dedup_engram(tmp_path)
    try:
        # Distinct content with disjoint bag-of-chars distributions.
        # The toy embedder is 16-d (ord(c) % 16); naive prefixes like
        # "aaaa unique-doc-NN" share the long " unique-doc-" suffix and end
        # up with cosine ≥ 0.92 across many pairs (false positives in the
        # *fixture*, not the dedup). Build texts as two-bucket bag-of-chars
        # signatures pulled from C(16,2) so pairwise cosine ≤ 0.5 by
        # construction, well below the 0.92 dedup threshold.
        import itertools as _it
        bucket_chars = ["p", "a", "b", "c", "d", "e", "f", "g",
                        "h", "i", "j", "k", "l", "m", "n", "o"]
        bucket_pairs = list(_it.combinations(range(16), 2))[:50]
        texts = [
            f"{bucket_chars[i] * 5} {bucket_chars[j] * 5}"
            for i, j in bucket_pairs
        ]
        N = len(texts)

        errors: list[BaseException] = []
        barrier = threading.Barrier(N)

        def writer(t: str):
            try:
                barrier.wait()
                eng.remember(t, salience=0.4)
            except BaseException as ex:  # noqa: BLE001
                errors.append(ex)

        with ThreadPoolExecutor(max_workers=N) as pool:
            futs = [pool.submit(writer, t) for t in texts]
            for f in as_completed(futs):
                f.result()

        assert not errors, f"errors: {errors[:3]}"
        landed = eng.status()["total_memories"]
        # With the disjoint-bucket fixture, pairwise cosine ≤ 0.5 by
        # construction, so the dedup gate must let *all* of them through.
        # Any drop is a real false positive.
        assert landed == N, \
            f"distinct content suffered false-positive dedup: {landed}/{N}"
    finally:
        eng.close()


def test_two_clusters_interleaved(tmp_path: Path):
    """Two clusters of identical-within-cluster, distinct-across-cluster writes.

    Final state: at most a small handful per cluster (ideally 1 each), but
    never zero, and the two clusters never merge.
    """
    eng = _mk_dedup_engram(tmp_path)
    try:
        cluster_a = "alpha alpha alpha alpha alpha"
        cluster_b = "zulu zulu zulu zulu zulu"
        N = 30
        errors: list[BaseException] = []
        barrier = threading.Barrier(N * 2)

        def writer(t: str):
            try:
                barrier.wait()
                eng.remember(t, salience=0.4)
            except BaseException as ex:  # noqa: BLE001
                errors.append(ex)

        with ThreadPoolExecutor(max_workers=N * 2) as pool:
            futs = (
                [pool.submit(writer, cluster_a) for _ in range(N)]
                + [pool.submit(writer, cluster_b) for _ in range(N)]
            )
            for f in as_completed(futs):
                f.result()

        assert not errors, f"errors: {errors[:3]}"

        # Both clusters represented (recall by content).
        a_hits = eng.recall("alpha", limit=5)
        b_hits = eng.recall("zulu", limit=5)
        assert len(a_hits) >= 1, "cluster A should have at least one survivor"
        assert len(b_hits) >= 1, "cluster B should have at least one survivor"

        # Total bounded.
        landed = eng.status()["total_memories"]
        assert landed <= int(N * 2 * 0.25), \
            f"too many duplicates leaked: {landed} of {N*2}"
    finally:
        eng.close()
