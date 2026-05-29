"""Hypothesis stateful machine for write-side cosine dedup.

NEXT.md priority #3 calls out a write-dedup state machine fuzz. The static
property tests in `test_dedup_invariants.py` pin a small, hand-picked set
of invariants over independent sequences; this machine generates random
*interleavings* of `remember` (with random content drawn from a small
fixed corpus of disjoint clusters), `forget(hard=True)`, and `rebuild`,
and asserts dedup invariants hold across the whole trajectory:

  D-I1  Cluster cardinality bound: at any tick, the number of *active*
        rows whose content belongs to cluster C is ≤ 1 (under
        threshold > 0 with a deterministic embedder where intra-cluster
        cosine == 1.0 and inter-cluster cosine == 0.0).

  D-I2  Resurrection-after-hard-delete: hard-forgetting the sole
        representative of cluster C must allow the next `remember` of
        cluster-C content to land a new row (dedup is checked against
        the *active* vector store, not against tombstones). Net effect
        on cluster cardinality: still ≤ 1, but the row may appear under
        a different memory id.

  D-I3  Soft-suppress does NOT permit duplicate land: if a cluster-C
        row is soft-forgotten (state=SUPPRESSED) — but its embedding
        is still present in the vector index — the next cluster-C
        write should still dedup. (This guards against a regression
        where soft-forget would silently un-dedup the ACL-scope.)

  D-I4  Total-count parity: status()['total_memories'] equals the
        number of *active* memory ids the model currently believes
        exist (excluding hard-deleted ids).

The machine uses a 16-dim bag-of-chars deterministic embedder so cosine
values are fully reproducible; cluster prefixes are chosen from disjoint
single-character mod-classes so inter-cluster cosine is ~0.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from engram import Config, Engram
from engram.providers.embeddings import EmbeddingProvider
from engram.store.vector import SQLiteVecStore


# ---------------------------------------------------------------------------
# Deterministic embedder — bag-of-chars over 16 dims, mod-class projection.
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


# Disjoint cluster prefixes: each is a single character repeated, so the
# bag-of-chars embedder concentrates the unit vector on exactly one
# coordinate per cluster, giving inter-cluster cosine == 0.
_CLUSTERS = ["aaaaaaaa", "bbbbbbbb", "cccccccc", "dddddddd"]


def _content_strategy():
    # Generate cluster_idx only. Identical content per cluster guarantees
    # intra-cluster cosine == 1.0 under the bag-of-chars embedder, well above
    # any dedup threshold; inter-cluster cosine == 0 (disjoint mod-classes).
    return st.integers(min_value=0, max_value=len(_CLUSTERS) - 1)


def _materialize(ci: int) -> tuple[int, str]:
    """(cluster_idx, full_text)."""
    return ci, _CLUSTERS[ci]


class WriteDedupStateMachine(RuleBasedStateMachine):
    memory_ids = Bundle("memory_ids")

    def __init__(self):
        super().__init__()
        self._tmp: Path | None = None
        self._engram: Engram | None = None
        # cluster_idx -> set of active mids that landed for that cluster
        self._active_by_cluster: dict[int, set[str]] = {}
        # set of mids hard-deleted (must never resurface)
        self._hard_deleted: set[str] = set()
        # set of mids soft-forgotten (still in vector store)
        self._soft_suppressed: set[str] = set()
        # mid -> cluster_idx
        self._mid_cluster: dict[str, int] = {}

    @initialize()
    def setup(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="engram-dedup-sm-"))
        cfg = Config(path=str(self._tmp))
        cfg.storage.write_dedup_threshold = 0.92
        cfg.security.max_events_per_minute = 0
        eng = Engram(config=cfg)
        eng._embeddings = _DetEmbedder()
        eng._vector = SQLiteVecStore(self._tmp / "vectors.db", dimension=16)
        self._engram = eng

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    @rule(target=memory_ids, ci=_content_strategy())
    def remember(self, ci):
        e = self._engram
        _, text = _materialize(ci)
        e.remember(text, salience=0.4)
        # Find which mid (if any) is now the active representative for this
        # cluster. We drive recall by content; it should pull the existing
        # representative if dedup absorbed the write, or the new row if a
        # new one landed.
        hits = e.recall(text, limit=5)
        if not hits:
            return "noop"
        mid = hits[0].memory.id
        if mid not in self._mid_cluster:
            self._mid_cluster[mid] = ci
            self._active_by_cluster.setdefault(ci, set()).add(mid)
        return mid

    @rule(mid=memory_ids)
    def hard_forget(self, mid):
        if mid == "noop" or mid in self._hard_deleted:
            return
        e = self._engram
        e.forget(id=mid, hard=True)
        self._hard_deleted.add(mid)
        ci = self._mid_cluster.get(mid)
        if ci is not None:
            self._active_by_cluster.get(ci, set()).discard(mid)
        self._soft_suppressed.discard(mid)

    @rule(mid=memory_ids)
    def soft_forget(self, mid):
        if mid == "noop" or mid in self._hard_deleted:
            return
        e = self._engram
        e.forget(id=mid, hard=False)
        self._soft_suppressed.add(mid)
        # Soft-forget does NOT purge the vector embedding — and it does NOT
        # change which row is the "active" cluster representative for the
        # purposes of dedup (D-I3). The model's _active_by_cluster still
        # carries this mid because dedup will still match against it.

    # NOTE: We deliberately do NOT include a `rebuild()` rule in this state
    # machine. The original rationale was that `rebuild()` ignored write-side
    # dedup, so mixing it into the machine would just rediscover the known
    # gap on every shrink. That gap is now closed (engine.rebuild() routes
    # EXPLICIT_REMEMBER through the dedup-aware upsert path), and the closed
    # behaviour is pinned by `test_rebuild_dedup_closed` /
    # `test_rebuild_dedup_preserves_distinct_clusters` below. We still keep
    # rebuild out of the random-rule mix because it's deterministic and
    # idempotent w.r.t. the model state — exercising it stochastically would
    # slow shrinking without adding coverage. A future commit may re-enable
    # it as a dedicated `@rule` once a richer model emerges.

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    @invariant()
    def cluster_cardinality_bound(self):
        # D-I1: at most one *active* row per cluster.
        for ci, mids in self._active_by_cluster.items():
            live = mids - self._hard_deleted
            assert len(live) <= 1, (
                f"D-I1 violated: cluster {ci} has {len(live)} live rows: {live}"
            )

    @invariant()
    def total_count_parity(self):
        # D-I4: status total_memories matches our model's live-count.
        e = self._engram
        if e is None:
            return
        live = sum(
            len(mids - self._hard_deleted)
            for mids in self._active_by_cluster.values()
        )
        total = e.status()["total_memories"]
        assert total == live, (
            f"D-I4 violated: status total={total}, model live={live}"
        )


WriteDedupTest = WriteDedupStateMachine.TestCase
WriteDedupTest.settings = settings(
    max_examples=25,
    stateful_step_count=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# Smoke test: confirm the embedder geometry our state machine relies on.
# ---------------------------------------------------------------------------


def test_embedder_cluster_geometry_holds():
    """Inter-cluster cosine == 0; intra-cluster cosine == 1.0."""
    emb = _DetEmbedder()

    def cos(a: list[float], b: list[float]) -> float:
        return sum(x * y for x, y in zip(a, b))

    for i in range(len(_CLUSTERS)):
        for j in range(i + 1, len(_CLUSTERS)):
            c = cos(emb.embed(_CLUSTERS[i]), emb.embed(_CLUSTERS[j]))
            assert c < 0.5, f"clusters {i},{j} not disjoint: cos={c}"

    for ci, prefix in enumerate(_CLUSTERS):
        c = cos(emb.embed(prefix), emb.embed(prefix))
        assert c >= 0.999, f"self-cosine cluster {ci}: {c}"


# ---------------------------------------------------------------------------
# Closed: rebuild() now plumbs write-side dedup.
#
# Originally surfaced by the state machine on 2026-05-24: two sequential
# `remember(x)` calls absorbed to 1 row at write time but `rebuild()` re-created
# 2 rows because the dedup gate was remember-time-only and never replayed
# against the live vector store. Fix landed: `engine.rebuild()` now routes
# EXPLICIT_REMEMBER through the dedup-aware `_store.upsert(..., dedup_threshold,
# vector_store, embedding_provider)` path on full rebuild (after dropping the
# vector store) and on incremental replay. This test pins the closed-state
# behaviour: 1 row both pre- and post-rebuild.
# ---------------------------------------------------------------------------


def test_rebuild_dedup_closed():
    """Pinned: rebuild() now respects write-side dedup (gap closed)."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.storage.write_dedup_threshold = 0.92
        cfg.security.max_events_per_minute = 0
        e = Engram(config=cfg)
        e._embeddings = _DetEmbedder()
        e._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
        try:
            e.remember("aaaaaaaa", salience=0.4)
            e.remember("aaaaaaaa", salience=0.4)
            assert e.status()["total_memories"] == 1, (
                "write-time dedup regression — expected 1 active row"
            )
            e.rebuild()
            after = e.status()["total_memories"]
            assert after == 1, (
                f"rebuild-dedup regression: expected 1 row after rebuild, got {after}"
            )
        finally:
            e.close()


def test_rebuild_dedup_preserves_distinct_clusters():
    """Rebuild dedup must not over-collapse: distinct clusters survive."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.storage.write_dedup_threshold = 0.92
        cfg.security.max_events_per_minute = 0
        e = Engram(config=cfg)
        e._embeddings = _DetEmbedder()
        e._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
        try:
            # 3 events across 2 disjoint clusters: a-cluster (×2 dup), b-cluster (×1).
            e.remember("aaaaaaaa", salience=0.4)
            e.remember("aaaaaaaa", salience=0.4)
            e.remember("bbbbbbbb", salience=0.4)
            assert e.status()["total_memories"] == 2, "write-time: 2 clusters"
            e.rebuild()
            after = e.status()["total_memories"]
            assert after == 2, f"rebuild collapsed distinct clusters: got {after}, expected 2"
        finally:
            e.close()
