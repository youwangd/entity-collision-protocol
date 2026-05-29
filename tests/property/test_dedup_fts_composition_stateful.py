"""Cross-state-machine composition: write-dedup × FTS5.

NEXT.md priority #4 (research thread, first sub-bullet):

  > Cross-state-machine composition: e.g. write-dedup × FTS5 (a
  > dedup-absorbed write should still leave FTS5 in a consistent
  > state — currently we test these in isolation).

The two state machines we already have pin invariants for each subsystem
in isolation:

  * `test_write_dedup_stateful.py`  — D-I1..D-I4 over remember/forget
    with deterministic embedder, dedup threshold > 0, but FTS5 is
    *not exercised* (the machine never queries text).
  * `test_fts_index_stateful.py`    — F-I1..F-I4 over remember/forget
    with deterministic FTS5 tokens but dedup threshold = 0 (every
    write lands a row, so FTS5 and the row table 1:1).

What's missing is the *coupling* between them: when dedup absorbs
a write, the FTS5 index must NOT create an orphan row, and when the
sole cluster representative is hard-forgotten, the MATCH-set for the
cluster's token must drop to empty (no leaked FTS5 ghosts from prior
dedup-absorbed writes).

Composition invariants:

  X-I1  Dedup–FTS5 cardinality coupling: for every cluster C, the
        number of rows hit by FTS5 MATCH(C_token) restricted to
        states=['active','fading'] equals the number of active
        rows the model believes exist for cluster C (≤ 1 under
        threshold > 0). This catches the regression where dedup
        skips the row insert but a stray FTS5 row is still indexed.

  X-I2  Hard-delete cascades through dedup absorptions: after the
        sole representative of cluster C is hard-forgotten,
        MATCH(C_token) over ALL states (active+fading+faded+suppressed)
        returns 0 hits. This catches the regression where multiple
        prior writes were dedup-absorbed but one of them left an
        FTS5 ghost.

  X-I3  Total-FTS-cardinality parity: sum over all clusters of
        |MATCH(C_token, [active,fading])| == status['total_memories'].
        Whole-index level safety net.

The deterministic embedder is the same bag-of-chars embedder used by
the write-dedup machine; cluster tokens are chosen so their bag-of-chars
projections lie on disjoint mod-16 classes. Each cluster's content is
literally the token repeated, so:

  * dedup sees identical cosine == 1.0 within a cluster (absorbs)
  * dedup sees cosine == 0.0 across clusters (lands new row)
  * FTS5 can MATCH the token and only hit rows from that cluster.

Running ~25 hypothesis examples × ~20 steps gives strong coverage of
the interleaving space without being slow.
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
# Same deterministic bag-of-chars embedder as the write-dedup machine.
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


# Each cluster's content is a single repeated alphabetic char. Constraints:
#   * disjoint mod-16 ord values (so inter-cluster cosine == 0)
#   * length ≥ 4 so FTS5 tokenizer treats them as searchable terms
#   * unique within the suite (not shared with FTS-only or dedup-only machines,
#     to keep failure attribution clean across files)
# ord('e')=101 %16=5 ; ord('f')=102 %16=6 ; ord('g')=103 %16=7 ; ord('h')=104 %16=8
_CLUSTERS = ["eeeeee", "ffffff", "gggggg", "hhhhhh"]


class DedupFTSCompositionMachine(RuleBasedStateMachine):
    memory_ids = Bundle("memory_ids")

    def __init__(self):
        super().__init__()
        self._tmp: Path | None = None
        self._engram: Engram | None = None
        # cluster_idx -> set of mids the engine ever surfaced as a representative
        self._active_by_cluster: dict[int, set[str]] = {}
        # mid -> cluster_idx
        self._mid_cluster: dict[str, int] = {}
        # mids that have been hard-forgotten
        self._hard_deleted: set[str] = set()
        # mids that have been soft-forgotten (still alive in FTS5)
        self._soft_suppressed: set[str] = set()

    @initialize()
    def setup(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="engram-dedupfts-sm-"))
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

    @rule(target=memory_ids, ci=st.integers(min_value=0, max_value=len(_CLUSTERS) - 1))
    def remember(self, ci):
        e = self._engram
        text = _CLUSTERS[ci]
        e.remember(text, salience=0.4)
        # Identify the active cluster representative via FTS5 — note the
        # critical choice: we use the *FTS5 path*, not vector recall, so
        # that this rule both drives traffic AND probes the index.
        hits = e._store.search_text(text, limit=10, states=["active", "fading"])
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
        # Note: per D-I3 from the dedup machine, soft-forget does NOT
        # un-dedup the cluster (the embedding is still in the vector
        # store), so we do NOT remove the mid from _active_by_cluster.
        # That said, the mid IS suppressed for FTS5-default-state
        # purposes, so X-I1 must account for it: a suppressed mid no
        # longer counts as an "active rep" for cluster C.

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    def _live_active_for_cluster(self, ci: int) -> set[str]:
        """Mids that are not hard-deleted AND not soft-suppressed."""
        return (
            self._active_by_cluster.get(ci, set())
            - self._hard_deleted
            - self._soft_suppressed
        )

    @invariant()
    def dedup_fts_cardinality_coupling(self):
        # X-I1
        e = self._engram
        if e is None:
            return
        for ci, _ in enumerate(_CLUSTERS):
            token = _CLUSTERS[ci]
            hits = e._store.search_text(
                token, limit=50, states=["active", "fading"]
            )
            fts_mids = {h.memory.id for h in hits}
            model_active = self._live_active_for_cluster(ci)
            assert fts_mids == model_active, (
                f"X-I1 violated: cluster {ci} ({token!r}) "
                f"FTS5 active hits {fts_mids} != model active {model_active}\n"
                f"  hard_deleted intersection: "
                f"{fts_mids & self._hard_deleted}\n"
                f"  suppressed intersection: "
                f"{fts_mids & self._soft_suppressed}"
            )

    @invariant()
    def hard_delete_cascades_through_dedup(self):
        # X-I2: if every mid the model has ever attributed to cluster C
        # is hard-deleted, MATCH(C_token) over ALL states must be empty.
        e = self._engram
        if e is None:
            return
        for ci, _ in enumerate(_CLUSTERS):
            attributed = self._active_by_cluster.get(ci, set())
            if not attributed:
                continue
            if not attributed.issubset(self._hard_deleted):
                continue
            token = _CLUSTERS[ci]
            hits = e._store.search_text(
                token, limit=50,
                states=["active", "fading", "faded", "suppressed"],
            )
            ghost_mids = {h.memory.id for h in hits} & attributed
            assert not ghost_mids, (
                f"X-I2 violated: cluster {ci} ({token!r}) — all "
                f"attributed mids hard-deleted but FTS5 still "
                f"surfaces ghosts: {ghost_mids}"
            )

    @invariant()
    def total_fts_cardinality_parity(self):
        # X-I3
        e = self._engram
        if e is None:
            return
        total_fts_active = 0
        for ci, _ in enumerate(_CLUSTERS):
            token = _CLUSTERS[ci]
            hits = e._store.search_text(
                token, limit=50, states=["active", "fading"]
            )
            total_fts_active += len(hits)
        # status total_memories includes suppressed rows too (they're
        # still rows, just in a different state). What we want here is:
        # FTS5 active+fading hit count == #rows in active/fading state.
        # Easier to compute from status['by_state']:
        bs = e.status().get("by_state", {})
        active_or_fading = bs.get("active", 0) + bs.get("fading", 0)
        assert total_fts_active == active_or_fading, (
            f"X-I3 violated: FTS5 active-state hits across all clusters "
            f"= {total_fts_active}, status active+fading = {active_or_fading}"
        )


DedupFTSCompositionTest = DedupFTSCompositionMachine.TestCase
DedupFTSCompositionTest.settings = settings(
    max_examples=25,
    stateful_step_count=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# Closed-state pin: the simplest concrete trace exercising the coupling.
# Useful as a fast smoke and as a witness if the state machine ever fails.
# ---------------------------------------------------------------------------


def test_dedup_absorbed_write_does_not_create_fts_orphan_smoke():
    """Two writes of the same cluster -> 1 row in DB AND 1 row in FTS5."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.storage.write_dedup_threshold = 0.92
        cfg.security.max_events_per_minute = 0
        e = Engram(config=cfg)
        e._embeddings = _DetEmbedder()
        e._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
        try:
            e.remember("eeeeee", salience=0.4)
            e.remember("eeeeee", salience=0.4)  # absorbed
            assert e.status()["total_memories"] == 1
            hits = e._store.search_text(
                "eeeeee", limit=10, states=["active", "fading"]
            )
            assert len(hits) == 1, (
                f"FTS5 orphan: expected 1 hit after dedup absorption, got {len(hits)}"
            )
        finally:
            e.close()


def test_hard_forget_after_dedup_absorption_purges_fts_smoke():
    """After several dedup-absorbed writes, hard-forget purges FTS5."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.storage.write_dedup_threshold = 0.92
        cfg.security.max_events_per_minute = 0
        e = Engram(config=cfg)
        e._embeddings = _DetEmbedder()
        e._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
        try:
            for _ in range(5):
                e.remember("ffffff", salience=0.4)
            hits = e._store.search_text(
                "ffffff", limit=10, states=["active", "fading"]
            )
            assert len(hits) == 1
            mid = hits[0].memory.id
            e.forget(id=mid, hard=True)
            after = e._store.search_text(
                "ffffff", limit=10,
                states=["active", "fading", "faded", "suppressed"],
            )
            assert not after, f"FTS5 ghost after hard-forget: {after}"
        finally:
            e.close()
