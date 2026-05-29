"""Cross-state-machine composition: write-dedup × schema-lifecycle.

Closes the last open corner of the cross-state-machine matrix flagged
in NEXT.md priority #1. The cross_channel_coupling_audit verdict for
E×G is "I" (independent by design), but "independent by design" must
still be *pinned by interleaved fuzz* — otherwise a future refactor
that introduces an unintended coupling (e.g. lifecycle emission
reading dedup state, or a shared in-memory cache) goes silently wrong.

The two channels share exactly one substrate: `JSONLBufferStore`. Both
append events to it. Channel E (write-dedup) emits MEMORY_UPSERT (or
not, on dedup-absorption). Channel G (schema-lifecycle) emits
CONSOLIDATION_SCHEMA_LIFECYCLE. The projection (snapshot_from_buffer)
filters by event_type, so by construction dedup events should never
perturb the lifecycle snapshot — and lifecycle events should never
perturb dedup decisions (write-dedup reads only the vector store).

Invariants:

  EG-I1  Filter invariance — projection independent of E traffic:
         snapshot_from_buffer(buffer) is byte-for-byte equal to
         reduce_events(direct lifecycle event list), regardless of
         how many MEMORY_UPSERT / forget / RECALL_REQUEST events
         interleave around the lifecycle events. Pins that the
         scan(event_type=...) filter is exact.

  EG-I2  Cache parity — incremental == full replay under mixed traffic:
         CachedLifecycleSnapshot.get(buffer) == snapshot_from_buffer(buffer)
         at every observation point. Pins that interleaved non-lifecycle
         appends do not break the byte-offset bookkeeping.

  EG-I3  Lifecycle DAG holds — no illegal transitions surface in the
         snapshot under mixed traffic. (The reducer enforces this in
         strict mode; here we run lenient and assert the DAG ourselves.)

  EG-I4  Dedup independence from lifecycle — total_memories from status()
         depends only on the dedup-absorbing remember/forget calls, not
         on intervening lifecycle emissions. We track an expected count
         in the model and assert parity with status().

The strategy generates random interleavings of:
  * remember(cluster_text) — dedup-absorbed under threshold>0
  * hard-forget the sole rep of a cluster
  * append a CREATE/PROMOTE/DEPRECATE/RECOVER/BUMP_VERSION lifecycle event
This is fuzz-equivalent to two operators racing on the buffer.
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
from engram.consolidation.lifecycle_projection import (
    CachedLifecycleSnapshot,
    make_lifecycle_event,
    snapshot_from_buffer,
)
from engram.consolidation.schema_lifecycle import (
    EventKind,
    SchemaLifecycleEvent,
    SchemaStatus,
    reduce_events,
)
from engram.providers.embeddings import EmbeddingProvider
from engram.store.vector import SQLiteVecStore


# Same deterministic embedder as the dedup state machine.
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


# Disjoint clusters from the dedup-only and dedup×FTS5 machines so
# failure attribution stays clean (these don't appear elsewhere).
# ord('i')=105%16=9; 'j'=10; 'k'=11; 'l'=12.
_CLUSTERS = ["iiiiii", "jjjjjj", "kkkkkk", "llllll"]
_SCHEMA_IDS = ["sigma_alpha", "sigma_beta", "sigma_gamma"]
_WINDOW_IDS = ["w_red", "w_blue", "w_green", "w_violet"]

# DAG for EG-I3 (lenient reducer never lands on out-of-DAG transitions).
_ALLOWED = {
    (SchemaStatus.INFERRED, SchemaStatus.PROMOTED),
    (SchemaStatus.INFERRED, SchemaStatus.DEPRECATED),
    (SchemaStatus.PROMOTED, SchemaStatus.DEPRECATED),
    (SchemaStatus.DEPRECATED, SchemaStatus.INFERRED),
}


class DedupLifecycleCompositionMachine(RuleBasedStateMachine):
    memory_ids = Bundle("memory_ids")

    def __init__(self):
        super().__init__()
        self._tmp: Path | None = None
        self._engram: Engram | None = None
        self._cache: CachedLifecycleSnapshot | None = None
        # Direct ledger of every lifecycle event we appended, in order
        # — the canonical input to the reducer for EG-I1 comparison.
        self._direct_events: list[SchemaLifecycleEvent] = []
        # Dedup model state
        self._active_by_cluster: dict[int, set[str]] = {}
        self._mid_cluster: dict[str, int] = {}
        self._hard_deleted: set[str] = set()
        # Snapshot of per-schema status before each lifecycle append, for
        # EG-I3: every observed status change must lie in _ALLOWED.
        self._prior_snapshot: dict = {}

    @initialize()
    def setup(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="engram-eg-sm-"))
        cfg = Config(path=str(self._tmp))
        cfg.storage.write_dedup_threshold = 0.92
        cfg.security.max_events_per_minute = 0
        eng = Engram(config=cfg)
        eng._embeddings = _DetEmbedder()
        eng._vector = SQLiteVecStore(self._tmp / "vectors.db", dimension=16)
        self._engram = eng
        self._cache = CachedLifecycleSnapshot()

    # --- Rules (channel E: dedup) ---

    @rule(target=memory_ids, ci=st.integers(min_value=0, max_value=len(_CLUSTERS) - 1))
    def remember(self, ci):
        e = self._engram
        text = _CLUSTERS[ci]
        e.remember(text, salience=0.4)
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
        self._engram.forget(id=mid, hard=True)
        self._hard_deleted.add(mid)
        ci = self._mid_cluster.get(mid)
        if ci is not None:
            self._active_by_cluster.get(ci, set()).discard(mid)

    # --- Rules (channel G: lifecycle) ---

    @rule(
        sid=st.sampled_from(_SCHEMA_IDS),
        kind=st.sampled_from(list(EventKind)),
        win=st.sampled_from(_WINDOW_IDS),
    )
    def emit_lifecycle(self, sid, kind, win):
        e = self._engram
        ev = make_lifecycle_event(schema_id=sid, kind=kind, window_id=win)
        # Snapshot before so we can verify EG-I3 (DAG-respect) on the
        # delta induced by this single append.
        before = snapshot_from_buffer(e._buffer, strict=False)
        e._buffer.append(ev)
        self._direct_events.append(
            SchemaLifecycleEvent(schema_id=sid, kind=kind, window_id=win, ts=0)
        )
        after = snapshot_from_buffer(e._buffer, strict=False)
        # EG-I3: any status delta must be in the DAG.
        b = before.get(sid)
        a = after.get(sid)
        if b is not None and a is not None and b.status != a.status:
            assert (b.status, a.status) in _ALLOWED, (
                f"EG-I3: illegal lifecycle transition {b.status} -> {a.status} "
                f"via {kind} on {sid}"
            )

    # --- Invariants ---

    @invariant()
    def filter_invariance(self):
        # EG-I1
        e = self._engram
        if e is None:
            return
        snap = snapshot_from_buffer(e._buffer, strict=False)
        direct = reduce_events(self._direct_events, strict=False)
        assert set(snap.keys()) == set(direct.keys()), (
            f"EG-I1 keys: snap={set(snap.keys())} direct={set(direct.keys())}"
        )
        for sid, st_d in direct.items():
            st_s = snap[sid]
            assert (st_s.status, st_s.version, st_s.last_window_id) == (
                st_d.status, st_d.version, st_d.last_window_id
            ), (
                f"EG-I1 mismatch on {sid}: snap={st_s} direct={st_d}"
            )

    @invariant()
    def cache_parity(self):
        # EG-I2
        e = self._engram
        if e is None or self._cache is None:
            return
        full = snapshot_from_buffer(e._buffer, strict=False)
        cached = self._cache.get(e._buffer, strict=False)
        assert set(full.keys()) == set(cached.keys()), (
            f"EG-I2 keys: full={set(full.keys())} cached={set(cached.keys())}"
        )
        for sid, st_f in full.items():
            st_c = cached[sid]
            assert (st_f.status, st_f.version, st_f.last_window_id) == (
                st_c.status, st_c.version, st_c.last_window_id
            ), (
                f"EG-I2 mismatch on {sid}: full={st_f} cached={st_c}"
            )

    @invariant()
    def dedup_independence_from_lifecycle(self):
        # EG-I4: total_memories driven by dedup model only.
        e = self._engram
        if e is None:
            return
        live = sum(
            len(mids - self._hard_deleted)
            for mids in self._active_by_cluster.values()
        )
        total = e.status()["total_memories"]
        assert total == live, (
            f"EG-I4: status total={total}, model live={live} — "
            f"lifecycle traffic perturbed dedup count?"
        )


DedupLifecycleCompositionTest = DedupLifecycleCompositionMachine.TestCase
DedupLifecycleCompositionTest.settings = settings(
    max_examples=25,
    stateful_step_count=20,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# Closed-state smoke: a deterministic mini-trace exercising the coupling.
def test_lifecycle_projection_unchanged_by_dedup_absorption_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.storage.write_dedup_threshold = 0.92
        cfg.security.max_events_per_minute = 0
        e = Engram(config=cfg)
        e._embeddings = _DetEmbedder()
        e._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
        try:
            # Three identical writes -> one row (dedup absorbs 2nd and 3rd)
            for _ in range(3):
                e.remember("iiiiii", salience=0.4)
            # Interleave lifecycle CREATE + PROMOTE on a schema
            e._buffer.append(make_lifecycle_event(
                schema_id="sigma_x", kind=EventKind.CREATE, window_id="w_red"))
            e.remember("iiiiii", salience=0.4)  # absorbed
            e._buffer.append(make_lifecycle_event(
                schema_id="sigma_x", kind=EventKind.PROMOTE, window_id="w_red"))
            e.remember("jjjjjj", salience=0.4)  # new cluster, lands

            snap = snapshot_from_buffer(e._buffer, strict=False)
            assert "sigma_x" in snap
            assert snap["sigma_x"].status == SchemaStatus.PROMOTED
            assert snap["sigma_x"].version == 1
            # Dedup count: 2 unique cluster reps
            assert e.status()["total_memories"] == 2
            # Cache agrees with full replay
            cache = CachedLifecycleSnapshot()
            assert cache.get(e._buffer)["sigma_x"].status == SchemaStatus.PROMOTED
        finally:
            e.close()


def test_dedup_decisions_unchanged_by_lifecycle_emissions_smoke():
    """Lifecycle events between two writes do NOT cause the second to land."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.storage.write_dedup_threshold = 0.92
        cfg.security.max_events_per_minute = 0
        e = Engram(config=cfg)
        e._embeddings = _DetEmbedder()
        e._vector = SQLiteVecStore(Path(tmp) / "vectors.db", dimension=16)
        try:
            e.remember("kkkkkk", salience=0.4)
            for kind in (EventKind.CREATE, EventKind.PROMOTE, EventKind.DEPRECATE):
                e._buffer.append(make_lifecycle_event(
                    schema_id="noise", kind=kind, window_id="w_blue"))
            e.remember("kkkkkk", salience=0.4)  # MUST be absorbed
            assert e.status()["total_memories"] == 1, (
                "lifecycle emissions perturbed dedup decision"
            )
        finally:
            e.close()
