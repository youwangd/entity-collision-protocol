"""§6.16 quorum-gate phase B — call-site wiring tests.

Phase A (commit 4ea3589) threaded `StageContext.consolidator_id` →
`make_lifecycle_event(emitter_id=...)` → wire → reducer. The default
``consolidator_id=""`` made every call-site emit emitter_id=None, so
phase A was bit-stable for legacy logs but did not actually feed the
quorum gate.

Phase B wires:

  - ``Config.consolidator_id`` (default "") →
    ``ConsolidationPipeline.run`` →
    ``StageContext.consolidator_id`` →
    every SchemaUpdate emission has ``metadata['emitter_id']`` set
    when ``Config.consolidator_id`` is non-empty.

  - ``RetrievalConfig.deprecate_quorum_k`` (default 1) →
    ``RetrievalEngine.search`` →
    ``CachedLifecycleSnapshot.get(deprecate_quorum_k=...)`` →
    ``reduce_events(deprecate_quorum_k=...)``. The cache must
    invalidate (and rebuild the snapshot under the new k) when the
    knob changes between calls.

These tests are deliberately surgical — full schema-DEPRECATE end-to-
end is exercised by ``test_schema_deprecate_quorum.py``; here we
only pin the wiring at each plumbing boundary.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

import pytest

from engram.consolidation.lifecycle_projection import (
    CachedLifecycleSnapshot,
    make_lifecycle_event,
)
from engram.consolidation.schema_lifecycle import EventKind, SchemaStatus
from engram.core.config import Config, RetrievalConfig
from engram.core.types import EventType
from engram.store.buffer import JSONLBufferStore


# ---------------------------------------------------------------------------
# B-1: Config.consolidator_id default is "" (legacy bit-stable behaviour).
# ---------------------------------------------------------------------------
def test_b1_config_consolidator_id_default():
    cfg = Config()
    assert cfg.consolidator_id == ""


# ---------------------------------------------------------------------------
# B-2: Config.consolidator_id round-trips through YAML.
# ---------------------------------------------------------------------------
def test_b2_config_consolidator_id_yaml(tmp_path):
    yml = tmp_path / "engram.yaml"
    yml.write_text("path: ~/.engram\nconsolidator_id: node-alpha-7\n")
    cfg = Config.from_yaml(str(yml))
    assert cfg.consolidator_id == "node-alpha-7"


# ---------------------------------------------------------------------------
# B-3: RetrievalConfig.deprecate_quorum_k default is 1 (legacy).
# ---------------------------------------------------------------------------
def test_b3_retrieval_quorum_k_default():
    rc = RetrievalConfig()
    assert rc.deprecate_quorum_k == 1


# ---------------------------------------------------------------------------
# B-4: ConsolidationPipeline.run threads Config.consolidator_id into
# StageContext.consolidator_id (whitebox check via the constructed ctx).
# We don't run the full pipeline; we just instantiate a pipeline and
# poke at the run() implementation by stubbing the stages list to one
# stage that captures the ctx. This pins the wiring at the boundary
# without dragging in 12 other stages.
# ---------------------------------------------------------------------------
def test_b4_pipeline_threads_consolidator_id(tmp_path):
    from unittest.mock import MagicMock

    from engram.consolidation.pipeline import ConsolidationPipeline, StageContext

    cfg = Config(path=str(tmp_path), consolidator_id="node-beta")
    buf = JSONLBufferStore(tmp_path / "buffer.jsonl")
    pipeline = ConsolidationPipeline(
        buffer=buf, store=None, audit=MagicMock(), config=cfg
    )

    captured: list[StageContext] = []

    class _Capture:
        name = "capture"

        def run(self, ctx):  # noqa: ANN001
            captured.append(ctx)
            return ctx

    pipeline.stages = [_Capture()]
    pipeline.run()

    assert captured, "stage was not invoked"
    assert captured[0].consolidator_id == "node-beta"


# ---------------------------------------------------------------------------
# B-5: empty Config.consolidator_id → StageContext.consolidator_id == ""
# (bit-stable legacy default; emitter_id key omitted from wire).
# ---------------------------------------------------------------------------
def test_b5_pipeline_empty_consolidator_id_legacy(tmp_path):
    from unittest.mock import MagicMock

    from engram.consolidation.pipeline import ConsolidationPipeline, StageContext

    cfg = Config(path=str(tmp_path))  # default consolidator_id=""
    buf = JSONLBufferStore(tmp_path / "buffer.jsonl")
    pipeline = ConsolidationPipeline(
        buffer=buf, store=None, audit=MagicMock(), config=cfg
    )

    captured: list[StageContext] = []

    class _Capture:
        name = "capture"

        def run(self, ctx):  # noqa: ANN001
            captured.append(ctx)
            return ctx

    pipeline.stages = [_Capture()]
    pipeline.run()

    assert captured[0].consolidator_id == ""


# ---------------------------------------------------------------------------
# B-6: CachedLifecycleSnapshot rebuilds when deprecate_quorum_k changes.
# Same event log: with k=1 we must see DEPRECATED; with k=2 (after a
# second get with the new k), the same single-emitter ballot must
# *not* fire — the schema stays INFERRED with one pending vote.
# ---------------------------------------------------------------------------
def _emit(buf: JSONLBufferStore, *, schema_id: str, kind: EventKind,
          window_id: str | None = None, emitter_id: str | None = None) -> None:
    ev = make_lifecycle_event(
        schema_id=schema_id,
        kind=kind,
        window_id=window_id,
        emitter_id=emitter_id,
    )
    buf.append(ev)


def test_b6_cache_invalidates_on_quorum_k_change(tmp_path):
    buf = JSONLBufferStore(tmp_path / "buffer.jsonl")
    _emit(buf, schema_id="s1", kind=EventKind.CREATE, window_id="w1")
    _emit(buf, schema_id="s1", kind=EventKind.DEPRECATE,
          window_id="w2", emitter_id="alpha")

    cache = CachedLifecycleSnapshot()

    # Under k=1: legacy single-vote semantics → DEPRECATED.
    snap_k1 = cache.get(buf, strict=False, deprecate_quorum_k=1)
    assert snap_k1["s1"].status is SchemaStatus.DEPRECATED
    misses_after_k1 = cache.stats["misses"]
    assert misses_after_k1 == 1

    # Switch to k=2: cache must rebuild and the ballot must hold pending.
    snap_k2 = cache.get(buf, strict=False, deprecate_quorum_k=2)
    assert snap_k2["s1"].status is not SchemaStatus.DEPRECATED
    assert "alpha" in snap_k2["s1"].pending_deprecate_emitters
    assert cache.stats["misses"] == misses_after_k1 + 1, (
        "k change must trigger a full rebuild"
    )

    # Stay at k=2: no work, hit-path.
    hits_before = cache.stats["hits"]
    snap_k2b = cache.get(buf, strict=False, deprecate_quorum_k=2)
    assert snap_k2b["s1"].status is not SchemaStatus.DEPRECATED
    assert cache.stats["hits"] == hits_before + 1
    assert cache.stats["misses"] == misses_after_k1 + 1


# ---------------------------------------------------------------------------
# B-7: cache rebuild on k change is symmetric — k=2 → k=1 should fire
# the previously-pending DEPRECATE.
# ---------------------------------------------------------------------------
def test_b7_cache_k_change_symmetric(tmp_path):
    buf = JSONLBufferStore(tmp_path / "buffer.jsonl")
    _emit(buf, schema_id="s1", kind=EventKind.CREATE, window_id="w1")
    _emit(buf, schema_id="s1", kind=EventKind.DEPRECATE,
          window_id="w2", emitter_id="alpha")

    cache = CachedLifecycleSnapshot()
    snap = cache.get(buf, strict=False, deprecate_quorum_k=2)
    assert snap["s1"].status is not SchemaStatus.DEPRECATED

    snap2 = cache.get(buf, strict=False, deprecate_quorum_k=1)
    assert snap2["s1"].status is SchemaStatus.DEPRECATED


# ---------------------------------------------------------------------------
# B-8: invalid k is rejected.
# ---------------------------------------------------------------------------
def test_b8_invalid_quorum_k_rejected(tmp_path):
    buf = JSONLBufferStore(tmp_path / "buffer.jsonl")
    cache = CachedLifecycleSnapshot()
    with pytest.raises(ValueError):
        cache.get(buf, deprecate_quorum_k=0)


# ---------------------------------------------------------------------------
# B-9: end-to-end retrieval — RetrievalConfig.deprecate_quorum_k=2 must
# *not* hide a SCHEMA candidate under a single-emitter DEPRECATE ballot.
# Pins the engine→cache→reducer plumbing.
# ---------------------------------------------------------------------------
def test_b9_retrieval_engine_quorum_k_blocks_single_vote(tmp_path):
    from engram.core import Memory, MemoryState, MemoryType
    from engram.retrieval.engine import RetrievalEngine
    from engram.store.memory import SQLiteMemoryStore

    buf = JSONLBufferStore(tmp_path / "buffer.jsonl")
    store = SQLiteMemoryStore(tmp_path / "memory.db")

    schema_id = "schema-quorum-test"
    now = datetime.now(timezone.utc)
    schema_mem = Memory(
        id=schema_id,
        type=MemoryType.SCHEMA,
        state=MemoryState.ACTIVE,
        content="people who like coffee also like espresso",
        summary="coffee→espresso",
        salience=0.5,
        confidence=0.9,
        decay_rate=0.01,
        created_at=now,
        last_accessed=now,
    )
    store.upsert(schema_mem)

    _emit(buf, schema_id=schema_id, kind=EventKind.CREATE, window_id="w1")
    _emit(buf, schema_id=schema_id, kind=EventKind.DEPRECATE,
          window_id="w2", emitter_id="alpha")

    # k=1 (default): SCHEMA filtered out as DEPRECATED.
    eng_k1 = RetrievalEngine(
        store=store,
        config=RetrievalConfig(deprecate_quorum_k=1),
        buffer=buf,
    )
    res_k1 = eng_k1.search("coffee espresso", limit=10)
    assert all(m.memory.id != schema_id for m in res_k1), (
        "k=1: single-emitter DEPRECATE should suppress the SCHEMA"
    )

    # k=2: ballot is one short → SCHEMA still surfaced.
    eng_k2 = RetrievalEngine(
        store=store,
        config=RetrievalConfig(deprecate_quorum_k=2),
        buffer=buf,
    )
    res_k2 = eng_k2.search("coffee espresso", limit=10)
    assert any(m.memory.id == schema_id for m in res_k2), (
        "k=2: pending ballot must NOT suppress the SCHEMA"
    )


# ---------------------------------------------------------------------------
# B-10: legacy pipeline (consolidator_id="") still emits bit-stable
# pre-quorum wire format — no `emitter_id` key in metadata.
# ---------------------------------------------------------------------------
def test_b10_legacy_pipeline_wire_format_byte_stable(tmp_path):
    from unittest.mock import MagicMock

    from engram.consolidation.pipeline import ConsolidationPipeline, StageContext

    cfg = Config(path=str(tmp_path))
    buf = JSONLBufferStore(tmp_path / "buffer.jsonl")
    pipeline = ConsolidationPipeline(
        buffer=buf, store=None, audit=MagicMock(), config=cfg
    )

    captured: list[StageContext] = []

    class _EmitOne:
        name = "emitone"

        def run(self, ctx):  # noqa: ANN001
            ev = make_lifecycle_event(
                schema_id="s1",
                kind=EventKind.CREATE,
                window_id="w1",
                emitter_id=ctx.consolidator_id or None,
            )
            ctx.buffer.append(ev)
            captured.append(ctx)
            return ctx

    pipeline.stages = [_EmitOne()]
    pipeline.run()

    # Read raw bytes back; assert no emitter_id key in metadata.
    raw = buf.path.read_text().splitlines()
    lifecycle_lines = [
        l for l in raw
        if json.loads(l).get("type") == EventType.CONSOLIDATION_SCHEMA_LIFECYCLE.value
    ]
    assert len(lifecycle_lines) == 1
    meta = json.loads(lifecycle_lines[0])["metadata"]
    assert "emitter_id" not in meta, (
        "default Config.consolidator_id='' must yield byte-stable legacy "
        "wire format (no emitter_id key)"
    )
