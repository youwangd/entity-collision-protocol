"""Integration test: SchemaUpdate (Stage 6) → buffer → snapshot reducer.

Verifies the wiring change in consolidation/pipeline.py: when
SchemaUpdate creates a new schema memory, it must also append a
CONSOLIDATION_SCHEMA_LIFECYCLE CREATE event so the projection sees the
schema as INFERRED.

We use a stub LLM provider that returns a fixed schemas payload, then
read the buffer back through `snapshot_from_buffer` and check status.
"""
from __future__ import annotations

from pathlib import Path

from engram.consolidation.lifecycle_projection import snapshot_from_buffer
from engram.consolidation.pipeline import SchemaUpdate, StageContext
from engram.consolidation.schema_lifecycle import SchemaStatus
from engram.core.types import Event, EventType, Memory, MemoryType, generate_event_id
from engram.providers.llm import LLMProvider
from engram.store.buffer import JSONLBufferStore
from engram.store.memory import SQLiteMemoryStore


class _StubLLM(LLMProvider):
    """Returns a single schema with 3+ supporting facts on extract_json.

    Uses an 80+ char pattern so summary (pattern[:80]) is a stable
    identity key independent of trailing refinements (used in
    BUMP_VERSION tests below).
    """

    _PATTERN = (
        "users prefer postgres for transactional workloads with strict "
        "ACID guarantees and pg_partman partitioning"
    )

    def __init__(self):
        assert len(self._PATTERN) >= 80
        self._payload = {
            "schemas": [
                {"pattern": self._PATTERN,
                 "facts": ["alice uses postgres", "bob uses postgres",
                           "carol uses postgres"]}
            ]
        }

    def extract_json(self, prompt: str, *, system: str = "", **_kw):
        return self._payload

    # Methods the abstract base requires; not used in this test.
    def complete(self, prompt: str, **_kw) -> str:  # pragma: no cover
        return ""


def _seed_facts(store: SQLiteMemoryStore) -> None:
    from datetime import datetime, timezone
    for i in range(3):
        ev = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.EVENT_CAPTURE,
            content=f"user-{i} uses postgres",
        )
        m = Memory.from_event(ev, memory_type=MemoryType.FACT)
        store.upsert(m)


class _SubThresholdLLM(_StubLLM):
    """Returns a single schema with only 2 supporting facts (sub-promote)."""

    def __init__(self):
        self._payload = {
            "schemas": [
                {"pattern": "users prefer postgres for transactional workloads",
                 "facts": ["alice uses postgres", "bob uses postgres"]}
            ]
        }


def test_schema_update_emits_lifecycle_create_and_promote(tmp_path: Path):
    """3 supporting facts hit the default promote threshold → expect
    a CREATE followed by a PROMOTE for the same schema in the same
    consolidation window."""
    buf = JSONLBufferStore(base_path=tmp_path)
    store = SQLiteMemoryStore(base_path=tmp_path)
    _seed_facts(store)

    ctx = StageContext(
        buffer=buf, store=store, llm=_StubLLM(),
        consolidation_id="cycle-42",
    )
    out = SchemaUpdate().run(ctx)

    assert out.stats["schemas_created"] == 1
    schema_mem = out.schemas_created[0]

    snap = snapshot_from_buffer(buf, strict=False)
    assert schema_mem.id in snap, "lifecycle CREATE event was not emitted"
    state = snap[schema_mem.id]
    # 3 facts >= default Thresholds.promote=3 → schema_decision.decide
    # emits PROMOTE in the same window. CREATE+PROMOTE collapse to
    # status=PROMOTED, version=1, promote_count=1.
    assert state.status is SchemaStatus.PROMOTED
    assert state.version == 1
    assert state.promote_count == 1
    assert state.last_window_id == "cycle-42"


def test_schema_update_inferred_when_below_promote_threshold(tmp_path: Path):
    """2 supporting facts < default promote=3 → CREATE only, status
    stays INFERRED."""
    buf = JSONLBufferStore(base_path=tmp_path)
    store = SQLiteMemoryStore(base_path=tmp_path)
    _seed_facts(store)

    ctx = StageContext(
        buffer=buf, store=store, llm=_SubThresholdLLM(),
        consolidation_id="cycle-43",
    )
    out = SchemaUpdate().run(ctx)
    schema_mem = out.schemas_created[0]

    snap = snapshot_from_buffer(buf, strict=False)
    state = snap[schema_mem.id]
    assert state.status is SchemaStatus.INFERRED
    assert state.promote_count == 0
    assert state.last_window_id == "cycle-43"


def test_schema_update_no_emit_without_buffer(tmp_path: Path):
    """If ctx.buffer is None, the stage must still create the schema
    memory and not crash trying to emit a lifecycle event."""
    store = SQLiteMemoryStore(base_path=tmp_path)
    _seed_facts(store)
    ctx = StageContext(buffer=None, store=store, llm=_StubLLM())
    out = SchemaUpdate().run(ctx)
    assert out.stats["schemas_created"] == 1


class _RefinedPatternLLM(_StubLLM):
    """Same summary (pattern[:80]) as _StubLLM but a refined longer
    body — drives BUMP_VERSION on the second pass."""

    def __init__(self):
        # Identical first 80 chars (the summary), but different
        # content past that prefix. Refined trailing clause.
        prefix = _StubLLM._PATTERN[:80]
        assert len(prefix) == 80
        self._payload = {
            "schemas": [
                {"pattern": prefix + " (refined: also analytics)",
                 "facts": ["alice", "bob"]}
            ]
        }


def test_schema_update_emits_bump_version_on_refined_pattern(tmp_path: Path):
    """Two consolidation cycles: the first creates+promotes the
    schema, the second sees the SAME summary with refined content
    and must emit BUMP_VERSION (not a duplicate CREATE)."""
    buf = JSONLBufferStore(base_path=tmp_path)
    store = SQLiteMemoryStore(base_path=tmp_path)
    _seed_facts(store)

    # Cycle 1: original pattern.
    ctx1 = StageContext(
        buffer=buf, store=store, llm=_StubLLM(),
        consolidation_id="cycle-1",
    )
    out1 = SchemaUpdate().run(ctx1)
    assert out1.stats["schemas_created"] == 1
    assert out1.stats["schemas_bumped"] == 0
    schema_mem = out1.schemas_created[0]
    # Persist the schema so cycle 2 can find it.
    store.upsert(schema_mem)

    # Cycle 2: refined pattern, same summary.
    ctx2 = StageContext(
        buffer=buf, store=store, llm=_RefinedPatternLLM(),
        consolidation_id="cycle-2",
    )
    out2 = SchemaUpdate().run(ctx2)
    assert out2.stats["schemas_created"] == 0, "should reuse, not re-create"
    assert out2.stats["schemas_bumped"] == 1

    snap = snapshot_from_buffer(buf, strict=False)
    state = snap[schema_mem.id]
    # version=1 from CREATE, +1 from BUMP_VERSION
    assert state.version == 2
    # status is preserved across BUMP_VERSION (invariant #3)
    assert state.status is SchemaStatus.PROMOTED


def test_schema_update_idempotent_on_same_pattern(tmp_path: Path):
    """Re-running SchemaUpdate with the exact same pattern is a no-op:
    no CREATE, no BUMP_VERSION."""
    buf = JSONLBufferStore(base_path=tmp_path)
    store = SQLiteMemoryStore(base_path=tmp_path)
    _seed_facts(store)

    ctx1 = StageContext(buffer=buf, store=store, llm=_StubLLM(),
                       consolidation_id="cycle-1")
    out1 = SchemaUpdate().run(ctx1)
    schema_mem = out1.schemas_created[0]
    store.upsert(schema_mem)

    ctx2 = StageContext(buffer=buf, store=store, llm=_StubLLM(),
                       consolidation_id="cycle-2")
    out2 = SchemaUpdate().run(ctx2)
    assert out2.stats["schemas_created"] == 0
    assert out2.stats["schemas_bumped"] == 0
