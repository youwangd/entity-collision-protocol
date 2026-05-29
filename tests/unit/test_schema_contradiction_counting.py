"""Stage 6 contradiction-counting heuristic.

When Stage 5 (InterferenceDetection) flags ``"conflict"`` actions in
the current window, Stage 6 (SchemaUpdate) attributes those
contradictions to a newly-created schema whose supporting facts
overlap the conflicting memory's content. With enough contradictions
to clear the deprecate threshold, the schema's terminal state in the
window is DEPRECATED — closing the loop on the lifecycle DAG without
needing typed-property machinery.

Heuristic under test: word-set Jaccard ≥ 0.3 between a conflict's
``new`` memory content and a supporting fact string. One contradiction
per conflict (not per supporting-fact match), so the count is
upper-bounded by ``len(conflict_actions)``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from engram.consolidation.lifecycle_projection import snapshot_from_buffer
from engram.consolidation.pipeline import (
    SchemaUpdate,
    StageContext,
    _count_schema_contradictions,
)
from engram.consolidation.schema_lifecycle import SchemaStatus
from engram.core.types import (
    Event,
    EventType,
    Memory,
    MemoryType,
    generate_event_id,
)
from engram.providers.llm import LLMProvider
from engram.store.buffer import JSONLBufferStore
from engram.store.memory import SQLiteMemoryStore


# ---------------------------------------------------------------------
# Pure-helper unit tests (no LLM, no buffer, no Stage 6 wiring)
# ---------------------------------------------------------------------

def _mk_fact(store: SQLiteMemoryStore, text: str) -> Memory:
    ev = Event(
        id=generate_event_id(),
        ts=datetime.now(timezone.utc),
        type=EventType.EVENT_CAPTURE,
        content=text,
    )
    m = Memory.from_event(ev, memory_type=MemoryType.FACT)
    store.upsert(m)
    return m


def test_count_zero_when_no_actions(tmp_path: Path):
    store = SQLiteMemoryStore(base_path=tmp_path)
    ctx = StageContext(store=store)
    assert _count_schema_contradictions(["alice uses postgres"], ctx) == 0


def test_count_zero_when_no_supporting_facts(tmp_path: Path):
    store = SQLiteMemoryStore(base_path=tmp_path)
    m = _mk_fact(store, "alice uses postgres")
    ctx = StageContext(store=store)
    ctx.interference_actions.append({"action": "conflict", "new": m.id, "old": "x"})
    assert _count_schema_contradictions([], ctx) == 0


def test_count_ignores_non_conflict_actions(tmp_path: Path):
    store = SQLiteMemoryStore(base_path=tmp_path)
    m = _mk_fact(store, "alice uses postgres for transactions")
    ctx = StageContext(store=store)
    ctx.interference_actions.append({
        "action": "supersede", "new": m.id, "old": "x"
    })
    assert _count_schema_contradictions(
        ["alice uses postgres for transactions"], ctx
    ) == 0


def test_count_one_per_conflict_not_per_fact_match(tmp_path: Path):
    """A single conflict that overlaps multiple supporting facts should
    contribute exactly one contradiction (avoid double-counting)."""
    store = SQLiteMemoryStore(base_path=tmp_path)
    m = _mk_fact(store, "alice uses postgres for transactions")
    ctx = StageContext(store=store)
    ctx.interference_actions.append({"action": "conflict", "new": m.id, "old": "x"})
    facts = [
        "alice uses postgres for transactions",
        "alice prefers postgres for transactions",
        "alice runs postgres for transactions",
    ]
    assert _count_schema_contradictions(facts, ctx) == 1


def test_count_aggregates_across_distinct_conflicts(tmp_path: Path):
    store = SQLiteMemoryStore(base_path=tmp_path)
    a = _mk_fact(store, "alice uses postgres for transactions")
    b = _mk_fact(store, "bob uses postgres for transactions")
    ctx = StageContext(store=store)
    ctx.interference_actions.extend([
        {"action": "conflict", "new": a.id, "old": "x"},
        {"action": "conflict", "new": b.id, "old": "y"},
    ])
    facts = [
        "alice uses postgres for transactions",
        "bob uses postgres for transactions",
    ]
    assert _count_schema_contradictions(facts, ctx) == 2


def test_count_zero_when_overlap_below_jaccard_threshold(tmp_path: Path):
    store = SQLiteMemoryStore(base_path=tmp_path)
    # 'redis' shares 0 words with the supporting fact below.
    m = _mk_fact(store, "team migrated to redis cluster sharding")
    ctx = StageContext(store=store)
    ctx.interference_actions.append({"action": "conflict", "new": m.id, "old": "x"})
    facts = ["alice uses postgres for transactions"]
    assert _count_schema_contradictions(facts, ctx) == 0


def test_count_zero_when_memory_missing_from_store(tmp_path: Path):
    store = SQLiteMemoryStore(base_path=tmp_path)
    ctx = StageContext(store=store)
    ctx.interference_actions.append({
        "action": "conflict", "new": "nonexistent-id", "old": "x"
    })
    assert _count_schema_contradictions(["alice uses postgres"], ctx) == 0


# ---------------------------------------------------------------------
# Integration: contradictions ≥ deprecate threshold → DEPRECATE emitted
# ---------------------------------------------------------------------

class _StubLLM(LLMProvider):
    """Returns a single schema with 3 supporting facts.

    Same prefix structure as the existing emission tests so summary
    is stable and ≥80 chars.
    """

    _PATTERN = (
        "users prefer postgres for transactional workloads with strict "
        "ACID guarantees and pg_partman partitioning"
    )

    def __init__(self):
        self._payload = {
            "schemas": [
                {"pattern": self._PATTERN,
                 "facts": [
                     "alice uses postgres for transactions",
                     "bob uses postgres for transactions",
                     "carol uses postgres for transactions",
                 ]}
            ]
        }

    def extract_json(self, prompt: str, *, system: str = "", **_kw):
        return self._payload

    def complete(self, prompt: str, **_kw) -> str:  # pragma: no cover
        return ""


def _seed_facts(store: SQLiteMemoryStore) -> list[Memory]:
    out = []
    for text in [
        "alice uses postgres for transactions",
        "bob uses postgres for transactions",
        "carol uses postgres for transactions",
    ]:
        ev = Event(
            id=generate_event_id(),
            ts=datetime.now(timezone.utc),
            type=EventType.EVENT_CAPTURE,
            content=text,
        )
        m = Memory.from_event(ev, memory_type=MemoryType.FACT)
        store.upsert(m)
        out.append(m)
    return out


def test_schema_update_deprecates_when_contradictions_clear_threshold(tmp_path: Path):
    """Two conflict actions on facts that all overlap the schema's
    supporting-fact set should drive contradictions ≥ Thresholds.deprecate
    (default 2), causing schema_decision.decide to return DEPRECATE
    instead of PROMOTE — even though supports also clears the
    promote threshold. (Policy precedence: contradictions win on
    INFERRED — see schema_decision.decide docstring.)"""
    buf = JSONLBufferStore(base_path=tmp_path)
    store = SQLiteMemoryStore(base_path=tmp_path)
    facts = _seed_facts(store)

    ctx = StageContext(
        buffer=buf, store=store, llm=_StubLLM(),
        consolidation_id="cycle-deprecate",
    )
    # Inject two upstream conflict actions overlapping the supporting
    # facts. In production these come from Stage 5; here we synthesize
    # the same shape directly.
    ctx.interference_actions.extend([
        {"action": "conflict", "new": facts[0].id, "old": "old-a"},
        {"action": "conflict", "new": facts[1].id, "old": "old-b"},
    ])

    out = SchemaUpdate().run(ctx)
    assert out.stats["schemas_created"] == 1
    schema_mem = out.schemas_created[0]

    snap = snapshot_from_buffer(buf, strict=False)
    state = snap[schema_mem.id]
    # CREATE then DEPRECATE in the same window. Status terminates
    # at DEPRECATED; promote_count stays 0.
    assert state.status is SchemaStatus.DEPRECATED
    assert state.promote_count == 0
    assert state.last_window_id == "cycle-deprecate"


def test_schema_update_promotes_when_only_one_contradiction(tmp_path: Path):
    """One conflict < default deprecate=2 → still PROMOTE."""
    buf = JSONLBufferStore(base_path=tmp_path)
    store = SQLiteMemoryStore(base_path=tmp_path)
    facts = _seed_facts(store)

    ctx = StageContext(
        buffer=buf, store=store, llm=_StubLLM(),
        consolidation_id="cycle-promote",
    )
    ctx.interference_actions.append(
        {"action": "conflict", "new": facts[0].id, "old": "old-a"}
    )

    out = SchemaUpdate().run(ctx)
    schema_mem = out.schemas_created[0]
    snap = snapshot_from_buffer(buf, strict=False)
    state = snap[schema_mem.id]
    assert state.status is SchemaStatus.PROMOTED
    assert state.promote_count == 1
