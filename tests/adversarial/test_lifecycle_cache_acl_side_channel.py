"""Adversarial — schema-lifecycle cache ACL side-channel audit (§D-lifecycle-acl).

The shared ``RetrievalEngine._lifecycle_cache`` replays the buffer's
``CONSOLIDATION_SCHEMA_LIFECYCLE`` event stream into a
``{schema_id: SchemaState}`` snapshot, then uses it to drop DEPRECATED
SCHEMA candidates from the result set (engine.py:435–462).

This audit pins three closure invariants:

  ACL-LC-1  Alice's recall ranking over **FACT** memories is invariant to
            arbitrary lifecycle-event traffic injected on Bob-owned SCHEMA
            ids. The lifecycle filter only fires on candidates of type
            SCHEMA, so cross-agent FACT rankings cannot depend on Bob's
            schema status.

  ACL-LC-2  When a SCHEMA candidate's ``schema_id`` collides with a
            DEPRECATED entry in the snapshot, it is suppressed
            *regardless of who emitted the lifecycle event*. This is the
            INTENDED behaviour — schemas are agent_id='' (system-wide
            patterns) — but we pin it explicitly so a future "scope
            lifecycle by emitter" change has to refresh this test.

  ACL-LC-3  ``respect_schema_lifecycle=False`` is a no-op: lifecycle
            traffic from any actor cannot perturb any candidate set.

These three together close the last open NEXT-list audit item:
"Lifecycle-cache: confirm no other recall paths feed cross-agent texts
into a learned signal."
"""
from __future__ import annotations

from pathlib import Path

import pytest

from datetime import datetime, timezone

from engram import Engram
from engram.core import (
    DECAY_RATES,
    Memory,
    MemoryState,
    MemoryType,
)
from engram.core.config import Config
from engram.consolidation.lifecycle_projection import make_lifecycle_event
from engram.consolidation.schema_lifecycle import EventKind


def _make_schema_memory(*, sid: str, content: str, summary: str) -> Memory:
    now = datetime.now(timezone.utc)
    return Memory(
        id=sid,
        type=MemoryType.SCHEMA,
        state=MemoryState.ACTIVE,
        content=content,
        summary=summary,
        salience=0.8,
        confidence=1.0,
        decay_rate=DECAY_RATES.get(MemoryType.SCHEMA, 0.001),
        created_at=now,
        last_accessed=now,
        agent_id="",
    )


def _make_engram(tmp: Path, *, acl: bool, respect_lifecycle: bool = True,
                 actor: str = "alice") -> Engram:
    cfg = Config(path=str(tmp))
    cfg.retrieval.respect_schema_lifecycle = respect_lifecycle
    # Disable PRF + share_prior so we isolate the lifecycle channel.
    cfg.retrieval.query_expansion_min_dominance = None
    cfg.retrieval.reranker = None
    if acl:
        cfg.acl = {
            "enabled": True,
            "grants": {
                "alice": {"permissions": ["read", "write"], "scope": "own"},
                "bob": {"permissions": ["read", "write"], "scope": "own"},
            },
        }
    return Engram(config=cfg, actor=actor)


def _seed_alice_facts(e: Engram) -> None:
    e.remember("alpha apollo project notes", agent_id="alice")
    e.remember("beta beacon project notes", agent_id="alice")
    e.remember("gamma generic project notes", agent_id="alice")
    e.remember("delta default project notes", agent_id="alice")


def _ranks(e: Engram, query: str = "project notes", limit: int = 10):
    # Compare by content+score so we don't false-fail on per-engram random IDs.
    return [(r.memory.content, round(r.score, 6)) for r in e.recall(query, limit=limit)]


# ---------- ACL-LC-1: FACT-ranking invariance under Bob's lifecycle ----------

@pytest.mark.parametrize(
    "kinds",
    [
        # baseline (no lifecycle traffic)
        [],
        # Bob deprecates one of his own schemas
        [(EventKind.CREATE, "bob_schema_x"), (EventKind.DEPRECATE, "bob_schema_x")],
        # Bob churns lots of schema lifecycle events
        [
            (EventKind.CREATE, "bob_schema_a"),
            (EventKind.PROMOTE, "bob_schema_a"),
            (EventKind.CREATE, "bob_schema_b"),
            (EventKind.DEPRECATE, "bob_schema_b"),
            (EventKind.CREATE, "bob_schema_c"),
            (EventKind.BUMP_VERSION, "bob_schema_c"),
        ],
    ],
)
def test_acl_lc_1_fact_ranking_invariant_to_bob_lifecycle(tmp_path, kinds):
    """Alice's FACT-only ranking is invariant under Bob-emitted lifecycle events."""
    base = _make_engram(tmp_path / "base", acl=True, actor="alice")
    _seed_alice_facts(base)
    expected = _ranks(base)

    e = _make_engram(tmp_path / "with_bob_lifecycle", acl=True, actor="alice")
    _seed_alice_facts(e)

    # Inject Bob's schema lifecycle traffic into the SHARED buffer.
    window = "bob_window_1"
    for i, (kind, sid) in enumerate(kinds):
        # RECOVER needs a fresh window_id; each event gets its own to be
        # safe across kinds.
        ev = make_lifecycle_event(
            schema_id=sid, kind=kind,
            window_id=f"{window}_{i}", content="bob private pattern",
        )
        e._buffer.append(ev)

    got = _ranks(e)
    assert got == expected, (
        f"Bob's lifecycle traffic perturbed Alice's FACT ranking:\n"
        f"  baseline: {expected}\n  with bob: {got}\n"
        f"  (lifecycle filter must only apply to SCHEMA candidates)"
    )


# ---------- ACL-LC-2: SCHEMA-id collision pins intended global suppression ----------

def test_acl_lc_2_schema_deprecation_is_global(tmp_path):
    """A DEPRECATE event suppresses the matching SCHEMA candidate regardless
    of which actor emitted it. Schemas are agent_id='' (system-wide) by
    design; this test pins the intended global lifecycle scope so any
    future "scope lifecycle by emitter" change has to update this test
    rather than silently change semantics.
    """
    e = _make_engram(tmp_path, acl=True, actor="alice")
    # Insert a SCHEMA memory directly (bypass consolidation) with a fixed id.
    sid = "shared_schema_xyz"
    schema_mem = _make_schema_memory(
        sid=sid,
        content="schema: project notes pattern",
        summary="project notes schema",
    )
    e._store.upsert(schema_mem)

    # Pre-deprecation: schema candidate is reachable.
    pre = e.recall("project notes pattern schema", limit=10)
    assert any(r.memory.id == sid for r in pre), \
        "schema must be retrievable before deprecation"

    # "Bob" appends a CREATE+DEPRECATE on this schema_id to the shared buffer.
    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.CREATE, window_id="w1",
        content="schema: project notes pattern",
    ))
    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.DEPRECATE, window_id="w1",
        content="schema: project notes pattern",
    ))

    post = e.recall("project notes pattern schema", limit=10)
    assert not any(r.memory.id == sid for r in post), (
        "schema deprecation must apply globally (intended): the deprecated "
        "schema id must be suppressed from the candidate set."
    )


# ---------- ACL-LC-3: respect_schema_lifecycle=False neutralizes the channel ----------

def test_acl_lc_3_lifecycle_off_is_inert(tmp_path):
    """With respect_schema_lifecycle=False, no recall path consults the
    snapshot — confirms the cache is not silently feeding any other signal."""
    e = _make_engram(tmp_path, acl=True, respect_lifecycle=False, actor="alice")
    _seed_alice_facts(e)

    sid = "shared_schema_xyz"
    schema_mem = _make_schema_memory(
        sid=sid,
        content="schema: project notes pattern",
        summary="project notes schema",
    )
    e._store.upsert(schema_mem)

    # Lifecycle traffic that *would* deprecate the schema if the gate were on.
    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.CREATE, window_id="w1",
        content="schema: project notes pattern",
    ))
    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.DEPRECATE, window_id="w1",
        content="schema: project notes pattern",
    ))

    # Schema must STILL be reachable: gate is off.
    out = e.recall("project notes pattern schema", limit=10)
    assert any(r.memory.id == sid for r in out), (
        "with respect_schema_lifecycle=False the deprecated schema must "
        "still surface (gate off → no lifecycle effect on retrieval)."
    )


# ---------- positive control: cache is actually consulted when on ----------

def test_acl_lc_positive_control_cache_is_consulted(tmp_path):
    """Sanity: with the gate on, an explicit DEPRECATE *does* suppress.
    Pairs with ACL-LC-3: together they prove the cache is reachable when
    on and inert when off."""
    e = _make_engram(tmp_path, acl=True, respect_lifecycle=True, actor="alice")
    sid = "ctrl_schema"
    schema_mem = _make_schema_memory(
        sid=sid,
        content="schema: control pattern",
        summary="control schema",
    )
    e._store.upsert(schema_mem)

    pre = e.recall("control pattern schema", limit=10)
    assert any(r.memory.id == sid for r in pre)

    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.CREATE, window_id="w1",
        content="schema: control pattern",
    ))
    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.DEPRECATE, window_id="w1",
        content="schema: control pattern",
    ))

    post = e.recall("control pattern schema", limit=10)
    assert not any(r.memory.id == sid for r in post)
