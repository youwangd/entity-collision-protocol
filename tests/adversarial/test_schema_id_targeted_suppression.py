"""Adversarial — cross-actor SCHEMA-id targeted suppression (§D-schema-id-suppress).

Follow-up audit named in NEXT.md after the §6.9 lifecycle-cache audit
flagged it as a quiet implementation detail. We pin it here as an
explicit, named threat-model statement instead.

Background
----------
``RetrievalEngine`` filters DEPRECATED schemas from a recall using a
global ``{schema_id: SchemaState}`` snapshot replayed from the
``CONSOLIDATION_SCHEMA_LIFECYCLE`` event stream (see paper §6.9). The
lifecycle DAG is **intentionally global** — schemas are
``agent_id=''`` (system-wide patterns), so a single canonical
deprecation must apply to every actor's recall.

That global semantics has a consequence:

    *If* a malicious actor learns a victim schema's ``schema_id``,
    they can append one DEPRECATE event to the buffer and silently
    suppress that schema from every other actor's recalls.

This test pins three things:

  SC-ID-1  **Targeted suppression works.** Given a known schema_id,
           a single DEPRECATE event from any actor suppresses the
           schema from a third actor's recall. (Worst case — pinned
           so we don't lose track of the attack surface.)

  SC-ID-2  **Random-id guessing is infeasible.** A DEPRECATE event
           whose schema_id does NOT match any extant schema is a
           no-op. Schema ids are ``mem-sc-<uuid4 hex[:12]>`` (48 bits
           of entropy), so a blind-guess attack needs ~2^47 writes on
           average to suppress a single victim schema.

  SC-ID-3  **The id is observable through normal recall.** Any actor
           that recalls a schema sees ``result.memory.id`` — meaning
           an observed schema is a suppressible schema. This is the
           realistic attack path; documenting it here so future
           "redact ids by actor" work has a starting hook.

Mitigation surface (deferred — not implemented):
- Scope lifecycle events by emitter (per-agent DAGs); breaks the
  intended global-schema semantics, so it would need a parallel
  "shared-pattern" type.
- Require an audit-trail signature on DEPRECATE events.
- Quarantine externally-emitted DEPRECATEs behind a quorum gate.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

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


def _engram(tmp: Path, *, actor: str) -> Engram:
    cfg = Config(path=str(tmp))
    cfg.retrieval.respect_schema_lifecycle = True
    cfg.retrieval.query_expansion_min_dominance = None
    cfg.retrieval.reranker = None
    cfg.acl = {
        "enabled": True,
        "grants": {
            "alice": {"permissions": ["read", "write"], "scope": "own"},
            "bob": {"permissions": ["read", "write"], "scope": "own"},
            "carol": {"permissions": ["read", "write"], "scope": "own"},
        },
    }
    return Engram(config=cfg, actor=actor)


def _seed_schema(e: Engram, *, sid: str, content: str, summary: str) -> Memory:
    now = datetime.now(timezone.utc)
    mem = Memory(
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
    e._store.upsert(mem)
    return mem


# ---------- SC-ID-1: targeted suppression by a known id ----------

def test_sc_id_1_targeted_suppression_known_id(tmp_path):
    """A single DEPRECATE on a known schema_id suppresses the schema
    globally. Pinned worst-case attack surface."""
    e = _engram(tmp_path, actor="alice")
    sid = "mem-sc-victimschema01"
    _seed_schema(
        e, sid=sid,
        content="schema: rare apricot meridian protocol",
        summary="apricot meridian schema",
    )

    pre = e.recall("apricot meridian protocol", limit=10)
    assert any(r.memory.id == sid for r in pre), "schema must be reachable pre-attack"

    # "Bob" — having somehow learned `sid` — appends CREATE+DEPRECATE.
    # (Reducer in lenient mode drops DEPRECATE on unknown schemas; an
    # attacker would emit CREATE first, exactly as the system does.)
    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.CREATE, window_id="bob_attack",
        content="(attacker create)",
    ))
    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.DEPRECATE, window_id="bob_attack",
        content="(attacker deprecate)",
    ))

    post = e.recall("apricot meridian protocol", limit=10)
    assert not any(r.memory.id == sid for r in post), (
        "ATTACK SURFACE: a single DEPRECATE on a known schema_id "
        "globally suppresses the schema (intended consequence of "
        "global lifecycle DAG; documented threat-model item)."
    )


# ---------- SC-ID-2: blind-guess DEPRECATE on a non-existent id is inert ----------

@pytest.mark.parametrize(
    "guess",
    [
        "mem-sc-aaaaaaaaaaaa",
        "mem-sc-deadbeefcafe",
        "mem-sc-000000000000",
        "mem-sc-ffffffffffff",
        "totally-fabricated-id",
    ],
)
def test_sc_id_2_random_guess_is_inert(tmp_path, guess):
    """A DEPRECATE on an id that doesn't match any extant schema
    is a no-op. Schema ids are mem-sc-<uuid4().hex[:12]> = 48 bits of
    entropy, so a blind-guess attack needs ~2^47 writes on average to
    land a single suppression."""
    e = _engram(tmp_path, actor="alice")
    sid = "mem-sc-realschema123"
    _seed_schema(
        e, sid=sid,
        content="schema: rare cobalt periwinkle directive",
        summary="cobalt directive schema",
    )

    expected = {r.memory.id for r in e.recall("cobalt periwinkle directive", limit=10)}
    assert sid in expected, "victim schema must be in baseline result set"

    e._buffer.append(make_lifecycle_event(
        schema_id=guess, kind=EventKind.DEPRECATE, window_id="blind_guess",
        content="(blind-guess attack)",
    ))

    after = {r.memory.id for r in e.recall("cobalt periwinkle directive", limit=10)}
    assert after == expected, (
        f"blind-guess DEPRECATE on {guess!r} perturbed the result set; "
        "non-matching schema_ids must be inert (no collision side-effects)."
    )


# ---------- SC-ID-3: schema_id is observable through normal recall ----------

def test_sc_id_3_observed_then_suppressed_attack_chain(tmp_path):
    """End-to-end: an actor recalls a system-wide schema (learns its id),
    then writes a DEPRECATE on that id; subsequent recalls miss it.
    This is the realistic attack path the threat-model item names."""
    e = _engram(tmp_path, actor="alice")
    sid = "mem-sc-globalsharedX"
    _seed_schema(
        e, sid=sid,
        content="schema: shared lattice propagation rule pattern",
        summary="lattice propagation",
    )

    # Step 1 — observability: a recall that surfaces the schema also
    # exposes its id (this is why "guess the id" isn't required).
    pre = e.recall("lattice propagation rule pattern schema", limit=10)
    observed_ids = [r.memory.id for r in pre if r.memory.type == MemoryType.SCHEMA]
    assert sid in observed_ids, (
        "schemas are agent_id='' (system-wide), so any actor's recall "
        "must expose the schema id — this is the observability that "
        "makes targeted suppression realistic."
    )

    # Step 2 — attacker writes CREATE+DEPRECATE on the observed id.
    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.CREATE, window_id="bob_attack",
        content="(attacker create)",
    ))
    e._buffer.append(make_lifecycle_event(
        schema_id=sid, kind=EventKind.DEPRECATE, window_id="bob_attack",
        content="(attacker deprecate)",
    ))

    # Step 3 — schema is now suppressed from every actor's recall.
    post = e.recall("lattice propagation rule pattern schema", limit=10)
    assert not any(r.memory.id == sid for r in post), (
        "ATTACK CHAIN: id observed via normal recall → suppressed with "
        "one DEPRECATE event. Pin to keep threat-model item explicit."
    )
