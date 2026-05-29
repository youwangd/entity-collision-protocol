"""Adversarial — FactExtraction agent_id ACL inheritance audit
(§D-fact-extraction-acl).

The Stage 4c ``FactExtraction`` consolidator distils EPISODE memories
into FACT rows via an LLM call. The synthesized
``CONSOLIDATION_EXTRACT`` event carries no actor context, so the
default ``Memory.agent_id`` for the resulting fact is ``''`` (empty —
"system memory" in the v0.2 ACL model).

Pre-fix, this meant every extracted fact was globally readable:

  - Alice writes "I am meeting Mallory at 3pm at the Whitebridge."
  - Consolidation extracts the fact "Alice meets Mallory at the Whitebridge".
  - The fact is stored with agent_id='' → Bob's ``recall("Mallory")``
    surfaces it under his own ACL grant (which permits agent_id='' as
    a system-shared scope, see ``security/acl.py:Grant.can_access``).

This is strictly worse than the §6.11 / §6.12 channels: those leak a
*signal* about Alice's content; this one leaks the **distilled
content itself**.

The fix in ``FactExtraction.run`` is one line:
``fact_memory.agent_id = memory.agent_id`` (inherit from the source
episode). Schemas keep ``agent_id=''`` intentionally — they are
system-shared patterns, matching the §6.9 lifecycle-cache audit's
treatment of schemas as agent-less.

Invariants pinned:

  FE-ACL-1  Facts extracted from an Alice-owned episode inherit
            ``agent_id='alice'``.
  FE-ACL-2  Facts extracted from a Bob-owned episode inherit
            ``agent_id='bob'``.
  FE-ACL-3  Facts extracted from a system episode (``agent_id=''``)
            stay ``agent_id=''``.
  FE-ACL-4  Mixed batch: per-fact attribution is correct (no
            cross-contamination across episodes processed in the
            same stage invocation).
"""
from __future__ import annotations

from datetime import datetime, timezone


from engram.consolidation.pipeline import FactExtraction, StageContext
from engram.core import Memory, MemoryType
from engram.core.types import Event, EventType, generate_event_id


class _FixedLLM:
    """LLM that always returns the same fact payload."""

    def __init__(self, payload):
        self._payload = payload

    def complete(self, prompt: str, system: str = "", max_tokens: int = 0) -> str:
        return ""

    def extract_json(self, prompt: str, system: str = "") -> dict:
        return self._payload


def _episode(content: str, agent_id: str) -> Memory:
    ev = Event(
        id=generate_event_id(),
        ts=datetime.now(timezone.utc),
        type=EventType.EXPLICIT_REMEMBER,
        content=content,
    )
    mem = Memory.from_event(ev, memory_type=MemoryType.EPISODE)
    mem.agent_id = agent_id
    return mem


def _run(episodes, payload):
    ctx = StageContext(memories_created=list(episodes), llm=_FixedLLM(payload))
    FactExtraction().run(ctx)
    return [m for m in ctx.memories_created if m.type == MemoryType.FACT]


def test_fe_acl_1_alice_episode_yields_alice_fact():
    ep = _episode("Alice meeting at Whitebridge", "alice")
    payload = {"facts": [{"text": "Alice meets Mallory", "confidence": 0.9}]}
    facts = _run([ep], payload)
    assert len(facts) == 1
    assert facts[0].agent_id == "alice", (
        f"extracted fact lost ACL ownership; agent_id={facts[0].agent_id!r}"
    )


def test_fe_acl_2_bob_episode_yields_bob_fact():
    ep = _episode("Bob's calendar entry", "bob")
    payload = {"facts": [{"text": "Bob has a 3pm meeting", "confidence": 0.85}]}
    facts = _run([ep], payload)
    assert len(facts) == 1
    assert facts[0].agent_id == "bob"


def test_fe_acl_3_system_episode_stays_system():
    """agent_id='' is the legitimate system-owned case (e.g. seeded data).

    The fact should also stay agent_id='' — we are only forbidding
    *silent promotion* of agent-owned content into the system pool,
    not blocking system→system extraction.
    """
    ep = _episode("System policy: meetings are 30 min by default.", "")
    payload = {"facts": [{"text": "Default meeting length is 30 min", "confidence": 1.0}]}
    facts = _run([ep], payload)
    assert len(facts) == 1
    assert facts[0].agent_id == ""


def test_fe_acl_4_mixed_batch_no_cross_contamination():
    """Two episodes from different agents in one stage run → per-fact attribution."""
    alice_ep = _episode("Alice meeting at Whitebridge", "alice")
    bob_ep = _episode("Bob calendar entry", "bob")
    payload = {"facts": [{"text": "extracted statement", "confidence": 0.9}]}
    facts = _run([alice_ep, bob_ep], payload)
    # One fact per episode, since extract_json returns the same payload
    # for each call.
    assert len(facts) == 2
    owners = sorted(f.agent_id for f in facts)
    assert owners == ["alice", "bob"], (
        f"per-fact attribution wrong; owners={owners}"
    )
    # And source_events must point to the correct episode in each case.
    by_owner = {f.agent_id: f for f in facts}
    assert by_owner["alice"].source_events == [alice_ep.id]
    assert by_owner["bob"].source_events == [bob_ep.id]
