"""§D3-collateral-(b) — entity-aware interference detector unit tests."""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

from engram.consolidation.pipeline import InterferenceDetection
from engram.core.types import Memory, MemoryType, MemoryState, generate_memory_id


def _mk(content: str, *, t: datetime) -> Memory:
    return Memory(
        id=generate_memory_id(MemoryType.FACT),
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content=content,
        summary=content[:32],
        salience=0.5,
        confidence=1.0,
        decay_rate=0.005,
        created_at=t,
    )


def test_content_tokens_strips_stop_words_and_short():
    toks = InterferenceDetection._content_tokens(
        "User user_001 now prefers using dark mode for daily debugging work.")
    # Boilerplate should be gone; entity tokens kept.
    assert toks == {"dark", "debugging", "mode", "user001"}


def test_entity_aware_blocks_cross_slot_template_overlap():
    """Two memories from different slots that share template boilerplate
    but disagree on the entity (user) should NOT be classified as supersede
    when entity_aware=True."""
    t0 = datetime.now(timezone.utc)
    a = _mk("User user_001 now prefers using dark mode for daily debugging work.",
            t=t0)
    b = _mk("User user_002 now prefers using tmux for daily debugging work.",
            t=t0 + timedelta(seconds=1))
    stage = InterferenceDetection()
    # Permissive: classic Jaccard catches it.
    assert stage._detect_interference(b, a, llm=None) == "supersede"
    # Strict (τ=0.7): entity-token overlap below threshold, blocked.
    out = stage._detect_interference(b, a, llm=None,
                                      entity_aware=True, entity_min=0.7)
    assert out == ""


def test_entity_aware_preserves_within_slot_supersede():
    """Two memories about the same entity (high entity-token overlap)
    differing only in the value field should still fire under
    entity_aware=True at τ=0.5.

    Tokens A: {host001, runs, version, nginx, 142}
    Tokens B: {host001, runs, version, nginx, 201}
    Entity-Jaccard = 4/6 ≈ 0.67 ≥ 0.5 → supersede preserved.
    """
    t0 = datetime.now(timezone.utc)
    a = _mk("Server host_001 runs version 1.4.2 of nginx daily.", t=t0)
    b = _mk("Server host_001 runs version 2.0.1 of nginx daily.",
            t=t0 + timedelta(seconds=1))
    stage = InterferenceDetection()
    out = stage._detect_interference(b, a, llm=None,
                                      entity_aware=True, entity_min=0.5)
    assert out == "supersede"


def test_entity_aware_threshold_zero_is_passthrough():
    """At entity_min=0.0 the gate should be a no-op (any non-empty union passes)."""
    t0 = datetime.now(timezone.utc)
    a = _mk("Server host-001 runs version 1.4.2 of nginx.", t=t0)
    b = _mk("Server host-002 runs version 2.0.1 of nginx.", t=t0 + timedelta(seconds=1))
    stage = InterferenceDetection()
    classic = stage._detect_interference(b, a, llm=None)
    permissive = stage._detect_interference(b, a, llm=None,
                                             entity_aware=True, entity_min=0.0)
    assert classic == permissive
