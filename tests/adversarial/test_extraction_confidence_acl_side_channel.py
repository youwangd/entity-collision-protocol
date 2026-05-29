"""Adversarial — Governed-Memory ``extraction_confidence`` ACL side-channel
audit (§D-extraction-confidence-acl).

The ``extraction_confidence`` (EC) field is set per-fact in
``FactExtraction.run`` (consolidation/pipeline.py:399) from the LLM's
parsed confidence on the source episode, and consumed at recall time as
a per-candidate score multiplier in ``RetrievalEngine._final_score``
(retrieval/engine.py:680–686). The Governed Memory paper
(arXiv:2603.17787) treats EC as a per-fact downweight on uncertainty.

Audit risk: would the EC channel ever consume cross-agent state, either
on the write side (extractor reading another actor's memories) or the
read side (recall scoring weighting Alice's candidates by Bob's EC
distribution)?

Reading the code, both paths look clean:
- write: extractor only reads the source episode's content; EC is the
  parsed LLM confidence and carries no cross-agent context.
- read: ``_final_score`` multiplies in only the *candidate's own* EC;
  no aggregate, no peer lookup.

This audit pins three closure invariants so any future refactor that
introduces a peer-aware EC normaliser (e.g. "calibrate EC against
corpus mean") has to refresh this test:

  EC-ACL-1  Alice's recall ranking over her own FACT memories is
            bit-identical (content, score) regardless of Bob's EC
            distribution. Whether Bob's facts are all 0.0, all 1.0,
            or a 5-quantile spread, Alice's results are unchanged.

  EC-ACL-2  EC acts as a strict per-candidate multiplier: the score of
            an Alice candidate at EC=c is exactly c × the score at
            EC=1.0 (modulo float rounding tolerance), independent of
            other Alice or Bob candidates' ECs.

  EC-ACL-3  ``use_extraction_confidence=False`` is fully inert: EC
            traffic from any actor cannot perturb any candidate.
            Confirms EC has no second consumer that bypasses the gate.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engram import Engram
from engram.core import DECAY_RATES, Memory, MemoryState, MemoryType
from engram.core.config import Config


def _make_engram(tmp: Path, *, use_ec: bool = True, actor: str = "alice") -> Engram:
    cfg = Config(path=str(tmp))
    cfg.retrieval.use_extraction_confidence = use_ec
    # Isolate the EC channel.
    cfg.retrieval.query_expansion_min_dominance = None
    cfg.retrieval.reranker = None
    cfg.acl = {
        "enabled": True,
        "grants": {
            "alice": {"permissions": ["read", "write"], "scope": "own"},
            "bob": {"permissions": ["read", "write"], "scope": "own"},
        },
    }
    return Engram(config=cfg, actor=actor)


def _put_fact(engram: Engram, *, content: str, agent_id: str, ec: float) -> str:
    """Write a FACT memory with a controlled extraction_confidence."""
    now = datetime.now(timezone.utc)
    mem_id = f"fact-{agent_id}-{abs(hash(content)) % 10**9}"
    mem = Memory(
        id=mem_id,
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content=content,
        summary=content[:80],
        salience=0.6,
        confidence=1.0,
        decay_rate=DECAY_RATES.get(MemoryType.FACT, 0.0005),
        created_at=now,
        last_accessed=now,
        agent_id=agent_id,
        extraction_confidence=ec,
    )
    engram._store.upsert(mem)
    return mem_id


def _alice_ranks(e: Engram, query: str = "project notes", limit: int = 10):
    return [(r.memory.content, round(r.score, 8)) for r in e.recall(query, limit=limit)]


# ---------- EC-ACL-1: Alice ranking invariant under Bob's EC distribution ----------

@pytest.mark.parametrize(
    "bob_ecs",
    [
        # baseline — Bob has no facts at all (control)
        [],
        # Bob's facts all maximally confident
        [1.0, 1.0, 1.0, 1.0, 1.0],
        # Bob's facts all zero confidence (would be heavily downweighted if leaked)
        [0.0, 0.0, 0.0, 0.0, 0.0],
        # Bob's facts spread across the full quantile range
        [0.05, 0.25, 0.50, 0.75, 0.95],
    ],
    ids=["no_bob", "all_max", "all_zero", "quantile_spread"],
)
def test_ec_acl_1_alice_ranking_invariant_to_bob_ec(tmp_path, bob_ecs):
    """Alice's FACT-only recall is bit-identical under any Bob EC distribution."""
    base = _make_engram(tmp_path / "base", actor="alice")
    _put_fact(base, content="alpha apollo project notes", agent_id="alice", ec=0.9)
    _put_fact(base, content="beta beacon project notes", agent_id="alice", ec=0.4)
    _put_fact(base, content="gamma generic project notes", agent_id="alice", ec=0.7)
    _put_fact(base, content="delta default project notes", agent_id="alice", ec=0.6)
    baseline = _alice_ranks(base)

    perturbed = _make_engram(tmp_path / "perturb", actor="alice")
    _put_fact(perturbed, content="alpha apollo project notes", agent_id="alice", ec=0.9)
    _put_fact(perturbed, content="beta beacon project notes", agent_id="alice", ec=0.4)
    _put_fact(perturbed, content="gamma generic project notes", agent_id="alice", ec=0.7)
    _put_fact(perturbed, content="delta default project notes", agent_id="alice", ec=0.6)
    for i, ec in enumerate(bob_ecs):
        _put_fact(perturbed, content=f"bob_fact_{i} project notes payload", agent_id="bob", ec=ec)

    assert _alice_ranks(perturbed) == baseline, (
        "Alice's FACT ranking changed under Bob's EC perturbation — leak."
    )


# ---------- EC-ACL-2: EC is a strict per-candidate multiplier ----------

def test_ec_acl_2_strict_per_candidate_multiplier(tmp_path):
    """For an Alice candidate, score(EC=c) == c × score(EC=1.0)."""
    e_full = _make_engram(tmp_path / "full", actor="alice")
    fid = _put_fact(e_full, content="zeta zenith project notes", agent_id="alice", ec=1.0)
    full_score = next(
        r.score for r in e_full.recall("zeta zenith project notes", limit=5)
        if r.memory.id == fid
    )

    for c in (0.1, 0.25, 0.5, 0.75, 0.9):
        e_c = _make_engram(tmp_path / f"c_{c}", actor="alice")
        fid_c = _put_fact(e_c, content="zeta zenith project notes", agent_id="alice", ec=c)
        sub = next(
            r.score for r in e_c.recall("zeta zenith project notes", limit=5)
            if r.memory.id == fid_c
        )
        # EC enters as final *= clamp(ec). So sub == c * full.
        assert math.isclose(sub, c * full_score, rel_tol=1e-6, abs_tol=1e-9), (
            f"EC multiplier non-linear at c={c}: sub={sub} expected≈{c*full_score}"
        )


# ---------- EC-ACL-3: use_extraction_confidence=False is fully inert ----------

def test_ec_acl_3_disabled_flag_is_inert(tmp_path):
    """With use_extraction_confidence=False, no Alice candidate's score
    depends on any EC value — its own or Bob's."""
    e_low = _make_engram(tmp_path / "low", use_ec=False, actor="alice")
    _put_fact(e_low, content="alpha apollo project notes", agent_id="alice", ec=0.01)
    _put_fact(e_low, content="beta beacon project notes", agent_id="alice", ec=0.02)
    _put_fact(e_low, content="bob_fact_x project notes payload", agent_id="bob", ec=0.99)
    low_ranks = _alice_ranks(e_low)

    e_high = _make_engram(tmp_path / "high", use_ec=False, actor="alice")
    _put_fact(e_high, content="alpha apollo project notes", agent_id="alice", ec=0.99)
    _put_fact(e_high, content="beta beacon project notes", agent_id="alice", ec=0.99)
    _put_fact(e_high, content="bob_fact_x project notes payload", agent_id="bob", ec=0.01)
    high_ranks = _alice_ranks(e_high)

    assert low_ranks == high_ranks, (
        "With EC gate off, Alice's ranking still moved — there is a "
        "second EC consumer bypassing use_extraction_confidence."
    )


# ---------- EC-ACL-4: positive control — gate ON does respond to own EC ----------

def test_ec_acl_4_positive_control_gate_on_responds_to_own_ec(tmp_path):
    """Sanity: with the gate ON, lowering Alice's own EC measurably
    drops her score. Confirms EC-ACL-3 is a real null, not a dead gate."""
    e_high = _make_engram(tmp_path / "high", use_ec=True, actor="alice")
    fid = _put_fact(e_high, content="omega oracle project notes", agent_id="alice", ec=0.95)
    high = next(r.score for r in e_high.recall("omega oracle project notes", limit=5)
                if r.memory.id == fid)

    e_low = _make_engram(tmp_path / "low", use_ec=True, actor="alice")
    fid2 = _put_fact(e_low, content="omega oracle project notes", agent_id="alice", ec=0.05)
    low = next(r.score for r in e_low.recall("omega oracle project notes", limit=5)
               if r.memory.id == fid2)

    assert low < high * 0.5, (
        f"Gate-on positive control failed: low={low} should be << high={high}"
    )
