"""Adversarial — C×F composition: mech-merge × extraction_confidence (EC).

Cross-channel coupling audit catalogues this pair as **C+gap**: the
question is whether a peer's EC distribution can influence Alice's
merge survivor or its surviving EC. Reading
``MechanicalMerge.run`` (consolidation/pipeline.py:1366), the survivor
is selected purely by ``salience`` — EC is never consulted. So the pair
is *structurally* independent, but pinning the closure invariant here
makes any future "EC-aware survivor selection" refactor trip a test
rather than silently introduce a peer-state read on the write side.

Threat model
------------
A future refactor adds "tie-break by EC" or "weighted salience × EC"
to the merge survivor rule. If EC normalisation goes through a
*global* statistic (corpus mean, peer-agent percentile), Bob's EC
distribution starts to influence which of Alice's near-duplicates wins
the merge. That's a write-side cross-tenant leak: Bob can DoS Alice's
canonical FACT by skewing his own EC distribution.

Invariants
----------
  CXF-1  Alice's merge-survivor identity is invariant under any Bob
         EC distribution. Whether Bob's facts are all-zero, all-one,
         or quantile-spread, the same Alice row wins.

  CXF-2  The surviving Alice row's EC is bit-identical (no peer-aware
         EC normalisation has been silently introduced).

  CXF-3  Inversely-ordered salience: when two Alice near-dups have
         (sal=0.9, ec=0.1) and (sal=0.1, ec=0.9), the high-salience /
         low-EC one wins regardless of Bob's EC. This is the
         "salience-only survivor rule" positive control.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from engram.consolidation.pipeline import MechanicalMerge, StageContext
from engram.core import DECAY_RATES, Memory, MemoryState, MemoryType
from engram.store.memory import SQLiteMemoryStore
from engram.store.vector import SQLiteVecStore


class _FakeEmbeddings:
    dimension = 8

    def __init__(self, table):
        self._table = table

    def embed(self, content):
        return self._table.get(content, [0.0] * self.dimension)

    def embed_batch(self, contents):  # pragma: no cover
        return [self.embed(c) for c in contents]


def _mem(*, mid, content, salience, agent_id, ec=1.0):
    now = datetime.now(timezone.utc)
    return Memory(
        id=mid,
        type=MemoryType.FACT,
        state=MemoryState.ACTIVE,
        content=content,
        summary=content[:80],
        salience=salience,
        confidence=1.0,
        decay_rate=DECAY_RATES.get(MemoryType.FACT, 0.001),
        created_at=now,
        last_accessed=now,
        agent_id=agent_id,
        extraction_confidence=ec,
    )


def _setup(tmp_path: Path, memories):
    store = SQLiteMemoryStore(tmp_path / "mem.sqlite")
    vec = SQLiteVecStore(tmp_path / "vec.sqlite", dimension=_FakeEmbeddings.dimension)
    near = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    table = {}
    for m in memories:
        store.upsert(m)
        vec.upsert(m.id, near)
        table[m.content] = near
    stage = MechanicalMerge(vector_store=vec, embedding_provider=_FakeEmbeddings(table), threshold=0.90)
    return StageContext(store=store), stage


@pytest.mark.parametrize(
    "bob_ecs,label",
    [
        ([], "no_bob"),
        ([1.0, 1.0], "all_max"),
        ([0.0, 0.0], "all_zero"),
        ([0.05, 0.95], "extreme_spread"),
    ],
)
def test_cxf_1_alice_survivor_invariant_to_bob_ec(tmp_path: Path, bob_ecs, label):
    """Alice's merge survivor is invariant under any Bob EC distribution."""
    a1 = _mem(mid="alice-1", content="payload P v1", salience=0.9, agent_id="alice", ec=0.7)
    a2 = _mem(mid="alice-2", content="payload P v2", salience=0.3, agent_id="alice", ec=0.2)
    bobs = [
        _mem(mid=f"bob-{i}", content=f"bob payload {i}", salience=0.5,
             agent_id="bob", ec=ec)
        for i, ec in enumerate(bob_ecs)
    ]
    ctx, stage = _setup(tmp_path / label, [a1, a2, *bobs])
    stage.run(ctx)

    surv = ctx.store.get("alice-1")
    loser = ctx.store.get("alice-2")
    assert surv.state == MemoryState.ACTIVE, (
        f"CXF-1 [{label}]: high-salience Alice row was suppressed; "
        f"surv.state={surv.state}, bob_ecs={bob_ecs}"
    )
    assert loser.state == MemoryState.SUPPRESSED, (
        f"CXF-1 [{label}]: low-salience Alice row not suppressed under "
        f"composition; bob_ecs={bob_ecs}"
    )


def test_cxf_2_alice_surviving_ec_bit_identical(tmp_path: Path):
    """Surviving Alice EC is bit-identical regardless of Bob's EC."""
    a1 = _mem(mid="alice-1", content="payload P v1", salience=0.9, agent_id="alice", ec=0.7)
    a2 = _mem(mid="alice-2", content="payload P v2", salience=0.3, agent_id="alice", ec=0.2)

    # Arm: Alice alone.
    ctx_alone, st_alone = _setup(tmp_path / "alone", [a1, a2])
    st_alone.run(ctx_alone)
    ec_alone = ctx_alone.store.get("alice-1").extraction_confidence

    # Arm: Alice + Bob with adversarial EC distribution.
    a1b = _mem(mid="alice-1", content="payload P v1", salience=0.9, agent_id="alice", ec=0.7)
    a2b = _mem(mid="alice-2", content="payload P v2", salience=0.3, agent_id="alice", ec=0.2)
    b1 = _mem(mid="bob-1", content="bob payload x", salience=0.99, agent_id="bob", ec=0.0)
    b2 = _mem(mid="bob-2", content="bob payload y", salience=0.01, agent_id="bob", ec=1.0)
    ctx_with, st_with = _setup(tmp_path / "with", [a1b, b1, a2b, b2])
    st_with.run(ctx_with)
    ec_with = ctx_with.store.get("alice-1").extraction_confidence

    assert ec_alone == ec_with == 0.7, (
        f"CXF-2 LEAK: Alice's surviving EC drifted under Bob perturbation. "
        f"alone={ec_alone} with_bob={ec_with} (expected 0.7)"
    )


def test_cxf_3_salience_only_survivor_rule(tmp_path: Path):
    """High-salience / low-EC Alice row wins over low-salience / high-EC.

    Positive control that the merge survivor rule reads salience only and
    not EC. If a future refactor introduces EC-weighted salience, this
    test fails.
    """
    a_hi_sal = _mem(mid="alice-hi", content="payload Q v1", salience=0.9,
                    agent_id="alice", ec=0.1)
    a_hi_ec = _mem(mid="alice-ec", content="payload Q v2", salience=0.1,
                   agent_id="alice", ec=0.9)
    ctx, stage = _setup(tmp_path, [a_hi_sal, a_hi_ec])
    stage.run(ctx)
    assert ctx.store.get("alice-hi").state == MemoryState.ACTIVE
    assert ctx.store.get("alice-ec").state == MemoryState.SUPPRESSED, (
        "CXF-3: survivor rule consulted EC; salience-only invariant broken"
    )
