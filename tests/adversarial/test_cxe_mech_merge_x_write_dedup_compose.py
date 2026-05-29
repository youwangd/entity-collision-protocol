"""Adversarial — C×E composition: mech-merge × write-dedup interleave.

Cross-channel coupling audit (research_notes/cross_channel_coupling_audit.md)
flagged this pair as **C+gap**: both stages are write-time write-decision
channels, and a Bob-write → Alice-write → mech-merge interleave is the
canonical composition path. Each channel has a per-channel ACL audit
already (`test_write_dedup_acl_side_channel.py`,
`test_mechanical_merge_acl_side_channel.py`); this file pins their
*composition* — that the two fixes do not silently re-introduce a leak
when chained.

Threat model
------------
Suppose Alice and Bob both want to write the same payload P. Under the
write-dedup ACL fix, Bob's write lands in Bob's slice and Alice's later
write lands in Alice's slice (both rows present). Then mech-merge runs.
Under the mech-merge ACL fix, the cross-agent pair must NOT be
suppressed. But two composition-only failure modes are conceivable:

  CXE-1 **Candidate-pool leak.** Mech-merge's vector neighbourhood query
        returns Bob's row when scanning Alice's row. If the candidate
        pool size shrinks because of the cross-agent reject (i.e.
        same-agent neighbours get dropped), Alice's *intra-actor*
        merges could become non-deterministic relative to Bob's write
        history.

  CXE-2 **Sequencing leak.** If write-dedup state (the audit event log)
        flowing into mech-merge changes the iteration order of
        ``store.all_active()``, Bob's writes can perturb the order in
        which Alice's facts are merged, and which side wins on
        salience-tied pairs.

Invariants
----------
  CXE-1  Alice's same-agent merge outcome is BIT-IDENTICAL whether or
         not Bob has written cross-agent near-duplicates beforehand.
         (Compares the (id → state) map after merge.)

  CXE-2  Bob's row remains ACTIVE through the merge (positive control
         that the per-channel ACL fix is still effective under
         composition).

  CXE-3  Alice's intra-actor near-duplicates ARE still merged (positive
         control: composition does not disable the stage).
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

from engram.consolidation.pipeline import MechanicalMerge, StageContext
from engram.core import DECAY_RATES, Memory, MemoryState, MemoryType
from engram.store.memory import SQLiteMemoryStore
from engram.store.vector import SQLiteVecStore


class _FakeEmbeddings:
    dimension = 8

    def __init__(self, table: dict[str, list[float]]):
        self._table = table

    def embed(self, content: str) -> list[float]:
        return self._table.get(content, [0.0] * self.dimension)

    def embed_batch(self, contents):  # pragma: no cover
        return [self.embed(c) for c in contents]


def _mem(*, mid: str, content: str, salience: float, agent_id: str) -> Memory:
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
    )


def _setup(tmp_path: Path, memories: list[Memory]) -> tuple[StageContext, MechanicalMerge]:
    store = SQLiteMemoryStore(tmp_path / "mem.sqlite")
    vec = SQLiteVecStore(tmp_path / "vec.sqlite", dimension=_FakeEmbeddings.dimension)
    near_dup = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    table: dict[str, list[float]] = {}
    for m in memories:
        store.upsert(m)
        vec.upsert(m.id, near_dup)
        table[m.content] = near_dup
    emb = _FakeEmbeddings(table)
    stage = MechanicalMerge(vector_store=vec, embedding_provider=emb, threshold=0.90)
    ctx = StageContext(store=store)
    return ctx, stage


def _state_map(ctx: StageContext, ids: list[str]) -> dict[str, MemoryState]:
    return {i: ctx.store.get(i).state for i in ids}


def test_cxe_1_alice_merge_invariant_to_bob_writes(tmp_path: Path):
    """Alice's intra-actor merge outcome is identical with/without Bob's writes."""
    # Arm 1: Alice alone — two near-duplicates, salience-ordered.
    a1 = _mem(mid="alice-1", content="payload P v1", salience=0.9, agent_id="alice")
    a2 = _mem(mid="alice-2", content="payload P v2", salience=0.3, agent_id="alice")
    ctx_alone, stage_alone = _setup(tmp_path / "alone", [a1, a2])
    stage_alone.run(ctx_alone)
    alone = _state_map(ctx_alone, ["alice-1", "alice-2"])

    # Arm 2: Alice + Bob's cross-agent near-duplicates interleaved.
    a1b = _mem(mid="alice-1", content="payload P v1", salience=0.9, agent_id="alice")
    a2b = _mem(mid="alice-2", content="payload P v2", salience=0.3, agent_id="alice")
    b1 = _mem(mid="bob-1", content="payload P b1", salience=0.95, agent_id="bob")
    b2 = _mem(mid="bob-2", content="payload P b2", salience=0.05, agent_id="bob")
    ctx_with, stage_with = _setup(tmp_path / "with", [a1b, b1, a2b, b2])
    stage_with.run(ctx_with)
    with_bob = _state_map(ctx_with, ["alice-1", "alice-2"])

    assert alone == with_bob, (
        f"CXE-1 LEAK: Alice's same-agent merge outcome changed when Bob's "
        f"cross-agent near-dups were present. alone={alone} with_bob={with_bob}"
    )


def test_cxe_2_bob_rows_active_through_composition(tmp_path: Path):
    """Bob's rows survive composition — per-channel ACL fix still effective."""
    a1 = _mem(mid="alice-1", content="payload P v1", salience=0.9, agent_id="alice")
    a2 = _mem(mid="alice-2", content="payload P v2", salience=0.3, agent_id="alice")
    b1 = _mem(mid="bob-1", content="payload P b1", salience=0.95, agent_id="bob")
    b2 = _mem(mid="bob-2", content="payload P b2", salience=0.05, agent_id="bob")
    ctx, stage = _setup(tmp_path, [a1, b1, a2, b2])
    stage.run(ctx)
    # Bob has two near-dup rows under his own agent_id — same-agent merge
    # SHOULD fire within Bob's slice (positive control), but neither row
    # may flip due to a cross-agent comparison with Alice. We assert at
    # least one Bob row is ACTIVE (intra-Bob merge keeps the high-salience
    # winner) and the keeper is the high-salience one.
    states = {m: ctx.store.get(m).state for m in ["bob-1", "bob-2"]}
    assert states["bob-1"] == MemoryState.ACTIVE, (
        f"CXE-2: Bob's high-salience row was suppressed under composition; "
        f"states={states}"
    )


def test_cxe_3_alice_intra_actor_merge_still_fires(tmp_path: Path):
    """Composition does not disable the merge stage — Alice's dups still merged."""
    a1 = _mem(mid="alice-1", content="payload P v1", salience=0.9, agent_id="alice")
    a2 = _mem(mid="alice-2", content="payload P v2", salience=0.3, agent_id="alice")
    b1 = _mem(mid="bob-1", content="payload P b1", salience=0.95, agent_id="bob")
    ctx, stage = _setup(tmp_path, [a1, b1, a2])
    stage.run(ctx)
    a1_after = ctx.store.get("alice-1").state
    a2_after = ctx.store.get("alice-2").state
    # The low-salience Alice row must be suppressed by the high-salience one.
    assert a1_after == MemoryState.ACTIVE
    assert a2_after == MemoryState.SUPPRESSED, (
        f"CXE-3: composition disabled the merge stage; alice-2 not suppressed "
        f"(state={a2_after})"
    )
