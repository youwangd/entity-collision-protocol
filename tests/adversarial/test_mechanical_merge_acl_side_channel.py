"""Adversarial — MechanicalMerge ACL side-channel audit (§D-mech-merge-acl).

The Stage 12 ``MechanicalMerge`` consolidator iterates
``ctx.store.all_active()`` (global) and, for each memory, asks the vector
store for any other memory with cosine similarity above its threshold
(default 0.95). When a pair crosses the threshold, the lower-salience
side is unconditionally moved to ``MemoryState.SUPPRESSED``.

The pre-fix implementation did not consult ``agent_id`` on the matched
pair. That is two bugs in one:

  1. **Existence leak.** Bob's silent near-duplicate of Alice's content
     gets suppressed at consolidation time. Alice's later "did Bob ever
     remember <thing>?" probe — which she should not be able to answer
     — is now answerable: Bob's row is in SUPPRESSED state, distinguishable
     from "never written" via lifecycle metadata.

  2. **Cross-tenant DoS.** Any agent that writes a high-salience version
     of content can suppress another agent's near-duplicate. In a
     multi-tenant deployment this lets the noisier tenant erase the
     quieter tenant's memories.

The fix in ``MechanicalMerge.run`` over-fetches the candidate pool to
20 and skips any pair whose ``agent_id`` strings differ. System-owned
memories (``agent_id=''``, e.g. SCHEMA prototypes) remain mergeable
globally — that is the intended behaviour for shared system patterns.

Invariants pinned here:

  MM-ACL-1  Cross-agent near-duplicates are NEVER suppressed by mechanical
            merge, regardless of salience ordering.

  MM-ACL-2  Same-agent near-duplicates ARE still suppressed (positive
            control — the fix is not just disabling the stage).

  MM-ACL-3  System (agent_id='') near-duplicates remain mergeable
            globally — schemas are intentionally shared.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path


from engram.consolidation.pipeline import MechanicalMerge, StageContext
from engram.core import DECAY_RATES, Memory, MemoryState, MemoryType
from engram.store.memory import SQLiteMemoryStore
from engram.store.vector import SQLiteVecStore


class _FakeEmbeddings:
    """Deterministic embedder: identical content → identical vector.

    We don't need real semantics — the merge stage just calls
    ``embed(memory.content)`` and pumps it through cosine on the vector
    store, which we control.
    """

    dimension = 8

    def __init__(self, table: dict[str, list[float]]):
        self._table = table

    def embed(self, content: str) -> list[float]:
        return self._table.get(content, [0.0] * self.dimension)

    def embed_batch(self, contents):  # pragma: no cover - unused here
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
    # Single shared near-duplicate vector for every memory in the test —
    # cosine ~1.0 for all pairs, so the merge stage will see them as
    # near-duplicates regardless of agent_id.
    near_dup = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    table = {}
    for m in memories:
        store.upsert(m)
        vec.upsert(m.id, near_dup)
        table[m.content] = near_dup
    emb = _FakeEmbeddings(table)
    stage = MechanicalMerge(vector_store=vec, embedding_provider=emb, threshold=0.90)
    ctx = StageContext(store=store)
    return ctx, stage


def test_mm_acl_1_cross_agent_near_dup_not_suppressed(tmp_path: Path):
    """Alice's high-salience near-duplicate must not suppress Bob's row."""
    alice = _mem(mid="alice-1", content="meeting notes A", salience=0.9, agent_id="alice")
    bob = _mem(mid="bob-1", content="meeting notes B", salience=0.1, agent_id="bob")
    ctx, stage = _setup(tmp_path, [alice, bob])
    stage.run(ctx)
    a = ctx.store.get("alice-1")
    b = ctx.store.get("bob-1")
    assert a.state == MemoryState.ACTIVE, "Alice should remain active"
    assert b.state == MemoryState.ACTIVE, (
        f"Bob's near-duplicate must NOT be suppressed across ACL boundaries; "
        f"got state={b.state}"
    )
    assert ctx.stats.get("mechanical_merged", 0) == 0


def test_mm_acl_1_symmetric_high_salience_bob(tmp_path: Path):
    """Symmetry: high-salience BOB must not suppress low-salience ALICE either."""
    alice = _mem(mid="alice-1", content="x", salience=0.05, agent_id="alice")
    bob = _mem(mid="bob-1", content="y", salience=0.95, agent_id="bob")
    ctx, stage = _setup(tmp_path, [alice, bob])
    stage.run(ctx)
    assert ctx.store.get("alice-1").state == MemoryState.ACTIVE
    assert ctx.store.get("bob-1").state == MemoryState.ACTIVE
    assert ctx.stats.get("mechanical_merged", 0) == 0


def test_mm_acl_2_same_agent_dup_still_merges(tmp_path: Path):
    """Positive control: within one agent, mechanical merge still fires."""
    a1 = _mem(mid="alice-hi", content="alpha", salience=0.9, agent_id="alice")
    a2 = _mem(mid="alice-lo", content="beta", salience=0.1, agent_id="alice")
    ctx, stage = _setup(tmp_path, [a1, a2])
    stage.run(ctx)
    states = {m: ctx.store.get(m).state for m in ("alice-hi", "alice-lo")}
    assert states["alice-hi"] == MemoryState.ACTIVE
    assert states["alice-lo"] == MemoryState.SUPPRESSED, (
        f"same-agent merge must still suppress lower-salience side; got {states}"
    )
    assert ctx.stats.get("mechanical_merged", 0) >= 1


def test_mm_acl_3_system_memories_merge_globally(tmp_path: Path):
    """System memories (agent_id='') stay globally mergeable — schemas are shared."""
    s1 = _mem(mid="sys-hi", content="proto-A", salience=0.8, agent_id="")
    s2 = _mem(mid="sys-lo", content="proto-B", salience=0.2, agent_id="")
    ctx, stage = _setup(tmp_path, [s1, s2])
    stage.run(ctx)
    assert ctx.store.get("sys-hi").state == MemoryState.ACTIVE
    assert ctx.store.get("sys-lo").state == MemoryState.SUPPRESSED
    assert ctx.stats.get("mechanical_merged", 0) >= 1


def test_mm_acl_4_system_does_not_suppress_agent_owned(tmp_path: Path):
    """A system memory (agent_id='') must not suppress an agent-owned near-dup."""
    sys_hi = _mem(mid="sys", content="shared", salience=0.99, agent_id="")
    alice = _mem(mid="alice", content="private", salience=0.1, agent_id="alice")
    ctx, stage = _setup(tmp_path, [sys_hi, alice])
    stage.run(ctx)
    assert ctx.store.get("sys").state == MemoryState.ACTIVE
    assert ctx.store.get("alice").state == MemoryState.ACTIVE, (
        "agent_id='' must not be treated as a wildcard owner during merge"
    )
