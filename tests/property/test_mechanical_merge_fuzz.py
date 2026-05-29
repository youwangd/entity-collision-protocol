"""Property-based fuzz for MechanicalMerge candidate selection (§D8 fuzz expansion).

NEXT.md (2026-05-24) called out mechanical-merge candidate selection as one of
the next fuzz targets after the lifecycle-cache fuzz caught a real bug
(961dfad). This module pins five aggregate invariants of the Stage-12
``MechanicalMerge`` consolidator under randomly generated multi-agent
populations.

Each Memory is assigned a ``(agent_id, cluster_id)`` tuple. Cluster IDs index
into a one-hot embedding axis, so:

  * Two memories with the same cluster_id ⇒ cosine == 1.0  (near-duplicates)
  * Two memories with different cluster_ids ⇒ cosine == 0.0  (orthogonal)

This decouples the merge decision logic from any messy embedding semantics
and lets us state crisp invariants over the multiset of (agent, cluster) tags.

Invariants pinned:

  MM-INV-1  Idempotence: running ``MechanicalMerge.run`` twice yields the same
            ACTIVE set as running it once.

  MM-INV-2  Per-(agent, cluster)-bucket survivor: after merge, each non-empty
            same-agent near-duplicate bucket leaves exactly one ACTIVE memory,
            and the survivor's salience equals the max salience in the bucket
            (tie-break is winner-takes-the-tie).

  MM-INV-3  Cross-agent non-suppression: a memory whose (agent, cluster)
            bucket has size 1 is NEVER suppressed, regardless of how many
            other agents own near-duplicates in the same cluster_id.

  MM-INV-4  Sub-threshold preservation: when every cluster has size 1
            per-agent (no within-agent near-duplicates), no memory is
            suppressed even if cross-agent near-duplicates exist.

  MM-INV-5  Active-count = distinct (agent_id, cluster_id) pairs, when the
            embedder produces orthogonal-or-identical vectors as set up here.

System-owned memories (agent_id='') are excluded from the random population
because their merge contract is "global" rather than per-bucket and is
covered by the deterministic adversarial suite
``tests/adversarial/test_mechanical_merge_acl_side_channel.py``.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.pipeline import MechanicalMerge, StageContext
from engram.core import DECAY_RATES, Memory, MemoryState, MemoryType
from engram.store.memory import SQLiteMemoryStore
from engram.store.vector import SQLiteVecStore


# ---------------------------------------------------------------------------
# Population strategy
# ---------------------------------------------------------------------------

# Keep dimensions small — we want a wide variety of compositions, not a wide
# variety of cosines. The merge contract is independent of dimensionality.
_DIM = 8
_AGENTS = ("alice", "bob", "carol")
_CLUSTERS = tuple(range(_DIM))  # cluster_id is the one-hot axis index


def _onehot(cluster_id: int) -> list[float]:
    v = [0.0] * _DIM
    v[cluster_id] = 1.0
    return v


def _mem(*, mid: str, cluster_id: int, salience: float, agent_id: str) -> Memory:
    now = datetime.now(timezone.utc)
    # Make content unique per id so the audit channel sees distinct rows;
    # the merge stage hashes on the embedding, not on content equality.
    content = f"{agent_id}:c{cluster_id}:{mid}"
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


class _ClusterEmbeddings:
    """Embedder that returns the one-hot vector for the memory's cluster.

    The merge stage calls ``embed(memory.content)``; we encode the cluster_id
    in the content prefix and parse it back here so the embedder is purely a
    function of content, matching the production contract.
    """

    dimension = _DIM

    @staticmethod
    def _parse_cluster(content: str) -> int:
        # content format: "<agent>:c<cluster>:<mid>"
        try:
            mid_part = content.split(":c", 1)[1]
            cluster_str = mid_part.split(":", 1)[0]
            return int(cluster_str)
        except (IndexError, ValueError):  # pragma: no cover — strategy controls inputs
            return 0

    def embed(self, content: str) -> list[float]:
        return _onehot(self._parse_cluster(content))

    def embed_batch(self, contents):  # pragma: no cover - unused here
        return [self.embed(c) for c in contents]


@st.composite
def _populations(draw, *, max_n: int = 12, threshold: float = 0.90):
    """Generate a random list of memory specs.

    Each spec is (mid, agent_id, cluster_id, salience). Mids are unique;
    salience is drawn from a discrete ladder so ties are common (tie-break
    correctness matters for MM-INV-2).
    """
    n = draw(st.integers(min_value=1, max_value=max_n))
    salience_ladder = [0.10, 0.30, 0.50, 0.70, 0.90]
    specs = []
    for i in range(n):
        agent = draw(st.sampled_from(_AGENTS))
        cluster_id = draw(st.sampled_from(_CLUSTERS))
        sal = draw(st.sampled_from(salience_ladder))
        specs.append((f"m-{i:02d}", agent, cluster_id, sal))
    return specs, threshold


def _setup_population(tmp_path: Path, specs, threshold: float):
    store = SQLiteMemoryStore(tmp_path / "mem.sqlite")
    vec = SQLiteVecStore(tmp_path / "vec.sqlite", dimension=_DIM)
    for (mid, agent, cluster_id, sal) in specs:
        m = _mem(mid=mid, cluster_id=cluster_id, salience=sal, agent_id=agent)
        store.upsert(m)
        vec.upsert(mid, _onehot(cluster_id))
    emb = _ClusterEmbeddings()
    stage = MechanicalMerge(vector_store=vec, embedding_provider=emb, threshold=threshold)
    ctx = StageContext(store=store)
    return ctx, stage, store


def _active_ids(store) -> set[str]:
    return {m.id for m in store.all_active()}


# ---------------------------------------------------------------------------
# MM-INV-1: idempotence
# ---------------------------------------------------------------------------


@given(pop=_populations())
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_mm_idempotent(tmp_path_factory, pop):
    specs, threshold = pop
    tmp = tmp_path_factory.mktemp("mm_idem")
    ctx, stage, store = _setup_population(tmp, specs, threshold)
    stage.run(ctx)
    after_first = _active_ids(store)
    stage.run(ctx)
    after_second = _active_ids(store)
    assert after_first == after_second, (
        f"MechanicalMerge.run should be idempotent at the active-set level; "
        f"first={sorted(after_first)} second={sorted(after_second)}"
    )


# ---------------------------------------------------------------------------
# MM-INV-2: per-(agent, cluster)-bucket survivor with max salience
# ---------------------------------------------------------------------------


@given(pop=_populations())
@settings(max_examples=60, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_mm_per_bucket_survivor_is_max_salience(tmp_path_factory, pop):
    specs, threshold = pop
    tmp = tmp_path_factory.mktemp("mm_bucket")
    ctx, stage, store = _setup_population(tmp, specs, threshold)
    stage.run(ctx)
    active = list(store.all_active())

    # Bucket the input by (agent, cluster) — these are the same-agent
    # near-duplicate equivalence classes under our orthogonal embedder.
    from collections import defaultdict
    buckets: dict[tuple[str, int], list[tuple[str, float]]] = defaultdict(list)
    for (mid, agent, cluster_id, sal) in specs:
        buckets[(agent, cluster_id)].append((mid, sal))

    # Each non-empty bucket should contribute exactly one survivor, and that
    # survivor's salience must equal the bucket's max salience.
    survivors_by_bucket: dict[tuple[str, int], list[Memory]] = defaultdict(list)
    for m in active:
        # Reverse-derive cluster from content prefix.
        cluster_id = _ClusterEmbeddings._parse_cluster(m.content)
        survivors_by_bucket[(m.agent_id, cluster_id)].append(m)

    for bucket_key, members in buckets.items():
        survivors = survivors_by_bucket.get(bucket_key, [])
        assert len(survivors) == 1, (
            f"bucket {bucket_key} had {len(members)} input memories but "
            f"{len(survivors)} survived; expected exactly 1. members={members}"
        )
        max_sal = max(s for (_mid, s) in members)
        assert survivors[0].salience == pytest.approx(max_sal), (
            f"bucket {bucket_key} survivor salience {survivors[0].salience} "
            f"should be max-of-bucket {max_sal}; members={members}"
        )


# ---------------------------------------------------------------------------
# MM-INV-3: cross-agent non-suppression
# ---------------------------------------------------------------------------


@given(pop=_populations())
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_mm_singletons_never_suppressed(tmp_path_factory, pop):
    specs, threshold = pop
    tmp = tmp_path_factory.mktemp("mm_singleton")
    ctx, stage, store = _setup_population(tmp, specs, threshold)

    # Memories whose (agent, cluster) bucket has size 1 are singletons under
    # the same-agent near-duplicate relation. They must survive merge.
    from collections import Counter
    bucket_counts = Counter((agent, c) for (_mid, agent, c, _s) in specs)
    singleton_ids = {
        mid for (mid, agent, c, _s) in specs if bucket_counts[(agent, c)] == 1
    }

    stage.run(ctx)
    after = _active_ids(store)
    missing = singleton_ids - after
    assert not missing, (
        f"singletons must never be suppressed by mechanical merge; "
        f"suppressed singletons={sorted(missing)} specs={specs}"
    )


# ---------------------------------------------------------------------------
# MM-INV-4: no within-agent dups ⇒ no suppression
# ---------------------------------------------------------------------------


@given(pop=_populations())
@settings(max_examples=30, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_mm_no_within_agent_dups_means_no_suppression(tmp_path_factory, pop):
    specs, _threshold = pop
    # Filter to specs where every (agent, cluster) is unique — this is the
    # "within-agent disjoint" precondition; cross-agent overlap is allowed.
    from collections import Counter
    counts = Counter((a, c) for (_mid, a, c, _s) in specs)
    filtered = [s for s in specs if counts[(s[1], s[2])] == 1]
    if not filtered:
        return  # vacuous example
    tmp = tmp_path_factory.mktemp("mm_disjoint")
    ctx, stage, store = _setup_population(tmp, filtered, threshold=0.90)
    stage.run(ctx)
    after = _active_ids(store)
    expected = {mid for (mid, *_rest) in filtered}
    assert after == expected, (
        f"under within-agent disjointness, merge should preserve all rows; "
        f"missing={sorted(expected - after)} extra={sorted(after - expected)}"
    )


# ---------------------------------------------------------------------------
# MM-INV-5: active count = distinct (agent, cluster) pairs
# ---------------------------------------------------------------------------


@given(pop=_populations())
@settings(max_examples=40, deadline=None,
          suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture])
def test_mm_active_count_equals_distinct_buckets(tmp_path_factory, pop):
    specs, threshold = pop
    tmp = tmp_path_factory.mktemp("mm_count")
    ctx, stage, store = _setup_population(tmp, specs, threshold)
    stage.run(ctx)
    after = _active_ids(store)
    distinct_buckets = {(agent, c) for (_mid, agent, c, _s) in specs}
    assert len(after) == len(distinct_buckets), (
        f"post-merge active count should equal #distinct (agent, cluster) "
        f"buckets; got {len(after)} expected {len(distinct_buckets)}; "
        f"specs={specs}"
    )
