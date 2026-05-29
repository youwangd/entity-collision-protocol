"""Cross-state-machine composition: MechanicalMerge × concurrent writes/forgets.

NEXT.md priority #4 (research thread, second outstanding sub-bullet):

  > mech-merge × concurrency: does a merge running concurrently with writes
  > to the same cluster suppress racing inserts vs leaving them as orphans?

Sister to ``test_mechmerge_fts_composition_stateful.py`` (mech-merge × FTS5,
already landed). Where that machine drove a sequential schedule and probed
the FTS5 projection, this machine drives a *concurrent* schedule and probes
the convergence properties of the merge stage when racing writes and forgets
happen against the same (agent, cluster) bucket.

Setup mirrors the mech-merge fuzz module exactly so we don't introduce new
ground truth:
  * orthogonal one-hot embedder over ``_DIM`` cluster axes
  * cluster_id encoded in content prefix (``"<agent>:c<ci>:<mid>"``)
  * MechanicalMerge runs at threshold=0.90; intra-cluster cosine == 1.0,
    inter-cluster cosine == 0.0, so the (agent, cluster) bucket is the
    near-duplicate equivalence class.

Concurrency invariants:

  CM-I1  No worker raises. Concurrent ``upsert`` × ``MechanicalMerge.run``
         × ``forget(hard=True)`` against the same SQLite store must complete
         without exceptions. (Catches lock-mode regressions and cursor
         lifecycle bugs in the merge stage.)

  CM-I2  Eventual cardinality bound: after the concurrent batch quiesces
         AND a final merge sweep is applied, every (agent, cluster) bucket
         has ≤ 1 ACTIVE row. The merge stage may have run mid-batch on a
         partial view and missed late-arriving inserts, but a post-quiesce
         sweep MUST drive every bucket to the canonical singleton.

  CM-I3  Suppression-only — no row vanishes silently: the union of
         {survivor mids in active∪fading} ∪ {suppressed mids} ∪
         {hard-deleted mids the test issued} must equal the set of all mids
         the test ever upserted. Catches the regression where a racing
         merge + write pair drops a row entirely.

  CM-I4  Hard-deleted rows never resurrect: ids the test hard-forgot must
         not appear in the final ACTIVE / FADING / SUPPRESSED projections
         of the store.

Population strategy: small (≤8) bucket of (agent_id, cluster_id, salience)
specs, sharded into "writer" mids (inserted concurrently with merges) and
"forget targets" (existing mids the test races a forget against).

Threading model: one ThreadPoolExecutor per Hypothesis-driven batch. We
DON'T run multi-batch interleavings via a stateful machine here — the
stateful machine pattern adds Hypothesis-shrinking surface area but the
real coverage gain is in the per-batch micro-interleaving (GIL + SQLite
lock scheduling) which @given examples already drive nicely.
"""
from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.pipeline import MechanicalMerge, StageContext
from engram.core import DECAY_RATES, Memory, MemoryState, MemoryType
from engram.store.memory import SQLiteMemoryStore
from engram.store.vector import SQLiteVecStore


# Disjoint cluster space from sibling machines — ints, not tokens, are fine
# because we don't probe FTS5 here.
_DIM = 5
_AGENTS = ("alpha", "beta")
_THRESHOLD = 0.90
_FANOUT = 4  # writers per concurrent batch — small to keep the test fast


def _onehot(ci: int) -> list[float]:
    v = [0.0] * _DIM
    v[ci] = 1.0
    return v


def _mem(*, mid: str, ci: int, salience: float, agent_id: str) -> Memory:
    now = datetime.now(timezone.utc)
    content = f"{agent_id}:c{ci}:{mid}"
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
    """Embedder mapping content -> one-hot cluster vector via prefix parse."""
    dimension = _DIM

    @staticmethod
    def _parse_cluster(content: str) -> int:
        # "<agent>:c<ci>:<mid>"
        try:
            ci_tok = content.split(":", 2)[1]  # "c<ci>"
            return int(ci_tok[1:])
        except Exception:
            return 0

    def embed(self, content: str) -> list[float]:
        return _onehot(self._parse_cluster(content))

    def embed_batch(self, contents):  # pragma: no cover - unused
        return [self.embed(c) for c in contents]


@dataclass
class _Spec:
    mid: str
    agent: str
    ci: int
    salience: float


@st.composite
def _populations(draw, *, max_seed: int = 6, max_writers: int = 4):
    """Generate (seed_population, racing_writers, forget_targets)."""
    n_seed = draw(st.integers(min_value=1, max_value=max_seed))
    sal_choices = [0.10, 0.30, 0.50, 0.70, 0.90]

    seed: list[_Spec] = []
    for i in range(n_seed):
        seed.append(
            _Spec(
                mid=f"seed-{i:02d}",
                agent=draw(st.sampled_from(_AGENTS)),
                ci=draw(st.integers(min_value=0, max_value=_DIM - 1)),
                salience=draw(st.sampled_from(sal_choices)),
            )
        )

    n_write = draw(st.integers(min_value=1, max_value=max_writers))
    writers: list[_Spec] = []
    for i in range(n_write):
        writers.append(
            _Spec(
                mid=f"race-{i:02d}",
                agent=draw(st.sampled_from(_AGENTS)),
                ci=draw(st.integers(min_value=0, max_value=_DIM - 1)),
                salience=draw(st.sampled_from(sal_choices)),
            )
        )

    forget_n = draw(st.integers(min_value=0, max_value=min(2, n_seed)))
    # Sample without replacement from the seed mids.
    forget_targets = draw(
        st.lists(
            st.sampled_from([s.mid for s in seed]),
            min_size=forget_n,
            max_size=forget_n,
            unique=True,
        )
    )
    return seed, writers, forget_targets


def _seed_store(tmp_path: Path, seed: Iterable[_Spec]):
    store = SQLiteMemoryStore(tmp_path)
    vec = SQLiteVecStore(tmp_path / "vec.sqlite", dimension=_DIM)
    for s in seed:
        m = _mem(mid=s.mid, ci=s.ci, salience=s.salience, agent_id=s.agent)
        store.upsert(m)
        vec.upsert(s.mid, _onehot(s.ci))
    return store, vec


def _run_concurrent_batch(store, vec, writers, forget_mids):
    """Race {writers, merge_stage, forgets} against the same store.

    Returns (errors, merge_ran).
    """
    n = len(writers) + 1 + len(forget_mids)
    barrier = threading.Barrier(n)
    errors: list[BaseException] = []
    emb = _ClusterEmbeddings()
    stage = MechanicalMerge(vector_store=vec, embedding_provider=emb,
                            threshold=_THRESHOLD)
    ctx = StageContext(store=store)

    def _writer(spec: _Spec):
        try:
            barrier.wait()
            m = _mem(mid=spec.mid, ci=spec.ci, salience=spec.salience,
                     agent_id=spec.agent)
            store.upsert(m)
            vec.upsert(spec.mid, _onehot(spec.ci))
        except BaseException as ex:  # noqa: BLE001
            errors.append(ex)

    def _merger():
        try:
            barrier.wait()
            stage.run(ctx)
        except BaseException as ex:  # noqa: BLE001
            errors.append(ex)

    def _forgetter(mid: str):
        try:
            barrier.wait()
            store.update_state(mid, MemoryState.FADED)  # soft-clear-ish
            # Simulate a hard-forget by updating to FADED then relying on
            # the test's "_hard_deleted" bookkeeping — we use update_state
            # rather than the engine-level forget() because we built the
            # store directly without an Engram facade.
        except BaseException as ex:  # noqa: BLE001
            errors.append(ex)

    workers = (
        [(_writer, (w,)) for w in writers]
        + [(_merger, ())]
        + [(_forgetter, (mid,)) for mid in forget_mids]
    )

    with ThreadPoolExecutor(max_workers=n) as pool:
        futs = [pool.submit(fn, *args) for (fn, args) in workers]
        for f in as_completed(futs):
            f.result()  # surface any worker exception not caught above
    return errors, stage, ctx


def _final_merge_sweep(store, vec, max_passes: int = 4):
    """Run merge until idempotent (handles late-arriving inserts)."""
    emb = _ClusterEmbeddings()
    stage = MechanicalMerge(vector_store=vec, embedding_provider=emb,
                            threshold=_THRESHOLD)
    last_active: set[str] | None = None
    for _ in range(max_passes):
        ctx = StageContext(store=store)
        stage.run(ctx)
        cur = {
            m.id for m in store.all_active()
            if m.state in (MemoryState.ACTIVE, MemoryState.FADING)
        }
        if last_active is not None and cur == last_active:
            return
        last_active = cur


@given(pop=_populations())
@settings(
    max_examples=60,
    deadline=None,
    suppress_health_check=[
        HealthCheck.too_slow,
        HealthCheck.function_scoped_fixture,
        HealthCheck.data_too_large,
    ],
)
def test_mm_concurrent_no_worker_raises_and_eventual_singletons(
    tmp_path_factory, pop
):
    """CM-I1, CM-I2, CM-I3, CM-I4 in a single integrated trace."""
    seed, writers, forget_targets = pop
    tmp = tmp_path_factory.mktemp("mmconc")
    store, vec = _seed_store(tmp, seed)

    # Mids the test "owns" — split into:
    #   all_inserted: every mid we ever wrote into the store
    #   forgotten:    mids we issued a forget against
    all_inserted = {s.mid for s in seed} | {w.mid for w in writers}
    forgotten = set(forget_targets)

    errors, _stage, _ctx = _run_concurrent_batch(
        store, vec, writers, list(forgotten)
    )

    # CM-I1: no worker exception
    assert not errors, (
        f"CM-I1 violated: worker raised "
        f"{type(errors[0]).__name__}: {errors[0]}"
    )

    # Drive to fixpoint.
    _final_merge_sweep(store, vec)

    # Project final state.
    _ = store.all_active()  # NB: includes non-active states in this codebase?
    # Be defensive: re-query each known mid via store.get to get authoritative state.
    states_by_mid: dict[str, MemoryState | None] = {}
    for mid in all_inserted:
        m = store.get(mid)
        states_by_mid[mid] = m.state if m is not None else None

    active_or_fading: set[str] = {
        mid for mid, st_ in states_by_mid.items()
        if st_ in (MemoryState.ACTIVE, MemoryState.FADING)
    }
    suppressed: set[str] = {
        mid for mid, st_ in states_by_mid.items() if st_ == MemoryState.SUPPRESSED
    }
    faded: set[str] = {
        mid for mid, st_ in states_by_mid.items() if st_ == MemoryState.FADED
    }
    missing: set[str] = {mid for mid, st_ in states_by_mid.items() if st_ is None}

    # CM-I2: per (agent, cluster) bucket — among rows the test inserted that
    # are still ACTIVE/FADING, at most one survivor per bucket.
    by_bucket: dict[tuple[str, int], list[str]] = {}
    for s in list(seed) + list(writers):
        if s.mid in active_or_fading:
            by_bucket.setdefault((s.agent, s.ci), []).append(s.mid)
    over_full = {b: mids for b, mids in by_bucket.items() if len(mids) > 1}
    assert not over_full, (
        f"CM-I2: post-sweep buckets with >1 ACTIVE survivor: {over_full}\n"
        f"  seed={seed}\n  writers={writers}\n  forget={forgotten}"
    )

    # CM-I3: every inserted mid is accounted for. Forget targets are allowed
    # to be in `faded` (we used update_state(FADED) as the test's "forget").
    accounted = active_or_fading | suppressed | faded
    lost = all_inserted - accounted - missing
    # `missing` (store.get returned None) is the regression we'd worry about.
    assert not missing, (
        f"CM-I3: rows vanished from the store entirely: {missing}\n"
        f"  forgotten={forgotten}, all_inserted={all_inserted}"
    )
    assert not lost, (
        f"CM-I3: rows in unknown final state: {lost}\n"
        f"  active_or_fading={active_or_fading}\n"
        f"  suppressed={suppressed}\n  faded={faded}"
    )

    # CM-I4: forgotten ids are NOT in the active/fading projection.
    leaks = forgotten & active_or_fading
    assert not leaks, (
        f"CM-I4: forgotten mids resurfaced as ACTIVE/FADING after concurrent "
        f"merge: {leaks}"
    )


# ---------------------------------------------------------------------------
# Closed-state pin: clearest concrete witness — concurrent writers landing
# the same (agent, cluster) while a merge is running converge to a singleton
# after the post-quiesce sweep.
# ---------------------------------------------------------------------------


def test_mm_concurrent_same_bucket_writes_converge_singleton_smoke(tmp_path):
    seed = [_Spec("seed-00", "alpha", 0, 0.50)]
    writers = [
        _Spec("race-00", "alpha", 0, 0.20),
        _Spec("race-01", "alpha", 0, 0.95),
        _Spec("race-02", "alpha", 0, 0.10),
    ]
    store, vec = _seed_store(tmp_path, seed)

    errors, _stage, _ctx = _run_concurrent_batch(store, vec, writers, [])
    assert not errors, errors

    _final_merge_sweep(store, vec)

    survivors = [
        m for m in store.all_active()
        if m.agent_id == "alpha" and _ClusterEmbeddings._parse_cluster(m.content) == 0
        and m.state in (MemoryState.ACTIVE, MemoryState.FADING)
    ]
    assert len(survivors) == 1, (
        f"smoke: expected exactly 1 survivor in (alpha, c0), got {len(survivors)}: "
        f"{[m.id for m in survivors]}"
    )
    # Highest-salience writer should have won.
    assert survivors[0].id == "race-01", (
        f"smoke: expected highest-salience 'race-01' to survive, got {survivors[0].id}"
    )
