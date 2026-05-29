"""Cross-state-machine composition: MechanicalMerge × FTS5.

NEXT.md priority #4 (research thread, second sub-bullet, mech-merge × FTS5):

  > a merge that rewrites canonical content should leave the FTS5 index
  > aligned with the merged row, no ghost from the absorbed row.

Companion to ``test_dedup_fts_composition_stateful.py`` (write-side dedup ×
FTS5, landed earlier this tick-arc). Where dedup is a write-time decision
that *prevents* a row from being inserted, MechanicalMerge runs as a
consolidation stage *after* the rows are already in the SQLite table and
indexed by FTS5 — it suppresses (state→SUPPRESSED) the lower-salience
near-duplicates rather than removing them. The FTS5 invariants therefore
differ in shape from the dedup case:

  M-I1  Survivor reachable via FTS5: for each (agent, cluster) bucket
        with ≥1 input row, exactly one row is reachable via
        MATCH(cluster_token, states=['active','fading']) — the survivor
        the merge picked. (Cardinality coupling at the active-state
        projection.)

  M-I2  Suppressed rows excluded from default-state search: rows that
        MechanicalMerge suppresses must NOT appear in
        MATCH(token, states=['active','fading']). They MAY still appear
        when the caller explicitly includes 'suppressed' in the state
        filter — that's the audit/forensics path and is part of the
        contract (suppression is reversible; hard-delete is not).

  M-I3  Whole-index parity: sum over all clusters of
        |MATCH(C_token, ['active','fading'])| == #rows in (active∪fading)
        states post-merge. Catches FTS5 row-leak bugs where suppression
        fails to update the FTS5 index for some rows.

  M-I4  Idempotence at FTS5 projection: running merge twice doesn't
        change the per-token MATCH multisets at the active-state filter.

Population strategy: same one-hot embedder as the existing mech-merge
fuzz module, paired with a single distinctive content token per cluster
so FTS5 MATCH is unambiguous. Tokens are disjoint from the dedup×FTS5
machine's tokens for clean failure attribution across files.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.pipeline import MechanicalMerge, StageContext
from engram.core import DECAY_RATES, Memory, MemoryState, MemoryType
from engram.store.memory import SQLiteMemoryStore
from engram.store.vector import SQLiteVecStore


# Disjoint cluster tokens: distinct from the dedup×FTS5 set
# ({eeeeee, ffffff, gggggg, hhhhhh}) and the FTS-only machine
# ({zappa, qwerty, fjord, krypton, synapse, obelisk, tundra, marrow}).
_TOKENS = ("nimbus", "cobalt", "lichen", "vellum", "borealis", "quartzite")
_DIM = len(_TOKENS)
_AGENTS = ("alice", "bob")


def _onehot(ci: int) -> list[float]:
    v = [0.0] * _DIM
    v[ci] = 1.0
    return v


def _mem(*, mid: str, ci: int, salience: float, agent_id: str) -> Memory:
    now = datetime.now(timezone.utc)
    # Embed the cluster token literally in content so FTS5 can MATCH on it,
    # plus a unique mid suffix so rows aren't text-identical (FTS5 still
    # MATCHes on the token).
    content = f"{_TOKENS[ci]} entry for {agent_id} mid {mid}"
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
    """Embedder mapping content prefix -> one-hot cluster vector."""
    dimension = _DIM

    @staticmethod
    def _parse_cluster(content: str) -> int:
        # First whitespace-delimited token is the cluster token.
        head = content.split(" ", 1)[0]
        try:
            return _TOKENS.index(head)
        except ValueError:  # pragma: no cover — strategy controls inputs
            return 0

    def embed(self, content: str) -> list[float]:
        return _onehot(self._parse_cluster(content))

    def embed_batch(self, contents):  # pragma: no cover - unused
        return [self.embed(c) for c in contents]


@st.composite
def _populations(draw, *, max_n: int = 10):
    n = draw(st.integers(min_value=1, max_value=max_n))
    salience_ladder = [0.10, 0.30, 0.50, 0.70, 0.90]
    specs = []
    for i in range(n):
        agent = draw(st.sampled_from(_AGENTS))
        ci = draw(st.integers(min_value=0, max_value=_DIM - 1))
        sal = draw(st.sampled_from(salience_ladder))
        specs.append((f"m-{i:02d}", agent, ci, sal))
    return specs


def _setup(tmp_path: Path, specs):
    store = SQLiteMemoryStore(tmp_path / "mem.sqlite")
    vec = SQLiteVecStore(tmp_path / "vec.sqlite", dimension=_DIM)
    for (mid, agent, ci, sal) in specs:
        m = _mem(mid=mid, ci=ci, salience=sal, agent_id=agent)
        store.upsert(m)
        vec.upsert(mid, _onehot(ci))
    stage = MechanicalMerge(
        vector_store=vec, embedding_provider=_ClusterEmbeddings(), threshold=0.90
    )
    return store, vec, stage


def _fts_active_mids_by_token(store) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for tok in _TOKENS:
        hits = store.search_text(tok, limit=200, states=["active", "fading"])
        out[tok] = {h.memory.id for h in hits}
    return out


# ---------------------------------------------------------------------------
# M-I1: per-bucket survivor reachable via FTS5
# ---------------------------------------------------------------------------


@given(specs=_populations())
@settings(
    max_examples=40, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_mm_fts_survivor_reachable(tmp_path_factory, specs):
    tmp = tmp_path_factory.mktemp("mmfts_survivor")
    store, _vec, stage = _setup(tmp, specs)
    ctx = StageContext(store=store)
    stage.run(ctx)

    # Bucket inputs by (agent, cluster) — these are the same-agent
    # near-duplicate equivalence classes under the orthogonal embedder.
    buckets: dict[tuple[str, int], list[tuple[str, float]]] = defaultdict(list)
    for (mid, agent, ci, sal) in specs:
        buckets[(agent, ci)].append((mid, sal))

    # Group survivor counts by cluster (across agents): each (agent, cluster)
    # bucket leaves exactly 1 ACTIVE survivor; FTS5(token) should hit all of
    # them (one per agent that owned ≥1 row in that cluster).
    expected_per_cluster: dict[int, int] = defaultdict(int)
    for (_agent, ci), members in buckets.items():
        if members:
            expected_per_cluster[ci] += 1

    fts = _fts_active_mids_by_token(store)
    for ci, tok in enumerate(_TOKENS):
        got = len(fts[tok])
        exp = expected_per_cluster.get(ci, 0)
        assert got == exp, (
            f"M-I1: cluster {ci} ({tok!r}) FTS5 active hits = {got}, "
            f"expected {exp} (one per agent with ≥1 row). specs={specs}"
        )


# ---------------------------------------------------------------------------
# M-I2: suppressed rows excluded from default search but visible w/ filter
# ---------------------------------------------------------------------------


@given(specs=_populations())
@settings(
    max_examples=40, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_mm_fts_suppressed_excluded_from_default(tmp_path_factory, specs):
    tmp = tmp_path_factory.mktemp("mmfts_suppressed")
    store, _vec, stage = _setup(tmp, specs)
    ctx = StageContext(store=store)
    stage.run(ctx)

    # Identify suppressed rows from the store directly.
    suppressed_ids: set[str] = set()
    for (mid, *_rest) in specs:
        m = store.get(mid)
        if m is not None and m.state == MemoryState.SUPPRESSED:
            suppressed_ids.add(mid)

    if not suppressed_ids:
        return  # vacuous

    fts_active = _fts_active_mids_by_token(store)
    all_active = set().union(*fts_active.values())
    leaks = suppressed_ids & all_active
    assert not leaks, (
        f"M-I2: suppressed mids {leaks} appeared in default-state FTS5 search; "
        f"specs={specs}"
    )

    # Reverse direction: suppressed rows MUST still be findable when the
    # caller explicitly opts into the suppressed state. This is the audit
    # path; suppression is reversible, hard-delete is not.
    found_with_filter: set[str] = set()
    for tok in _TOKENS:
        hits = store.search_text(
            tok, limit=200, states=["active", "fading", "faded", "suppressed"]
        )
        for h in hits:
            if h.memory.id in suppressed_ids:
                found_with_filter.add(h.memory.id)
    missing = suppressed_ids - found_with_filter
    assert not missing, (
        f"M-I2 audit-path: suppressed mids {missing} not findable even with "
        f"explicit ['active','fading','faded','suppressed'] filter; specs={specs}"
    )


# ---------------------------------------------------------------------------
# M-I3: whole-index parity — FTS active hits == #rows in active∪fading
# ---------------------------------------------------------------------------


@given(specs=_populations())
@settings(
    max_examples=40, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_mm_fts_whole_index_parity(tmp_path_factory, specs):
    tmp = tmp_path_factory.mktemp("mmfts_parity")
    store, _vec, stage = _setup(tmp, specs)
    ctx = StageContext(store=store)
    stage.run(ctx)

    fts_active_total = sum(len(s) for s in _fts_active_mids_by_token(store).values())
    active_or_fading = len([
        m for m in store.all_active() if m.state in (MemoryState.ACTIVE, MemoryState.FADING)
    ])
    assert fts_active_total == active_or_fading, (
        f"M-I3: FTS5 active-state hits across all clusters = {fts_active_total}, "
        f"#(active∪fading) rows = {active_or_fading}; specs={specs}"
    )


# ---------------------------------------------------------------------------
# M-I4: idempotence at the FTS5 projection
# ---------------------------------------------------------------------------


@given(specs=_populations())
@settings(
    max_examples=30, deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)
def test_mm_fts_idempotent_projection(tmp_path_factory, specs):
    tmp = tmp_path_factory.mktemp("mmfts_idem")
    store, _vec, stage = _setup(tmp, specs)
    ctx = StageContext(store=store)
    stage.run(ctx)
    after_first = _fts_active_mids_by_token(store)
    stage.run(ctx)
    after_second = _fts_active_mids_by_token(store)
    assert after_first == after_second, (
        f"M-I4: MechanicalMerge not idempotent at FTS5 projection.\n"
        f"  first={after_first}\n  second={after_second}\n  specs={specs}"
    )


# ---------------------------------------------------------------------------
# Closed-state pin: clearest concrete witness that a same-cluster merge
# leaves no FTS5 ghost from the absorbed (lower-salience) row.
# ---------------------------------------------------------------------------


def test_mm_no_fts_ghost_from_suppressed_smoke(tmp_path):
    specs = [
        ("m-hi", "alice", 0, 0.90),  # nimbus, high salience -> survivor
        ("m-lo", "alice", 0, 0.10),  # nimbus, low salience -> suppressed
    ]
    store, _vec, stage = _setup(tmp_path, specs)
    ctx = StageContext(store=store)
    stage.run(ctx)

    hits_default = store.search_text("nimbus", limit=10, states=["active", "fading"])
    hit_ids = {h.memory.id for h in hits_default}
    assert hit_ids == {"m-hi"}, (
        f"expected only survivor 'm-hi' in default search, got {hit_ids}"
    )

    # And the suppressed row is still recoverable via explicit state filter
    # (audit channel — suppression is reversible).
    hits_full = store.search_text(
        "nimbus", limit=10, states=["active", "fading", "faded", "suppressed"]
    )
    hit_ids_full = {h.memory.id for h in hits_full}
    assert hit_ids_full == {"m-hi", "m-lo"}, (
        f"audit channel: expected both rows, got {hit_ids_full}"
    )
