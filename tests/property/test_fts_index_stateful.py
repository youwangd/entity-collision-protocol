"""Hypothesis stateful machine for FTS5 incremental update consistency.

NEXT.md priority #1 (last remaining target of the property-fuzz expansion
plan). Mech-merge (779f1e2) and write-dedup (9876af2 + 64fd716) are
already covered by their own state machines. This one fuzzes random
interleavings of the FTS5-mutating operations — `remember`,
`forget(hard=False)`, `forget(hard=True)`, and `rebuild()` — and asserts:

  F-I1  Searchability of active rows: every row whose state ∈
        {active, fading} can be retrieved by an FTS5 MATCH on a
        distinctive token from its content.

  F-I2  Hard-delete is observable in FTS5: a memory id that has been
        hard-forgotten must NOT appear in any FTS5 search hit, even
        when its old distinctive token is queried.

  F-I3  Soft-delete is *not* visible in default search: a memory id
        that has been soft-forgotten (state=SUPPRESSED) must not
        appear in `search_text` calls that filter to ['active',
        'fading'] (the engine's default for `recall`).

  F-I4  Rebuild idempotence on FTS5: calling `engine.rebuild()` does
        not change the set of (active-rowids reachable by their
        distinctive token) — i.e. the FTS5 index is fully rebuilt
        from JSONL events and matches the pre-rebuild state.

The model uses 8 disjoint single-character clusters; every memory's
content embeds the cluster's distinctive token (e.g. `zappa_3`) so the
search query is unambiguous. We use the engine's own `recall` /
`store.search_text` so we exercise the production path including the
sanitizer and triggers.
"""
from __future__ import annotations

import tempfile
from pathlib import Path

from hypothesis import HealthCheck, settings, strategies as st
from hypothesis.stateful import (
    Bundle,
    RuleBasedStateMachine,
    initialize,
    invariant,
    rule,
)

from engram import Config, Engram


# Disjoint, FTS5-friendly tokens — alphabetic, length>2, not stopwords,
# not FTS5 operators. Each cluster gets a unique token so a MATCH on
# that token only hits rows from that cluster.
_TOKENS = [
    "zappa", "qwerty", "fjord", "krypton",
    "synapse", "obelisk", "tundra", "marrow",
]


class FTSIndexStateMachine(RuleBasedStateMachine):
    memory_ids = Bundle("memory_ids")

    def __init__(self):
        super().__init__()
        self._tmp: Path | None = None
        self._engram: Engram | None = None
        # mid -> token (so we know what query should retrieve it)
        self._mid_token: dict[str, str] = {}
        # mid -> "active" | "suppressed" | "hard_deleted"
        self._mid_state: dict[str, str] = {}
        self._counter = 0

    @initialize()
    def setup(self):
        self._tmp = Path(tempfile.mkdtemp(prefix="engram-fts-sm-"))
        cfg = Config(path=str(self._tmp))
        # Disable rate-limit so the machine can hammer freely.
        cfg.security.max_events_per_minute = 0
        # Dedup off — each remember() lands a row regardless of content.
        cfg.storage.write_dedup_threshold = 0.0
        self._engram = Engram(config=cfg)

    # ------------------------------------------------------------------
    # Rules
    # ------------------------------------------------------------------

    @rule(target=memory_ids, ti=st.integers(min_value=0, max_value=len(_TOKENS) - 1))
    def remember(self, ti):
        e = self._engram
        token = _TOKENS[ti]
        self._counter += 1
        # Embed the token plus a unique disambiguator so rows are not
        # text-identical (we still want F-I1 to MATCH on `token`).
        content = f"the {token} entry number {self._counter} stands alone"
        e.remember(content, salience=0.5)
        # Find the mid we just landed by searching for our unique counter.
        # Use store-level search to keep things deterministic.
        hits = e._store.search_text(f"{token}", limit=50, states=["active", "fading"])
        for h in hits:
            mid = h.memory.id
            if str(self._counter) in h.memory.content and mid not in self._mid_token:
                self._mid_token[mid] = token
                self._mid_state[mid] = "active"
                return mid
        return "noop"

    @rule(mid=memory_ids)
    def soft_forget(self, mid):
        if mid == "noop":
            return
        if self._mid_state.get(mid) != "active":
            return
        self._engram.forget(id=mid, hard=False)
        self._mid_state[mid] = "suppressed"

    @rule(mid=memory_ids)
    def hard_forget(self, mid):
        if mid == "noop":
            return
        if self._mid_state.get(mid) == "hard_deleted":
            return
        self._engram.forget(id=mid, hard=True)
        self._mid_state[mid] = "hard_deleted"

    # NOTE: We deliberately do NOT include a `rebuild()` rule. `rebuild()`
    # replays JSONL events into a fresh SQLite, regenerating memory ids,
    # which would invalidate every mid this state machine has tracked.
    # The same shrink-pathology argument that excluded rebuild from the
    # write-dedup machine applies here. F-I4 (rebuild idempotence on
    # FTS5) is pinned by `test_fts_rebuild_preserves_searchability_smoke`
    # below, which compares *content sets* rather than mid sets.

    # ------------------------------------------------------------------
    # Invariants
    # ------------------------------------------------------------------

    @invariant()
    def active_rows_are_searchable(self):
        # F-I1
        e = self._engram
        if e is None:
            return
        for mid, token in self._mid_token.items():
            if self._mid_state.get(mid) != "active":
                continue
            hits = e._store.search_text(token, limit=200, states=["active", "fading"])
            mids = {h.memory.id for h in hits}
            assert mid in mids, (
                f"F-I1 violated: active mid {mid} (token={token!r}) "
                f"not retrievable via FTS5 MATCH"
            )

    @invariant()
    def hard_deleted_invisible(self):
        # F-I2
        e = self._engram
        if e is None:
            return
        for mid, token in self._mid_token.items():
            if self._mid_state.get(mid) != "hard_deleted":
                continue
            # All states — even tombstones must not surface.
            for state_set in (
                ["active", "fading"],
                ["active", "fading", "faded", "suppressed"],
            ):
                hits = e._store.search_text(token, limit=200, states=state_set)
                mids = {h.memory.id for h in hits}
                assert mid not in mids, (
                    f"F-I2 violated: hard-deleted mid {mid} resurfaced "
                    f"in FTS5 with states={state_set}"
                )

    @invariant()
    def soft_deleted_not_in_default_search(self):
        # F-I3
        e = self._engram
        if e is None:
            return
        for mid, token in self._mid_token.items():
            if self._mid_state.get(mid) != "suppressed":
                continue
            hits = e._store.search_text(token, limit=200, states=["active", "fading"])
            mids = {h.memory.id for h in hits}
            assert mid not in mids, (
                f"F-I3 violated: soft-forgotten mid {mid} appeared in "
                f"default-state FTS5 search"
            )


FTSIndexTest = FTSIndexStateMachine.TestCase
FTSIndexTest.settings = settings(
    max_examples=40,
    stateful_step_count=25,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
)


# ---------------------------------------------------------------------------
# Closed-state pins — the invariants above expressed as flat regression tests.
# Useful when a shrink fails: these run fast and surface immediately.
# ---------------------------------------------------------------------------


def test_fts_active_row_searchable_smoke():
    """F-I1 baseline: a remembered row is FTS5-reachable by a unique token."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.storage.write_dedup_threshold = 0.0
        e = Engram(config=cfg)
        try:
            e.remember("the zappa entry stands alone")
            hits = e._store.search_text("zappa", limit=10, states=["active", "fading"])
            assert any("zappa" in h.memory.content for h in hits), (
                "active row not retrievable via FTS5 MATCH"
            )
        finally:
            e.close()


def test_fts_hard_delete_purges_from_index_smoke():
    """F-I2 baseline: hard_forget removes the row from FTS5."""
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.storage.write_dedup_threshold = 0.0
        e = Engram(config=cfg)
        try:
            e.remember("the qwerty entry stands alone")
            hits = e._store.search_text("qwerty", limit=10, states=["active", "fading"])
            assert hits, "precondition: row searchable before delete"
            mid = hits[0].memory.id
            e.forget(id=mid, hard=True)
            after = e._store.search_text(
                "qwerty", limit=10,
                states=["active", "fading", "faded", "suppressed"],
            )
            assert mid not in {h.memory.id for h in after}, (
                f"F-I2: hard-deleted mid {mid} still in FTS5"
            )
        finally:
            e.close()


def test_fts_rebuild_preserves_searchability_smoke():
    """F-I4 baseline: rebuild() reproduces the FTS5 index from events.

    We compare the *content set* reachable per token — `rebuild()`
    regenerates memory ids (it replays events into a fresh SQLite),
    so id-level comparison is the wrong projection. What matters is
    that for every distinctive token, exactly the same number of rows
    surface, and their content is identical.
    """
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.storage.write_dedup_threshold = 0.0
        e = Engram(config=cfg)
        try:
            e.remember("the fjord entry stands alone")
            e.remember("the krypton entry stands alone")
            before = {
                tok: sorted(
                    h.memory.content
                    for h in e._store.search_text(tok, limit=10, states=["active", "fading"])
                )
                for tok in ("fjord", "krypton")
            }
            assert all(len(v) == 1 for v in before.values()), before
            e.rebuild()
            after = {
                tok: sorted(
                    h.memory.content
                    for h in e._store.search_text(tok, limit=10, states=["active", "fading"])
                )
                for tok in ("fjord", "krypton")
            }
            assert before == after, (
                f"F-I4: rebuild altered FTS5-reachable content set:\n"
                f"  before={before}\n  after={after}"
            )
        finally:
            e.close()
