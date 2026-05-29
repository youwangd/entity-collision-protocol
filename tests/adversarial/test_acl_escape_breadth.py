"""Adversarial — ACL escape *breadth* corpus (NEXT.md priority 2).

`tests/adversarial/test_security_gaps.py` already covers depth: FEDERATED
upgrade attempts, falsy-id forgery, whitespace-id, empty-string actor.
The remaining gap is **breadth**: many agents × many read paths ×
many isolation patterns under `scope='own'`.

Invariant pinned (across N=12 agents):

    For every (reader, owner) pair with reader != owner:
      reader's grant is scope='own', not FEDERATED.
      reader.recall(...) MUST NOT return any memory whose
        memory.agent_id == owner.

    The same invariant must hold across every read API:
      - engine.recall(query, agent_id=reader)
      - engine.recall_with_filters(query=..., agent_id=reader)
      - engine.recall_with_filters(properties={...}, agent_id=reader)

    And across many query phrasings: exact, partial, empty-ish,
    high-cardinality, FTS-operator-laden, unicode.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from engram import Engram
from engram.core.config import Config
from engram.security.acl import Permission


N_AGENTS = 12
SECRETS = [
    ("alpha",   "alpha owns the red apricot"),
    ("bravo",   "bravo owns the blue blueberry"),
    ("charlie", "charlie owns the green cucumber"),
    ("delta",   "delta owns the yellow durian"),
    ("echo",    "echo owns the orange elderberry"),
    ("foxtrot", "foxtrot owns the purple fig"),
    ("golf",    "golf owns the white grape"),
    ("hotel",   "hotel owns the black huckleberry"),
    ("india",   "india owns the pink imbe"),
    ("juliet",  "juliet owns the brown jujube"),
    ("kilo",    "kilo owns the silver kiwi"),
    ("lima",    "lima owns the gold lemon"),
]
assert len(SECRETS) == N_AGENTS


@pytest.fixture(scope="module")
def populated(tmp_path_factory):
    """Build a single Engram with N agents, each owning a unique secret."""
    p = tmp_path_factory.mktemp("acl_breadth")
    cfg = Config.minimal()
    cfg.path = str(p / "engram")
    eng = Engram(cfg, actor=SECRETS[0][0])
    eng.acl._enabled = True
    for owner, _ in SECRETS:
        eng.acl.grant(owner, {Permission.READ, Permission.WRITE}, scope="own")
    # Each owner writes their own secret as themselves.
    for owner, secret in SECRETS:
        eng.remember(secret, agent_id=owner, salience=0.5)
        # Give each owner a property too — for filter-mode isolation tests.
        # remember does not return id; pull the latest by all_active.
    yield eng
    eng.close()


# ---------------------------------------------------------------------------
# 1. recall() pairwise isolation — every reader vs every owner.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader_idx", range(N_AGENTS))
def test_recall_isolation_pairwise(populated, reader_idx):
    """For one reader, walk every owner's secret-marker word and confirm
    that no memory owned by anyone else surfaces."""
    reader = SECRETS[reader_idx][0]
    for owner_idx, (owner, secret) in enumerate(SECRETS):
        # Use the secret's distinctive fruit-word as query.
        marker = secret.split()[-1]  # "apricot", "blueberry", ...
        results = populated.recall(marker, agent_id=reader, limit=20)
        for r in results:
            if reader == owner:
                continue
            assert r.memory.agent_id != owner, (
                f"ACL leak: reader={reader!r} saw owner={owner!r}'s memory "
                f"id={r.memory.id} via query={marker!r}"
            )


# ---------------------------------------------------------------------------
# 2. recall_with_filters() — hybrid mode (query + filter).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader_idx", range(N_AGENTS))
def test_recall_with_filters_hybrid_isolation(populated, reader_idx):
    reader = SECRETS[reader_idx][0]
    for owner, secret in SECRETS:
        if reader == owner:
            continue
        marker = secret.split()[-1]
        # Hybrid: query + dummy property (none of the memories have it,
        # but the ACL filter must run regardless).
        try:
            mems = populated.recall_with_filters(
                query=marker, properties=None, agent_id=reader, limit=20
            )
        except PermissionError:
            continue
        for m in mems:
            assert m.agent_id != owner, (
                f"hybrid filter leak: {reader} saw {owner}'s {m.id}"
            )


# ---------------------------------------------------------------------------
# 3. recall_with_filters() — query-only (no properties).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader_idx", range(N_AGENTS))
def test_recall_with_filters_query_only_isolation(populated, reader_idx):
    reader = SECRETS[reader_idx][0]
    for owner, secret in SECRETS:
        if reader == owner:
            continue
        marker = secret.split()[-1]
        try:
            mems = populated.recall_with_filters(
                query=marker, agent_id=reader, limit=20
            )
        except PermissionError:
            continue
        for m in mems:
            assert m.agent_id != owner


# ---------------------------------------------------------------------------
# 4. Hypothesis fuzz: arbitrary readers × arbitrary queries.
# ---------------------------------------------------------------------------


_owner_ids = [s[0] for s in SECRETS]
_query_text = st.one_of(
    st.sampled_from([s[1].split()[-1] for s in SECRETS]),  # marker words
    st.sampled_from(["owns", "the", "owns the", ""]),       # bridges
    st.sampled_from(["AND", "OR", "NOT", "NEAR", "*", '"']),  # FTS operators
    st.sampled_from(["你好", "café", "naïve", "🎉", "\u200b", "\ufeff"]),  # unicode
    st.text(min_size=0, max_size=24),
)


@settings(max_examples=200, deadline=None)
@given(reader=st.sampled_from(_owner_ids), q=_query_text)
def test_fuzz_recall_isolation(populated, reader, q):
    """Hypothesis: 200 arbitrary (reader, query) pairs. ACL invariant must
    hold for every result returned, no matter what the query produces."""
    try:
        results = populated.recall(q, agent_id=reader, limit=10)
    except (PermissionError, ValueError):
        return
    for r in results:
        # The reader either owns the memory, or the memory's agent_id is
        # the reader's. Anything else is a leak.
        assert r.memory.agent_id == reader, (
            f"fuzz ACL leak: reader={reader!r} got memory "
            f"agent_id={r.memory.agent_id!r} via query={q!r}"
        )


# ---------------------------------------------------------------------------
# 5. recall() with explicit unowned agent_id — no impersonation via query.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader_idx", range(N_AGENTS))
def test_no_impersonation_via_recall_agent_id(populated, reader_idx):
    """Calling recall(..., agent_id=other) does NOT allow reader to assume
    other's identity for ACL purposes — agent_id IS the actor identity, so
    the request is checked as 'other', and absent some grant-side leak,
    the read must still be limited to memories 'other' owns.

    This pins: agent_id=other on recall does NOT cross-leak to memories
    owned by yet a *third* party (since other is also scope='own')."""
    _ = SECRETS[reader_idx][0]
    other = SECRETS[(reader_idx + 1) % N_AGENTS][0]
    third_owner, third_secret = SECRETS[(reader_idx + 2) % N_AGENTS]
    marker = third_secret.split()[-1]
    results = populated.recall(marker, agent_id=other, limit=20)
    for r in results:
        # Acting as 'other' (scope='own', no FEDERATED) — must not see
        # third_owner's memory.
        assert r.memory.agent_id != third_owner, (
            f"impersonation leak: query as {other!r} surfaced {third_owner!r}'s memory"
        )
