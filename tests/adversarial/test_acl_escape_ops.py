"""Adversarial — ACL escape across **metadata-channel ops** (NEXT.md prio 2).

`test_acl_escape_breadth.py` already pins isolation across the *recall*
surfaces. This file closes the remaining gap: per-id metadata channels
that historically bypassed ACL READ enforcement —

    - ``Engram.get(memory_id)``           (returns full Memory)
    - ``Engram.provenance(memory_id)``    (returns lineage chain)
    - ``Engram.trace(memory_id)``         (returns full diagnostic dict)

Before this commit, all three resolved the memory by id and returned it
verbatim regardless of `self._actor` — i.e. an actor with ``scope='own'``
on its own grant could fetch any other agent's full memory record by
guessing or enumerating ids. Recall surfaces filtered correctly; these
metadata channels did not.

Invariants pinned (across N=8 agents, two reading APIs, three ops):

  AO-1  For every (reader, owner) pair with reader != owner under
        scope='own': ``Engram(actor=reader).get(owner_mem_id)`` is None.
  AO-2  Same for ``provenance(owner_mem_id)``.
  AO-3  Same for ``trace(owner_mem_id)``.
  AO-4  Self-access still works — the owning actor still gets its
        own data back from all three ops (no over-correction).
  AO-5  Hypothesis fuzz: arbitrary (reader, target_id) pairs over the
        full id pool. Reader sees data iff reader == target.agent_id.
  AO-6  ``get_properties`` (currently a public-ish channel via the
        store) is not promoted to a leak vector by these tests; if it
        is later wired through the engine, this test should be
        extended.

Cross-ref: this addresses the metadata-channel gap noted in NEXT.md
priority 2 ("extend to the metadata channel: get_properties, lifecycle
ops, schema queries"). schemas() returns SCHEMA-typed memories which
are aggregated and not per-agent-owned in the current model — separate
test would be needed if/when schemas become agent-scoped.
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from engram import Engram
from engram.core.config import Config
from engram.security.acl import Permission


N_AGENTS = 8
SECRETS = [
    ("alpha",   "alpha owns the red apricot"),
    ("bravo",   "bravo owns the blue blueberry"),
    ("charlie", "charlie owns the green cucumber"),
    ("delta",   "delta owns the yellow durian"),
    ("echo",    "echo owns the orange elderberry"),
    ("foxtrot", "foxtrot owns the purple fig"),
    ("golf",    "golf owns the white grape"),
    ("hotel",   "hotel owns the black huckleberry"),
]
assert len(SECRETS) == N_AGENTS


@pytest.fixture(scope="module")
def populated(tmp_path_factory):
    """One Engram, N agents under scope='own', each owning one secret.

    Yields (engine, [(owner, mem_id), ...]) so tests can target ids.
    """
    p = tmp_path_factory.mktemp("acl_ops")
    cfg = Config.minimal()
    cfg.path = str(p / "engram")
    # Bootstrap actor is the system; we'll override per-test by spawning
    # fresh Engram handles bound to different actors against the same
    # store path (sqlite handles concurrent process opens fine for our
    # single-tick read scenario).
    eng = Engram(cfg, actor=SECRETS[0][0])
    eng.acl._enabled = True
    for owner, _ in SECRETS:
        eng.acl.grant(owner, {Permission.READ, Permission.WRITE}, scope="own")
    owner_to_mem: dict[str, str] = {}
    for owner, secret in SECRETS:
        eng.remember(secret, agent_id=owner, salience=0.5)
    # Walk the store once and pick one memory per owner.
    for m in eng._store.all_active():
        if m.agent_id and m.agent_id in dict(SECRETS) and m.agent_id not in owner_to_mem:
            owner_to_mem[m.agent_id] = m.id
    assert len(owner_to_mem) == N_AGENTS, owner_to_mem
    yield eng, owner_to_mem, str(p / "engram")
    eng.close()


def _reader_engine(store_path: str, actor: str) -> Engram:
    """Fresh Engram handle bound to `actor` against the populated store."""
    cfg = Config.minimal()
    cfg.path = store_path
    e = Engram(cfg, actor=actor)
    e.acl._enabled = True
    # Re-grant scope='own' for every agent on this fresh handle (ACL is
    # not persisted; each handle starts empty).
    for owner, _ in SECRETS:
        e.acl.grant(owner, {Permission.READ, Permission.WRITE}, scope="own")
    return e


# ---------------------------------------------------------------------------
# AO-1, AO-2, AO-3: pairwise (reader != owner) MUST get None from each op.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("reader_idx", range(N_AGENTS))
def test_get_blocks_cross_agent(populated, reader_idx):
    eng, owner_to_mem, path = populated
    reader = SECRETS[reader_idx][0]
    e = _reader_engine(path, reader)
    try:
        for owner, mem_id in owner_to_mem.items():
            if owner == reader:
                continue
            assert e.get(mem_id) is None, (
                f"AO-1 leak: {reader!r}.get({mem_id!r}) returned a memory "
                f"owned by {owner!r}"
            )
    finally:
        e.close()


@pytest.mark.parametrize("reader_idx", range(N_AGENTS))
def test_provenance_blocks_cross_agent(populated, reader_idx):
    eng, owner_to_mem, path = populated
    reader = SECRETS[reader_idx][0]
    e = _reader_engine(path, reader)
    try:
        for owner, mem_id in owner_to_mem.items():
            if owner == reader:
                continue
            assert e.provenance(mem_id) is None, (
                f"AO-2 leak: {reader!r}.provenance({mem_id!r}) returned data "
                f"owned by {owner!r}"
            )
    finally:
        e.close()


@pytest.mark.parametrize("reader_idx", range(N_AGENTS))
def test_trace_blocks_cross_agent(populated, reader_idx):
    eng, owner_to_mem, path = populated
    reader = SECRETS[reader_idx][0]
    e = _reader_engine(path, reader)
    try:
        for owner, mem_id in owner_to_mem.items():
            if owner == reader:
                continue
            assert e.trace(mem_id) is None, (
                f"AO-3 leak: {reader!r}.trace({mem_id!r}) returned data "
                f"owned by {owner!r}"
            )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# AO-4: self-access still works — own owner can still get / provenance / trace.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("idx", range(N_AGENTS))
def test_self_access_still_works(populated, idx):
    eng, owner_to_mem, path = populated
    actor = SECRETS[idx][0]
    mem_id = owner_to_mem[actor]
    e = _reader_engine(path, actor)
    try:
        m = e.get(mem_id)
        assert m is not None and m.agent_id == actor, "self get() regressed"
        prov = e.provenance(mem_id)
        assert prov is not None and prov["memory_id"] == mem_id, \
            "self provenance() regressed"
        tr = e.trace(mem_id)
        assert tr is not None and tr["memory_id"] == mem_id, \
            "self trace() regressed"
    finally:
        e.close()


# ---------------------------------------------------------------------------
# AO-5: Hypothesis fuzz across (reader, mem_id) pairs.
# ---------------------------------------------------------------------------


_actor_names = [s[0] for s in SECRETS]


@settings(max_examples=200, deadline=None)
@given(
    reader=st.sampled_from(_actor_names),
    target_owner=st.sampled_from(_actor_names),
    op=st.sampled_from(["get", "provenance", "trace"]),
)
def test_fuzz_metadata_ops_isolation(populated, reader, target_owner, op):
    """For 200 random (reader, owner, op) triples: result MUST be None
    iff reader != owner; otherwise it MUST be non-None and reference the
    correct memory id."""
    eng, owner_to_mem, path = populated
    mem_id = owner_to_mem[target_owner]
    e = _reader_engine(path, reader)
    try:
        result = getattr(e, op)(mem_id)
        if reader != target_owner:
            assert result is None, (
                f"AO-5 fuzz leak: op={op!r} reader={reader!r} owner="
                f"{target_owner!r} mem_id={mem_id!r} returned non-None"
            )
        else:
            assert result is not None, (
                f"AO-5 self-access regression: op={op!r} actor={reader!r} "
                f"got None for own memory {mem_id!r}"
            )
    finally:
        e.close()


# ---------------------------------------------------------------------------
# AO-6: unknown / forged memory ids must return None (not crash, not leak).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("op", ["get", "provenance", "trace"])
@pytest.mark.parametrize("fake_id", [
    "",
    "mem-fa-deadbeefdeadbeef",
    "../../../etc/passwd",
    "' OR 1=1 --",
    "\x00",
    "🎉",
    "a" * 1024,
])
def test_metadata_ops_unknown_id(populated, op, fake_id):
    eng, _, path = populated
    e = _reader_engine(path, "alpha")
    try:
        out = getattr(e, op)(fake_id)
        assert out is None, f"{op}({fake_id!r}) leaked: {out!r}"
    finally:
        e.close()
