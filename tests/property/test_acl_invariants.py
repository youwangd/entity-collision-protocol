"""Property-based invariants for the ACL / AccessPolicy layer.

Mission item 2a (governed-memory v0.2 testing framework) — close the gap
flagged in NEXT.md: ACL transitivity, scope semantics, permission lattice.

Invariants under test:

  I1. ACL disabled ⇒ all checks pass (regardless of grants).
  I2. No grant ⇒ any check raises PermissionError.
  I3. ADMIN permission ⇒ has() returns True for every Permission value.
  I4. Permission lattice has no implicit upgrades: WRITE alone does not
      satisfy READ; READ alone does not satisfy WRITE; etc. (only ADMIN
      is the wildcard.)
  I5. scope='own' ⇒ READ allowed iff (memory_agent_id == agent_id) OR
      (memory_agent_id == "")  OR  Permission.FEDERATED is held.
  I6. scope='*' ⇒ READ allowed for every memory_agent_id (given READ perm).
  I7. revoke is idempotent and undoes grant (post-revoke ≡ no-grant).
  I8. to_dict / from_dict round-trips losslessly (grants, scope, enabled).

These are *runtime* invariants — they encode the contract the rest of the
system depends on. A regression here is a security regression.
"""

from __future__ import annotations

import pytest
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st

from engram.security.acl import AccessPolicy, Grant, Permission

# All non-ADMIN permissions (ADMIN is the wildcard, tested separately).
_NON_ADMIN_PERMS = [p for p in Permission if p is not Permission.ADMIN]

# Strategy: a non-empty subset of non-admin permissions.
_perm_subset = st.sets(st.sampled_from(_NON_ADMIN_PERMS), min_size=1, max_size=len(_NON_ADMIN_PERMS))

_agent_id = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126, blacklist_characters=":\\\""),
    min_size=1,
    max_size=24,
).filter(lambda s: s.strip())

_scope = st.sampled_from(["own", "*"])


# ---------------------------------------------------------------- I1
@given(_agent_id, st.sampled_from(list(Permission)), _agent_id)
@settings(max_examples=50, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_disabled_policy_never_raises(agent_id: str, perm: Permission, mem_agent_id: str):
    """I1: disabled policy is a no-op for every check."""
    policy = AccessPolicy(enabled=False)
    # No grants, but disabled ⇒ must not raise.
    policy.check(agent_id, perm, memory_agent_id=mem_agent_id)


# ---------------------------------------------------------------- I2
@given(_agent_id, st.sampled_from(list(Permission)))
@settings(max_examples=50)
def test_no_grant_always_denies_when_enabled(agent_id: str, perm: Permission):
    """I2: enabled policy with no grants denies every operation."""
    policy = AccessPolicy(enabled=True)
    with pytest.raises(PermissionError, match="no access grants"):
        policy.check(agent_id, perm)


# ---------------------------------------------------------------- I3
@given(_agent_id, _scope, st.sampled_from(list(Permission)))
@settings(max_examples=50)
def test_admin_grant_satisfies_every_permission(agent_id: str, scope: str, perm: Permission):
    """I3: ADMIN is the wildcard — Grant.has(p) is True for every p."""
    g = Grant(agent_id=agent_id, permissions={Permission.ADMIN}, scope=scope)
    assert g.has(perm) is True


# ---------------------------------------------------------------- I4
@given(_agent_id, _perm_subset, st.sampled_from(_NON_ADMIN_PERMS))
@settings(max_examples=100)
def test_permission_lattice_no_implicit_upgrade(
    agent_id: str, perms: set[Permission], probe: Permission
):
    """I4: holding perms ⊆ P does NOT imply holding any p ∉ P (no upgrades).

    Only ADMIN is wildcard. Excluding ADMIN, has(p) ⇔ p ∈ perms.
    """
    assume(Permission.ADMIN not in perms)
    g = Grant(agent_id=agent_id, permissions=perms, scope="own")
    assert g.has(probe) == (probe in perms)


# ---------------------------------------------------------------- I5
@given(_agent_id, _agent_id)
@settings(max_examples=80)
def test_scope_own_read_semantics(agent_id: str, mem_agent_id: str):
    """I5: scope='own' READ allowed iff same-agent or anonymous owner.

    With FEDERATED present, cross-agent reads must succeed.
    Without FEDERATED, cross-agent reads must raise.
    """
    assume(agent_id != mem_agent_id and mem_agent_id != "")
    policy = AccessPolicy(enabled=True)
    policy.grant(agent_id, {"read"}, scope="own")

    # cross-agent read must raise (no federated).
    with pytest.raises(PermissionError):
        policy.check(agent_id, Permission.READ, memory_agent_id=mem_agent_id)

    # same-agent must succeed.
    policy.check(agent_id, Permission.READ, memory_agent_id=agent_id)
    # anonymous owner ('') must succeed under 'own'.
    policy.check(agent_id, Permission.READ, memory_agent_id="")

    # add FEDERATED ⇒ cross-agent read now allowed.
    policy.grant(agent_id, {"read", "federated"}, scope="own")
    policy.check(agent_id, Permission.READ, memory_agent_id=mem_agent_id)


# ---------------------------------------------------------------- I6
@given(_agent_id, _agent_id)
@settings(max_examples=50)
def test_scope_star_allows_all_memory_owners(agent_id: str, mem_agent_id: str):
    """I6: scope='*' with READ permits every memory_agent_id."""
    policy = AccessPolicy(enabled=True)
    policy.grant(agent_id, {"read"}, scope="*")
    policy.check(agent_id, Permission.READ, memory_agent_id=mem_agent_id)


# ---------------------------------------------------------------- I7
@given(_agent_id, _perm_subset, _scope)
@settings(max_examples=50)
def test_revoke_is_idempotent_and_undoes_grant(
    agent_id: str, perms: set[Permission], scope: str
):
    """I7: post-revoke state is observationally equal to no-grant state."""
    policy = AccessPolicy(enabled=True)
    policy.grant(agent_id, perms, scope=scope)
    policy.revoke(agent_id)
    # second revoke is a no-op (idempotent).
    policy.revoke(agent_id)
    # any check denies, just like a fresh policy.
    with pytest.raises(PermissionError, match="no access grants"):
        policy.check(agent_id, next(iter(perms)))


# ---------------------------------------------------------------- I8
@given(
    st.lists(
        st.tuples(_agent_id, _perm_subset, _scope),
        min_size=0,
        max_size=6,
        unique_by=lambda t: t[0],
    ),
    st.booleans(),
)
@settings(max_examples=40)
def test_to_dict_from_dict_roundtrip(grants: list, enabled: bool):
    """I8: serialization round-trip is lossless."""
    p1 = AccessPolicy(enabled=enabled)
    for agent_id, perms, scope in grants:
        p1.grant(agent_id, {x.value for x in perms}, scope=scope)

    p2 = AccessPolicy.from_dict(p1.to_dict())

    assert p2.enabled == p1.enabled
    assert p2.to_dict() == p1.to_dict()
    # behavioural equivalence: every (agent, perm) check has the same outcome.
    if enabled:
        for agent_id, perms, _scope in grants:
            for perm in perms:
                # both must accept (with self-owned memory).
                p1.check(agent_id, perm, memory_agent_id=agent_id)
                p2.check(agent_id, perm, memory_agent_id=agent_id)


# ---------------------------------------------------------------- I9
@given(_agent_id, _perm_subset, _perm_subset, _scope, _scope)
@settings(max_examples=80)
def test_regrant_replaces_prior_state(
    agent_id: str,
    perms_a: set[Permission],
    perms_b: set[Permission],
    scope_a: str,
    scope_b: str,
):
    """I9: re-grant fully REPLACES prior grant — no merge of perms or scope.

    Security-critical: a downgrade (e.g., admin→read) must not leak the old
    permissions, and a scope tightening ('*'→'own') must not leak the old scope.
    """
    assume(Permission.ADMIN not in perms_a and Permission.ADMIN not in perms_b)
    policy = AccessPolicy(enabled=True)
    policy.grant(agent_id, perms_a, scope=scope_a)
    policy.grant(agent_id, perms_b, scope=scope_b)

    # Equivalent to a fresh policy with only the second grant.
    ref = AccessPolicy(enabled=True)
    ref.grant(agent_id, perms_b, scope=scope_b)
    assert policy.to_dict() == ref.to_dict()

    # Behavioural: any perm in (perms_a − perms_b) must now be denied.
    for leaked in perms_a - perms_b:
        with pytest.raises(PermissionError):
            policy.check(agent_id, leaked, memory_agent_id=agent_id)


# ---------------------------------------------------------------- I10
@given(_agent_id, _agent_id, _perm_subset)
@settings(max_examples=80)
def test_cross_agent_grant_isolation(
    agent_a: str, agent_b: str, perms: set[Permission]
):
    """I10: granting to agent_A confers nothing on agent_B.

    Multi-namespace tenancy: each agent_id is an isolated namespace; no
    "shared bucket" semantics, no implicit propagation.
    """
    assume(agent_a != agent_b)
    assume(Permission.ADMIN not in perms)
    policy = AccessPolicy(enabled=True)
    policy.grant(agent_a, perms, scope="*")

    # agent_b has no grant ⇒ every check must raise.
    for p in perms:
        with pytest.raises(PermissionError, match="no access grants"):
            policy.check(agent_b, p, memory_agent_id=agent_b)

    # And revoking agent_a does not affect a (still-absent) agent_b.
    policy.revoke(agent_a)
    with pytest.raises(PermissionError, match="no access grants"):
        policy.check(agent_b, next(iter(perms)))


# ---------------------------------------------------------------- I11 (concurrency)
def test_concurrent_grant_revoke_no_torn_state():
    """I11: concurrent grant/revoke from many threads leaves consistent state.

    Not Hypothesis (threading + Hypothesis → flaky). Deterministic stress
    test that exercises the in-memory dict against the GIL: 16 threads ×
    200 ops each. Final state must be a valid AccessPolicy (no half-written
    Grant, no exception during check), and serialization must round-trip.
    """
    import threading

    policy = AccessPolicy(enabled=True)
    agents = [f"agent-{i}" for i in range(8)]
    barrier = threading.Barrier(16)
    errors: list[BaseException] = []

    def worker(seed: int):
        import random

        rng = random.Random(seed)
        try:
            barrier.wait()
            for _ in range(200):
                a = rng.choice(agents)
                op = rng.random()
                if op < 0.45:
                    perms = rng.sample(_NON_ADMIN_PERMS, k=rng.randint(1, 3))
                    policy.grant(a, set(perms), scope=rng.choice(["own", "*"]))
                elif op < 0.65:
                    policy.revoke(a)
                else:
                    # check should never raise anything other than PermissionError.
                    try:
                        policy.check(a, Permission.READ, memory_agent_id=a)
                    except PermissionError:
                        pass
        except BaseException as e:  # pragma: no cover — diagnostic
            errors.append(e)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert not errors, f"worker raised non-PermissionError: {errors[:3]}"

    # Final state must serialize and round-trip cleanly.
    snapshot = policy.to_dict()
    rebuilt = AccessPolicy.from_dict(snapshot)
    assert rebuilt.to_dict() == snapshot

    # Every surviving grant must have a valid (non-empty) perm set and valid scope.
    for entry in policy.list_grants():
        assert entry["permissions"], f"empty perm set for {entry['agent_id']}"
        assert entry["scope"] in ("own", "*"), f"bad scope: {entry}"
