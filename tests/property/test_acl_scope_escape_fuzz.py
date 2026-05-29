"""ACL scope-string escape fuzz — close NEXT.md gap (paper §6 threat model).

The Grant.can_access() and AccessPolicy.check() codepaths gate cross-agent
reads on a string match against `scope`. The contract pinned here:

    S-1  Scope is fail-closed for every string except the two literals
         "own" and "*". Any other string (homoglyph asterisk, unicode
         confusable, NFKC near-miss, surrounding whitespace, mixed case,
         zero-width joiner, BIDI marks, NUL injection, "ALL"/"all"/"OWN")
         must result in `can_access(other_agent_id)` returning False.

    S-2  The literal "*" is the ONLY scope value that allows
         `can_access(memory_agent_id)` for a memory_agent_id != grantee.

    S-3  AccessPolicy.check() with a hostile-scope grant denies READ on a
         3rd-party memory_agent_id with PermissionError, unless the
         scope is exactly "*" or Permission.FEDERATED is also granted.

    S-4  No scope string ever causes can_access() or check() to raise
         anything other than PermissionError — the gate must be total.

Together these pin the surface paper §6 leans on: "a misconfigured /
adversarial scope string cannot escalate cross-agent read access".
"""
from __future__ import annotations

import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from engram.security.acl import AccessPolicy, Grant, Permission

pytestmark = pytest.mark.property


# Asterisk look-alikes and other "*"-confusable codepoints. None of these
# is the literal U+002A and none must satisfy the wildcard branch.
_ASTERISK_CONFUSABLES = [
    "\uFF0A",  # FULLWIDTH ASTERISK
    "\u2217",  # ASTERISK OPERATOR
    "\u204E",  # LOW ASTERISK
    "\u22C6",  # STAR OPERATOR
    "\u2731",  # HEAVY ASTERISK
    "\u2733",  # EIGHT SPOKED ASTERISK
    "\u2734",  # EIGHT POINTED BLACK STAR
    "\u2736",  # SIX POINTED BLACK STAR
    "\u2605",  # BLACK STAR
    "\u2606",  # WHITE STAR
    "\u066D",  # ARABIC FIVE POINTED STAR
    "**",
    " *",
    "* ",
    "*\u200B",
    "\u200B*",
    "*\u200D",
    "*\u00A0",
    "\uFEFF*",
    "*\u0000",
    "*\n",
    "*\t",
]

_OWN_CONFUSABLES = [
    "OWN",
    "Own",
    "oWn",
    " own",
    "own ",
    "own\u200B",
    "own\u0301",   # combining acute
    "оwn",         # leading Cyrillic 'o'
    "ｏｗｎ",        # fullwidth
    "own.",
    "own\u0000",
    "own\n",
]

_BENIGN_OTHER = [
    "",
    "world",
    "ALL",
    "all",
    "any",
    "everyone",
    "0",
    "false",
    "/",
    "*.read",
    "own:*",
    "*/own",
]


@st.composite
def hostile_scope(draw) -> str:
    """A scope string that must NOT satisfy the wildcard branch."""
    pool = _ASTERISK_CONFUSABLES + _OWN_CONFUSABLES + _BENIGN_OTHER
    base = draw(st.sampled_from(pool))
    # Optionally prepend/append additional unicode noise.
    noise = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Cf", "Mn", "Zs")),
            min_size=0,
            max_size=3,
        )
    )
    where = draw(st.sampled_from(["pre", "post", "none"]))
    if where == "pre":
        return noise + base
    if where == "post":
        return base + noise
    return base


_AGENT = st.text(
    alphabet=st.characters(min_codepoint=33, max_codepoint=126, blacklist_characters=':\\"'),
    min_size=1,
    max_size=12,
).filter(lambda s: s.strip())


# ---------------------------------------------------------------- S-1, S-2
@given(grantee=_AGENT, other=_AGENT, scope=hostile_scope())
@settings(max_examples=300, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hostile_scope_denies_cross_agent_read(grantee: str, other: str, scope: str):
    """S-1/S-2: only literal '*' opens cross-agent access."""
    if grantee == other:
        return
    if scope == "*":
        return  # hostile_scope() should never emit this — defensive.
    g = Grant(agent_id=grantee, permissions={Permission.READ}, scope=scope)
    assert g.can_access(other) is False, (
        f"scope {scope!r} unexpectedly satisfied wildcard branch for "
        f"grantee={grantee!r} other={other!r}"
    )


# ---------------------------------------------------------------- S-3
@given(grantee=_AGENT, other=_AGENT, scope=hostile_scope())
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hostile_scope_check_raises_permission_error(grantee: str, other: str, scope: str):
    """S-3: AccessPolicy.check() denies READ on a 3rd-party memory when the
    scope is anything other than literal '*' (and FEDERATED is off)."""
    if grantee == other or scope == "*":
        return
    policy = AccessPolicy(enabled=True)
    # Bypass AccessPolicy.grant() so we can install a hostile scope verbatim;
    # this is the lower-level gate the threat model cares about.
    policy._grants[grantee] = Grant(
        agent_id=grantee, permissions={Permission.READ}, scope=scope
    )
    with pytest.raises(PermissionError):
        policy.check(grantee, Permission.READ, memory_agent_id=other)


# ---------------------------------------------------------------- S-4
@given(scope=hostile_scope(), other=_AGENT)
@settings(max_examples=200, suppress_health_check=[HealthCheck.function_scoped_fixture])
def test_hostile_scope_never_raises_unexpected_exception(scope: str, other: str):
    """S-4: can_access is total — never raises (Type/Unicode/IndexError)."""
    g = Grant(agent_id="grantee", permissions={Permission.READ}, scope=scope)
    result = g.can_access(other)
    assert isinstance(result, bool)


# ---------------------------------------------------------------- S-2 positive
def test_literal_star_is_the_unique_wildcard():
    """Pin the positive case for S-2: literal '*' DOES grant cross-agent access."""
    g = Grant(agent_id="a", permissions={Permission.READ}, scope="*")
    assert g.can_access("b") is True
    assert g.can_access("") is True
    g2 = Grant(agent_id="a", permissions={Permission.READ}, scope="own")
    assert g2.can_access("a") is True
    assert g2.can_access("") is True
    assert g2.can_access("b") is False


# ---------------------------------------------------------------- S-3 federated escape hatch
def test_federated_permits_cross_agent_read_under_own_scope():
    """Pin the documented escape hatch: scope='own' + FEDERATED ⇒ cross-agent OK.
    (This is the inverse of S-3 — confirms the gate isn't over-aggressive.)"""
    policy = AccessPolicy(enabled=True)
    policy.grant("reviewer", {"read", "federated"}, scope="own")
    # Should NOT raise.
    policy.check("reviewer", Permission.READ, memory_agent_id="other-agent")
