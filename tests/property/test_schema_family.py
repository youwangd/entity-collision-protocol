"""Property tests for `engram.consolidation.schema_family`.

Locks invariants F1-F6 from the module docstring.
"""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from engram.consolidation.schema_family import (
    cluster,
    cluster_by_cooccurrence,
    jaccard,
)


# Strategies ---------------------------------------------------------------

prop_names = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),  # a-z
    min_size=1,
    max_size=4,
)

prop_set = st.frozensets(prop_names, max_size=6)

# At most ~8 schemas per case; n^2 clustering doesn't need stress here, the
# stress is on coverage of the metric and the partition logic.
schema_dict = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(min_codepoint=65, max_codepoint=90),  # A-Z
        min_size=1,
        max_size=3,
    ),
    values=prop_set,
    min_size=1,
    max_size=8,
)

tau_st = st.floats(min_value=0.0, max_value=1.0, allow_nan=False)


# F1: partition --------------------------------------------------------------


@given(schema_dict, tau_st)
@settings(max_examples=200, deadline=None)
def test_F1_partition(schemas, tau):
    out = cluster(schemas, tau=tau)
    flat = [sid for c in out for sid in c]
    assert sorted(flat) == sorted(schemas.keys())
    assert len(flat) == len(set(flat))  # no duplicates


# F2: tau=1.0 → only identical non-empty prop sets group ---------------------


@given(schema_dict)
@settings(max_examples=200, deadline=None)
def test_F2_tau_one_groups_only_identical(schemas):
    out = cluster(schemas, tau=1.0)
    for c in out:
        non_empty = [sid for sid in c if schemas[sid]]
        # All non-empty schemas in the cluster must have the same prop set
        if len(non_empty) > 1:
            sets = {frozenset(schemas[s]) for s in non_empty}
            assert len(sets) == 1


# F3: empty-prop schemas are singletons --------------------------------------


@given(schema_dict, tau_st)
@settings(max_examples=200, deadline=None)
def test_F3_empty_schemas_are_singletons(schemas, tau):
    out = cluster(schemas, tau=tau)
    for c in out:
        for sid in c:
            if not schemas[sid]:
                assert len(c) == 1, f"empty schema {sid} should be singleton, got {c}"


# F4: input order doesn't change output --------------------------------------


@given(schema_dict, tau_st)
@settings(max_examples=100, deadline=None)
def test_F4_canonical_order_invariant(schemas, tau):
    # Reverse insertion order
    reordered = dict(reversed(list(schemas.items())))
    assert cluster(schemas, tau=tau) == cluster(reordered, tau=tau)


# F6: reflexivity ------------------------------------------------------------


@given(schema_dict, tau_st)
@settings(max_examples=50, deadline=None)
def test_F6_every_schema_in_some_cluster(schemas, tau):
    out = cluster(schemas, tau=tau)
    seen = set().union(*out) if out else set()
    assert seen == set(schemas.keys())


# Concrete smoke tests -------------------------------------------------------


def test_smoke_identical_props_group_at_tau_1():
    schemas = {
        "A": frozenset({"name", "age"}),
        "B": frozenset({"name", "age"}),
        "C": frozenset({"name"}),
    }
    out = cluster(schemas, tau=1.0)
    # A and B together; C alone
    assert frozenset({"A", "B"}) in out
    assert frozenset({"C"}) in out
    assert len(out) == 2


def test_smoke_partial_overlap_tau_half():
    # |A∩B|/|A∪B| = 2/3 ≈ 0.67 → groups at tau=0.5
    schemas = {
        "A": frozenset({"x", "y", "z"}),
        "B": frozenset({"x", "y", "w"}),
        "C": frozenset({"q"}),
    }
    out = cluster(schemas, tau=0.5)
    assert frozenset({"A", "B"}) in out
    assert frozenset({"C"}) in out


def test_smoke_transitive_chain():
    # Single-link should chain across pairwise edges even when A↔C is 0.
    # A↔B = |{x,y}|/|{x,y,z}| = 2/3
    # B↔C = |{y,z}|/|{x,y,z,w}| = 2/4 = 0.5
    # A↔C = |{}|/|{x,y,z,w}| = 0
    schemas = {
        "A": frozenset({"x", "y"}),
        "B": frozenset({"x", "y", "z"}),
        "C": frozenset({"y", "z", "w"}),
    }
    out = cluster(schemas, tau=0.5)
    assert any({"A", "B", "C"} <= set(c) for c in out)


def test_jaccard_empty_convention():
    assert jaccard(frozenset(), frozenset({"a"})) == 0.0
    assert jaccard(frozenset({"a"}), frozenset()) == 0.0
    assert jaccard(frozenset(), frozenset()) == 0.0


def test_jaccard_disjoint_is_zero():
    assert jaccard(frozenset({"a"}), frozenset({"b"})) == 0.0


def test_jaccard_identical_is_one():
    assert jaccard(frozenset({"a", "b"}), frozenset({"a", "b"})) == 1.0


def test_invalid_tau():
    import pytest
    with pytest.raises(ValueError):
        cluster({"A": frozenset({"x"})}, tau=1.5)
    with pytest.raises(ValueError):
        cluster({"A": frozenset({"x"})}, tau=-0.1)


def test_F5_unrelated_schema_does_not_disturb():
    # Two perfectly identical schemas + one totally disjoint one
    base = {"A": frozenset({"x", "y"}), "B": frozenset({"x", "y"})}
    out_base = cluster(base, tau=0.5)
    extended = {**base, "Z": frozenset({"q", "r"})}
    out_ext = cluster(extended, tau=0.5)
    # A and B should still co-cluster
    ab_base = next(c for c in out_base if "A" in c)
    ab_ext = next(c for c in out_ext if "A" in c)
    assert ab_base == ab_ext
    # Z is its own cluster
    assert frozenset({"Z"}) in out_ext


# Co-occurrence-in-evidence-window invariants -------------------------------
#
# `cluster_by_cooccurrence` is the same single-link Jaccard core, just keyed
# on shared evidence windows instead of property names. F1-F6 still hold; we
# spot-check a few via the same generator, then add C1 (no shared windows ⇒
# all singletons) and a parity check vs `cluster`.

# Reuse prop_set as a generic frozenset[str] strategy for window ids.
window_dict = st.dictionaries(
    keys=st.text(
        alphabet=st.characters(min_codepoint=65, max_codepoint=90),  # A-Z
        min_size=1,
        max_size=3,
    ),
    values=prop_set,  # plays the role of \"set of window ids\"
    min_size=1,
    max_size=8,
)


@given(window_dict, tau_st)
@settings(max_examples=200, deadline=None)
def test_cooccur_F1_partition(membership, tau):
    out = cluster_by_cooccurrence(membership, tau=tau)
    flat = [sid for c in out for sid in c]
    assert sorted(flat) == sorted(membership.keys())
    assert len(flat) == len(set(flat))


@given(window_dict, tau_st)
@settings(max_examples=100, deadline=None)
def test_cooccur_F3_empty_membership_singletons(membership, tau):
    out = cluster_by_cooccurrence(membership, tau=tau)
    for c in out:
        for sid in c:
            if not membership[sid]:
                assert len(c) == 1


@given(window_dict, tau_st)
@settings(max_examples=100, deadline=None)
def test_cooccur_F4_canonical_order_invariant(membership, tau):
    reordered = dict(reversed(list(membership.items())))
    assert (
        cluster_by_cooccurrence(membership, tau=tau)
        == cluster_by_cooccurrence(reordered, tau=tau)
    )


@given(schema_dict, tau_st)
@settings(max_examples=100, deadline=None)
def test_cooccur_parity_with_cluster(schemas, tau):
    """The two functions are the same core; given identical feature sets
    they must produce identical clusters. This guards against accidental
    divergence if either is edited later.
    """
    assert cluster(schemas, tau=tau) == cluster_by_cooccurrence(schemas, tau=tau)


# C1: no shared windows ⇒ every schema a singleton at any tau in (0, 1].
def test_cooccur_C1_disjoint_windows_force_singletons():
    membership = {
        "A": frozenset({"w1"}),
        "B": frozenset({"w2"}),
        "C": frozenset({"w3", "w4"}),
        "D": frozenset({"w5", "w6", "w7"}),
    }
    for tau in (0.01, 0.25, 0.5, 0.75, 1.0):
        out = cluster_by_cooccurrence(membership, tau=tau)
        assert len(out) == 4, f"tau={tau} should give 4 singletons, got {out}"
        for c in out:
            assert len(c) == 1


def test_cooccur_smoke_shared_windows_group():
    # A and B share 2/3 of windows → group at tau=0.5
    membership = {
        "A": frozenset({"w1", "w2", "w3"}),
        "B": frozenset({"w1", "w2", "w4"}),
        "C": frozenset({"w9"}),
    }
    out = cluster_by_cooccurrence(membership, tau=0.5)
    assert frozenset({"A", "B"}) in out
    assert frozenset({"C"}) in out


def test_cooccur_window_id_relabeling_invariant():
    # Bijective relabeling of window ids must not change cluster structure.
    membership = {
        "A": frozenset({"w1", "w2"}),
        "B": frozenset({"w1", "w2"}),
        "C": frozenset({"w3"}),
    }
    relabeled = {
        "A": frozenset({"alpha", "beta"}),
        "B": frozenset({"alpha", "beta"}),
        "C": frozenset({"gamma"}),
    }
    assert (
        cluster_by_cooccurrence(membership, tau=0.5)
        == cluster_by_cooccurrence(relabeled, tau=0.5)
    )


def test_cooccur_invalid_tau():
    import pytest
    with pytest.raises(ValueError):
        cluster_by_cooccurrence({"A": frozenset({"w1"})}, tau=1.5)
    with pytest.raises(ValueError):
        cluster_by_cooccurrence({"A": frozenset({"w1"})}, tau=-0.1)
