"""Property tests for runtime cluster-contamination diagnostic."""
from __future__ import annotations

import math

import pytest
from hypothesis import HealthCheck, given, settings, strategies as st

from engram.consolidation.schema_family import cluster, jaccard
from engram.consolidation.schema_family_contamination import (
    cluster_contamination,
    contamination_rate,
    fragmentation_rate,
    min_within_jaccard,
)


# ---------- strategies ----------

_token = st.text(alphabet="abcdefgh", min_size=1, max_size=2)
_features = st.sets(_token, min_size=0, max_size=5).map(frozenset)


def _gen_features(draw, *, n_min=1, n_max=8):
    n = draw(st.integers(min_value=n_min, max_value=n_max))
    return {f"s{i}": draw(_features) for i in range(n)}


@st.composite
def features_strategy(draw):
    return _gen_features(draw)


# ---------- K1: singletons contribute zero weight ----------

def test_K1_singleton_contamination_is_zero():
    feats = {"a": frozenset({"x"})}
    assert cluster_contamination(feats, ["a"], tau=0.5) == 0.0


def test_K1_all_singletons_rate_zero():
    feats = {f"s{i}": frozenset({f"t{i}"}) for i in range(5)}
    clusters = tuple(frozenset({k}) for k in feats)
    assert contamination_rate(feats, clusters, tau=0.5) == 0.0


# ---------- K2: rate is in [0, 1] ----------

@given(features_strategy(), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
def test_K2_rate_in_unit_interval(feats, tau):
    clusters = cluster(feats, tau=tau)
    rate = contamination_rate(feats, clusters, tau)
    assert 0.0 <= rate <= 1.0


# ---------- K3: identical-feature clusters score 0 ----------

@given(_features.filter(lambda s: len(s) > 0), st.integers(min_value=2, max_value=6))
@settings(max_examples=50)
def test_K3_identical_features_zero_contamination(feat, n):
    feats = {f"s{i}": feat for i in range(n)}
    # All pairwise Jaccards are 1.0; any tau ≤ 1 ⇒ rate = 0.
    clusters = (frozenset(feats.keys()),)
    for tau in (0.0, 0.25, 0.5, 0.99, 1.0):
        assert contamination_rate(feats, clusters, tau) == 0.0


# ---------- K4: determinism / order independence ----------

@given(features_strategy(), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=100, suppress_health_check=[HealthCheck.too_slow])
def test_K4_determinism(feats, tau):
    clusters = cluster(feats, tau=tau)
    rate1 = contamination_rate(feats, clusters, tau)
    # Reverse iteration order
    feats_rev = dict(reversed(list(feats.items())))
    clusters_rev = cluster(feats_rev, tau=tau)
    rate2 = contamination_rate(feats_rev, clusters_rev, tau)
    assert rate1 == rate2


# ---------- K5: partition validation ----------

def test_K5_missing_schema_raises():
    feats = {"a": frozenset({"x"}), "b": frozenset({"y"})}
    with pytest.raises(ValueError, match="partition"):
        contamination_rate(feats, [frozenset({"a"})], tau=0.5)


def test_K5_extra_schema_raises():
    feats = {"a": frozenset({"x"})}
    with pytest.raises(ValueError, match="partition"):
        contamination_rate(feats, [frozenset({"a", "b"})], tau=0.5)


def test_K5_duplicate_schema_raises():
    feats = {"a": frozenset({"x"}), "b": frozenset({"y"})}
    with pytest.raises(ValueError, match="appears in"):
        contamination_rate(feats, [frozenset({"a"}), frozenset({"a", "b"})], tau=0.5)


# ---------- K6: clean cluster dilutes contaminated cluster ----------

def test_K6_clean_dilutes_contaminated():
    # Two tight pairs (zero contamination each), plus one transitive chain.
    # Tight: {a,b} share {x,y}; {c,d} share {p,q}.
    # Contaminated: {e,f,g} where e-f overlap, f-g overlap, but e-g disjoint.
    feats = {
        "a": frozenset({"x", "y"}),
        "b": frozenset({"x", "y"}),
        "c": frozenset({"p", "q"}),
        "d": frozenset({"p", "q"}),
        "e": frozenset({"m", "n"}),
        "f": frozenset({"m", "n", "o"}),
        "g": frozenset({"o", "r"}),  # disjoint from e
    }
    contaminated_only = (frozenset({"e", "f", "g"}),)
    contaminated_feats = {k: feats[k] for k in ("e", "f", "g")}
    rate_dirty = contamination_rate(contaminated_feats, contaminated_only, tau=0.5)
    assert rate_dirty > 0.0  # e-g pair fails

    clusters_all = (
        frozenset({"a", "b"}),
        frozenset({"c", "d"}),
        frozenset({"e", "f", "g"}),
    )
    rate_mixed = contamination_rate(feats, clusters_all, tau=0.5)
    assert rate_mixed < rate_dirty
    # Math: 5 pairs total — (a,b)=1.0, (c,d)=1.0, (e,f)=2/3, (e,g)=0, (f,g)=1/4.
    # At tau=0.5: 2 below (e,g and f,g) → 2/5.
    assert math.isclose(rate_mixed, 2 / 5, rel_tol=1e-9)
    assert math.isclose(rate_dirty, 2 / 3, rel_tol=1e-9)


# ---------- K7: empty cluster raises ----------

def test_K7_empty_cluster_raises_in_helpers():
    with pytest.raises(ValueError):
        cluster_contamination({}, [], tau=0.5)
    with pytest.raises(ValueError):
        min_within_jaccard({}, [])


# ---------- min_within_jaccard semantics ----------

def test_min_within_singleton_is_one():
    feats = {"a": frozenset({"x"})}
    assert min_within_jaccard(feats, ["a"]) == 1.0


def test_min_within_finds_floor():
    feats = {
        "a": frozenset({"x", "y"}),
        "b": frozenset({"x", "y"}),       # 1.0 vs a
        "c": frozenset({"y", "z"}),       # 1/3 vs a, 1/3 vs b
    }
    floor = min_within_jaccard(feats, ["a", "b", "c"])
    # min over (a,b)=1.0, (a,c)=1/3, (b,c)=1/3 → 1/3
    assert math.isclose(floor, 1 / 3, rel_tol=1e-9)


# ---------- §69 deployment rule: clean cluster ⇒ rate=0 ⇒ share=0.75 safe ----------

def test_clean_cluster_passes_69_rule():
    """Tight cluster: every pair Jaccard ≥ tau directly. Rate = 0."""
    feats = {
        "s0": frozenset({"a", "b", "c"}),
        "s1": frozenset({"a", "b", "c", "d"}),  # J=3/4 vs s0
        "s2": frozenset({"a", "b", "d"}),       # J=2/4 vs s0, 3/4 vs s1
    }
    clusters = cluster(feats, tau=0.5)
    rate = contamination_rate(feats, clusters, tau=0.5)
    # All three pairs ≥ 0.5: 3/4, 1/2, 3/4 → rate=0.
    assert rate == 0.0
    # §69 rule: rate ≤ 0.10 ⇒ share=0.75 cleared.
    assert rate <= 0.10


def test_transitive_cluster_fails_69_rule_at_high_tau():
    """Chain held by single-link only: e-f ok, f-g ok, e-g not. Rate>0."""
    feats = {
        "e": frozenset({"m", "n"}),
        "f": frozenset({"m", "n", "o"}),  # J(e,f)=2/3
        "g": frozenset({"o"}),            # J(f,g)=1/3, J(e,g)=0
    }
    # tau=0.3 picks up e-f and f-g but not e-g; single-link unions all three.
    clusters = cluster(feats, tau=0.3)
    assert len(clusters) == 1 and clusters[0] == frozenset({"e", "f", "g"})
    rate = contamination_rate(feats, clusters, tau=0.3)
    # 1 of 3 pairs (e-g, J=0) below tau → rate = 1/3.
    assert math.isclose(rate, 1 / 3, rel_tol=1e-9)
    assert rate > 0.10  # §69 rule: would reject share=0.75 here


# ---------- jaccard sanity (already tested elsewhere; spot check) ----------

def test_jaccard_alignment():
    a = frozenset({"x", "y", "z"})
    b = frozenset({"x", "y"})
    assert math.isclose(jaccard(a, b), 2 / 3, rel_tol=1e-9)


# ---------- F1-F5: fragmentation_rate invariants (§74 companion meter) ----------


def test_F1_all_singletons_frag_one():
    feats = {f"s{i}": frozenset({f"t{i}"}) for i in range(5)}
    clusters = [frozenset({f"s{i}"}) for i in range(5)]
    assert fragmentation_rate(feats, clusters) == 1.0


def test_F2_one_big_cluster_frag_zero():
    feats = {f"s{i}": frozenset({"a", "b"}) for i in range(4)}
    clusters = [frozenset({"s0", "s1", "s2", "s3"})]
    assert fragmentation_rate(feats, clusters) == 0.0


def test_F2_empty_features_frag_zero():
    assert fragmentation_rate({}, []) == 0.0


@settings(suppress_health_check=[HealthCheck.too_slow], deadline=None, max_examples=100)
@given(features_strategy())
def test_F3_range_in_unit_interval(feats):
    clusters = cluster(feats, tau=0.5)
    f = fragmentation_rate(feats, clusters)
    assert 0.0 <= f <= 1.0


def test_F4_partition_mismatch_raises():
    feats = {"a": frozenset({"x"}), "b": frozenset({"y"})}
    with pytest.raises(ValueError):
        fragmentation_rate(feats, [frozenset({"a"})])  # missing b


def test_F5_order_independent():
    feats = {f"s{i}": frozenset({"a"}) if i < 2 else frozenset({chr(98 + i)})
             for i in range(5)}
    c_forward = [frozenset({"s0", "s1"}), frozenset({"s2"}),
                 frozenset({"s3"}), frozenset({"s4"})]
    c_shuffled = list(reversed(c_forward))
    assert fragmentation_rate(feats, c_forward) == fragmentation_rate(feats, c_shuffled)


def test_F_intermediate_count():
    """3 singletons + 1 pair from 5 schemas → 3/5."""
    feats = {f"s{i}": frozenset({f"t{i}"}) for i in range(5)}
    clusters = [
        frozenset({"s0", "s1"}),
        frozenset({"s2"}),
        frozenset({"s3"}),
        frozenset({"s4"}),
    ]
    assert fragmentation_rate(feats, clusters) == 3 / 5
