"""Property tests for schema_family_evidence (E1-E5)."""
from __future__ import annotations

from hypothesis import given, settings, strategies as st

from engram.consolidation.schema_decision import (
    EvidenceWindow,
    Thresholds,
    decide,
)
from engram.consolidation.schema_family_decision import decide_with_family
from engram.consolidation.schema_family_evidence import (
    all_owner_siblings,
    siblings_for,
)
from engram.consolidation.schema_lifecycle import SchemaState, SchemaStatus


# Strategies ----------------------------------------------------------------

schema_id_st = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=4,
).map(lambda s: f"sch-{s}")


@st.composite
def clustering(draw):
    """Generate a (clusters, evidence) pair where clusters partitions a set
    of schema_ids and evidence is keyed on a (possibly different) subset.
    """
    sids = draw(st.lists(schema_id_st, unique=True, min_size=1, max_size=8))
    # Random partition into 1..len(sids) groups
    n_groups = draw(st.integers(min_value=1, max_value=len(sids)))
    group_assignment = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_groups - 1),
            min_size=len(sids),
            max_size=len(sids),
        )
    )
    buckets: dict[int, set[str]] = {}
    for sid, g in zip(sids, group_assignment):
        buckets.setdefault(g, set()).add(sid)
    clusters = tuple(sorted(
        (frozenset(b) for b in buckets.values()),
        key=lambda c: min(c),
    ))
    # Evidence over a (possibly proper) subset
    ev_keys = draw(st.lists(st.sampled_from(sids), unique=True, min_size=0, max_size=len(sids)))
    win = draw(st.text(min_size=1, max_size=6))
    evidence = {
        sid: EvidenceWindow(
            window_id=f"w-{win}",
            supports=draw(st.integers(min_value=0, max_value=10)),
            contradictions=draw(st.integers(min_value=0, max_value=10)),
        )
        for sid in ev_keys
    }
    return clusters, evidence


# E1: owner exclusion --------------------------------------------------------

@given(clustering())
@settings(max_examples=100, deadline=None)
def test_E1_owner_excluded(case):
    clusters, evidence = case
    for owner in evidence:
        sibs = siblings_for(owner, clusters, evidence)
        assert evidence[owner] not in sibs or all(
            id(s) != id(evidence[owner]) for s in sibs
        )
        # Stronger: no sibling EvidenceWindow object can be the owner's own.
        owner_ev = evidence[owner]
        for s in sibs:
            assert s is not owner_ev


# E2: singleton -> empty -----------------------------------------------------

@given(clustering())
@settings(max_examples=100, deadline=None)
def test_E2_singletons_empty(case):
    clusters, evidence = case
    for owner in evidence:
        cluster = next((c for c in clusters if owner in c), None)
        if cluster is not None and len(cluster) == 1:
            assert siblings_for(owner, clusters, evidence) == ()


# E3: deterministic order ----------------------------------------------------

@given(clustering())
@settings(max_examples=100, deadline=None)
def test_E3_deterministic_by_sibling_id(case):
    clusters, evidence = case
    for owner in evidence:
        cluster = next((c for c in clusters if owner in c), None)
        if cluster is None:
            continue
        sibs_ids = sorted(s for s in cluster if s != owner and s in evidence)
        expected = tuple(evidence[s] for s in sibs_ids)
        assert siblings_for(owner, clusters, evidence) == expected


# E4: regression-safe with decide_with_family at share=0 --------------------

@given(clustering())
@settings(max_examples=80, deadline=None)
def test_E4_share_zero_identity(case):
    clusters, evidence = case
    th = Thresholds()
    for owner, own_ev in evidence.items():
        state = SchemaState(
            schema_id=owner,
            status=SchemaStatus.INFERRED,
            version=1,
            last_window_id=own_ev.window_id,
        )
        sibs = siblings_for(owner, clusters, evidence)
        bare = decide(state, own_ev, th)
        with_fam = decide_with_family(state, own_ev, sibs, th, share=0.0)
        assert bare == with_fam


# E5: every sibling is a cluster-mate ---------------------------------------

@given(clustering())
@settings(max_examples=100, deadline=None)
def test_E5_partition_honored(case):
    clusters, evidence = case
    for owner in evidence:
        cluster = next((c for c in clusters if owner in c), None)
        if cluster is None:
            continue
        sibs_ids = {
            s for s in cluster if s != owner and s in evidence
        }
        # Reconstruct from the EvidenceWindow tuple by looking up identities
        sib_evs = siblings_for(owner, clusters, evidence)
        # Each sibling EvidenceWindow must be evidence[some sid in cluster].
        for s_ev in sib_evs:
            owners = [k for k, v in evidence.items() if v is s_ev]
            assert owners, "sibling EvidenceWindow not traceable to evidence map"
            assert any(k in sibs_ids for k in owners)


# Smoke / unit tests --------------------------------------------------------

def test_smoke_basic_two_member_cluster():
    a = EvidenceWindow(window_id="w1", supports=2, contradictions=0)
    b = EvidenceWindow(window_id="w1", supports=3, contradictions=1)
    clusters = (frozenset({"a", "b"}),)
    assert siblings_for("a", clusters, {"a": a, "b": b}) == (b,)
    assert siblings_for("b", clusters, {"a": a, "b": b}) == (a,)


def test_smoke_unclustered_returns_empty():
    a = EvidenceWindow(window_id="w1", supports=1, contradictions=0)
    # owner 'a' isn't in any cluster
    assert siblings_for("a", (), {"a": a}) == ()


def test_smoke_sibling_without_evidence_skipped():
    a = EvidenceWindow(window_id="w1", supports=1, contradictions=0)
    clusters = (frozenset({"a", "b", "c"}),)
    # only 'a' has evidence
    assert siblings_for("a", clusters, {"a": a}) == ()


def test_all_owner_siblings_bulk():
    a = EvidenceWindow(window_id="w1", supports=1, contradictions=0)
    b = EvidenceWindow(window_id="w1", supports=2, contradictions=0)
    c = EvidenceWindow(window_id="w1", supports=3, contradictions=0)
    clusters = (frozenset({"a", "b"}), frozenset({"c"}))
    out = all_owner_siblings(clusters, {"a": a, "b": b, "c": c})
    assert out == {"a": (b,), "b": (a,), "c": ()}


def test_smoke_share_one_with_one_sibling_doubles_when_evidence_matches():
    """E4-adjacent: share=1.0 with sibling=own -> effective doubled own."""
    own = EvidenceWindow(window_id="w1", supports=4, contradictions=1)
    sib = EvidenceWindow(window_id="w1", supports=4, contradictions=1)
    clusters = (frozenset({"a", "b"}),)
    sibs = siblings_for("a", clusters, {"a": own, "b": sib})
    state = SchemaState(
        schema_id="a",
        status=SchemaStatus.INFERRED,
        version=1,
        last_window_id="w1",
    )
    th = Thresholds()
    doubled = EvidenceWindow(window_id="w1", supports=8, contradictions=2)
    assert decide_with_family(state, own, sibs, th, share=1.0) == decide(
        state, doubled, th
    )
