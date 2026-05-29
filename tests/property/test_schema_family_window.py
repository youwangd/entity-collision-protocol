"""Property tests for schema_family_window.decide_window (W1-W6)."""
from __future__ import annotations

import pytest
from hypothesis import given, settings, strategies as st

from engram.consolidation.schema_decision import (
    EvidenceWindow,
    Thresholds,
    decide,
)
from engram.consolidation.schema_family_window import decide_window
from engram.consolidation.schema_lifecycle import SchemaState, SchemaStatus


schema_id_st = st.text(
    alphabet=st.characters(min_codepoint=97, max_codepoint=122),
    min_size=1,
    max_size=4,
).map(lambda s: f"sch-{s}")


@st.composite
def world(draw, with_clusters: bool = True):
    sids = draw(st.lists(schema_id_st, unique=True, min_size=1, max_size=6))
    win = "w-" + draw(st.text(min_size=1, max_size=4))
    states = {
        sid: SchemaState(
            schema_id=sid,
            status=draw(st.sampled_from(list(SchemaStatus))),
            version=1,
            last_window_id=win,
        )
        for sid in sids
    }
    evidence = {
        sid: EvidenceWindow(
            window_id=win,
            supports=draw(st.integers(min_value=0, max_value=8)),
            contradictions=draw(st.integers(min_value=0, max_value=8)),
        )
        for sid in sids
    }
    if not with_clusters:
        return states, evidence, ()
    n_groups = draw(st.integers(min_value=1, max_value=len(sids)))
    assignment = draw(
        st.lists(
            st.integers(min_value=0, max_value=n_groups - 1),
            min_size=len(sids),
            max_size=len(sids),
        )
    )
    buckets: dict[int, set[str]] = {}
    for sid, g in zip(sids, assignment):
        buckets.setdefault(g, set()).add(sid)
    clusters = tuple(frozenset(b) for b in buckets.values())
    return states, evidence, clusters


# ---------------------------------------------------------------------------
# Properties
# ---------------------------------------------------------------------------


@given(world())
@settings(max_examples=200, deadline=None)
def test_w1_share_zero_matches_bare_decide(args) -> None:
    """W1: share=0 ⇒ key-by-key identical to bare decide()."""
    states, evidence, clusters = args
    th = Thresholds()
    got = decide_window(states, evidence, clusters, thresholds=th, share=0.0)
    expected = {sid: decide(states[sid], evidence[sid], th) for sid in evidence}
    assert got == expected


@given(world(with_clusters=False), st.floats(min_value=0.0, max_value=1.0))
@settings(max_examples=100, deadline=None)
def test_w2_no_clusters_share_invariant(args, share) -> None:
    """W2: empty clusters ⇒ same as bare decide() for any share."""
    states, evidence, clusters = args
    th = Thresholds()
    got = decide_window(states, evidence, clusters, thresholds=th, share=share)
    expected = {sid: decide(states[sid], evidence[sid], th) for sid in evidence}
    assert got == expected


@given(world())
@settings(max_examples=100, deadline=None)
def test_w3_determinism_under_iteration_order(args) -> None:
    """W3: shuffled-input dicts produce identical output (incl. order)."""
    states, evidence, clusters = args
    a = decide_window(states, evidence, clusters, share=0.5)
    # Reverse insertion order
    states2 = dict(reversed(list(states.items())))
    evidence2 = dict(reversed(list(evidence.items())))
    b = decide_window(states2, evidence2, clusters, share=0.5)
    assert a == b
    assert list(a.keys()) == list(b.keys())  # same iteration order


@given(world())
@settings(max_examples=100, deadline=None)
def test_w4_keys_are_evidence_keys(args) -> None:
    """W4: result keys == evidence keys, regardless of states or clusters."""
    states, evidence, clusters = args
    got = decide_window(states, evidence, clusters, share=0.5)
    assert set(got.keys()) == set(evidence.keys())


def test_w5_missing_state_raises() -> None:
    """W5: evidenced schema with no state → KeyError(schema_id)."""
    ev = {"sch-a": EvidenceWindow(window_id="w", supports=1, contradictions=0)}
    with pytest.raises(KeyError) as exc_info:
        decide_window(states_by_schema={}, evidence_by_schema=ev)
    assert "sch-a" in str(exc_info.value)


def test_w6_share_out_of_range_raises() -> None:
    """W6: share outside [0, 1] raises ValueError."""
    states = {
        "sch-a": SchemaState(
            schema_id="sch-a",
            status=SchemaStatus.INFERRED,
            version=1,
            last_window_id="w",
        )
    }
    ev = {"sch-a": EvidenceWindow(window_id="w", supports=1, contradictions=0)}
    with pytest.raises(ValueError):
        decide_window(states, ev, share=-0.1)
    with pytest.raises(ValueError):
        decide_window(states, ev, share=1.5)


# Smoke tests ---------------------------------------------------------------


def test_smoke_share_promotes_via_siblings() -> None:
    """End-to-end §8: an INFERRED owner with insufficient own-supports gets
    PROMOTED when share=1.0 borrows enough from a sibling."""
    th = Thresholds(promote=3, deprecate=2, recover=3)
    states = {
        "sch-a": SchemaState(
            schema_id="sch-a",
            status=SchemaStatus.INFERRED,
            version=1,
            last_window_id="w1",
        ),
        "sch-b": SchemaState(
            schema_id="sch-b",
            status=SchemaStatus.INFERRED,
            version=1,
            last_window_id="w1",
        ),
    }
    evidence = {
        "sch-a": EvidenceWindow(window_id="w1", supports=1, contradictions=0),
        "sch-b": EvidenceWindow(window_id="w1", supports=3, contradictions=0),
    }
    clusters = (frozenset({"sch-a", "sch-b"}),)

    # share=0 → no promotion (a only has 1, threshold 3)
    no_share = decide_window(states, evidence, clusters, th, share=0.0)
    assert no_share["sch-a"] is None

    # share=1 → a borrows b's 3 supports, total 4 ≥ 3 → PROMOTE
    full_share = decide_window(states, evidence, clusters, th, share=1.0)
    from engram.consolidation.schema_lifecycle import EventKind
    assert full_share["sch-a"] == EventKind.PROMOTE


def test_smoke_singleton_cluster_no_borrowing() -> None:
    """Singleton-cluster schemas behave exactly like bare decide()."""
    th = Thresholds()
    states = {
        "sch-a": SchemaState(
            schema_id="sch-a",
            status=SchemaStatus.INFERRED,
            version=1,
            last_window_id="w1",
        ),
    }
    evidence = {
        "sch-a": EvidenceWindow(window_id="w1", supports=5, contradictions=0),
    }
    clusters = (frozenset({"sch-a"}),)
    got = decide_window(states, evidence, clusters, th, share=1.0)
    assert got == {"sch-a": decide(states["sch-a"], evidence["sch-a"], th)}
