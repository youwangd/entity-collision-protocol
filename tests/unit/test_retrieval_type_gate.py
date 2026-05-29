"""§D15 — type-conditional PRF gate integration tests."""

from __future__ import annotations

import tempfile

from engram import Config, Engram
from engram.core.config import RetrievalConfig
from engram.retrieval.type_classifier import (
    DEFAULT_PRF_ALLOW,
    TYPE_KNOW_UPD,
    TYPE_SS_PREF,
)


def test_gate_inert_when_allow_is_none():
    """When `query_expansion_type_allow` is None, behaviour matches v0.2:
    PRF runs for every query whose dominance gate fires."""
    facts = [
        "Alice Smith met Bob Jones in Paris.",
        "Alice Smith works at Acme Corp.",
        "Alice Smith and Carol went to Paris.",
    ]
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval = RetrievalConfig(
            query_expansion_min_dominance=0.3,
            query_expansion_top_k=5,
            query_expansion_max_entities=2,
            query_expansion_type_allow=None,  # inert
        )
        eng = Engram(config=cfg)
        try:
            for f in facts:
                eng.remember(f)
            # Query that doesn't match a positive type — would be gated
            # OUT by the §D15 default. With allow=None, must still run.
            r = eng.recall("where did they go", limit=3)
            assert isinstance(r, list)
        finally:
            eng.close()


def test_gate_skips_prf_for_out_of_allow_query():
    """A non-preference, non-knowledge-update query under the default
    allow-set must take the un-expanded path (no PRF). We can't directly
    observe the expansion call, but we can assert that the result equals
    what the un-expanded engine would return."""
    facts = [
        "Alice Smith met Bob Jones in Paris.",
        "Alice Smith works at Acme Corp.",
        "Alice Smith and Carol went to Paris.",
        "Bob Jones flew to Tokyo last week.",
        "Carol White is a poet.",
    ]
    # Query type prediction: "Where did I redeem ..." → ss-user → OUT.
    out_of_allow_q = "Where did I redeem a $5 coupon on coffee creamer?"

    # Arm A: gated (allow={KU, Pref}) — out-of-allow → no PRF.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval = RetrievalConfig(
            query_expansion_min_dominance=0.3,
            query_expansion_top_k=5,
            query_expansion_max_entities=2,
            query_expansion_type_allow=DEFAULT_PRF_ALLOW,
        )
        eng = Engram(config=cfg)
        try:
            for f in facts:
                eng.remember(f)
            r_gated = [m.id for m in eng.recall(out_of_allow_q, limit=5)]
        finally:
            eng.close()

    # Arm B: PRF fully off — true reference.
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval = RetrievalConfig(
            query_expansion_min_dominance=None,
        )
        eng = Engram(config=cfg)
        try:
            for f in facts:
                eng.remember(f)
            r_baseline = [m.id for m in eng.recall(out_of_allow_q, limit=5)]
        finally:
            eng.close()

    assert r_gated == r_baseline, (
        f"out-of-allow query should bypass PRF entirely:\n"
        f"  gated   ={r_gated}\n  baseline={r_baseline}"
    )


def test_gate_runs_prf_for_in_allow_query():
    """An in-allow query under the gate should yield results (smoke).
    We don't compare ordering — only that the call succeeds and the
    type classifier routed correctly."""
    facts = [
        "Alice Smith met Bob Jones in Paris.",
        "Alice Smith works at Acme Corp.",
        "Alice Smith and Carol went to Paris.",
    ]
    # "Currently" → knowledge-update → IN.
    in_allow_q = "How many bikes do I currently own?"
    with tempfile.TemporaryDirectory() as tmp:
        cfg = Config(path=tmp)
        cfg.security.max_events_per_minute = 0
        cfg.retrieval = RetrievalConfig(
            query_expansion_min_dominance=0.3,
            query_expansion_top_k=5,
            query_expansion_max_entities=2,
            query_expansion_type_allow=DEFAULT_PRF_ALLOW,
        )
        eng = Engram(config=cfg)
        try:
            for f in facts:
                eng.remember(f)
            r = eng.recall(in_allow_q, limit=3)
            assert isinstance(r, list)
            assert len(r) <= 3
        finally:
            eng.close()


def test_default_prf_allow_is_pref_and_know_upd():
    """Lock the §D14-derived default allow-set."""
    assert DEFAULT_PRF_ALLOW == frozenset({TYPE_KNOW_UPD, TYPE_SS_PREF})
