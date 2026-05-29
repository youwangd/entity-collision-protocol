"""Stage 6 §8 prepass: cluster LLM patterns and credit cluster-mate evidence.

These tests cover the cluster-wiring step that replaces the prior
`siblings=()` placeholder. The prepass `_build_schema_family_siblings`
takes the LLM-returned `schemas` list, fingerprints supporting facts,
clusters them, and yields a `summary → siblings` map consumed at both
RECOVER and CREATE call sites.

Invariants:
  P1. share=0.0 (default) ⇒ prepass yields empty maps (regression-safe).
  P2. window_id=None ⇒ prepass yields empty maps.
  P3. With share>0 and overlapping-vocab patterns, summaries that
      cluster together each see the *other* in their siblings tuple.
  P4. Disjoint-vocab patterns get empty siblings (singleton clusters).
  P5. _decide_with_share with explicit siblings >0 supports actually
      crosses promote threshold when bare decide() would not (G3).
  P6. siblings tuple is owner-excluded (E1) and deterministic (E3).
"""
from __future__ import annotations

from engram.consolidation.pipeline import (
    StageContext,
    _build_schema_family_siblings,
    _decide_with_share,
)
from engram.consolidation.schema_decision import EvidenceWindow, Thresholds, decide
from engram.consolidation.schema_lifecycle import EventKind, SchemaState, SchemaStatus
from engram.core.config import Config, ConsolidationConfig


def _ctx(share: float, tau: float = 0.5, cid: str | None = "w1") -> StageContext:
    cfg = Config()
    cfg.consolidation = ConsolidationConfig(
        schema_family_share=share, schema_family_tau=tau,
    )
    ctx = StageContext(config=cfg)
    ctx.consolidation_id = cid
    return ctx


def _state(sid="s1", status=SchemaStatus.INFERRED):
    return SchemaState(schema_id=sid, status=status, version=1, last_window_id="w0")


def test_p1_share_zero_yields_empty_maps():
    schemas = [
        {"pattern": "user prefers dark mode", "facts": ["a alpha bravo", "b alpha bravo"]},
        {"pattern": "another distinct pattern xx", "facts": ["c gamma delta"]},
    ]
    sibs, evs = _build_schema_family_siblings(schemas, "w1", _ctx(0.0))
    assert sibs == {}
    assert evs == {}


def test_p2_window_id_none_yields_empty_maps():
    schemas = [{"pattern": "user prefers dark mode", "facts": ["a alpha bravo"]}]
    sibs, evs = _build_schema_family_siblings(schemas, None, _ctx(0.5))
    assert sibs == {}
    assert evs == {}


def test_p3_overlapping_facts_cluster_together():
    schemas = [
        # Both patterns share tokens "alpha" and "bravo" in their
        # supporting facts → high Jaccard → same cluster at tau=0.3.
        {"pattern": "pattern AAAAAAAAAAAA", "facts": ["alpha bravo charlie"]},
        {"pattern": "pattern BBBBBBBBBBBB", "facts": ["alpha bravo delta"]},
    ]
    sibs, evs = _build_schema_family_siblings(schemas, "w1", _ctx(0.5, tau=0.3))
    a = "pattern AAAAAAAAAAAA"
    b = "pattern BBBBBBBBBBBB"
    assert set(evs) == {a, b}
    assert len(sibs[a]) == 1 and sibs[a][0] is evs[b]  # owner-excluded (E1)
    assert len(sibs[b]) == 1 and sibs[b][0] is evs[a]


def test_p4_disjoint_facts_yield_empty_siblings():
    schemas = [
        {"pattern": "pattern AAAAAAAAAAAA", "facts": ["alpha bravo charlie"]},
        {"pattern": "pattern BBBBBBBBBBBB", "facts": ["xray yankee zulu"]},
    ]
    sibs, _ = _build_schema_family_siblings(schemas, "w1", _ctx(0.5, tau=0.3))
    a = "pattern AAAAAAAAAAAA"
    b = "pattern BBBBBBBBBBBB"
    assert sibs[a] == ()
    assert sibs[b] == ()


def test_p5_explicit_siblings_can_cross_promote_threshold():
    """Cross-pattern lift: bare decide() returns no PROMOTE; with a
    cluster-mate carrying enough supports, decide_with_family does."""
    th = Thresholds()  # default promote_supports
    state = _state(status=SchemaStatus.INFERRED)
    own = EvidenceWindow(window_id="w1", supports=1, contradictions=0)
    # Bare decide: too weak.
    assert decide(state, own, th) != EventKind.PROMOTE
    # Sibling carries enough to push (own + floor(share * sib)) >= threshold.
    sib_supports = max(2 * th.promote, 8)
    sib = EvidenceWindow(window_id="w1", supports=sib_supports, contradictions=0)
    kind = _decide_with_share(state, own, th, _ctx(1.0), siblings=(sib,))
    assert kind == EventKind.PROMOTE


def test_p6_siblings_owner_excluded_and_deterministic():
    schemas = [
        {"pattern": ("AAA " * 20)[:80], "facts": ["alpha bravo charlie"]},
        {"pattern": ("BBB " * 20)[:80], "facts": ["alpha bravo charlie"]},
        {"pattern": ("CCC " * 20)[:80], "facts": ["alpha bravo charlie"]},
    ]
    sibs1, _ = _build_schema_family_siblings(schemas, "w1", _ctx(0.5, tau=0.3))
    sibs2, _ = _build_schema_family_siblings(schemas, "w1", _ctx(0.5, tau=0.3))
    # Owner-excluded: each entry has exactly 2 siblings.
    for owner, sibs in sibs1.items():
        assert len(sibs) == 2
    # Deterministic across calls.
    assert sibs1 == sibs2


def test_default_config_disables_prepass_completely():
    """Regression-safety guarantee: out-of-the-box engram = empty prepass."""
    schemas = [{"pattern": "pattern xxxxxxxxxxxx", "facts": ["alpha bravo"]}]
    cfg = Config()
    cfg.consolidation = ConsolidationConfig()  # all defaults
    ctx = StageContext(config=cfg)
    ctx.consolidation_id = "w1"
    sibs, evs = _build_schema_family_siblings(schemas, "w1", ctx)
    assert sibs == {}
    assert evs == {}
