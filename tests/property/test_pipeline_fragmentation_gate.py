"""Stage 6 §74 fragmentation gate: schema_family_fragmentation_max.

Companion to the §69 contamination gate. SCALE_REPORT §74 calibration
showed that under realistic generative regimes the contamination meter
reads ~0.0 across c ∈ [0, 1] (outsiders fall out as singletons rather
than gluing in), while singletons/n_schemas tracks c almost 1:1.
This module wires the fragmentation meter into the same prepass.

Invariants:
  G1. fmax=None (default) ⇒ no fragmentation stat emitted; behavior
      identical to ungated path.
  G2. fmax set, observed frag ≤ fmax ⇒ siblings emitted; stats record
      the rate; no `share_gated` flag.
  G3. fmax set, observed frag > fmax ⇒ empty maps; stats record rate
      AND `schema_family_share_gated=True` AND
      `schema_family_fragmentation_gated=True`.
  G4. Either gate (contamination OR fragmentation) tripping collapses
      share — independent OR-of-gates semantics, not AND.
  G5. fmax=1.0 never trips (frag ≤ 1.0 by construction).
  G6. Both gates configured + neither tripped ⇒ siblings emitted, both
      stats recorded.
  G7. YAML roundtrip preserves the new key; default omits it.
"""
from __future__ import annotations

from engram.consolidation.pipeline import (
    StageContext,
    _build_schema_family_siblings,
)
from engram.core.config import Config, ConsolidationConfig


def _ctx(share=0.5, tau=0.3, cmax=None, fmax=None):
    cfg = Config()
    cfg.consolidation = ConsolidationConfig(
        schema_family_share=share,
        schema_family_tau=tau,
        schema_family_contamination_max=cmax,
        schema_family_fragmentation_max=fmax,
    )
    ctx = StageContext(config=cfg)
    ctx.consolidation_id = "w1"
    return ctx


def _tight_two():
    """Two patterns that cluster together at tau=0.3 → frag = 0/2."""
    return [
        {"pattern": "pattern AAAAAAAAAAAA", "facts": ["alpha bravo charlie"]},
        {"pattern": "pattern BBBBBBBBBBBB", "facts": ["alpha bravo charlie"]},
    ]


def _all_singleton_three():
    """Three patterns with disjoint vocabularies → 3 singletons → frag=1.0."""
    return [
        {"pattern": "pattern AAAAAAAAAAAA", "facts": ["alpha bravo charlie"]},
        {"pattern": "pattern BBBBBBBBBBBB", "facts": ["delta echo foxtrot"]},
        {"pattern": "pattern CCCCCCCCCCCC", "facts": ["golf hotel india"]},
    ]


def _two_singletons_one_pair():
    """1 pair + 2 singletons over 4 schemas → frag = 2/4 = 0.5."""
    return [
        {"pattern": "pattern AAAAAAAAAAAA", "facts": ["alpha bravo charlie"]},
        {"pattern": "pattern BBBBBBBBBBBB", "facts": ["alpha bravo charlie"]},
        {"pattern": "pattern CCCCCCCCCCCC", "facts": ["delta echo foxtrot"]},
        {"pattern": "pattern DDDDDDDDDDDD", "facts": ["golf hotel india"]},
    ]


def test_g1_fmax_none_no_frag_stat():
    ctx = _ctx(fmax=None)
    sibs, evs = _build_schema_family_siblings(_tight_two(), "w1", ctx)
    assert "schema_family_fragmentation_rate" not in ctx.stats
    a = "pattern AAAAAAAAAAAA"
    b = "pattern BBBBBBBBBBBB"
    assert sibs[a][0] is evs[b]


def test_g2_below_fmax_passes():
    ctx = _ctx(fmax=0.5)
    sibs, evs = _build_schema_family_siblings(_tight_two(), "w1", ctx)
    assert ctx.stats["schema_family_fragmentation_rate"] == 0.0
    assert "schema_family_share_gated" not in ctx.stats
    a = "pattern AAAAAAAAAAAA"
    b = "pattern BBBBBBBBBBBB"
    assert sibs[a][0] is evs[b]


def test_g3_above_fmax_trips():
    ctx = _ctx(fmax=0.5)
    sibs, evs = _build_schema_family_siblings(_all_singleton_three(), "w1", ctx)
    assert sibs == {}
    assert evs == {}
    assert ctx.stats["schema_family_fragmentation_rate"] == 1.0
    assert ctx.stats["schema_family_share_gated"] is True
    assert ctx.stats["schema_family_fragmentation_gated"] is True


def test_g3_at_threshold_does_not_trip():
    """frag == fmax ⇒ pass (strict > comparison)."""
    ctx = _ctx(fmax=0.5)
    sibs, evs = _build_schema_family_siblings(_two_singletons_one_pair(), "w1", ctx)
    assert ctx.stats["schema_family_fragmentation_rate"] == 0.5
    assert "schema_family_share_gated" not in ctx.stats
    # The 2 clustered schemas still emit siblings.
    assert len(evs) == 4


def test_g4_either_gate_trips_collapses_share():
    """Contamination meter reads 0 under fragmentation regime; if only
    fmax catches it, the share still collapses."""
    ctx = _ctx(cmax=0.10, fmax=0.5)
    sibs, evs = _build_schema_family_siblings(_all_singleton_three(), "w1", ctx)
    # contamination=0 (singletons → no pairs → 0); fragmentation=1 → trip.
    assert ctx.stats["schema_family_contamination_rate"] == 0.0
    assert ctx.stats["schema_family_fragmentation_rate"] == 1.0
    assert sibs == {}
    assert ctx.stats["schema_family_share_gated"] is True
    assert ctx.stats["schema_family_fragmentation_gated"] is True


def test_g5_fmax_one_never_trips():
    ctx = _ctx(fmax=1.0)
    _, _ = _build_schema_family_siblings(_all_singleton_three(), "w1", ctx)
    # frag is in [0, 1] by construction; strict > 1.0 cannot occur.
    assert ctx.stats["schema_family_fragmentation_rate"] == 1.0
    assert "schema_family_share_gated" not in ctx.stats


def test_g6_both_gates_pass():
    ctx = _ctx(cmax=0.10, fmax=0.10)
    sibs, evs = _build_schema_family_siblings(_tight_two(), "w1", ctx)
    assert ctx.stats["schema_family_contamination_rate"] == 0.0
    assert ctx.stats["schema_family_fragmentation_rate"] == 0.0
    assert "schema_family_share_gated" not in ctx.stats
    a, b = "pattern AAAAAAAAAAAA", "pattern BBBBBBBBBBBB"
    assert sibs[a][0] is evs[b]


def test_g7_yaml_roundtrip_fragmentation_max():
    import tempfile, os
    cfg = Config()
    cfg.consolidation = ConsolidationConfig(
        schema_family_share=0.75,
        schema_family_tau=0.5,
        schema_family_fragmentation_max=0.20,
    )
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "c.yaml")
        cfg.save_yaml(path)
        loaded = Config.from_yaml(path)
    assert loaded.consolidation.schema_family_fragmentation_max == 0.20


def test_g7_yaml_default_omits_key():
    cfg = Config()
    cfg.consolidation = ConsolidationConfig()
    d = cfg.to_dict()
    cons = d.get("consolidation", {})
    assert "schema_family_fragmentation_max" not in cons
