"""§D3 — add_only consolidation mode (Mem0 v3 ADD-only ablation).

When ``consolidation.add_only=True`` the interference stage must be a
no-op: no supersede, no conflict mutation, both memories survive in
their original states. Mechanical-merge dedup is unaffected (separate
stage). This is the primitive that lets us A/B Engram-supersede vs
Mem0-style ADD-only on the same corpus.
"""
from __future__ import annotations

from pathlib import Path

from engram import Engram, Config
from engram.core.config import ConsolidationConfig
from engram.core.types import MemoryState


def _build(path: Path, *, add_only: bool) -> Engram:
    cfg = Config(path=str(path))
    cfg.security.max_events_per_minute = 0
    cfg.consolidation = ConsolidationConfig(
        schedule="manual",
        window_hours=24,
        add_only=add_only,
    )
    return Engram(config=cfg)


def _facts() -> list[str]:
    # Two near-identical facts that the heuristic interference detector
    # treats as supersede-eligible (overlap > 0.6, same type).
    return [
        "User Alice prefers dark roast coffee for morning meetings.",
        "User Alice prefers light roast coffee for morning meetings.",
    ]


def test_add_only_skips_supersede(tmp_path: Path):
    eng = _build(tmp_path / "add", add_only=True)
    try:
        for f in _facts():
            eng.remember(f)
        report = eng.consolidate()

        # Both memories survive in active state.
        actives = [m for m in eng._store.search_text("Alice", limit=10)]
        assert len(actives) >= 2
        for s in actives[:2]:
            assert s.memory.state == MemoryState.ACTIVE

        # Zero supersede / conflict actions recorded.
        assert report.state_transitions["interference"] == 0
    finally:
        eng.close()


def test_default_supersede_path_still_active(tmp_path: Path):
    """Sanity: with add_only=False (default), interference stage runs.
    We don't assert a specific action count (heuristic-dependent) — only
    that nothing crashes and the report shape matches."""
    eng = _build(tmp_path / "default", add_only=False)
    try:
        for f in _facts():
            eng.remember(f)
        report = eng.consolidate()
        assert "interference" in report.state_transitions
    finally:
        eng.close()


def test_config_add_only_roundtrip(tmp_path: Path):
    cfg = Config(path=str(tmp_path))
    cfg.consolidation = ConsolidationConfig(add_only=True)
    d = cfg.to_dict()
    assert d["consolidation"]["add_only"] is True

    cfg2 = Config._from_dict(d)
    assert cfg2.consolidation is not None
    assert cfg2.consolidation.add_only is True

    # Default round-trip is silent (no add_only key when False).
    cfg3 = Config(path=str(tmp_path))
    cfg3.consolidation = ConsolidationConfig(add_only=False)
    d3 = cfg3.to_dict()
    assert "add_only" not in d3.get("consolidation", {})
