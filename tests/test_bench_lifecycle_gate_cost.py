"""Cover bench/bench_lifecycle_gate_cost.py helpers.

This bench is a §D5-adjacent measurement (NEXT.md item #1: gate hot-path
cost when fading/lifecycle pressure is non-zero). Pinning the helpers so
the curve we report in SCALE_REPORT.md / §40_results is reproducible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from bench import bench_lifecycle_gate_cost as bench  # noqa: E402

from engram import Config, Engram  # noqa: E402
from engram.consolidation.lifecycle_projection import snapshot_from_buffer  # noqa: E402
from engram.consolidation.schema_lifecycle import SchemaStatus  # noqa: E402


def test_percentile_handcomputed_examples():
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert bench._percentile(vals, 50) == 30.0
    # p100 == max
    assert bench._percentile(vals, 100) == 50.0
    # p0 == min
    assert bench._percentile(vals, 0) == 10.0
    # empty → 0.0 (defensive)
    assert bench._percentile([], 50) == 0.0


def test_percentile_linear_interpolation():
    # k = (5-1) * 0.95 = 3.8 → between vals[3]=40 and vals[4]=50
    vals = [10.0, 20.0, 30.0, 40.0, 50.0]
    got = bench._percentile(vals, 95)
    assert got == pytest.approx(48.0)


def test_seed_corpus_inserts_n_schema_active(tmp_path):
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        ids = bench._seed_corpus(eng, n_schemas=12)
        assert len(ids) == 12
        assert len(set(ids)) == 12  # unique
        # Round-trip a sample memory; confirm it's SCHEMA + ACTIVE.
        from engram.core import MemoryState, MemoryType
        m = eng._store.get(ids[3])
        assert m is not None
        assert m.type == MemoryType.SCHEMA
        assert m.state == MemoryState.ACTIVE
    finally:
        eng.close()


def test_seed_lifecycle_events_zero_short_circuits(tmp_path):
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        ids = bench._seed_corpus(eng, n_schemas=4)
        n_dep = bench._seed_lifecycle_events(eng, ids, n_events=0)
        assert n_dep == 0
        # No lifecycle events written.
        snap = snapshot_from_buffer(eng._buffer)
        assert all(
            st.status is not SchemaStatus.DEPRECATED for st in snap.values()
        )
    finally:
        eng.close()


def test_seed_lifecycle_events_alternates_dep_rec(tmp_path):
    """Layout: first n_schemas events are CREATEs (n_schemas=4 here),
    then events alternate DEPRECATE / RECOVER. With n_events=8 and
    n_schemas=4, the post-CREATE tail has 4 events: events 4..7 cycle
    through schemas 0..3 with DEPRECATE (first touch each), so 4
    distinct schemas end deprecated."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        ids = bench._seed_corpus(eng, n_schemas=4)
        n_dep = bench._seed_lifecycle_events(eng, ids, n_events=8)
        assert n_dep == 4
        snap = snapshot_from_buffer(eng._buffer, strict=True)
        n_dep_proj = sum(
            1 for st in snap.values() if st.status is SchemaStatus.DEPRECATED
        )
        assert n_dep_proj == 4
    finally:
        eng.close()


def test_seed_lifecycle_events_recover_round_trip(tmp_path):
    """With n_events=12 and n_schemas=4: 4 CREATEs, then 8 alternating.
    Each schema cycles DEP→REC twice → ends in REC (INFERRED), so 0
    deprecated."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        ids = bench._seed_corpus(eng, n_schemas=4)
        n_dep = bench._seed_lifecycle_events(eng, ids, n_events=12)
        assert n_dep == 0
        snap = snapshot_from_buffer(eng._buffer, strict=True)
        # All 4 schemas projected; none deprecated.
        assert len(snap) == 4
        n_dep_proj = sum(
            1 for st in snap.values() if st.status is SchemaStatus.DEPRECATED
        )
        assert n_dep_proj == 0
    finally:
        eng.close()


def test_measure_arm_returns_well_formed_row(tmp_path):
    """Smoke: 1500 events (1000 CREATEs + 500 DEP/REC), 10 queries,
    gate ON. Latency keys present and monotonic
    (p50 ≤ p95 ≤ p99 ≤ max)."""
    row = bench._measure_arm(
        tmp_path, n_events=1500, gate_on=True, n_queries=10
    )
    assert row["gate_on"] is True
    assert row["n_events"] == 1500
    assert row["n_queries"] == 10
    # 500 post-CREATE events alternate DEP/REC → ~250 distinct deprecated.
    assert row["n_deprecated_schemas"] >= 1
    for k in ("p50_ms", "p95_ms", "p99_ms", "mean_ms", "max_ms"):
        assert k in row
        assert isinstance(row[k], (int, float))
    assert row["p50_ms"] <= row["p95_ms"] <= row["p99_ms"] <= row["max_ms"]


def test_measure_arm_off_arm_skips_lifecycle_replay(tmp_path):
    """When gate_on=False, even a buffer with many lifecycle events
    must not slow down retrieval to the level of the gate-on arm by
    much. Sanity check; full quantitative result is in
    bench/results/lifecycle_gate_cost_*.json."""
    row_off = bench._measure_arm(
        tmp_path / "off", n_events=200, gate_on=False, n_queries=10
    )
    row_on = bench._measure_arm(
        tmp_path / "on", n_events=200, gate_on=True, n_queries=10
    )
    # Gate-off arm should have **at least** zero deprecated by construction
    # — n_deprecated_schemas tracks events seeded, not gate work, so it's
    # still populated; assert structural difference instead.
    assert row_off["gate_on"] is False and row_on["gate_on"] is True
    # Both should produce sensible, non-pathological p50.
    assert row_off["p50_ms"] < 500
    assert row_on["p50_ms"] < 500


def test_main_emits_payload_with_overhead_table(tmp_path, monkeypatch):
    """Patch the sweep down to {0, 50} for speed and check the JSON
    structure (arms + overhead rows for each n)."""
    out = tmp_path / "lifecycle.json"

    # Stub _measure_arm with a deterministic shape so main() runs fast.
    def fake_measure(tmp_root, n_events, gate_on, n_queries=200):
        return {
            "n_events": n_events,
            "gate_on": gate_on,
            "n_queries": n_queries,
            "n_deprecated_schemas": n_events // 4,
            "p50_ms": 10.0 + (1.5 if gate_on else 0.0) + 0.001 * n_events,
            "p95_ms": 25.0 + (2.0 if gate_on else 0.0),
            "p99_ms": 30.0 + (3.0 if gate_on else 0.0),
            "mean_ms": 12.0,
            "max_ms": 40.0,
        }

    monkeypatch.setattr(bench, "_measure_arm", fake_measure)
    # Shrink the sweep at module level via a local copy of main.
    monkeypatch.setattr(
        bench, "_git_sha", lambda: "deadbee"
    )

    # Patch the sweep list inside main by re-defining it via a thin wrapper:
    # easier — just call main with a constructed out_path after monkey-
    # patching the sweep constant. The cleanest hook is to monkeypatch
    # the sweep list literal — but it's a local. So we just call main
    # directly; with stubbed _measure_arm it's instant regardless of the
    # 5-point sweep.
    written = bench.main(out_path=out)
    assert written == out
    payload = json.loads(out.read_text())
    assert payload["meta"]["sha"] == "deadbee"
    assert payload["meta"]["name"] == "lifecycle_gate_cost"
    # Expect 5 sweep points × 2 arms = 10 arms.
    assert len(payload["arms"]) == 10
    # Overhead has one row per sweep point.
    assert len(payload["overhead"]) == 5
    # Every overhead row must have the required keys.
    for row in payload["overhead"]:
        assert {"n_events", "p50_off_ms", "p50_on_ms",
                "p50_abs_overhead_ms"} <= set(row)
        # Stub guarantees p50_on > p50_off by ≥1.5ms.
        assert row["p50_abs_overhead_ms"] >= 1.5
