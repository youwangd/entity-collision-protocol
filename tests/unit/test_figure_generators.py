"""Smoke tests for evals/ingest_latency_figure.py and evals/lme_per_type_figure.py.

Both generators are wired into scripts/regen_figures.sh as T5. These tests
guard:
  - they accept the documented JSON schemas without throwing
  - they emit a non-empty PNG file
  - they're robust to the legacy `p99_9` key (alias for `p999`)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")

from evals.ingest_latency_figure import _plot as ingest_plot
from evals.lme_per_type_figure import _plot as lme_plot


def _ingest_payload() -> dict:
    return {
        "n": 1_000_000,
        "throughput_per_sec": 1390.7,
        "wall_seconds": 719.0,
        "latency_ms": {
            "p50": 0.468, "p95": 0.821, "p99": 3.429,
            "p999": 24.063, "max": 121.59,
        },
        "head_100k_p99_ms": 3.377,
        "tail_100k_p99_ms": 3.656,
        "meta": {"sha": "deadbee", "timestamp": "20260101T000000"},
    }


def _lme_payload(arm: str, base_acc: float) -> dict:
    return {
        "n_instances": 500,
        "k": 10,
        "session_hit_at_1": base_acc,
        "session_hit_at_k": base_acc + 0.05,
        "per_type_session_hit_at_1": {
            "single-session-user": base_acc + 0.10,
            "single-session-assistant": base_acc + 0.05,
            "single-session-preference": base_acc - 0.02,
            "multi-session": base_acc - 0.05,
            "temporal-reasoning": base_acc - 0.10,
            "knowledge-update": base_acc - 0.03,
        },
        "per_type_session_hit_at_k": {},
        "per_type_n": {
            "single-session-user": 30,
            "single-session-assistant": 56,
            "single-session-preference": 20,
            "multi-session": 100,
            "temporal-reasoning": 124,
            "knowledge-update": 78,
        },
        "ingest_ms": {"p50": 0.5, "mean": 0.7, "max": 100.0},
        "recall_ms": {"p50": 5.0, "mean": 6.2, "max": 90.0},
        "arm": arm,
        "embed": "st",
        "vector_weight_override": 0.3,
    }


def test_ingest_latency_figure_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "ingest.png"
    ingest_plot(_ingest_payload(), out)
    assert out.exists() and out.stat().st_size > 1000


def test_ingest_latency_figure_accepts_p99_9_alias(tmp_path: Path) -> None:
    payload = _ingest_payload()
    payload["latency_ms"]["p99_9"] = payload["latency_ms"].pop("p999")
    out = tmp_path / "ingest_alias.png"
    ingest_plot(payload, out)
    assert out.stat().st_size > 1000


def test_lme_per_type_figure_writes_png(tmp_path: Path) -> None:
    base = _lme_payload("baseline", 0.40)
    arm = _lme_payload("prfsp", 0.42)
    out = tmp_path / "lme.png"
    lme_plot(base, arm, out)
    assert out.exists() and out.stat().st_size > 1000


def test_lme_per_type_figure_handles_missing_arm_type(tmp_path: Path) -> None:
    base = _lme_payload("baseline", 0.40)
    arm = _lme_payload("prfsp", 0.42)
    arm["per_type_session_hit_at_1"].pop("knowledge-update")
    out = tmp_path / "lme_missing.png"
    lme_plot(base, arm, out)
    assert out.stat().st_size > 1000


def test_ingest_latency_figure_cli(tmp_path: Path) -> None:
    """End-to-end: write payload to disk, invoke __main__ via runpy."""
    import subprocess
    import sys
    payload_path = tmp_path / "p.json"
    payload_path.write_text(json.dumps(_ingest_payload()))
    out = tmp_path / "cli.png"
    res = subprocess.run(
        [sys.executable, "-m", "evals.ingest_latency_figure",
         "--input", str(payload_path), "--out", str(out)],
        capture_output=True, text=True, check=True,
    )
    assert "wrote" in res.stdout
    assert out.stat().st_size > 1000
