"""Tests for scripts/diff_results.py — acceptance gate for reproduce.sh."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT = REPO_ROOT / "scripts" / "diff_results.py"


def _write(p: Path, payload: dict) -> Path:
    p.write_text(json.dumps(payload))
    return p


def _run(ref: Path, new: Path, *extra: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT), str(ref), str(new), *extra],
        capture_output=True, text=True,
    )


def test_identical_files_pass(tmp_path: Path) -> None:
    payload = {"session_hit_at_1": 0.858, "session_hit_at_k": 0.95,
               "recall_ms": {"p50": 25.31, "mean": 27.04}}
    ref = _write(tmp_path / "ref.json", payload)
    new = _write(tmp_path / "new.json", payload)
    cp = _run(ref, new)
    assert cp.returncode == 0, cp.stdout + cp.stderr
    assert "PASS" in cp.stdout


def test_rate_drift_under_tolerance_passes(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.json", {"session_hit_at_1": 0.858})
    new = _write(tmp_path / "new.json", {"session_hit_at_1": 0.860})  # +0.2pp
    cp = _run(ref, new)
    assert cp.returncode == 0, cp.stdout


def test_rate_drift_over_tolerance_fails(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.json", {"session_hit_at_1": 0.858})
    new = _write(tmp_path / "new.json", {"session_hit_at_1": 0.836})  # -2.2pp
    cp = _run(ref, new)
    assert cp.returncode == 1, cp.stdout
    assert "FAIL" in cp.stdout


def test_latency_relative_tolerance(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.json", {"recall_ms": {"p50": 25.0}})
    # +20% — within default ±25% tol
    new = _write(tmp_path / "new.json", {"recall_ms": {"p50": 30.0}})
    assert _run(ref, new).returncode == 0
    # +30% — out of tol
    new = _write(tmp_path / "new.json", {"recall_ms": {"p50": 32.5}})
    assert _run(ref, new).returncode == 1


def test_ingest_shape(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.json",
                 {"writes_per_sec": 1400.0, "latency_ms": {"p99": 3.4}})
    new = _write(tmp_path / "new.json",
                 {"writes_per_sec": 1450.0, "latency_ms": {"p99": 3.5}})
    cp = _run(ref, new)
    assert cp.returncode == 0, cp.stdout
    assert "writes_per_sec" in cp.stdout
    assert "latency_ms.p99" in cp.stdout


def test_missing_file_returns_2(tmp_path: Path) -> None:
    new = _write(tmp_path / "new.json", {"session_hit_at_1": 0.5})
    cp = _run(tmp_path / "missing.json", new)
    assert cp.returncode == 2


def test_no_recognised_metrics_returns_2(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.json", {"unrelated": 1})
    new = _write(tmp_path / "new.json", {"unrelated": 2})
    cp = _run(ref, new)
    assert cp.returncode == 2


def test_custom_tolerance_flag(tmp_path: Path) -> None:
    ref = _write(tmp_path / "ref.json", {"session_hit_at_1": 0.858})
    new = _write(tmp_path / "new.json", {"session_hit_at_1": 0.836})
    # default fails, but with --rate-tol-abs 0.05 it should pass
    cp = _run(ref, new, "--rate-tol-abs", "0.05")
    assert cp.returncode == 0, cp.stdout
