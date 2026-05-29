"""Per-cell checkpoint + resume tests for `evals.sweep_vector_weight`.

The sweep used to lose all output if killed mid-run inside cron (see NEXT.md
runs #14 #15). It now writes `<out>.cell-<i>.partial.json` after each cell
and `--resume` reuses those files. These tests pin the behaviour so we
don't silently regress.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path


def _run_sweep(out_path: Path, weights: str, *extra: str) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable, "-m", "evals.sweep_vector_weight",
        "--embed", "hash",
        "--n-sessions", "3",
        "--facts", "2",
        "--distractors", "3",
        "--weights", weights,
        "--out", str(out_path),
        *extra,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, timeout=120)


def test_checkpoint_files_written(tmp_path):
    out = tmp_path / "sweep.json"
    res = _run_sweep(out, "0.1,0.5")
    assert res.returncode == 0, res.stderr
    assert out.exists()
    cell0 = tmp_path / "sweep.cell-0.partial.json"
    cell1 = tmp_path / "sweep.cell-1.partial.json"
    assert cell0.exists() and cell1.exists()
    d0 = json.loads(cell0.read_text())
    d1 = json.loads(cell1.read_text())
    assert d0["vector_weight"] == 0.1
    assert d1["vector_weight"] == 0.5
    # Per-cell row contains the same keys the merged output uses.
    for required in ("baseline_hit_at_1", "baseline_mrr", "baseline_ndcg_at_k"):
        assert required in d0


def test_resume_uses_existing_cells(tmp_path):
    out = tmp_path / "sweep.json"
    # First run: produce cells.
    res1 = _run_sweep(out, "0.1,0.5")
    assert res1.returncode == 0, res1.stderr
    # Delete the merged output so we know resume isn't reading it.
    out.unlink()
    # Second run with --resume: should announce reuse for both cells.
    res2 = _run_sweep(out, "0.1,0.5", "--resume")
    assert res2.returncode == 0, res2.stderr
    assert res2.stdout.count("[resume] using") == 2
    # Merged output should be re-emitted from cached cells.
    assert out.exists()
    merged = json.loads(out.read_text())
    assert [c["vector_weight"] for c in merged["sweep"]] == [0.1, 0.5]


def test_resume_reruns_on_vw_mismatch(tmp_path):
    out = tmp_path / "sweep.json"
    cell0 = tmp_path / "sweep.cell-0.partial.json"
    # Plant a stale cell with a different vector_weight.
    cell0.parent.mkdir(parents=True, exist_ok=True)
    cell0.write_text(json.dumps({"vector_weight": 0.9, "baseline_hit_at_1": 0.0}))
    res = _run_sweep(out, "0.1", "--resume")
    assert res.returncode == 0, res.stderr
    assert "vw mismatch" in res.stdout
    fresh = json.loads(cell0.read_text())
    assert fresh["vector_weight"] == 0.1


def test_no_checkpoint_flag_suppresses_files(tmp_path):
    out = tmp_path / "sweep.json"
    res = _run_sweep(out, "0.1", "--no-checkpoint")
    assert res.returncode == 0, res.stderr
    assert out.exists()
    assert not (tmp_path / "sweep.cell-0.partial.json").exists()
