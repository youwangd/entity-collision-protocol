"""Regression and design-invariant tests for `evals.schema_share_contam_sweep`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.schema_share_contam_sweep import run_grid, safe_envelope, main
from evals.schema_share_sweep import SimConfig, run_share


# Tiny config so tests run in <1s.
TINY = SimConfig(
    n_clusters=40,
    cluster_size=4,
    n_per_window=1,
    n_windows=10,
    seed=0xE17A11,
)


def test_run_grid_byte_identity_with_share_sweep_baseline():
    """The (share=0.0, c=0.0, t=1.0) cell must equal §62 baseline exactly."""
    grid = run_grid(TINY, shares=[0.0], contaminations=[0.0])
    cell = grid["cells"][0]
    baseline = run_share(TINY, 0.0).to_dict()
    for k in (
        "n_high_q", "n_low_q",
        "promote_rate_high", "deprecate_rate_low",
        "false_promote_low_q", "false_deprecate_high_q",
    ):
        assert cell[k] == baseline[k], k


def test_run_grid_share_zero_contam_invariance():
    """At share=0.0 the policy ignores siblings → contamination has no effect."""
    grid = run_grid(TINY, shares=[0.0], contaminations=[0.0, 0.5, 1.0])
    a, b, c = grid["cells"]
    for key in ("promote_rate_high", "deprecate_rate_low",
                "false_promote_low_q", "false_deprecate_high_q"):
        assert a[key] == b[key] == c[key], key


def test_run_grid_metadata_attached():
    grid = run_grid(TINY, shares=[0.0, 0.5], contaminations=[0.0, 0.25])
    assert len(grid["cells"]) == 4
    seen = {(x["share"], x["contamination"]) for x in grid["cells"]}
    assert seen == {(0.0, 0.0), (0.0, 0.25), (0.5, 0.0), (0.5, 0.25)}


def test_safe_envelope_progress_floor_kicks_share_zero_out():
    """share=0 has zero promotes; with progress_min>0 it must NOT win the envelope."""
    grid = run_grid(TINY, shares=[0.0, 0.75], contaminations=[0.0])
    # share=0 satisfies FP/FD trivially (both 0) but has promote_rate=0.
    # share=0.75 should promote and dominate when progress is required.
    env_no_floor = safe_envelope(grid, progress_min=0.0)
    env_with_floor = safe_envelope(grid, progress_min=0.5)
    # Without a progress floor, share=0.75 still wins on tie because we
    # take the *largest* share that satisfies all constraints.
    assert env_no_floor[0.0] == 0.75
    # With a progress floor, the answer must be share=0.75 (share=0 fails).
    assert env_with_floor[0.0] == 0.75
    # And if we crank progress beyond what share=0.75 produces, we should
    # get None.
    env_strict = safe_envelope(grid, progress_min=0.999)
    cell_75 = next(
        x for x in grid["cells"] if x["share"] == 0.75 and x["contamination"] == 0.0
    )
    expected = 0.75 if cell_75["promote_rate_high"] >= 0.999 else None
    assert env_strict[0.0] == expected


def test_safe_envelope_monotone_in_contamination():
    """As contamination rises (at fixed thresholds), max safe share is non-increasing."""
    cfg = SimConfig(
        n_clusters=200, cluster_size=4, n_per_window=1, n_windows=20,
        seed=0xE17A11,
    )
    grid = run_grid(
        cfg,
        shares=[0.0, 0.25, 0.5, 0.75, 1.0],
        contaminations=[0.0, 0.25, 0.5, 0.75, 1.0],
    )
    env = safe_envelope(grid, fp_max=0.01, fd_max=0.05, progress_min=0.5)
    # Convert None -> -1 for monotone comparison (None means "nothing safe").
    seq = [-1.0 if env[c] is None else env[c] for c in (0.0, 0.25, 0.5, 0.75, 1.0)]
    for prev, curr in zip(seq, seq[1:]):
        assert curr <= prev, f"non-monotone: {seq}"


def test_main_writes_json_with_safe_envelope_section(tmp_path: Path):
    out = tmp_path / "g.json"
    rc = main([
        "--out", str(out),
        "--n-clusters", "30",
        "--n-windows", "8",
        "--shares", "0.0,0.75",
        "--contaminations", "0.0,0.5",
    ])
    assert rc == 0
    data = json.loads(out.read_text())
    assert "safe_envelope" in data
    assert "envelope_thresholds" in data
    assert set(data["envelope_thresholds"]) == {"fp_max", "fd_max", "progress_min"}


def test_main_rejects_out_of_range(tmp_path: Path):
    out = tmp_path / "g.json"
    with pytest.raises(SystemExit):
        main(["--out", str(out), "--shares", "1.5"])
    with pytest.raises(SystemExit):
        main(["--out", str(out), "--contaminations", "-0.1"])
