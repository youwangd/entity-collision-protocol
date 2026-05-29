"""Unit tests for evals.locomo_percat_plot — verifies extraction and rendering."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.locomo_percat_plot import CATS, VWS, _extract, main


def _fake_percat(tmp_path: Path, name: str) -> Path:
    """Write a minimal percat sweep JSON shaped like locomo10_*_sweep_ci_percat.json."""
    rows = []
    for vw in [0.0, 0.3, 0.5, 0.7, 1.0]:
        per_cat = {
            c: {
                "n": 100 + int(c) * 10,
                "hit_at_1": {
                    "mean": -0.05 if vw > 0 else 0.0,
                    "ci_lo": -0.10 if vw > 0 else 0.0,
                    "ci_hi": 0.0 if vw > 0 else 0.0,
                    "baseline_mean": 0.5,
                },
            }
            for c in CATS
        }
        rows.append(
            {
                "vector_weight": vw,
                "embedder": "fake",
                "n_questions_scored": 1000,
                "ci": {"per_category_delta": per_cat},
            }
        )
    p = tmp_path / name
    p.write_text(json.dumps({"rows": rows}))
    return p


def test_extract_returns_only_relevant_vws(tmp_path: Path) -> None:
    p = _fake_percat(tmp_path, "fake.json")
    out = _extract(p)
    assert set(out.keys()) == set(VWS)
    for vw in VWS:
        assert set(out[vw].keys()) == set(CATS)
        for c in CATS:
            cell = out[vw][c]
            assert {"mean", "lo", "hi", "n"} <= cell.keys()


def test_main_renders_png(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    h = _fake_percat(tmp_path, "h.json")
    s = _fake_percat(tmp_path, "s.json")
    out = tmp_path / "fig.png"
    monkeypatch.setattr(
        "sys.argv",
        ["locomo_percat_plot", "--hash", str(h), "--st", str(s), "--out", str(out)],
    )
    rc = main()
    assert rc == 0
    assert out.exists() and out.stat().st_size > 1000
