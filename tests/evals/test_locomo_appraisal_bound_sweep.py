"""Tests for §94c-appraisal-bound cap sweep."""
from __future__ import annotations

import pytest

from evals import locomo_appraisal_bound_sweep as mod


def test_default_caps_includes_none_and_descending():
    caps = mod.DEFAULT_CAPS
    assert caps[0] is None
    rest = [c for c in caps if c is not None]
    assert rest == sorted(rest, reverse=True), "caps should descend"
    assert all(0.0 < c <= 1.0 for c in rest)


@pytest.mark.slow
def test_run_sweep_smoke(tmp_path):
    # Use the in-repo locomo10 fixture but with max_instances=1 to keep
    # the test fast (~10–20s on CI).
    import os
    ds = os.path.join(os.path.dirname(__file__), "..", "..",
                      "bench", "data", "locomo10.json")
    if not os.path.exists(ds):
        import pytest
        pytest.skip("locomo10 fixture not present")
    out = mod.run_sweep(
        ds, max_instances=1, k=5, embedder_name="hashtrigram",
        caps=[None, 0.5],
    )
    assert out["max_instances"] == 1
    assert len(out["rows"]) == 2
    cap_vals = [r["appraisal_salience_cap"] for r in out["rows"]]
    assert cap_vals == [None, 0.5]
    for r in out["rows"]:
        assert "delta_h1" in r and "delta_grk" in r
        assert isinstance(r["n_pairs"], int)


def test_render_markdown_table_shape():
    fake = {
        "dataset": "x.json",
        "max_instances": 2,
        "k": 10,
        "embedder": "hashtrigram",
        "wall_seconds": 1.23,
        "rows": [
            {
                "appraisal_salience_cap": None,
                "n_pairs": 301,
                "delta_h1": 0.0764,
                "delta_hk": 0.1528,
                "delta_prk": 0.0,
                "delta_grk": -0.0075,
                "delta_mrr": 0.05,
            },
            {
                "appraisal_salience_cap": 0.5,
                "n_pairs": 301,
                "delta_h1": 0.07,
                "delta_hk": 0.14,
                "delta_prk": 0.0,
                "delta_grk": 0.001,
                "delta_mrr": 0.05,
            },
        ],
    }
    md = mod.render_markdown(fake)
    assert "appraisal-bound" in md
    assert "| None |" in md
    assert "| 0.50 |" in md
    # signed format with explicit + on positives
    assert "+0.0764" in md
    assert "-0.0075" in md
