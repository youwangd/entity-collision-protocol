"""Smoke + regression tests for §77 LoCoMo fragmentation replication."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from evals.locomo_fragmentation_replication import _extract_schemas, run

DATA = Path(__file__).resolve().parents[2] / "bench" / "data" / "locomo10.json"


def test_extracts_nontrivial_schema_count():
    data = json.loads(DATA.read_text())
    fps, mem = _extract_schemas(data)
    # Exact expected for the shipped dataset; locks in the §77 headline n.
    assert len(fps) == 543
    assert set(fps.keys()) == set(mem.keys())
    assert all(isinstance(v, frozenset) and len(v) > 0 for v in fps.values())


def test_run_headline_locomo_fragments_above_gate():
    """At every operationally sensible tau, LoCoMo fragments far above 0.10.

    Locks in the §77 headline: real-corpus prop-name clustering on
    LoCoMo observations cannot find sibling structure; the §76
    `fragmentation_max=0.10` gate trips on every tau in {0.3, 0.4, 0.5}.
    Any future change to the fingerprint tokenizer or to `cluster()`
    that meaningfully softens this should re-state §77 explicitly.
    """
    res = run(DATA, taus=(0.3, 0.4, 0.5))
    assert res["n_schemas"] == 543
    rows = {row["tau"]: row for row in res["by_tau"]}
    for tau in (0.3, 0.4, 0.5):
        assert rows[tau]["fragmentation_rate"] > 0.99, rows[tau]
        assert rows[tau]["contamination_rate"] == pytest.approx(0.0)


def test_run_serializable():
    res = run(DATA, taus=(0.5,))
    json.dumps(res)  # must round-trip
