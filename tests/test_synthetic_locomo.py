"""Smoke test for the synthetic LoCoMo-shape generator.

The synthetic generator exists so paper §4 LoCoMo-style runs aren't
blocked on the upstream `data/locomo/` dataset landing. We assert two
contracts:

  (1) Output is byte-identical for a fixed seed (reproducibility — the
      paper appendix cites the exact seed used).
  (2) The output is consumable by `evals.locomo_adapter.load_locomo`
      without modification, and yields the categories we generate
      (single_hop / multi_hop / adversarial), with at least one
      evidence session per QA pair, all referencing real session ids.

If either contract breaks, the synthetic→adapter→retrieval pipeline
silently regresses and downstream numbers in SCALE_REPORT become
non-comparable across runs, so we keep the test in the default suite.
"""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from evals.locomo_adapter import load_locomo
from evals.synthetic_locomo import generate


def _digest(samples) -> str:
    return hashlib.sha256(
        json.dumps(samples, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_synthetic_locomo_seed_is_byte_stable():
    """Same seed → identical sample list. Different seed → different list."""
    a = generate(n_samples=6, n_sessions=5, turns_per_session=8, n_facts=5, seed=2026)
    b = generate(n_samples=6, n_sessions=5, turns_per_session=8, n_facts=5, seed=2026)
    assert _digest(a) == _digest(b)
    c = generate(n_samples=6, n_sessions=5, turns_per_session=8, n_facts=5, seed=2027)
    assert _digest(a) != _digest(c)


def test_synthetic_locomo_loads_through_adapter():
    samples = generate(n_samples=4, n_sessions=6, turns_per_session=8,
                       n_facts=5, seed=2026)
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "synth.json"
        p.write_text(json.dumps(samples))
        loaded = load_locomo(p)

    assert len(loaded) == 4
    cats: set[str] = set()
    for s in loaded:
        # every sample has at least one session with at least one turn
        assert s.sessions
        sids = {sess["id"] for sess in s.sessions}
        # every QA's evidence session(s) reference real ids
        assert s.qa
        for q in s.qa:
            cats.add(q.category)
            assert q.evidence_sessions, "every QA must carry evidence"
            for sid in q.evidence_sessions:
                assert sid in sids, f"QA evidence {sid} not in {sids}"

    # generator advertises three QA categories — keep this tight so
    # silent drops don't pass.
    assert cats == {"single_hop", "multi_hop", "adversarial"}


def test_synthetic_locomo_qa_anchor_in_evidence_session():
    """Each single_hop QA's gold answer string must appear in the turns
    of the session it cites as evidence (anchor invariant). Without
    this, recall@k becomes ill-defined — we'd be measuring noise."""
    samples = generate(n_samples=4, n_sessions=6, turns_per_session=8,
                       n_facts=5, seed=2026)
    for raw in samples:
        sess_text: dict[str, str] = {}
        for k, v in raw["conversation"].items():
            if k.startswith("session_") and not k.endswith("_date_time"):
                idx = int(k.split("_")[1])
                sid = f"D{idx}"
                sess_text[sid] = " ".join(t["text"] for t in v)
        for q in raw["qa"]:
            if q["category"] != "single_hop":
                continue
            assert len(q["evidence"]) == 1
            sid = q["evidence"][0]
            assert q["answer"] in sess_text[sid], (
                f"answer {q['answer']!r} not in evidence session {sid} text"
            )
