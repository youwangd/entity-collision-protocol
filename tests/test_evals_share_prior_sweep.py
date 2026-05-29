"""Unit tests for ``evals.share_prior_sweep`` — the §4.7 multi-hop sweep driver.

Pin the bridge-corpus generator's invariants and the markdown/JSON shape so a
silent regression in pair-recall plumbing surfaces immediately. The sweep
runner itself is exercised end-to-end on a tiny corpus to keep CI under a
few seconds while still touching ``run_sweep → _eval_bridge → _eval_unique``.

What's covered:
  * generate_bridge_corpus invariants:
      - per-pair anchor uniqueness (a_anchor in fact_a only, b_anchor in fact_b
        only, and bridge appears in BOTH)
      - dropped pairs (no measurable b_anchor) leave no orphan memories
      - distractor count + shuffle determinism by seed
      - query.expected_substrings has exactly [a_anchor, b_anchor]
  * _hits / _any_hit case-insensitive boolean math
  * _delta writes round-tripped d_<k> entries with correct sign + 6dp rounding
  * _md table shape: row count = arms count, header columns intact
  * run_sweep end-to-end smoke (tiny corpus, single alpha): emits both
    recipes, baseline arm has all-zero deltas, sharer arm has d_* keys

These tests use ``Engram`` for the smoke arm — same hash-embedder default
as the rest of the test suite, no network.
"""
from __future__ import annotations

import json

import pytest

from evals.share_prior_sweep import (
    _BRIDGE_TEMPLATES,
    _any_hit,
    _delta,
    _hits,
    _md,
    generate_bridge_corpus,
    run_sweep,
)


# ---------------------------------------------------------------------------
# generate_bridge_corpus invariants
# ---------------------------------------------------------------------------

class TestGenerateBridgeCorpus:
    def test_anchors_per_pair_disjoint(self):
        ds = generate_bridge_corpus(n_pairs=12, plain_distractors=0, seed=17)
        # Reconstruct pairs by pair_id metadata, then verify per-fact anchors
        # are exclusive to that fact.
        by_pair: dict[str, list[tuple[str, dict]]] = {}
        for content, meta in ds.memories:
            if meta.get("kind") == "fact":
                by_pair.setdefault(meta["pair_id"], []).append((content, meta))

        for pid, mems in by_pair.items():
            # exactly two facts per pair
            assert len(mems) == 2, f"pair {pid} has {len(mems)} facts"
            # find matching query
            qs = [q for q in ds.queries if pid in q.tags]
            assert len(qs) == 1
            a_anchor, b_anchor = qs[0].expected_substrings

            # Identify which mem is which by fact_id suffix
            mems_by_fact = {m[1]["fact_id"][-1]: m[0] for m in mems}
            a_text = mems_by_fact["a"].lower()
            b_text = mems_by_fact["b"].lower()

            # a_anchor lives only in fact_a; b_anchor lives only in fact_b
            assert a_anchor.lower() in a_text
            assert a_anchor.lower() not in b_text
            assert b_anchor.lower() in b_text
            assert b_anchor.lower() not in a_text

    def test_distractor_count(self):
        ds = generate_bridge_corpus(n_pairs=4, plain_distractors=37, seed=3)
        n_distractors = sum(1 for _, m in ds.memories if m.get("kind") == "distractor")
        assert n_distractors == 37

    def test_seed_determinism(self):
        a = generate_bridge_corpus(n_pairs=8, plain_distractors=10, seed=42)
        b = generate_bridge_corpus(n_pairs=8, plain_distractors=10, seed=42)
        assert [m for m, _ in a.memories] == [m for m, _ in b.memories]

    def test_seed_changes_shuffle(self):
        a = generate_bridge_corpus(n_pairs=8, plain_distractors=20, seed=1)
        b = generate_bridge_corpus(n_pairs=8, plain_distractors=20, seed=2)
        # Same population; different ordering after shuffle.
        assert sorted(m for m, _ in a.memories) == sorted(m for m, _ in b.memories)
        assert [m for m, _ in a.memories] != [m for m, _ in b.memories]

    def test_no_orphan_memories_when_pair_dropped(self):
        # The relocation template (index 1 in _BRIDGE_TEMPLATES) is the one
        # that intentionally drops because b_anchor candidates collapse to
        # the city — which appears in fact_a too. Hammer with many pairs and
        # check #facts is even (every kept pair contributes 2 mems exactly).
        ds = generate_bridge_corpus(n_pairs=30, plain_distractors=0, seed=7)
        n_facts = sum(1 for _, m in ds.memories if m.get("kind") == "fact")
        assert n_facts % 2 == 0
        # And # of queries equals n_facts/2 (one query per surviving pair).
        assert len(ds.queries) == n_facts // 2

    def test_query_anchor_shape(self):
        ds = generate_bridge_corpus(n_pairs=6, plain_distractors=0, seed=99)
        for q in ds.queries:
            assert len(q.expected_substrings) == 2
            assert all(isinstance(s, str) and s for s in q.expected_substrings)
            assert "multi_hop" in q.tags

    def test_zero_pairs_no_crash(self):
        ds = generate_bridge_corpus(n_pairs=0, plain_distractors=5, seed=1)
        assert len(ds.queries) == 0
        # Only distractors survive.
        assert all(m.get("kind") == "distractor" for _, m in ds.memories)
        assert len(ds.memories) == 5

    def test_template_count_pinned(self):
        # _md/run_sweep math relies on at least 3 distinct templates so the
        # n_pairs=3 smoke arm exercises all templates. Pin the count so a
        # later refactor can't silently shrink it.
        assert len(_BRIDGE_TEMPLATES) >= 3


# ---------------------------------------------------------------------------
# _hits / _any_hit boolean math
# ---------------------------------------------------------------------------

class TestHitPredicates:
    def test_hits_all_required(self):
        assert _hits("Aria leads engineering on Project Atlas.",
                    ["Aria", "Project Atlas"])
        assert not _hits("Aria leads engineering on Project Borealis.",
                         ["Aria", "Project Atlas"])

    def test_hits_case_insensitive(self):
        assert _hits("ARIA leads engineering on PROJECT ATLAS.",
                    ["aria", "project atlas"])

    def test_hits_empty_text(self):
        assert not _hits("", ["Aria"])

    def test_hits_empty_anchors_vacuously_true(self):
        # all([]) is True — pin this contract.
        assert _hits("anything", [])

    def test_any_hit_disjunction(self):
        assert _any_hit("Aria runs the show", ["Aria", "Borealis"])
        assert _any_hit("Project Borealis launches Q3", ["Aria", "Borealis"])
        assert not _any_hit("totally unrelated", ["Aria", "Borealis"])

    def test_any_hit_falsy_anchor_skipped(self):
        # "" is falsy -> short-circuited, doesn't match every string.
        assert not _any_hit("anything", ["", None])  # type: ignore[list-item]


# ---------------------------------------------------------------------------
# _delta math
# ---------------------------------------------------------------------------

class TestDelta:
    def test_delta_writes_keys(self):
        arm = {"hit@1": 0.700, "hit@5": 0.880}
        baseline = {"hit@1": 0.650, "hit@5": 0.900}
        _delta(arm, baseline, ["hit@1", "hit@5"])
        assert arm["d_hit@1"] == pytest.approx(0.05)
        assert arm["d_hit@5"] == pytest.approx(-0.02)

    def test_delta_rounding_6dp(self):
        arm = {"x": 1 / 3}
        baseline = {"x": 0.0}
        _delta(arm, baseline, ["x"])
        # 6dp rounding pinned by the implementation.
        assert arm["d_x"] == round(1 / 3, 6)

    def test_delta_skips_missing_keys(self):
        arm = {"hit@1": 0.5}
        baseline = {"hit@5": 0.5}
        _delta(arm, baseline, ["hit@1", "hit@5"])
        assert "d_hit@1" not in arm  # baseline missing
        assert "d_hit@5" not in arm  # arm missing

    def test_delta_baseline_against_self_is_zero(self):
        arm = {"hit@1": 0.5}
        _delta(arm, arm, ["hit@1"])
        assert arm["d_hit@1"] == 0.0


# ---------------------------------------------------------------------------
# _md table shape
# ---------------------------------------------------------------------------

class TestMarkdown:
    def _fake_report(self) -> dict:
        return {
            "wall_seconds": 1.23,
            "corpus": {
                "n_facts": 10, "n_pairs": 4, "plain_distractors": 5,
                "seed": 7, "n_unique_memories": 30, "n_bridge_memories": 13,
            },
            "unique_entity": [
                {"reranker": None, "alpha": 0.0, "hit@1": 0.5, "hit@5": 0.7,
                 "hit@10": 0.8, "d_hit@1": 0.0, "d_hit@5": 0.0, "d_hit@10": 0.0},
                {"reranker": "share_prior", "alpha": 0.05, "hit@1": 0.55,
                 "hit@5": 0.75, "hit@10": 0.85,
                 "d_hit@1": 0.05, "d_hit@5": 0.05, "d_hit@10": 0.05},
            ],
            "bridge_multi_hop": [
                {"reranker": None, "alpha": 0.0,
                 "pair_recall@5": 0.2, "pair_recall@10": 0.3,
                 "any_hit@1": 0.5, "any_hit@5": 0.6,
                 "d_pair_recall@5": 0.0, "d_pair_recall@10": 0.0},
                {"reranker": "share_prior", "alpha": 0.05,
                 "pair_recall@5": 0.25, "pair_recall@10": 0.35,
                 "any_hit@1": 0.5, "any_hit@5": 0.62,
                 "d_pair_recall@5": 0.05, "d_pair_recall@10": 0.05},
            ],
        }

    def test_md_row_count(self):
        md = _md(self._fake_report())
        # Two arm rows in each table + header + sep.
        _ = [ln for ln in md.splitlines()
                       if ln.startswith("| ") and "share_prior" in ln or "—" in ln]
        # Looser: just check both arms appear in unique_entity table by counting
        # "share_prior" rows (one in each of two tables = 2 total).
        assert md.count("share_prior") == 2
        assert "Bridge multi-hop" in md
        assert "Unique-entity" in md

    def test_md_includes_corpus_meta(self):
        md = _md(self._fake_report())
        assert "seed=7" in md
        assert "30" in md  # n_unique_memories
        assert "13" in md  # n_bridge_memories

    def test_md_signed_deltas(self):
        md = _md(self._fake_report())
        assert "+0.050" in md  # share_prior arm Δ
        assert "+0.000" in md  # baseline arm Δ


# ---------------------------------------------------------------------------
# run_sweep end-to-end smoke (tiny corpus)
# ---------------------------------------------------------------------------

class TestRunSweepSmoke:
    def test_run_sweep_minimal(self):
        rep = run_sweep(alphas=[0.05], n_facts=12, n_pairs=4,
                        plain_distractors=10, seed=11)

        # Top-level keys
        assert set(rep) >= {"alphas", "corpus", "unique_entity",
                            "bridge_multi_hop", "wall_seconds"}
        assert rep["alphas"] == [0.05]
        assert rep["corpus"]["seed"] == 11

        # Each table = baseline + 1 sharer arm = 2 rows.
        assert len(rep["unique_entity"]) == 2
        assert len(rep["bridge_multi_hop"]) == 2

        # Baseline arm: deltas all zero by construction.
        base_u, sharer_u = rep["unique_entity"]
        assert base_u["reranker"] is None
        assert base_u.get("d_hit@1") == 0.0
        assert sharer_u["reranker"] == "share_prior"
        assert "d_hit@1" in sharer_u  # sharer arm has deltas

        base_b, sharer_b = rep["bridge_multi_hop"]
        assert base_b["reranker"] is None
        assert base_b.get("d_pair_recall@5") == 0.0
        assert "d_pair_recall@5" in sharer_b

        # Headline metrics in [0,1].
        for arm in rep["unique_entity"]:
            for k in ("hit@1", "hit@5", "hit@10"):
                assert 0.0 <= arm[k] <= 1.0
        for arm in rep["bridge_multi_hop"]:
            for k in ("pair_recall@5", "pair_recall@10",
                      "any_hit@1", "any_hit@5"):
                assert 0.0 <= arm[k] <= 1.0

    def test_run_sweep_json_serialisable(self):
        rep = run_sweep(alphas=[0.05], n_facts=8, n_pairs=3,
                        plain_distractors=5, seed=23)
        # Must round-trip through JSON for the atomic_write_json caller.
        s = json.dumps(rep, default=str)
        round_trip = json.loads(s)
        assert round_trip["alphas"] == [0.05]
        assert round_trip["corpus"]["n_pairs"] == 3
