"""Unit tests for §D3-real synthetic supersede corpus + driver.

Covers:
  * generator determinism (same seed -> identical dataset)
  * planted-fact arithmetic (n_slots * updates_per_slot facts + distractors)
  * gold/stale labels are non-overlapping (a gold value never appears in stale)
  * lexical overlap between consecutive updates is > 0.6
    (precondition for the InterferenceDetection heuristic to fire)
  * driver smoke: small corpus, default vs add_only diverge in
    expected directions (hit@1 default >= addonly, stale@1 default <= addonly)
"""

from __future__ import annotations

import pytest

from evals.synthetic import generate_supersede_dataset


def test_generator_determinism():
    a = generate_supersede_dataset(n_slots=20, updates_per_slot=2, distractors=10, seed=7)
    b = generate_supersede_dataset(n_slots=20, updates_per_slot=2, distractors=10, seed=7)
    assert [m for m, _ in a.memories] == [m for m, _ in b.memories]
    assert [q.text for q in a.queries] == [q.text for q in b.queries]
    assert [q.expected_substrings for q in a.queries] == [q.expected_substrings for q in b.queries]


def test_generator_counts():
    n_slots, updates, dist = 25, 3, 50
    ds = generate_supersede_dataset(
        n_slots=n_slots, updates_per_slot=updates, distractors=dist, seed=1
    )
    n_fact_mems = sum(1 for _, m in ds.memories if m.get("kind") == "fact")
    n_dist_mems = sum(1 for _, m in ds.memories if m.get("kind") == "distractor")
    assert n_fact_mems == n_slots * updates
    assert n_dist_mems == dist
    assert len(ds.queries) == n_slots


def test_gold_and_stale_disjoint():
    ds = generate_supersede_dataset(n_slots=40, updates_per_slot=2, distractors=20, seed=3)
    for q in ds.queries:
        gold = q.expected_substrings[0]
        stale_tag = next((t for t in q.tags if t.startswith("stale=")), "stale=")
        stale_values = stale_tag[len("stale="):].split("|")
        assert gold not in stale_values, (
            f"gold value {gold!r} appears in stale list {stale_values!r}"
        )


def test_consecutive_updates_have_overlap_above_supersede_threshold():
    """The heuristic InterferenceDetection requires lexical similarity > 0.6
    on a Jaccard of word sets to classify an update as supersede. If we
    silently break that property in the templates, §D3-real becomes a
    null run with no diagnostic. Lock it down here."""
    ds = generate_supersede_dataset(n_slots=80, updates_per_slot=2, distractors=0, seed=11)

    # Group facts by slot_id, sorted by update_idx
    by_slot: dict[str, list[tuple[str, int]]] = {}
    for content, meta in ds.memories:
        if meta.get("kind") != "fact":
            continue
        by_slot.setdefault(meta["slot_id"], []).append((content, meta["update_idx"]))

    for sid, items in by_slot.items():
        items.sort(key=lambda x: x[1])
        for (a, _), (b, _) in zip(items, items[1:]):
            aw = set(a.lower().split())
            bw = set(b.lower().split())
            sim = len(aw & bw) / max(len(aw | bw), 1)
            assert sim > 0.6, f"slot {sid} consecutive sim={sim:.3f} <= 0.6: {a!r} / {b!r}"


@pytest.mark.slow
def test_driver_smoke_default_beats_addonly_on_hit_at_1():
    """Driver smoke: on the supersede-rich corpus, default consolidation
    must improve hit@1 over add_only AND reduce stale@1. We don't require
    a particular magnitude, just the directional finding."""
    from evals.synthetic_supersede_d3_real import run_d3_real

    rep = run_d3_real(
        n_slots=20, updates_per_slot=2, distractors=20,
        seed=42, k=10, resamples=200, boot_seed=42,
    )
    a, b = rep["arms"]["default"], rep["arms"]["addonly"]
    assert a["interference_actions"] > 0, "interference stage was inert — corpus broken"
    assert b["interference_actions"] == 0, "add_only ran interference (config wiring broken)"
    assert a["hit_at_1"] >= b["hit_at_1"], "default should not under-recall hit@1 vs add_only"
    assert a["stale_at_1"] <= b["stale_at_1"], "default should surface fewer stale values at rank 1"
