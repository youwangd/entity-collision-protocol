"""Tests for the paraphrase-density synthetic generator.

Locks in: realized overlap tracks requested T; T=1.0 is verbatim;
T=0.0 swaps every available content token; entity anchors survive
swapping; queries remain answerable (anchors not corrupted)."""
from __future__ import annotations

from evals.paraphrase_density import (
    generate_dataset,
    _SPECS,
)


def test_dataset_shape():
    ds = generate_dataset(n_facts=12, distractors_per_fact=2, seed=1)
    assert len(ds.queries) == 12
    # 12 facts + 24 distractors
    assert len(ds.memories) == 12 + 24
    fact_mems = [m for m, meta in ds.memories if meta.get("kind") == "fact"]
    assert len(fact_mems) == 12
    # Every fact has a tag from the spec
    for q in ds.queries:
        assert q.tags and q.tags[0] in _SPECS


def test_overlap_target_extremes():
    """T=1.0 → realized overlap is high (verbatim shared tokens kept).
    T=0.0 → realized overlap is markedly lower (synonyms swapped in)."""
    ds_hi = generate_dataset(n_facts=60, distractors_per_fact=0,
                             overlap_target=1.0, seed=7)
    ds_lo = generate_dataset(n_facts=60, distractors_per_fact=0,
                             overlap_target=0.0, seed=7)
    # We're measuring jaccard on non-entity content tokens. T=1.0 keeps
    # all shared tokens; T=0.0 swaps every swappable one. The gap should
    # be substantial — at least 0.15 jaccard points on average.
    assert ds_hi.realized_overlap > ds_lo.realized_overlap + 0.15, (
        f"hi={ds_hi.realized_overlap:.3f} lo={ds_lo.realized_overlap:.3f}"
    )
    # And T=1.0 should give meaningful overlap (>0.15 jaccard) — entities
    # are excluded so we're measuring purely content-word agreement.
    assert ds_hi.realized_overlap > 0.15


def test_overlap_monotone_on_average():
    """Sweeping T in coarse steps should be monotonically non-decreasing
    in mean realized overlap (with enough facts to wash out RNG noise)."""
    overlaps = []
    for t in [0.0, 0.5, 1.0]:
        ds = generate_dataset(n_facts=120, distractors_per_fact=0,
                              overlap_target=t, seed=11)
        overlaps.append(ds.realized_overlap)
    assert overlaps[0] <= overlaps[1] + 1e-9, overlaps
    assert overlaps[1] <= overlaps[2] + 1e-9, overlaps
    # And the spread should be non-trivial
    assert overlaps[2] - overlaps[0] > 0.1, overlaps


def test_anchors_in_memory():
    """Every query's expected_substrings must be findable in its planted
    memory — otherwise ground truth is broken and recall metrics lie."""
    ds = generate_dataset(n_facts=60, distractors_per_fact=0,
                          overlap_target=0.5, seed=3)
    # Build a fact_idx → memory map
    by_idx = {meta["fact_idx"]: mem for mem, meta in ds.memories
              if meta.get("kind") == "fact"}
    # Queries are emitted in fact order; recover idx by enumerating
    fact_idxs = sorted(by_idx.keys())
    assert len(fact_idxs) == len(ds.queries)
    for fi, q in zip(fact_idxs, ds.queries):
        mem = by_idx[fi].lower()
        for anchor in q.expected_substrings:
            assert anchor.lower() in mem, (anchor, mem)


def test_invalid_overlap_target_raises():
    import pytest
    with pytest.raises(ValueError):
        generate_dataset(overlap_target=1.5)
    with pytest.raises(ValueError):
        generate_dataset(overlap_target=-0.1)


def test_seed_reproducibility():
    a = generate_dataset(n_facts=20, overlap_target=0.4, seed=99)
    b = generate_dataset(n_facts=20, overlap_target=0.4, seed=99)
    assert [m for m, _ in a.memories] == [m for m, _ in b.memories]
    assert [q.text for q in a.queries] == [q.text for q in b.queries]


def test_different_seeds_differ():
    a = generate_dataset(n_facts=40, overlap_target=0.5, seed=1)
    b = generate_dataset(n_facts=40, overlap_target=0.5, seed=2)
    assert [q.text for q in a.queries] != [q.text for q in b.queries]
