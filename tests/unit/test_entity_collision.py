"""Tests for the entity-collision generator.

Locks in: collision degree controls #facts/entity; queries paraphrase
the discriminator (no surface overlap); answers retrievable from memory;
shape integrity; seed reproducibility; bad inputs raise.
"""
from __future__ import annotations

import pytest

from evals.entity_collision import (
    _ENTITIES,
    _SPECS,
    generate_dataset,
)


def test_dataset_shape():
    ds = generate_dataset(n_entities=4, collision_degree=3,
                          distractors_per_entity=2, seed=1)
    # 4 entities × 3 facts = 12 queries
    assert len(ds.queries) == 12
    fact_mems = [m for m, meta in ds.memories if meta.get("kind") == "fact"]
    assert len(fact_mems) == 12
    distractors = [m for m, meta in ds.memories if meta.get("kind") == "distractor"]
    assert len(distractors) == 4 * 2
    for q in ds.queries:
        assert q.collision_degree == 3
        assert q.tags[0] in _SPECS


def test_collision_degree_one_no_collisions():
    """K=1: each entity has exactly one fact."""
    ds = generate_dataset(n_entities=8, collision_degree=1,
                          distractors_per_entity=0, seed=2)
    assert len(ds.queries) == 8
    # Each entity appears in exactly one fact
    entities = [meta["entity"] for _, meta in ds.memories
                if meta.get("kind") == "fact"]
    assert len(entities) == len(set(entities)) == 8


def test_high_collision_each_entity_has_K_facts():
    K = 6
    ds = generate_dataset(n_entities=4, collision_degree=K,
                          distractors_per_entity=0, seed=3)
    by_entity: dict[str, list] = {}
    for _, meta in ds.memories:
        if meta.get("kind") == "fact":
            by_entity.setdefault(meta["entity"], []).append(meta["disc"])
    assert len(by_entity) == 4
    for ent, discs in by_entity.items():
        assert len(discs) == K
        assert len(set(discs)) == K  # discriminators unique within entity


def test_query_paraphrases_discriminator():
    """The query must NOT contain the memory's discriminator surface
    token — it should use the synonym instead. That's the whole point."""
    ds = generate_dataset(n_entities=8, collision_degree=4,
                          distractors_per_entity=0, seed=4)
    # Build memory by (entity, disc) lookup
    spec = _SPECS["preference"]
    syn_map = {disc: syn for disc, syn, _ in spec["discs"]}
    # Each query's disc-syn must appear in the query, and the matching
    # disc surface form must NOT appear in the query.
    seen_at_least_one = False
    for q in ds.queries:
        # Find which discriminator this query is asking about by checking
        # which synonym appears in the query text.
        matched_syn = None
        for disc, syn in syn_map.items():
            if syn.lower() in q.text.lower():
                matched_syn = (disc, syn)
                break
        if matched_syn is None:
            continue
        disc, syn = matched_syn
        # Surface-level: the discriminator token should not be in the query
        assert disc.lower() not in q.text.lower(), (disc, q.text)
        seen_at_least_one = True
    assert seen_at_least_one


def test_anchors_findable_in_memories():
    """For each query, at least one fact memory in the corpus contains
    the expected_substring (the answer). Otherwise ground truth is broken."""
    ds = generate_dataset(n_entities=6, collision_degree=4,
                          distractors_per_entity=2, seed=5)
    fact_text = " ".join(m.lower() for m, meta in ds.memories
                         if meta.get("kind") == "fact")
    for q in ds.queries:
        for anchor in q.expected_substrings:
            assert anchor.lower() in fact_text


def test_seed_reproducibility():
    a = generate_dataset(n_entities=4, collision_degree=3, seed=99)
    b = generate_dataset(n_entities=4, collision_degree=3, seed=99)
    assert [m for m, _ in a.memories] == [m for m, _ in b.memories]
    assert [q.text for q in a.queries] == [q.text for q in b.queries]


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        generate_dataset(collision_degree=0)
    with pytest.raises(ValueError):
        generate_dataset(tag="not-a-tag")
    with pytest.raises(ValueError):
        # discriminator vocab has 8 entries; asking for 9 must fail
        generate_dataset(n_entities=2, collision_degree=99)
    with pytest.raises(ValueError):
        generate_dataset(n_entities=10_000, collision_degree=1)


def test_all_tags_generate_cleanly():
    """Each tag has at least 8 discriminators and produces valid datasets."""
    for tag in _SPECS:
        ds = generate_dataset(n_entities=2, collision_degree=2,
                              distractors_per_entity=0, seed=7, tag=tag)
        assert len(ds.queries) == 4
        assert len(_SPECS[tag]["discs"]) >= 8


def test_entity_pool_large_enough():
    assert len(_ENTITIES) >= 8


def test_paraphrase_memory_uses_variants_and_preserves_anchors():
    """With paraphrase_memory=True, fact memories are drawn from
    spec.memory_variants (>1 distinct surface forms appear over a
    moderate sample), but every query's expected_substring (the answer)
    is still locatable in some fact memory — ground truth intact."""
    ds = generate_dataset(
        n_entities=8, collision_degree=4, distractors_per_entity=0,
        seed=11, tag="tool", paraphrase_memory=True,
    )
    fact_mems = [m for m, meta in ds.memories if meta.get("kind") == "fact"]
    # At least 2 distinct surface templates actually appear (otherwise
    # paraphrase has no effect on this seed).
    # Strip entity/disc/answer-specific tokens by counting unique
    # leading bigrams as a cheap surface proxy.
    leading_bigrams = {tuple(m.lower().split()[:2]) for m in fact_mems}
    assert len(leading_bigrams) >= 2, leading_bigrams
    # Anchors still findable.
    fact_text = " ".join(m.lower() for m in fact_mems)
    for q in ds.queries:
        for anchor in q.expected_substrings:
            assert anchor.lower() in fact_text


def test_paraphrase_memory_seed_reproducibility():
    a = generate_dataset(n_entities=4, collision_degree=3, seed=99,
                         tag="tool", paraphrase_memory=True)
    b = generate_dataset(n_entities=4, collision_degree=3, seed=99,
                         tag="tool", paraphrase_memory=True)
    assert [m for m, _ in a.memories] == [m for m, _ in b.memories]


def test_all_tags_have_memory_variants():
    """Every tag must declare ≥3 paraphrased memory variants for §6.1
    coverage. Locks the contract so we don't silently regress when a
    new tag is added."""
    for tag, spec in _SPECS.items():
        variants = spec.get("memory_variants")
        assert variants is not None, f"tag {tag} missing memory_variants"
        assert len(variants) >= 3, f"tag {tag} only has {len(variants)} variants"
        # All variants must use the same {entity}/{disc}/{answer} slots.
        for v in variants:
            assert "{entity}" in v and "{disc}" in v and "{answer}" in v, (tag, v)
