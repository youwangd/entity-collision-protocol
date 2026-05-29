"""Tests for the D1 v0.3 multi-entity-hard fixture.

Validates the *hardness* contract:
- BM25 (lexical) hit@5 stays below 0.7 on the default config, i.e. the
  fixture is not BM25-saturated. This is the property that makes the
  fixture worth running PRF / type-aware NER experiments on.
- Reproducibility under fixed seed.
- Shape sanity (every query has a unique gold fact_id, gold memory
  exists in the corpus, expected_substrings is non-empty).
"""
from __future__ import annotations

import math
import re


from evals.corpora.multi_entity_hard import (
    HardFixtureConfig,
    generate_multi_entity_hard,
)


def test_reproducible_under_fixed_seed():
    cfg = HardFixtureConfig(n_facts=40, n_sessions=4, seed=11)
    a = generate_multi_entity_hard(cfg)
    b = generate_multi_entity_hard(cfg)
    assert [m[0] for m in a.memories] == [m[0] for m in b.memories]
    assert [q.text for q in a.queries] == [q.text for q in b.queries]


def test_shape_invariants():
    cfg = HardFixtureConfig(
        n_facts=40, n_sessions=4,
        distractors_per_fact=3, high_overlap_per_fact=1, seed=1,
    )
    ds = generate_multi_entity_hard(cfg)
    # n_facts gold + n_facts*(distractors+high_overlap) distractors
    assert len(ds.memories) == 40 * (1 + 3 + 1)
    assert len(ds.queries) == 40
    # every query has a gold answer substring
    assert all(q.expected_substrings for q in ds.queries)
    # every fact_id has exactly one fact memory
    fact_counts: dict[str, int] = {}
    for _text, meta in ds.memories:
        if meta.get("kind") == "fact":
            fact_counts[meta["fact_id"]] = fact_counts.get(meta["fact_id"], 0) + 1
    assert all(c == 1 for c in fact_counts.values())
    assert len(fact_counts) == 40


def _bm25_like_score(doc: str, query_tokens: list[str]) -> float:
    """Pretend-BM25: bag-of-words overlap, normalized by doc length.

    Good enough to test the saturation property — we want to show that
    a lexical retriever cannot uniquely pick the gold from the
    type-collision distractors, since by construction every distractor
    re-uses query tokens.
    """
    toks = re.findall(r"[a-z0-9_]+", doc.lower())
    if not toks:
        return 0.0
    qset = set(t.lower() for t in query_tokens)
    overlap = sum(1 for t in toks if t in qset)
    # length-normalized so longer distractors don't auto-win
    return overlap / math.sqrt(len(toks))


def test_bm25_saturation_below_target():
    """The corpus must NOT be BM25-saturated.

    Acceptance: lexical hit@5 < 0.7 on the default-hardness config.
    If this drifts upward, the fixture has lost its discriminative
    power and PRF/NER experiments on it become uninteresting.
    """
    cfg = HardFixtureConfig(
        n_facts=200, n_sessions=20,
        lexical_collision_rate=1.0, ner_disambig_rate=1.0, seed=42,
    )
    ds = generate_multi_entity_hard(cfg)
    # Index gold fact_id by memory text identity for fast scoring.
    fact_text_by_id = {
        meta["fact_id"]: text
        for text, meta in ds.memories if meta.get("kind") == "fact"
    }

    hits_at_5 = 0
    for q in ds.queries:
        fact_id = next(t.split("=", 1)[1] for t in q.tags if t.startswith("fact_id="))
        gold_text = fact_text_by_id[fact_id]
        q_tokens = re.findall(r"[a-z0-9_]+", q.text.lower())
        scored = sorted(
            ds.memories,
            key=lambda m: _bm25_like_score(m[0], q_tokens),
            reverse=True,
        )
        top5 = [m[0] for m in scored[:5]]
        if gold_text in top5:
            hits_at_5 += 1

    bm25_hit5 = hits_at_5 / len(ds.queries)
    # Hardness contract: BM25 alone leaves >= 30% of queries unsolved at top-5
    assert bm25_hit5 < 0.7, (
        f"Fixture has lost hardness: BM25-like hit@5={bm25_hit5:.3f} "
        f"(want < 0.70). Re-tune lexical_collision_rate or distractor count."
    )


def test_collision_rate_zero_makes_fixture_easy():
    """Sanity: turning collisions OFF should make BM25 trivially win.

    This proves the hardness comes from the collision design, not from
    some other quirk of the generator.
    """
    cfg = HardFixtureConfig(
        n_facts=80, n_sessions=8,
        lexical_collision_rate=0.0, ner_disambig_rate=0.0,
        distractors_per_fact=2, high_overlap_per_fact=0, seed=42,
    )
    ds = generate_multi_entity_hard(cfg)
    fact_text_by_id = {
        meta["fact_id"]: text
        for text, meta in ds.memories if meta.get("kind") == "fact"
    }
    hits = 0
    for q in ds.queries:
        fact_id = next(t.split("=", 1)[1] for t in q.tags if t.startswith("fact_id="))
        gold_text = fact_text_by_id[fact_id]
        q_tokens = re.findall(r"[a-z0-9_]+", q.text.lower())
        scored = sorted(
            ds.memories,
            key=lambda m: _bm25_like_score(m[0], q_tokens),
            reverse=True,
        )
        if gold_text in [m[0] for m in scored[:5]]:
            hits += 1
    # When collisions are off, BM25 should crush this. Generous floor
    # because gold memories are *very* short ("X works at Y.") and
    # length normalization can still introduce ties. The point is just
    # that easy ≫ hard.
    assert hits / len(ds.queries) >= 0.6
