"""Smoke + sanity tests for the evals/ benchmark scaffold.

These run a TINY benchmark in the default suite to make sure the harness
itself is healthy. The big benchmark numbers come from `python -m evals.run`.
"""
from __future__ import annotations

import pytest

from evals.synthetic import generate_dataset, generate_cross_session_dataset
from evals.metrics import find_match_rank, hit_at_k, mrr, ndcg_at_k
from evals.run import run_eval


def test_synthetic_dataset_is_reproducible():
    a = generate_dataset(n_sessions=3, facts_per_session=2, distractors_per_session=2, seed=7)
    b = generate_dataset(n_sessions=3, facts_per_session=2, distractors_per_session=2, seed=7)
    assert [m[0] for m in a.memories] == [m[0] for m in b.memories]
    assert [q.text for q in a.queries] == [q.text for q in b.queries]


def test_synthetic_dataset_has_expected_shape():
    ds = generate_dataset(n_sessions=4, facts_per_session=3, distractors_per_session=5, seed=1)
    # 4 sessions * (3 facts + 5 distractors) = 32 memories, 12 queries
    assert len(ds.memories) == 32
    assert len(ds.queries) == 12
    # Every query should have at least one expected_substring
    assert all(q.expected_substrings for q in ds.queries)


def test_metrics_basic():
    # 3 queries: ranks 0, 2, None
    ranks = [0, 2, None]
    assert hit_at_k(ranks, 1) == pytest.approx(1 / 3)
    assert hit_at_k(ranks, 5) == pytest.approx(2 / 3)
    # MRR: (1/1 + 1/3 + 0) / 3
    assert mrr(ranks) == pytest.approx((1.0 + 1 / 3) / 3)
    # nDCG@5 with binary relevance, single doc: (1/log2(2) + 1/log2(4) + 0) / 3
    import math
    expected_ndcg = (1.0 + 1.0 / math.log2(4)) / 3
    assert ndcg_at_k(ranks, 5) == pytest.approx(expected_ndcg)


def test_find_match_rank():
    class _R:
        def __init__(self, content: str):
            self.content = content
    results = [_R("alpha beta"), _R("gamma delta epsilon"), _R("zeta")]
    assert find_match_rank(results, ["gamma", "delta"]) == 1
    assert find_match_rank(results, ["nope"]) is None
    assert find_match_rank(results, []) is None


def test_strict_paraphrase_has_minimal_non_entity_overlap():
    """Strict-paraphrase queries should share only entity tokens with their facts.

    This is the whole point of the mode: it kills lexical retrieval so the
    benefit of semantic embedding can be measured.
    """
    import re
    ds = generate_dataset(
        n_sessions=5, facts_per_session=4, distractors_per_session=0,
        seed=23, strict_paraphrase=True,
    )
    fact_memories = [m for m, meta in ds.memories if meta.get("kind") == "fact"]
    assert len(fact_memories) == len(ds.queries) == 20

    # Stop-words we don't count as overlap.
    STOP = {"the", "a", "an", "is", "are", "was", "were", "of", "for", "on", "in",
            "to", "by", "and", "or", "do", "does", "did", "what", "when", "where",
            "who", "which", "how", "that", "this", "it", "its", "be", "been", "at"}
    tok = lambda s: set(w for w in re.findall(r"[a-zA-Z0-9_]+", s.lower()) if w not in STOP)

    overlaps_per_query = []
    # ds.memories is shuffled, but ds.queries is in fact-creation order.
    # Match each query to its fact via the answer-anchor: the anchor uniquely
    # identifies the source memory (entity tokens are random per fact).
    for query in ds.queries:
        # Find the fact memory containing all of this query's answer anchors
        candidates = [
            m for m in fact_memories
            if all(a in m for a in query.expected_substrings)
        ]
        if not candidates:
            continue
        mem = candidates[0]
        m_tokens = tok(mem)
        q_tokens = tok(query.text)
        anchor_set = {a.lower() for a in query.expected_substrings}
        overlap = (m_tokens & q_tokens) - anchor_set
        overlaps_per_query.append(len(overlap))

    avg = sum(overlaps_per_query) / max(1, len(overlaps_per_query))
    # Entity tokens (user, service, host, bug-id digits) are necessarily shared
    # between memory and query — they're the question's referent. After
    # subtracting the answer anchor, ~1-2 entity tokens remain. The point of
    # strict-paraphrase is that the *predicate* words don't overlap; we test
    # that loosely here.
    assert avg < 2.5, f"strict-paraphrase too lexically similar: avg overlap = {avg:.2f}"


@pytest.mark.evals
def test_tiny_eval_runs_end_to_end():
    """Smoke: tiny eval completes and metrics are in [0, 1]."""
    metrics = run_eval(n_sessions=2, facts_per_session=2, distractors_per_session=3, seed=11, k=5)
    assert metrics["n_queries"] == 4
    for m in ("hit_at_1", "hit_at_5", "hit_at_k", "mrr", "ndcg_at_k"):
        assert 0.0 <= metrics[m] <= 1.0


def test_cross_session_dataset_is_reproducible():
    a = generate_cross_session_dataset(n_facts=10, n_sessions=4,
                                       distractors_per_session=2, seed=7)
    b = generate_cross_session_dataset(n_facts=10, n_sessions=4,
                                       distractors_per_session=2, seed=7)
    assert [m[0] for m in a.memories] == [m[0] for m in b.memories]
    assert [q.text for q in a.queries] == [q.text for q in b.queries]


def test_cross_session_dataset_shape_and_pairing():
    """Each fact has TWO planted halves in TWO distinct sessions, plus
    answer anchors that appear in both halves."""
    ds = generate_cross_session_dataset(
        n_facts=12, n_sessions=5, distractors_per_session=3, seed=1,
    )
    facts = [(t, m) for t, m in ds.memories if m.get("kind") == "fact"]
    assert len(facts) == 24  # 12 pairs × 2 halves
    assert len(ds.queries) == 12

    # Group fact memories by pair_id and verify two distinct sessions per pair.
    by_pair: dict[str, list[tuple[str, dict]]] = {}
    for text, meta in facts:
        by_pair.setdefault(meta["pair_id"], []).append((text, meta))
    for pid, halves in by_pair.items():
        assert len(halves) == 2, f"{pid} has {len(halves)} halves"
        sessions = {h[1]["session"] for h in halves}
        assert len(sessions) == 2, f"{pid} both halves in same session"
        # Verify the query for this pair has its sessions tagged
        q = next(q for q in ds.queries if f"pair_id={pid}" in q.tags)
        sess_tags = {t.split("=")[1] for t in q.tags if t.startswith("sess_")}
        assert {str(s) for s in sessions} == sess_tags
        # Each anchor must appear in BOTH halves (paraphrases of same fact)
        for a in q.expected_substrings:
            assert all(a in h[0] for h in halves), \
                f"anchor {a!r} missing from a half of {pid}"


def test_cross_session_rejects_too_few_sessions():
    with pytest.raises(ValueError):
        generate_cross_session_dataset(n_facts=3, n_sessions=1, seed=0)
