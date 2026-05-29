"""Hard-distractor dataset invariants.

The adversarial-distractor mode plants memories that share entity tokens
with a planted fact but do NOT contain the answer anchor. This is the
ammunition that keeps the bench from saturating on BM25 alone.
"""
from __future__ import annotations

import pytest

from evals.synthetic import generate_dataset


@pytest.mark.evals
def test_hard_distractors_do_not_contain_answer_anchors():
    ds = generate_dataset(
        n_sessions=5, facts_per_session=4,
        distractors_per_session=2, hard_distractors_per_fact=3, seed=13,
    )
    hards = [(c, m) for c, m in ds.memories if m.get("kind") == "hard_distractor"]
    assert hards, "expected hard distractors to be planted"
    # Every fact has a query whose expected_substrings are the answer anchors.
    # No hard distractor should contain *all* anchors of any query that shares
    # its tag (otherwise it would silently be a correct answer).
    for hd_text, hd_meta in hards:
        for q in ds.queries:
            if hd_meta["tag"] not in q.tags:
                continue
            needles = [s.lower() for s in q.expected_substrings if s]
            if needles and all(n in hd_text.lower() for n in needles):
                pytest.fail(
                    f"hard distractor accidentally answers a query: {hd_text!r} "
                    f"vs anchors {q.expected_substrings}"
                )


@pytest.mark.evals
def test_hard_distractor_count_scales_with_flag():
    base = generate_dataset(n_sessions=3, facts_per_session=4,
                            distractors_per_session=0, hard_distractors_per_fact=0, seed=1)
    hard = generate_dataset(n_sessions=3, facts_per_session=4,
                            distractors_per_session=0, hard_distractors_per_fact=4, seed=1)
    base_hd = sum(1 for _, m in base.memories if m.get("kind") == "hard_distractor")
    hard_hd = sum(1 for _, m in hard.memories if m.get("kind") == "hard_distractor")
    assert base_hd == 0
    # 3 sessions × 4 facts × 4 hard distractors = 48 max (some templates may be
    # filtered if they accidentally contain the answer anchor; allow slight slack)
    assert 40 <= hard_hd <= 48
    # Same number of facts and queries regardless of hard-distractor count
    assert len(base.queries) == len(hard.queries) == 12


@pytest.mark.evals
def test_hard_distractors_share_entity_tokens_with_facts():
    """Sanity: hard distractors should share entity tokens with the planted
    fact of the same tag — that's the whole point of being adversarial."""
    ds = generate_dataset(
        n_sessions=4, facts_per_session=5,
        distractors_per_session=0, hard_distractors_per_fact=2, seed=99,
    )
    facts_by_tag: dict[str, list[str]] = {}
    for c, m in ds.memories:
        if m.get("kind") == "fact":
            facts_by_tag.setdefault(m["tag"], []).append(c.lower())

    sharing = 0
    total = 0
    for c, m in ds.memories:
        if m.get("kind") != "hard_distractor":
            continue
        total += 1
        tag = m["tag"]
        hd_tokens = set(c.lower().split())
        # Stopwords/common-noise we don't count as "shared entity"
        hd_tokens -= {
            "the", "a", "an", "and", "or", "of", "for", "in", "to", "on",
            "at", "is", "was", "be", "but", "this", "that",
        }
        if any(hd_tokens & set(f.split()) for f in facts_by_tag.get(tag, [])):
            sharing += 1
    # Almost all hard distractors should share tokens with their tag's facts.
    assert sharing / max(total, 1) >= 0.9
