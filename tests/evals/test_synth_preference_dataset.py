"""Smoke tests for evals.synthetic.generate_preference_dataset (§D15c)."""
from __future__ import annotations

from evals.synthetic import generate_preference_dataset
from engram.retrieval.type_classifier import classify_question_type


def test_generate_preference_dataset_basic_shape():
    ds = generate_preference_dataset(
        n_facts=20, distractors_per_fact=2,
        hard_distractors_per_fact=2, seed=42,
    )
    # One query per fact; one fact memory per fact + distractors.
    assert len(ds.queries) == 20
    facts = [m for m, meta in ds.memories if meta.get("kind") == "fact"]
    assert len(facts) == 20
    # Every query carries a non-empty preference anchor.
    for q in ds.queries:
        assert q.expected_substrings
        assert q.expected_substrings[0]


def test_every_query_classifies_as_preference():
    """The whole point of this corpus: queries map to TYPE_SS_PREF.

    If this regresses, the §D15c corroboration loses its mechanism.
    """
    ds = generate_preference_dataset(
        n_facts=50, distractors_per_fact=0,
        hard_distractors_per_fact=0, seed=42,
    )
    misclassified = []
    for q in ds.queries:
        guess = classify_question_type(q.text)
        if guess.label != "single-session-preference":
            misclassified.append((q.text, guess.label))
    assert not misclassified, (
        f"{len(misclassified)} queries failed to classify as "
        f"single-session-preference; first few: {misclassified[:3]}"
    )


def test_seed_determinism():
    a = generate_preference_dataset(n_facts=10, seed=7)
    b = generate_preference_dataset(n_facts=10, seed=7)
    assert [m for m, _ in a.memories] == [m for m, _ in b.memories]
    assert [q.text for q in a.queries] == [q.text for q in b.queries]


def test_answer_anchor_tokens_token_count_matches():
    """When ``answer_anchor_tokens=k>0`` every planted anchor has k tokens."""
    import pytest

    for k in (1, 2, 3):
        ds = generate_preference_dataset(
            n_facts=30, distractors_per_fact=0,
            hard_distractors_per_fact=0, seed=42,
            answer_anchor_tokens=k,
        )
        for q in ds.queries:
            anchor = q.expected_substrings[0]
            assert len(anchor.split()) == k, (
                f"anchor {anchor!r} has {len(anchor.split())} tokens, want {k}"
            )

    with pytest.raises(ValueError):
        generate_preference_dataset(n_facts=2, answer_anchor_tokens=99)


def test_answer_anchor_tokens_default_is_legacy():
    """Default (k=0) reproduces the original 16-pref legacy corpus byte-for-byte."""
    ds_default = generate_preference_dataset(n_facts=10, seed=7)
    ds_explicit = generate_preference_dataset(
        n_facts=10, seed=7, answer_anchor_tokens=0,
    )
    assert [m for m, _ in ds_default.memories] == [
        m for m, _ in ds_explicit.memories
    ]
    assert [q.text for q in ds_default.queries] == [
        q.text for q in ds_explicit.queries
    ]
