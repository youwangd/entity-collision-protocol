"""Unit tests for the heuristic question-type classifier (§D15)."""

from __future__ import annotations

from engram.retrieval.type_classifier import (
    ALL_TYPES,
    DEFAULT_PRF_ALLOW,
    TYPE_KNOW_UPD,
    TYPE_MULTI,
    TYPE_SS_AST,
    TYPE_SS_PREF,
    TYPE_SS_USER,
    TYPE_TEMPORAL,
    classify_question_type,
    should_expand_for_type,
)


def test_label_set_matches_taxonomy():
    assert set(ALL_TYPES) == {
        TYPE_SS_USER,
        TYPE_SS_PREF,
        TYPE_SS_AST,
        TYPE_MULTI,
        TYPE_TEMPORAL,
        TYPE_KNOW_UPD,
    }


def test_empty_query_returns_unknown():
    g = classify_question_type("")
    assert g.label is None
    assert g.confidence == 0.0


def test_single_session_assistant_distinctive():
    qs = [
        "Can you remind me what was the chord progression for that sad song?",
        "I was going through our previous chat about the shift rotation sheet.",
        "I'm looking back at our previous conversation about cocktails.",
        "I remember you told me about CITGO's three refineries.",
    ]
    for q in qs:
        assert classify_question_type(q).label == TYPE_SS_AST, q


def test_temporal_reasoning_patterns():
    qs = [
        "How many days passed between my MoMA visit and the exhibit?",
        "Which book did I finish reading first, 'The Hate U Give' or 'The Nightingale'?",
        "How long did I use my new binoculars before I saw the goldfinches?",
        "How many weeks ago did I meet up with my aunt?",
    ]
    for q in qs:
        assert classify_question_type(q).label == TYPE_TEMPORAL, q


def test_preference_recommendation_asks():
    qs = [
        "Can you recommend some resources where I can learn more about video editing?",
        "Any documentary recommendations?",
        "Do you have any helpful tips?",
        "Any advice on getting better results with my slow cooker?",
    ]
    for q in qs:
        assert classify_question_type(q).label == TYPE_SS_PREF, q


def test_knowledge_update_currentness():
    qs = [
        "How many bikes do I currently own?",
        "What brand of BBQ sauce am I currently obsessed with?",
        "Where do I currently keep my old sneakers?",
        "What type of camera lens did I purchase most recently?",
    ]
    for q in qs:
        assert classify_question_type(q).label == TYPE_KNOW_UPD, q


def test_multi_session_aggregate_cues():
    qs = [
        "What is the total number of siblings I have?",
        "How many hours of jogging and yoga did I do last week in total?",
    ]
    for q in qs:
        assert classify_question_type(q).label == TYPE_MULTI, q


def test_fallback_is_low_confidence_user():
    g = classify_question_type("What degree did I graduate with?")
    # Falls through cascade → fallback bucket.
    assert g.label == TYPE_SS_USER
    assert g.confidence < 0.5


def test_should_expand_gating_logic():
    # No allow-set → never expand.
    assert should_expand_for_type(TYPE_KNOW_UPD, None) is False
    assert should_expand_for_type(None, DEFAULT_PRF_ALLOW) is False
    # Membership check.
    assert should_expand_for_type(TYPE_KNOW_UPD, DEFAULT_PRF_ALLOW) is True
    assert should_expand_for_type(TYPE_SS_PREF, DEFAULT_PRF_ALLOW) is True
    # The two §D14-regressing types are NOT in default allow-set.
    assert should_expand_for_type(TYPE_SS_USER, DEFAULT_PRF_ALLOW) is False
    assert should_expand_for_type(TYPE_TEMPORAL, DEFAULT_PRF_ALLOW) is False


def test_aggregate_accuracy_on_lme_s_held_in_check():
    """Sanity: classifier accuracy on the LongMemEval-S labels.

    Not a strict bound — just guards against accidental regressions.
    Measured 2026-05-23: overall acc=0.548 on n=500.
    """
    import json
    import pathlib

    p = pathlib.Path("data/longmemeval/longmemeval_s.json")
    if not p.exists():
        return  # skip silently when dataset not present
    data = json.loads(p.read_text())
    correct = sum(
        1
        for q in data
        if classify_question_type(q["question"]).label == q["question_type"]
    )
    # Lock in current behaviour (>50%); regressions below this fail.
    assert correct / len(data) >= 0.50, f"acc dropped: {correct}/{len(data)}"
