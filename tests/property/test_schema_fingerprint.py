"""Property tests for schema_fingerprint.

Locks invariants S1-S8 documented in
``src/engram/consolidation/schema_fingerprint.py``.
"""

from __future__ import annotations

import string

import pytest
from hypothesis import given, settings, strategies as st

from engram.consolidation.schema_fingerprint import (
    STOP_WORDS,
    fingerprint,
    fingerprints,
)


# Hypothesis strategies. Use printable ASCII to keep the tokenizer
# focused on what we actually care about (alphanumeric splits).
_PRINTABLE = st.text(alphabet=string.printable, min_size=0, max_size=80)
_FACTS = st.lists(_PRINTABLE, min_size=0, max_size=10)


@given(facts=_FACTS)
@settings(max_examples=200, deadline=None)
def test_S1_determinism(facts):
    """S1: same input list ⇒ same output frozenset."""
    assert fingerprint(facts) == fingerprint(facts)


@pytest.mark.parametrize("facts", [
    [],
    [""],
    ["", ""],
    ["the and for"],   # all stop-words
    ["a b c"],         # all length < 3
    ["!!! ??? ..."],   # no alphanumeric
])
def test_S2_empty_handling(facts):
    """S2: degenerate inputs ⇒ empty frozenset (singleton signal)."""
    assert fingerprint(facts) == frozenset()


@given(facts=_FACTS, extra=_FACTS)
@settings(max_examples=150, deadline=None)
def test_S3_subset_monotonicity(facts, extra):
    """S3: ``fp(facts) ⊆ fp(facts + extra)``. Adding facts never shrinks."""
    base = fingerprint(facts)
    grown = fingerprint(list(facts) + list(extra))
    assert base.issubset(grown)


@given(facts=_FACTS, seed=st.integers(min_value=0, max_value=2**31 - 1))
@settings(max_examples=150, deadline=None)
def test_S4_order_invariance(facts, seed):
    """S4: shuffling fact list ⇒ identical fingerprint."""
    import random
    shuffled = list(facts)
    random.Random(seed).shuffle(shuffled)
    assert fingerprint(shuffled) == fingerprint(facts)


@pytest.mark.parametrize("words", [
    ("hello", "Hello", "HELLO", "HeLLo"),
    ("python", "PYTHON", "Python", "pYthOn"),
])
def test_S5_case_insensitivity(words):
    """S5: case folding to lowercase before tokenizing."""
    fps = {fingerprint([w]) for w in words}
    assert len(fps) == 1


def test_S6_punctuation_invariance():
    """S6: punctuation is ignored; tokens are alphanumeric runs."""
    a = fingerprint(["hello, world!"])
    b = fingerprint(["hello world"])
    c = fingerprint(["hello---world"])
    assert a == b == c == frozenset({"hello", "world"})


@given(facts=_FACTS)
@settings(max_examples=150, deadline=None)
def test_S7_stopwords_excluded(facts):
    """S7: no token in STOP_WORDS ever appears in output."""
    out = fingerprint(facts)
    assert out.isdisjoint(STOP_WORDS)


@given(facts=_FACTS)
@settings(max_examples=150, deadline=None)
def test_S8_min_length(facts):
    """S8: tokens of length < 3 are excluded."""
    out = fingerprint(facts)
    assert all(len(t) >= 3 for t in out)


# --- Smoke / integration ---

def test_smoke_realistic_schema():
    """Realistic schema-update pattern: prefers, dietary, restrictions."""
    facts = [
        "Alice prefers tea over coffee in the morning.",
        "Alice has dietary restrictions: vegetarian.",
        "Alice avoids dairy products.",
    ]
    fp = fingerprint(facts)
    assert "alice" in fp
    assert "prefers" in fp
    assert "dietary" in fp
    # stop-words gone
    assert "the" not in fp
    assert "has" not in fp


def test_bulk_fingerprints_preserves_keys():
    """Bulk variant maps every input schema to a fingerprint."""
    inp = {
        "s1": ["Alice prefers tea."],
        "s2": ["Bob plays guitar."],
        "s3": [],
    }
    out = fingerprints(inp)
    assert set(out) == {"s1", "s2", "s3"}
    assert "alice" in out["s1"] and "prefers" in out["s1"]
    assert "bob" in out["s2"] and "guitar" in out["s2"]
    assert out["s3"] == frozenset()


def test_feeds_cluster_directly():
    """fingerprints() output is consumable by schema_family.cluster()."""
    from engram.consolidation.schema_family import cluster
    inp = {
        "s1": ["Alice prefers tea."],
        "s2": ["Alice prefers coffee."],   # shares "alice", "prefers"
        "s3": ["Bob plays guitar."],       # disjoint
    }
    fps = fingerprints(inp)
    parts = cluster(fps, tau=0.3)
    # s1 and s2 share 2 tokens out of (3+3-2)=4 → Jaccard 0.5 ≥ 0.3
    # s3 disjoint
    s1_cluster = next(c for c in parts if "s1" in c)
    assert "s2" in s1_cluster
    assert "s3" not in s1_cluster


def test_cross_metric_disjoint_vocab_singletons():
    """Schemas with disjoint vocab ⇒ all singletons even at low tau."""
    from engram.consolidation.schema_family import cluster
    inp = {
        "s1": ["alpha beta gamma"],
        "s2": ["delta epsilon zeta"],
        "s3": ["eta theta iota"],
    }
    fps = fingerprints(inp)
    parts = cluster(fps, tau=0.01)
    assert len(parts) == 3
