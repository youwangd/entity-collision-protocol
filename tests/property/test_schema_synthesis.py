"""§93 — schema_synthesis unit tests.

Pure-function invariants. No I/O, no clocks. Run in milliseconds.
"""
from __future__ import annotations

import pytest

from engram.consolidation.schema_synthesis import (
    _tokenize,
    synthesize_schemas,
)


class TestTokenize:
    def test_empty_string(self):
        assert _tokenize("") == frozenset()

    def test_min_length_3(self):
        # "a", "is", "to" are all dropped (< 3 chars or stopword)
        assert _tokenize("a is to be") == frozenset()

    def test_drops_stopwords(self):
        toks = _tokenize("the quick brown fox")
        assert "the" not in toks
        assert "quick" in toks
        assert "brown" in toks
        assert "fox" in toks

    def test_lowercase_and_punctuation(self):
        assert _tokenize("Alice's Pizza!") == frozenset({"alice", "pizza"})

    def test_deterministic(self):
        # Same input twice ⇒ same set (frozenset is unordered, but eq is content)
        a = _tokenize("Alice loves pizza")
        b = _tokenize("Alice loves pizza")
        assert a == b


class TestSynthesizeSchemas:
    def test_empty_input(self):
        assert synthesize_schemas([]) == []

    def test_below_min_supports(self):
        # 2 facts, min_supports=3 default → no schemas
        out = synthesize_schemas(["Alice loves pizza", "Alice eats pizza"])
        assert out == []

    def test_three_isolated_facts_no_schema(self):
        # Three facts with no shared tokens → all singletons → no schema
        out = synthesize_schemas([
            "the quick brown fox",
            "lazy dog sleeps deeply",
            "rainbow umbrella discovered",
        ])
        assert out == []

    def test_basic_clustering(self):
        out = synthesize_schemas([
            "Alice loves pizza",
            "Alice eats pizza often",
            "Pizza is what Alice prefers",
        ])
        assert len(out) == 1
        assert "alice" in out[0]["pattern"]
        assert "pizza" in out[0]["pattern"]
        assert len(out[0]["facts"]) == 3

    def test_multiple_clusters_deterministic_order(self):
        facts = [
            "Alice loves pizza", "Alice eats pizza", "Pizza Alice prefers",
            "Bob hates spinach", "Bob refuses spinach", "Spinach Bob avoids",
            "Carol drinks coffee", "Carol enjoys coffee", "Coffee Carol favorite",
        ]
        out_a = synthesize_schemas(facts)
        out_b = synthesize_schemas(list(reversed(facts)))
        # Same set of clusters regardless of input order. Sort by
        # pattern to compare (note: facts order within a cluster
        # depends on input position, so we compare sorted-fact-sets).
        def _sig(out):
            return sorted(
                (d["pattern"], frozenset(d["facts"])) for d in out
            )
        assert _sig(out_a) == _sig(out_b)
        assert len(out_a) == 3

    def test_pattern_has_recurring_prefix(self):
        out = synthesize_schemas([
            "Alice loves pizza", "Alice eats pizza", "Pizza Alice loves",
        ])
        assert out[0]["pattern"].startswith("recurring: ")

    def test_summary_slot_stable_across_runs(self):
        # SchemaUpdate uses pattern[:80] as `summary`. Identical input
        # must produce identical summary slots so re-emission is
        # detected as BUMP_VERSION not duplicate CREATE.
        facts = ["Alice loves pizza", "Alice eats pizza", "Pizza Alice loves"]
        a = synthesize_schemas(facts)[0]["pattern"][:80]
        b = synthesize_schemas(facts)[0]["pattern"][:80]
        assert a == b

    def test_invalid_tau_raises(self):
        with pytest.raises(ValueError):
            synthesize_schemas(["a b c", "d e f", "g h i"], tau=1.5)

    def test_min_supports_floor_respected(self):
        # min_supports=4 with only 3 cluster members → no output
        out = synthesize_schemas(
            ["Alice pizza", "Alice pizza", "Alice pizza"],
            min_supports=4,
        )
        assert out == []

    def test_size_ordering(self):
        # Larger cluster comes first (sorted by -len(facts)).
        facts = [
            "alpha shared theme", "beta shared theme",  # 2-cluster
            "gamma other group", "delta other group",
            "epsilon other group", "zeta other group",  # 4-cluster
        ]
        out = synthesize_schemas(facts, min_supports=2)
        assert len(out) == 2
        assert len(out[0]["facts"]) >= len(out[1]["facts"])
