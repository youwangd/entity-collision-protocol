"""Schema fingerprinting — supporting-facts → token-set adapter.

The §8 prior-sharing wire-up needs a per-schema feature set to feed
`schema_family.cluster()`. The most defensible feature today is the
**bag of distinctive content tokens** drawn from the schema's own
supporting_facts in the current window — schemas that fire on
overlapping vocabulary are the ones whose lifecycle decisions should
share priors.

This module is the **pure adapter** between Stage 6's per-schema
``supporting_facts: list[str]`` and `schema_family.cluster()`'s
``dict[schema_id, frozenset[token]]`` input. No clocks, no RNG, no I/O.

**Why not typed properties?** ``MemoryType.SCHEMA`` memories don't yet
carry typed property scaffolding (NEXT.md run #64). Until they do,
content tokens are the only schema-level signal available.

**Why not raw words?** Raw whitespace splits would let stop-words
("the", "is", "a") dominate Jaccard — every schema would be in one
giant cluster. We strip a small, conservative stop-list and minimum-
length-3 tokens, lowercased, alphanumeric only. This matches the
heuristic Stage 5 (InterferenceDetection) and Stage 6's own
contradiction-counter use, so the fingerprint is consistent with how
the rest of the consolidation pipeline reasons about schema content.

Design invariants (locked in
``tests/property/test_schema_fingerprint.py``):

  S1. Determinism: same input ⇒ same output (frozensets are unordered,
      iteration over the supporting-facts list does not change tokens).
  S2. Empty-input handling: empty list / empty strings / all-stopword
      facts ⇒ empty frozenset (the canonical "no fingerprint" signal,
      which `schema_family.cluster()` correctly turns into a singleton).
  S3. Subset monotonicity: adding facts can only grow the token set,
      never shrink it. ``fp(facts) ⊆ fp(facts + extra)``.
  S4. Order-invariance: ``fp(facts)`` is independent of the order of
      facts in the list.
  S5. Case-insensitivity: ``fp(["Hello"]) == fp(["hello"]) == fp(["HELLO"])``.
  S6. Punctuation-invariance: tokenizer ignores non-alphanumeric.
  S7. Stop-words excluded: tokens in STOP_WORDS never appear in output.
  S8. Min-length: tokens of length < 3 never appear in output.

What's NOT in this module:

  * Stage-6 plumbing. That call site (in ``pipeline.py``) is still
    deferred until we collect per-schema EvidenceWindows in one place
    (NEXT.md "Next pickup").
  * Property weighting / TF-IDF. Pure presence/absence is enough for
    Jaccard at the cluster granularity §8 needs.
  * Cross-window aggregation. One window's facts are one window's
    fingerprint; multi-window decay is `schema_decay`'s job.
"""

from __future__ import annotations

import re
from typing import Iterable, Mapping

# Conservative stop-list. Smaller is safer: false-merging is worse
# than false-splitting for §8 (false-merge means a deprecate on schema
# A leaks priors into unrelated schema B).
STOP_WORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "any",
    "had", "her", "was", "one", "our", "out", "day", "get", "has",
    "him", "his", "how", "man", "new", "now", "old", "see", "two",
    "way", "who", "boy", "did", "its", "let", "put", "say", "she",
    "too", "use", "with", "this", "that", "from", "have", "been",
    "they", "were", "what", "your", "when", "will", "would", "there",
    "their", "them", "these", "those", "then", "than", "into", "such",
    "some", "more", "most", "much", "very", "also", "just", "only",
})

_TOKEN_RE = re.compile(r"[a-z0-9]+")
_MIN_LEN = 3


def fingerprint(supporting_facts: Iterable[str]) -> frozenset[str]:
    """Convert one schema's supporting_facts to a content-token frozenset.

    Pure. Deterministic. See module docstring for invariants S1-S8.
    """
    tokens: set[str] = set()
    for fact in supporting_facts:
        if not fact:
            continue
        for tok in _TOKEN_RE.findall(fact.lower()):
            if len(tok) < _MIN_LEN:
                continue
            if tok in STOP_WORDS:
                continue
            tokens.add(tok)
    return frozenset(tokens)


def fingerprints(
    supporting_facts_by_schema: Mapping[str, Iterable[str]],
) -> dict[str, frozenset[str]]:
    """Bulk variant: ``schema_id → supporting_facts`` to ``schema_id → tokens``.

    Output is a fresh dict; safe to feed directly to
    `schema_family.cluster()`. Order of keys is preserved from input
    (Python dict insertion order); `cluster()` itself canonicalizes.
    """
    return {sid: fingerprint(facts) for sid, facts in supporting_facts_by_schema.items()}
