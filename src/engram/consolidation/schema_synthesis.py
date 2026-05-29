"""§93 — Deterministic, non-LLM schema synthesis from a fact corpus.

The LLM-driven schema-extraction path in `SchemaUpdate` is the only
place patterns get materialised. In the cron / no-LLM environment this
makes every downstream lifecycle step (CREATE, BUMP_VERSION, RECOVER,
schema_family share, contamination, fragmentation gates) physically
unreachable from the production write path. §91 surfaced this as the
root cause of the §90 vacuous-Δ on LoCoMo.

This module ships a deterministic fallback so the §85 / §87 / §90 / §91
calibration results can be re-validated under a *live* pipeline without
network/LLM dependence.

Algorithm
---------
1. Tokenise each fact into a feature set of "content tokens": lowercase
   alphanumeric words of length ≥ 3, minus a small ENGLISH-STOPWORDS
   list. Punctuation stripped. No stemming (we want
   reproducibility-without-NLTK).
2. Build a `fact_id → token_set` mapping (fact_id is the index into the
   input list — pure, no clocks, no UUIDs).
3. Run `schema_family._cluster_by_jaccard` at threshold `tau` to get a
   single-link agglomerative partition. This is the same primitive
   §85/§87 calibrated.
4. Drop clusters with size < `min_supports` (default 3, matching the
   prompt's "Only report patterns with 3+ supporting facts").
5. For each surviving cluster, build a `pattern` string from the
   highest-frequency shared tokens (token must appear in ≥
   `pattern_token_min_share` of the cluster's facts). Tokens are
   ordered by (-count, token) for determinism, then joined as
   `"<token1>, <token2>, ..."` and prefixed with the constant
   `"recurring: "` so the summary slot (= `pattern[:80]`) is stable
   across runs.
6. Output is sorted by `(-len(facts), pattern)` so two corpora that
   produce the same clusters always emit the same ordered list (LLM
   path is order-stable too via dict iteration of a list literal).

Pure: no I/O, no time, no RNG. Same input ⇒ same output.

Returns the same shape SchemaUpdate already consumes:

    [{"pattern": str, "facts": [str, ...]}, ...]

The integration point in `pipeline.py` is the existing `else:` branch
that runs when `ctx.llm` is `None` or `NoLLMProvider`. We feed the
synthesizer's output through the *same* loop the LLM path uses, so the
schema-family / contamination / fragmentation gates fire identically.

What this is *not*
------------------
- An LLM substitute. It will not surface abstract themes ("user
  prefers vegetarian food"). It surfaces lexical-overlap clusters.
- A retrieval reranker. SchemaUpdate produces SCHEMA-typed memories;
  retrieval is the existing hybrid path.
- An evaluation. Whether this synthesizer translates §87's calibration
  into actual recall lift is the §94 question.
"""
from __future__ import annotations

from collections import Counter
from typing import Iterable, Sequence

from engram.consolidation.schema_family import _cluster_by_jaccard


_ENGLISH_STOPWORDS: frozenset[str] = frozenset({
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had",
    "her", "was", "one", "our", "out", "day", "get", "has", "him", "his",
    "how", "man", "new", "now", "old", "see", "two", "way", "who", "boy",
    "did", "its", "let", "put", "say", "she", "too", "use", "also", "with",
    "from", "this", "that", "have", "they", "will", "what", "when", "your",
    "been", "into", "than", "them", "well", "were", "where", "which",
    "would", "there", "their", "could", "should", "about", "after", "again",
    "before", "being", "between", "during", "every", "while", "these",
    "those", "very", "much", "more", "most", "some", "such", "only", "other",
    "over", "under", "just", "then", "than", "still", "even", "any", "many",
    "like", "make", "made", "made", "make", "makes",
})


def _tokenize(text: str) -> frozenset[str]:
    """Bag-of-word tokenizer. Pure: no stemming, no NLTK, no clocks."""
    if not text:
        return frozenset()
    out: set[str] = set()
    cur: list[str] = []
    for ch in text.lower():
        if ch.isalnum():
            cur.append(ch)
        else:
            if cur:
                tok = "".join(cur)
                if len(tok) >= 3 and tok not in _ENGLISH_STOPWORDS:
                    out.add(tok)
                cur = []
    if cur:
        tok = "".join(cur)
        if len(tok) >= 3 and tok not in _ENGLISH_STOPWORDS:
            out.add(tok)
    return frozenset(out)


def _pattern_from_cluster(
    fact_texts: Sequence[str],
    pattern_token_min_share: float,
    max_tokens: int,
) -> str:
    """Build a deterministic pattern string from a fact cluster.

    Picks tokens that occur in ≥ `pattern_token_min_share` of the
    cluster's facts. Ties broken by (-count, token) for determinism.
    Falls back to the cluster's first fact's tokens when no token
    clears the share floor (e.g. cluster of 2 with only 1 shared
    token at exactly 0.5 share).
    """
    if not fact_texts:
        return "recurring: (empty)"
    counts: Counter[str] = Counter()
    for ft in fact_texts:
        for tok in _tokenize(ft):
            counts[tok] += 1
    n = len(fact_texts)
    floor = max(2, int(round(pattern_token_min_share * n)))
    candidates = [(c, tok) for tok, c in counts.items() if c >= floor]
    if not candidates:
        # Fall back to highest-count tokens at all (≥2 docs).
        candidates = [(c, tok) for tok, c in counts.items() if c >= 2]
    if not candidates:
        # Nothing shared above singleton level; punt to first fact.
        first_tokens = sorted(_tokenize(fact_texts[0]))[:max_tokens]
        return "recurring: " + ", ".join(first_tokens) if first_tokens \
            else "recurring: (no-shared-tokens)"
    # Deterministic order: -count then token alpha
    candidates.sort(key=lambda ct: (-ct[0], ct[1]))
    picked = [tok for _, tok in candidates[:max_tokens]]
    return "recurring: " + ", ".join(picked)


def synthesize_schemas(
    facts: Iterable[str],
    *,
    tau: float = 0.3,
    min_supports: int = 3,
    pattern_token_min_share: float = 0.5,
    max_tokens_per_pattern: int = 8,
) -> list[dict]:
    """Deterministically derive `[{pattern, facts}]` from a fact corpus.

    Args:
      facts: iterable of fact content strings.
      tau: Jaccard threshold for cluster() on token-set features.
        Lower = looser clusters. 0.3 is the §85-style default.
      min_supports: drop clusters smaller than this. Matches the
        LLM prompt's "3+ supporting facts" floor.
      pattern_token_min_share: a token must appear in ≥ this share of a
        cluster's facts to enter its pattern string.
      max_tokens_per_pattern: cap on tokens per pattern (bound the
        summary[:80] slice).

    Returns:
      List of `{"pattern": str, "facts": [str, ...]}` dicts in
      deterministic order: by (-cluster_size, pattern).

    Raises:
      ValueError: tau outside [0,1] (delegated from
        `_cluster_by_jaccard`).
    """
    fact_list = [str(f) for f in facts]
    if len(fact_list) < min_supports:
        return []
    feature_sets: dict[str, frozenset[str]] = {}
    for i, ft in enumerate(fact_list):
        feature_sets[str(i)] = _tokenize(ft)
    clusters = _cluster_by_jaccard(feature_sets, tau)

    out: list[dict] = []
    for cluster in clusters:
        if len(cluster) < min_supports:
            continue
        # Skip clusters where every member has zero tokens (they'd all be
        # singletons under F3 anyway, but defensive).
        cluster_facts = [fact_list[int(sid)] for sid in sorted(cluster, key=int)]
        if not any(_tokenize(f) for f in cluster_facts):
            continue
        pattern = _pattern_from_cluster(
            cluster_facts,
            pattern_token_min_share=pattern_token_min_share,
            max_tokens=max_tokens_per_pattern,
        )
        out.append({"pattern": pattern, "facts": cluster_facts})

    out.sort(key=lambda d: (-len(d["facts"]), d["pattern"]))
    return out
