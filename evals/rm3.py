"""RM3 pseudo-relevance feedback (Lavrenko & Croft 2001).

Two-pass query expansion baseline for the AUDIT-D arm. Operates over
BM25 outputs externally to Engram core (no src/engram/ changes).

Reference: Lavrenko & Croft (2001), "Relevance Models in Information
Retrieval", SIGIR 2001. Anserini implementation:
https://github.com/castorini/anserini

Algorithm (RM1 with uniform doc prior, then RM3 mixture):
  1. First pass: run BM25 on the original query, get top-k documents.
  2. Expansion: P_RM(t) = (1/k) * sum_{d in top-k} (tf(t,d) / |d|).
     Absent docs contribute 0 to a term's mean. Keep top-N candidates
     after excluding original-query terms.
  3. Second pass: new query = lambda * original_terms +
     (1-lambda) * expansion_terms; re-run BM25.

Hyperparameters use Anserini defaults (k=10, num_terms=10, lambda=0.5,
epsilon=0.01).

Usage from a runner::

    from evals.rm3 import expand_query, RM3Config, build_expanded_query_string

    cfg = RM3Config(top_k=10, num_terms=10, lambda_orig=0.5)
    expanded = expand_query(
        original_query="why does my mortgage rate change",
        first_pass_doc_ids=bm25_top_k_ids,
        get_doc_text=corpus_lookup,  # doc_id -> str
        cfg=cfg,
    )
    new_query = build_expanded_query_string(original_query, expanded, cfg)
    # Re-run BM25 with new_query.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]+")


def _tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization. Matches typical BM25 indexers."""
    return [t.lower() for t in _TOKEN_RE.findall(text or "")]


@dataclass
class RM3Config:
    top_k: int = 10
    num_terms: int = 10
    lambda_orig: float = 0.5
    epsilon: float = 0.01
    stopwords: frozenset[str] = field(default_factory=frozenset)


def expand_query(
    original_query: str,
    first_pass_doc_ids: list[str],
    get_doc_text: Callable[[str], str],
    cfg: RM3Config | None = None,
) -> Counter[str]:
    """Return a Counter[term -> normalized_weight] for the expansion terms.

    Weights are non-negative and sum to 1.0 across the returned terms.
    Original-query terms are excluded from the expansion pool (RM3
    convention; they are added back via build_expanded_query_string with
    weight lambda_orig).

    Returns an empty Counter if first_pass_doc_ids is empty or no
    expansion candidates survive filtering.
    """
    cfg = cfg or RM3Config()
    if not first_pass_doc_ids:
        return Counter()

    top_k_ids = first_pass_doc_ids[: cfg.top_k]

    # Lavrenko-Croft RM1 with uniform document prior P(d|q) = 1/|top_k|:
    #   P_RM(t) = sum_d P(d|q) * P(t|d)
    #          = (1/k) * sum_{d in top-k} (tf(t,d) / |d|)
    # Absent docs contribute 0, so terms appearing across many top-k docs
    # outrank terms appearing in just one with high local TF. This matches
    # Anserini's RM3 implementation. We add a small epsilon floor only to
    # avoid zero-prob ties in degenerate cases.
    term_scores: dict[str, float] = {}
    n_docs_used = 0
    for doc_id in top_k_ids:
        text = get_doc_text(doc_id)
        if not text:
            continue
        tokens = _tokenize(text)
        if not tokens:
            continue
        n_docs_used += 1
        doc_len = len(tokens)
        tf = Counter(tokens)
        for term, count in tf.items():
            if term in cfg.stopwords:
                continue
            p_t_given_d = count / doc_len
            term_scores[term] = term_scores.get(term, 0.0) + p_t_given_d

    if n_docs_used == 0:
        return Counter()

    # Convert sums to mean P_RM(t) and add the epsilon floor.
    for t in term_scores:
        term_scores[t] = (term_scores[t] / n_docs_used) + cfg.epsilon

    # RM3 vs RM1: drop original-query terms from expansion candidates.
    orig_term_set = set(_tokenize(original_query))
    expansion_pool = {
        t: s for t, s in term_scores.items() if t not in orig_term_set
    }

    sorted_terms = sorted(
        expansion_pool.items(), key=lambda kv: kv[1], reverse=True
    )[: cfg.num_terms]
    if not sorted_terms:
        return Counter()

    # Normalize the kept top-N to sum to 1.0.
    total = sum(s for _, s in sorted_terms)
    if total <= 0:
        return Counter()
    return Counter({t: s / total for t, s in sorted_terms})


def build_expanded_query_string(
    original_query: str,
    expanded_terms: Counter[str],
    cfg: RM3Config | None = None,
    repetition_scale: int = 10,
) -> str:
    """Render the expanded query as a string suitable for BM25.

    Uses term-repetition encoding so the BM25 implementation does not
    need a weighted-query API. Original-query terms appear once each
    (weight lambda_orig is implicit in the BM25 TF math); expansion
    terms appear N times where N scales with their normalized weight.

    repetition_scale controls aggressiveness of expansion. Default 10
    means the highest-weight expansion term repeats up to ~5 times at
    lambda_orig=0.5.
    """
    cfg = cfg or RM3Config()
    parts: list[str] = list(_tokenize(original_query))
    for term, weight in expanded_terms.items():
        n = max(1, round((1 - cfg.lambda_orig) * weight * repetition_scale))
        parts.extend([term] * n)
    return " ".join(parts)
