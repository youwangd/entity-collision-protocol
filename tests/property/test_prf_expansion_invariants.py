"""Hypothesis property tests for ``engram.retrieval.expansion.expand_query``.

Pins invariants that the production PRF wire-in
(``RetrievalConfig.query_expansion_min_dominance``) relies on:

  P1. Identity bypass — top_k<=0 or max_entities<=0 returns input untouched.
  P2. Original-query prefix — the returned ``expanded`` always starts with the
      original ``query`` (we only ever *append*, never rewrite).
  P3. Chosen-entities cap — ``len(chosen) <= max_entities``.
  P4. Novelty — no chosen entity has all its words already in the query token set.
  P5. Pool truncation — texts past index ``top_k`` are inert (extending the
      tail with arbitrary noise must not change the output).
  P6. Idempotence under re-expansion — re-running expand_query against the
      already-expanded query never re-chooses the previous anchors (they are
      now in-query, filtered by the novelty gate).
  P7. Dominance-gate monotonicity — chosen@1.0 ⊆ chosen@0.0.
  P8. Anchor-share gate — pure-monoculture pool is skipped under
      ``anchor_share_max < 1.0``.

Heuristic backend only — spaCy backends are slow + non-deterministic for
fuzz-scale runs.
"""
from __future__ import annotations

import re

from hypothesis import given, settings, strategies as st

from engram.retrieval.expansion import expand_query


_NAME_CHARS = st.sampled_from(list("abcdefghij"))


@st.composite
def _capitalized_name(draw) -> str:
    n = draw(st.integers(min_value=2, max_value=6))
    body = "".join(draw(_NAME_CHARS) for _ in range(n))
    return body.capitalize()


@st.composite
def _doc_text(draw) -> str:
    n_names = draw(st.integers(min_value=1, max_value=4))
    names = [draw(_capitalized_name()) for _ in range(n_names)]
    parts = ["the"]
    for nm in names:
        parts.append(nm)
        parts.append("walked")
    return " ".join(parts)


_query_strat = st.text(
    alphabet=st.sampled_from("abcdefghijklmnopqrstuvwxyz "),
    min_size=1,
    max_size=20,
).filter(lambda s: s.strip() != "")

_texts_strat = st.lists(_doc_text(), min_size=0, max_size=15)


@given(query=_query_strat, texts=_texts_strat)
def test_p1_identity_when_top_k_zero(query, texts):
    expanded, chosen = expand_query(query, texts, top_k=0, max_entities=3, min_dominance=0.0)
    assert expanded == query
    assert chosen == []


@given(query=_query_strat, texts=_texts_strat)
def test_p1_identity_when_max_entities_zero(query, texts):
    expanded, chosen = expand_query(query, texts, top_k=10, max_entities=0, min_dominance=0.0)
    assert expanded == query
    assert chosen == []


@given(
    query=_query_strat,
    texts=_texts_strat,
    top_k=st.integers(min_value=1, max_value=15),
    max_entities=st.integers(min_value=1, max_value=5),
    min_dom=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=200)
def test_p2_p3_p4_output_shape(query, texts, top_k, max_entities, min_dom):
    expanded, chosen = expand_query(
        query, texts, top_k=top_k, max_entities=max_entities, min_dominance=min_dom
    )
    # P2 — prefix preservation
    assert expanded == query or expanded.startswith(query + " ")
    # P3 — cap
    assert len(chosen) <= max_entities
    # P4 — novelty: no chosen entity has all its words already in the query
    q_tokens = {t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 1}
    word_re = re.compile(r"[a-z0-9]+")
    for ent in chosen:
        words = set(word_re.findall(ent))
        if words:
            assert not words.issubset(q_tokens), (ent, q_tokens)


@given(
    query=_query_strat,
    head=st.lists(_doc_text(), min_size=1, max_size=8),
    tail=_texts_strat,
    max_entities=st.integers(min_value=1, max_value=3),
    min_dom=st.floats(min_value=0.0, max_value=0.6),
)
@settings(max_examples=150)
def test_p5_pool_truncation_inert(query, head, tail, max_entities, min_dom):
    top_k = len(head)
    short_out = expand_query(
        query, head, top_k=top_k, max_entities=max_entities, min_dominance=min_dom
    )
    long_out = expand_query(
        query, head + tail, top_k=top_k, max_entities=max_entities, min_dominance=min_dom
    )
    assert short_out == long_out


@given(
    query=_query_strat,
    texts=st.lists(_doc_text(), min_size=1, max_size=10),
    top_k=st.integers(min_value=1, max_value=10),
    max_entities=st.integers(min_value=1, max_value=4),
    min_dom=st.floats(min_value=0.0, max_value=0.5),
)
@settings(max_examples=150)
def test_p6_reexpansion_no_repeat(query, texts, top_k, max_entities, min_dom):
    expanded, chosen1 = expand_query(
        query, texts, top_k=top_k, max_entities=max_entities, min_dominance=min_dom
    )
    if not chosen1:
        return
    _, chosen2 = expand_query(
        expanded, texts, top_k=top_k, max_entities=max_entities, min_dominance=min_dom
    )
    assert set(chosen2).isdisjoint(set(chosen1)), (chosen1, chosen2)


@given(
    query=_query_strat,
    texts=st.lists(_doc_text(), min_size=1, max_size=8),
    max_entities=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=120)
def test_p7_strict_dominance_subset_of_open(query, texts, max_entities):
    _, chosen_open = expand_query(
        query, texts, top_k=len(texts), max_entities=max_entities, min_dominance=0.0
    )
    _, chosen_strict = expand_query(
        query, texts, top_k=len(texts), max_entities=max_entities, min_dominance=1.0
    )
    if chosen_strict:
        assert set(chosen_strict).issubset(set(chosen_open))


@given(
    query=_query_strat,
    anchor=_capitalized_name(),
    n_docs=st.integers(min_value=3, max_value=8),
)
@settings(max_examples=80)
def test_p8_anchor_share_max_gates_monoculture(query, anchor, n_docs):
    texts = [f"the {anchor} walked"] * n_docs
    _, chosen_off = expand_query(
        query, texts, top_k=n_docs, max_entities=3, min_dominance=0.0,
        anchor_share_max=None,
    )
    _, chosen_on = expand_query(
        query, texts, top_k=n_docs, max_entities=3, min_dominance=0.0,
        anchor_share_max=0.5,
    )
    if chosen_off:
        assert chosen_on == []
