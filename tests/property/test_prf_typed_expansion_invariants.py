"""Hypothesis property tests for ``expand_query_typed``.

Companion to ``test_prf_expansion_invariants.py`` — pins the type-aware
PRF contract used by §4.8.1 + §4.9 remediation. Heuristic backend only
(spaCy backends are slow and out-of-scope for fuzz-scale runs; the
heuristic backend assigns every entity label ``MISC`` so type-purity
== 1.0 by construction, which is the *intended* degeneration path).

Properties:

  T1. Shape carries over — P1/P2/P3/P4 from ``expand_query`` hold here too.
  T2. Parity (no purity gate) — when ``type_purity_min is None``, output
      equals the legacy ``expand_query`` for matching params.
  T3. Purity-gate inertness on heuristic — for any ``type_purity_min``
      in (0, 1.0], the heuristic backend treats the gate as a no-op
      (because every label is "MISC", so purity == 1.0).
  T4. Type-purity monotonicity — chosen@purity=1.0 ⊆ chosen@purity=0.0
      (strict gate is a subset of open). Trivial here on heuristic but
      pins the contract for spaCy backends.
  T5. Dominance-gate parity — the typed path enforces the same
      dominance gate as ``expand_query``: if legacy returns [] under
      min_dominance=d, the typed path also returns [] under
      (min_dominance=d, type_purity_min=None).
"""
from __future__ import annotations

import re

from hypothesis import given, settings, strategies as st

from engram.retrieval.expansion import expand_query, expand_query_typed


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

_texts_strat = st.lists(_doc_text(), min_size=0, max_size=12)


@given(
    query=_query_strat,
    texts=_texts_strat,
    top_k=st.integers(min_value=1, max_value=12),
    max_entities=st.integers(min_value=1, max_value=4),
    min_dom=st.floats(min_value=0.0, max_value=1.0),
    purity=st.one_of(st.none(), st.floats(min_value=0.0, max_value=1.0)),
)
@settings(max_examples=200, deadline=None)
def test_t1_shape_invariants(query, texts, top_k, max_entities, min_dom, purity):
    expanded, chosen = expand_query_typed(
        query,
        texts,
        top_k=top_k,
        max_entities=max_entities,
        min_dominance=min_dom,
        type_purity_min=purity,
        backend="heuristic",
    )
    # P2 — prefix
    assert expanded == query or expanded.startswith(query + " ")
    # P3 — cap
    assert len(chosen) <= max_entities
    # P4 — novelty
    q_tokens = {t for t in re.findall(r"[A-Za-z0-9]+", query.lower()) if len(t) > 1}
    word_re = re.compile(r"[a-z0-9]+")
    for ent in chosen:
        words = set(word_re.findall(ent))
        if words:
            assert not words.issubset(q_tokens), (ent, q_tokens)


@given(
    query=_query_strat,
    texts=_texts_strat,
    top_k=st.integers(min_value=1, max_value=12),
    max_entities=st.integers(min_value=1, max_value=4),
    min_dom=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=150, deadline=None)
def test_t2_parity_no_purity_gate(query, texts, top_k, max_entities, min_dom):
    """type_purity_min=None ⇒ identical output to expand_query (heuristic)."""
    legacy = expand_query(
        query, texts, top_k=top_k, max_entities=max_entities, min_dominance=min_dom
    )
    typed = expand_query_typed(
        query,
        texts,
        top_k=top_k,
        max_entities=max_entities,
        min_dominance=min_dom,
        type_purity_min=None,
        backend="heuristic",
    )
    assert legacy == typed, (legacy, typed)


@given(
    query=_query_strat,
    texts=_texts_strat,
    top_k=st.integers(min_value=1, max_value=10),
    max_entities=st.integers(min_value=1, max_value=4),
    min_dom=st.floats(min_value=0.0, max_value=0.5),
    purity=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=120, deadline=None)
def test_t3_purity_inert_on_heuristic(query, texts, top_k, max_entities, min_dom, purity):
    """On the heuristic backend, all labels are 'MISC' ⇒ purity == 1.0
    ⇒ the gate cannot block (purity_min<=1.0 always satisfied)."""
    open_ = expand_query_typed(
        query,
        texts,
        top_k=top_k,
        max_entities=max_entities,
        min_dominance=min_dom,
        type_purity_min=None,
        backend="heuristic",
    )
    gated = expand_query_typed(
        query,
        texts,
        top_k=top_k,
        max_entities=max_entities,
        min_dominance=min_dom,
        type_purity_min=purity,
        backend="heuristic",
    )
    # Heuristic ⇒ MISC monoculture ⇒ purity gate is a no-op.
    assert open_ == gated, (open_, gated, purity)


@given(
    query=_query_strat,
    texts=st.lists(_doc_text(), min_size=1, max_size=8),
    max_entities=st.integers(min_value=1, max_value=3),
    min_dom=st.floats(min_value=0.0, max_value=0.5),
)
@settings(max_examples=100, deadline=None)
def test_t4_purity_monotonic_subset(query, texts, max_entities, min_dom):
    """chosen@purity=1.0 ⊆ chosen@purity=0.0 (strict ⊆ open)."""
    _, open_ = expand_query_typed(
        query, texts,
        top_k=len(texts), max_entities=max_entities,
        min_dominance=min_dom, type_purity_min=0.0, backend="heuristic",
    )
    _, strict = expand_query_typed(
        query, texts,
        top_k=len(texts), max_entities=max_entities,
        min_dominance=min_dom, type_purity_min=1.0, backend="heuristic",
    )
    if strict:
        assert set(strict).issubset(set(open_)), (open_, strict)


@given(
    query=_query_strat,
    texts=_texts_strat,
    top_k=st.integers(min_value=1, max_value=10),
    max_entities=st.integers(min_value=1, max_value=4),
    min_dom=st.floats(min_value=0.0, max_value=1.0),
)
@settings(max_examples=120, deadline=None)
def test_t5_dominance_gate_parity(query, texts, top_k, max_entities, min_dom):
    """If legacy expand_query blocks (returns []), typed with the same
    dominance and type_purity_min=None must also block."""
    expanded_legacy, chosen_legacy = expand_query(
        query, texts, top_k=top_k, max_entities=max_entities, min_dominance=min_dom
    )
    if not chosen_legacy:
        expanded_typed, chosen_typed = expand_query_typed(
            query, texts,
            top_k=top_k, max_entities=max_entities,
            min_dominance=min_dom, type_purity_min=None, backend="heuristic",
        )
        assert chosen_typed == [], (chosen_legacy, chosen_typed)
        assert expanded_typed == expanded_legacy
