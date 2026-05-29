"""§5.4 angle 1 — pseudo-relevance-feedback (PRF) entity-based query expansion.

Production wire-in of the experimental driver in
``evals/query_entity_expansion.py``. Mines the top-K first-pass texts for
the most frequent novel entities (entities not already in the query),
applies a dominance gate, and returns the expanded query.

Operating point recommended by anchors 18 + 22 (paired-bootstrap CIs,
α=0.05, d=0.3, pool=20):

* ``top_k = 10``  (anchor 29/30)
* ``max_entities = 3``  (anchor 14)
* ``min_dominance = 0.3``  (anchor 18)

Inert when ``min_dominance is None``.
"""
from __future__ import annotations

import re

from engram.retrieval.entities import extract_entities, extract_entities_typed


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")
_ENTITY_WORD_RE = re.compile(r"[a-z0-9]+")


def _query_terms(q: str) -> set[str]:
    return {t for t in _TOKEN_RE.findall(q.lower()) if len(t) > 1}


def expand_query(
    query: str,
    first_pass_texts: list[str],
    *,
    top_k: int = 10,
    max_entities: int = 3,
    min_dominance: float = 0.3,
    backend: str = "heuristic",
    idf_min_rarity: float | None = None,
    rarity_lookup=None,
    anchor_share_max: float | None = None,
) -> tuple[str, list[str]]:
    """Append the most frequent novel entities from the top-K texts.

    Args:
        query: Original query string.
        first_pass_texts: List of candidate texts from the first-pass
            retrieval, ordered by relevance.
        top_k: How many first-pass texts to mine. Anchor 29/30 sweet spot=10.
        max_entities: Cap on appended entities. Anchor 14 sweep.
        min_dominance: Gate. Top entity must appear in
            ``≥ min_dominance × top_k`` documents, else no expansion.
            0.0 disables the gate.
        backend: NER backend (``"heuristic"`` or ``"spacy_sm"``).
        idf_min_rarity: §4.15g IDF-rarity filter. When non-None and
            ``rarity_lookup`` is provided, drop candidate entities whose
            rarity score < threshold *before* truncating to
            ``max_entities``. Inert when None or rarity_lookup is None.
        rarity_lookup: Optional callable ``(entity: str) -> float``
            returning corpus-rarity in [0,1] (1.0 = entity appears in 0
            docs, 0.0 = entity appears in every doc). Computed by the
            caller (engine) from the store's FTS index. Lenient: any
            exception raised by the callable for a given candidate
            treats that candidate as rarity=0.0 (i.e. filtered out when
            idf_min_rarity > 0). Skipped entirely when ``idf_min_rarity``
            is None.
        anchor_share_max: §D15d diagnostic gate. Skip PRF entirely
            when the dominant candidate entity's share of total candidate
            occurrences in the top-K pool exceeds this threshold (i.e.
            the pool is saturated by one anchor — the §D15c
            shared-anchor-density failure mode). Inert when None.

    Returns:
        ``(expanded_query, chosen_entities)``. When no expansion fires
        (no novel entities, or dominance-gated out, or all candidates
        filtered by IDF), returns ``(query, [])``.
    """
    if top_k <= 0 or max_entities <= 0:
        return query, []
    seen_in_q = _query_terms(query)
    counts: dict[str, int] = {}
    pool = first_pass_texts[:top_k]
    n_docs = max(len(pool), 1)
    for text in pool:
        try:
            ents = extract_entities(text or "", backend=backend)
        except Exception:
            # NER must never break retrieval. Lenient fail.
            continue
        for e in ents:
            words = set(_ENTITY_WORD_RE.findall(e))
            if words and words.issubset(seen_in_q):
                continue
            counts[e] = counts.get(e, 0) + 1
    ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    if not ranked:
        return query, []
    if min_dominance > 0.0 and ranked[0][1] / n_docs < min_dominance:
        return query, []
    # §D15d anchor-share gate. ``share`` = (occurrences of dominant
    # candidate entity in top-K pool) / (total candidate-entity
    # occurrences). When share > anchor_share_max, the pool is saturated
    # by one anchor — the §D15c failure mode (cross-fact entity confusion
    # under shared-anchor density). Skip PRF for this query. Computed
    # BEFORE the IDF filter and BEFORE max_entities truncation so it
    # fires on raw pool composition. Inert when anchor_share_max is None.
    if anchor_share_max is not None:
        total_occ = sum(counts.values())
        if total_occ > 0:
            share = ranked[0][1] / total_occ
            if share > anchor_share_max:
                return query, []
    # §4.15g IDF-rarity filter — drop low-IDF (corpus-common) entities.
    if idf_min_rarity is not None and rarity_lookup is not None:
        filtered = []
        for e, c in ranked:
            try:
                r = float(rarity_lookup(e))
            except Exception:
                r = 0.0
            if r >= idf_min_rarity:
                filtered.append((e, c))
        ranked = filtered
        if not ranked:
            return query, []
    chosen = [e for e, _ in ranked[:max_entities]]
    return query + " " + " ".join(chosen), chosen


def expand_query_typed(
    query: str,
    first_pass_texts: list[str],
    *,
    top_k: int = 10,
    max_entities: int = 3,
    min_dominance: float = 0.3,
    type_purity_min: float | None = None,
    backend: str = "heuristic",
) -> tuple[str, list[str]]:
    """Type-aware variant of :func:`expand_query`.

    Adds a *type-purity* gate on top of the frequency-dominance gate:
    among the candidate entities mined from the top-K first-pass texts,
    require the dominant entity-type to constitute at least
    ``type_purity_min`` of all candidate-entity occurrences. The intent
    (§4.8.1 + §4.9 remediation): expand only when the corpus signal is
    *type-coherent* (e.g. uniformly PERSON, or uniformly GPE), avoiding
    the failure mode where PRF concatenates a PERSON + GPE + ORG salad
    and degrades multi-entity-hard hit@k.

    Behavior:
      - When ``type_purity_min is None``: identical to :func:`expand_query`
        (purity gate inert; PRF still subject to dominance gate).
      - When backend == "heuristic": all entities have label "MISC", so
        purity == 1.0 trivially — the gate is inert by construction.
        This is intentional: heuristic NER cannot supply real types, so
        the type-aware path safely degenerates to the legacy path.
      - When backend == "spacy_sm": purity = (# occurrences of dominant
        label) / (total entity occurrences across the top-K pool, after
        novelty filter).

    Returns ``(expanded_query, chosen_entities)``. When *either* gate
    blocks expansion, returns ``(query, [])``.
    """
    if top_k <= 0 or max_entities <= 0:
        return query, []
    seen_in_q = _query_terms(query)
    # (entity, label) → count
    typed_counts: dict[tuple[str, str], int] = {}
    label_counts: dict[str, int] = {}
    pool = first_pass_texts[:top_k]
    n_docs = max(len(pool), 1)
    for text in pool:
        try:
            pairs = extract_entities_typed(text or "", backend=backend)
        except Exception:
            continue
        # Per-doc dedup — frequency = # docs containing the entity, matching
        # expand_query semantics.
        seen_in_doc: set[tuple[str, str]] = set()
        for e, lab in pairs:
            words = set(_ENTITY_WORD_RE.findall(e))
            if words and words.issubset(seen_in_q):
                continue
            key = (e, lab)
            if key in seen_in_doc:
                continue
            seen_in_doc.add(key)
            typed_counts[key] = typed_counts.get(key, 0) + 1
            label_counts[lab] = label_counts.get(lab, 0) + 1
    if not typed_counts:
        return query, []
    # Frequency-dominance gate (same as expand_query, computed from typed_counts).
    # Aggregate to entity-string level so a multi-label entity isn't double-counted.
    ent_freq: dict[str, int] = {}
    for (e, _lab), c in typed_counts.items():
        ent_freq[e] = max(ent_freq.get(e, 0), c)
    ranked = sorted(ent_freq.items(), key=lambda kv: (-kv[1], kv[0]))
    if min_dominance > 0.0 and ranked[0][1] / n_docs < min_dominance:
        return query, []
    # Type-purity gate.
    if type_purity_min is not None and label_counts:
        total = sum(label_counts.values())
        top_label_count = max(label_counts.values())
        purity = top_label_count / total if total else 0.0
        if purity < type_purity_min:
            return query, []
    chosen = [e for e, _ in ranked[:max_entities]]
    return query + " " + " ".join(chosen), chosen
