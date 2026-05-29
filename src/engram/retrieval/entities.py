"""Lightweight entity extraction for retrieval-side entity-link channel (D1).

Strategy: dependency-free heuristic NER. Captures multi-word capitalized
proper-noun spans (e.g. "New York", "Alice Smith") and ALL-CAPS acronyms
(e.g. "NASA", "MIT"). Returns a *normalized* set of entity tokens — lowercase,
whitespace-collapsed.

This is deliberately not spaCy. The mission for D1 is to ship the channel
end-to-end as a new fusion signal; a swap to spaCy `en_core_web_sm` is a
later optimization gated on a recall@k delta. Heuristic NER on news/chat
text typically scores 0.6–0.7 F1 vs. spaCy ~0.85 — good enough for a
weak supervision signal in a fused retrieval score.

Lazy + cheap (regex only). Safe to call per-query and per-memory.
"""

from __future__ import annotations

import re
from typing import Iterable

# Words that look proper-noun-shaped at sentence start but aren't entities.
# Kept tiny on purpose; the goal is signal, not exhaustive filtering.
_STOPWORDS: frozenset[str] = frozenset({
    "i", "the", "a", "an", "and", "or", "but", "if", "of", "on", "in", "at",
    "to", "for", "with", "by", "from", "as", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will", "would",
    "could", "should", "may", "might", "must", "can", "this", "that", "these",
    "those", "it", "its", "he", "she", "they", "we", "you", "my", "your",
    "his", "her", "their", "our", "what", "when", "where", "why", "how",
    "who", "whom", "which", "there", "here", "now", "then",
    # common day/month names — typically not the entities we want to link on
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
    "january", "february", "march", "april", "may", "june", "july", "august",
    "september", "october", "november", "december",
    "yesterday", "today", "tomorrow",
})

# Multi-word capitalized span: "Alice Smith", "New York City", "Acme Corp".
# Anchored to require the *first* token to start with a capital and not be
# at the absolute start of the string with a stopword (cheap check post-match).
_CAP_SPAN = re.compile(
    r"\b([A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*){0,3})\b"
)
# All-caps acronym (2+ chars) — "NASA", "MIT", "API".
_ACRONYM = re.compile(r"\b([A-Z]{2,})\b")


def extract_entities(text: str, backend: str = "heuristic") -> set[str]:
    """Extract a normalized set of entity strings from `text`.

    Returns lowercase, whitespace-collapsed tokens. Empty set on empty/None.

    Args:
        text: input string.
        backend: NER backend selector. One of:
            - "heuristic" (default): dependency-free regex-based NER.
            - "spacy_sm": spaCy `en_core_web_sm`. Lazy-imported on first call.
              Raises ImportError with install hint if the dep is absent.
              Recognized entity types: PERSON, ORG, GPE, LOC, NORP, FAC,
              EVENT, PRODUCT, WORK_OF_ART. (Skips DATE/TIME/QUANTITY which
              are typically retrieval-noise, matching the heuristic's
              stopword philosophy.)
    """
    if not text:
        return set()
    if backend == "heuristic":
        return _extract_heuristic(text)
    if backend in ("spacy_sm", "spacy_md", "spacy_lg"):
        return _extract_spacy(text, _SPACY_MODEL_NAMES[backend])
    raise ValueError(
        f"Unknown entity_ner backend: {backend!r}. "
        f"Expected 'heuristic', 'spacy_sm', 'spacy_md', or 'spacy_lg'."
    )


def _extract_heuristic(text: str) -> set[str]:
    out: set[str] = set()

    for m in _CAP_SPAN.finditer(text):
        span = m.group(1).strip()
        # Reject single-token spans that are stopwords (sentence-initial "The", etc.)
        norm = " ".join(span.split()).lower()
        if " " not in norm and norm in _STOPWORDS:
            continue
        # Reject single-token spans that are the very first word of the text —
        # it's usually a sentence-initial capital, not a name. Multi-word spans
        # at offset 0 are kept ("New York is great").
        if m.start() == 0 and " " not in norm:
            continue
        out.add(norm)

    for m in _ACRONYM.finditer(text):
        out.add(m.group(1).lower())

    return out


# spaCy entity-type allow-list. PERSON / ORG / GPE / LOC / NORP / FAC /
# EVENT / PRODUCT / WORK_OF_ART are the "thing-y" labels useful for retrieval
# linkage. DATE/TIME/QUANTITY/PERCENT/ORDINAL/CARDINAL/MONEY are excluded —
# they overlap with stopword classes in the heuristic and tend to add noise
# to a Jaccard channel.
_SPACY_ALLOWED_LABELS: frozenset[str] = frozenset({
    "PERSON", "ORG", "GPE", "LOC", "NORP", "FAC",
    "EVENT", "PRODUCT", "WORK_OF_ART", "LAW", "LANGUAGE",
})

# Module-level cache for the loaded spaCy pipelines, keyed by model name.
# Loading is ~1s + ~50MB (sm), ~50MB (md), ~750MB (lg); cached once per name.
_SPACY_MODEL_NAMES: dict[str, str] = {
    "spacy_sm": "en_core_web_sm",
    "spacy_md": "en_core_web_md",
    "spacy_lg": "en_core_web_lg",
}
_SPACY_NLP_CACHE: dict[str, object] = {}


def _load_spacy(model_name: str):
    if model_name in _SPACY_NLP_CACHE:
        return _SPACY_NLP_CACHE[model_name]
    try:
        import spacy  # type: ignore[import-not-found]
    except ImportError as e:
        raise ImportError(
            f"entity_ner spacy backend requires spaCy. "
            f"Install with: pip install -e '.[entity-ner]' "
            f"(then: python -m spacy download {model_name})"
        ) from e
    try:
        nlp = spacy.load(model_name)
    except OSError as e:
        raise ImportError(
            f"spaCy model {model_name!r} not found. "
            f"Install with: python -m spacy download {model_name}"
        ) from e
    _SPACY_NLP_CACHE[model_name] = nlp
    return nlp


def _extract_spacy(text: str, model_name: str) -> set[str]:
    nlp = _load_spacy(model_name)
    doc = nlp(text)
    out: set[str] = set()
    for ent in doc.ents:
        if ent.label_ not in _SPACY_ALLOWED_LABELS:
            continue
        norm = " ".join(ent.text.split()).lower()
        if not norm:
            continue
        out.add(norm)
    return out


# Back-compat alias retained for anything that might import the old name.
def _extract_spacy_sm(text: str) -> set[str]:
    return _extract_spacy(text, "en_core_web_sm")


def extract_entities_typed(
    text: str, backend: str = "heuristic"
) -> list[tuple[str, str]]:
    """Like :func:`extract_entities`, but returns ``(entity, label)`` pairs.

    Used by the type-aware PRF gate (§5.4 follow-up to anchor 18). When the
    backend cannot supply true labels (``"heuristic"``), every entity is
    tagged ``"MISC"`` — this makes a type-purity gate degenerate (purity=1.0)
    and therefore inert, which is the right behavior: the heuristic backend
    cannot disambiguate types, so the gate gracefully no-ops.

    Returns a *list* (not a set) so the same surface entity at multiple
    label positions is preserved for type-frequency counting. Order follows
    document order.
    """
    if not text:
        return []
    if backend == "heuristic":
        return [(e, "MISC") for e in sorted(_extract_heuristic(text))]
    if backend in ("spacy_sm", "spacy_md", "spacy_lg"):
        return _extract_spacy_typed(text, _SPACY_MODEL_NAMES[backend])
    raise ValueError(
        f"Unknown entity_ner backend: {backend!r}. "
        f"Expected 'heuristic', 'spacy_sm', 'spacy_md', or 'spacy_lg'."
    )


def _extract_spacy_typed(text: str, model_name: str) -> list[tuple[str, str]]:
    nlp = _load_spacy(model_name)
    doc = nlp(text)
    out: list[tuple[str, str]] = []
    for ent in doc.ents:
        if ent.label_ not in _SPACY_ALLOWED_LABELS:
            continue
        norm = " ".join(ent.text.split()).lower()
        if not norm:
            continue
        out.append((norm, ent.label_))
    return out


def _extract_spacy_sm_typed(text: str) -> list[tuple[str, str]]:
    return _extract_spacy_typed(text, "en_core_web_sm")


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """Jaccard similarity over two iterables of strings.

    Returns 0.0 if either side is empty. Symmetric. Bounded [0, 1].
    """
    sa = set(a) if not isinstance(a, set) else a
    sb = set(b) if not isinstance(b, set) else b
    if not sa or not sb:
        return 0.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union else 0.0
