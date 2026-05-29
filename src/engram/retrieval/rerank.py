"""Post-rerank stage for the retrieval pipeline (§96 hook).

A reranker is a callable that takes the scored, pre-limit candidate pool
plus retrieval-time context and returns a (possibly reordered) list of
ScoredMemory. Rerankers MAY mutate `score` and `sources`; they MUST NOT
mutate `memory` fields. They run AFTER fused scoring and BEFORE
`limit` truncation, so they see a wider pool.

Discovery is via a process-global registry keyed by name. The
`RetrievalConfig.reranker` string selects which one (None = identity =
no-op, default).

Wiring:
    register_reranker("my_reranker", my_reranker_fn)
    cfg.reranker = "my_reranker"

A reranker that raises is swallowed and the pre-rerank order is kept —
retrieval must never break because a rerank stage misbehaved.
"""

from __future__ import annotations

import logging
from typing import Callable, Iterable

from engram.core.types import ScoredMemory

logger = logging.getLogger(__name__)

# Reranker signature:
#   (results, *, query, intent, entity_cache) -> list[ScoredMemory]
# Use kwargs-only after `results` so we can extend the context dict
# without breaking existing rerankers.
Reranker = Callable[..., list[ScoredMemory]]

_REGISTRY: dict[str, Reranker] = {}


def register_reranker(name: str, fn: Reranker) -> None:
    """Register a reranker under `name`. Idempotent overwrite."""
    if not name or not isinstance(name, str):
        raise ValueError("reranker name must be a non-empty string")
    _REGISTRY[name] = fn


def get_reranker(name: str | None) -> Reranker | None:
    """Return the registered reranker, or None for unknown / falsy name."""
    if not name:
        return None
    return _REGISTRY.get(name)


def list_rerankers() -> list[str]:
    return sorted(_REGISTRY.keys())


def clear_rerankers() -> None:
    """Test helper. Wipes the registry."""
    _REGISTRY.clear()
    _register_builtins()


def _identity(results: list[ScoredMemory], **_: object) -> list[ScoredMemory]:
    """No-op reranker. Useful for benchmarks (forces the rerank code path)."""
    return list(results)


def _register_builtins() -> None:
    register_reranker("identity", _identity)
    # Lazy import to avoid circular import at module import time.
    try:
        from engram.retrieval.rerankers.share_prior import (  # noqa: F401
            share_prior_reranker,
        )

        register_reranker("share_prior", share_prior_reranker)
    except Exception:  # pragma: no cover - defensive
        pass


_register_builtins()


def apply_reranker(
    name: str | None,
    results: Iterable[ScoredMemory],
    **context: object,
) -> list[ScoredMemory]:
    """Apply named reranker to results. Lenient: returns input on miss/error."""
    results_list = list(results)
    fn = get_reranker(name)
    if fn is None:
        return results_list
    try:
        out = fn(results_list, **context)
        if not isinstance(out, list):
            out = list(out)
        return out
    except Exception as exc:  # pragma: no cover - logged, not raised
        logger.warning("reranker %r raised %s; falling back to pre-rerank order", name, exc)
        return results_list
