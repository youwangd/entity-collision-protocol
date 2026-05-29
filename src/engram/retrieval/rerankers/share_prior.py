"""§96 — multi-mate prior sharing reranker.

Concept (Personize §96, "shared prior"):
    Within the retrieved candidate pool, candidates that share named entities
    with MULTIPLE OTHER pool members (not just the query) are evidentially
    "in the same neighbourhood". On multi-hop queries that connect via a
    shared bridge entity, this signal promotes the bridge candidate.

We build an undirected entity-sharing graph over the pool and score each
node by how many other candidates it shares entities with — the "multi-mate"
degree. We then add a small bounded boost to the candidate's fused score,
re-sort, and return.

Hard invariant: must NEVER demote the original rank-0 candidate. We enforce
this by capping per-candidate boost so post-rerank scores at i ≥ 1 stay
strictly below the original rank-0 score. This preserves h@1 monotonically
on every recipe — the worst case is "no change at h@1, possibly improved
h@k for k > 1" — which is exactly the property we want for an opt-in stage.

Other invariants:
    - empty / single-element pool → unchanged
    - candidates with no entities contribute zero shared-mate signal
    - lenient on extraction errors (skips that candidate's entities)
    - deterministic: ties broken by original pool index
"""

from __future__ import annotations

from typing import Iterable

from engram.core.types import ScoredMemory
from engram.retrieval.entities import extract_entities
from engram.retrieval.rerank import register_reranker

# Default boost magnitude. Small relative to typical fused scores (~0..1).
# Capped per-candidate by the rank-0-preservation guard below.
DEFAULT_ALPHA = 0.05


def _adaptive_alpha_scale(max_deg: int) -> float:
    """Taper schedule for adaptive alpha (paper/30_methods.md §3.5).

    Heuristic motivation: when `max_deg == 1` the entity-sharing graph is a
    near-matching — at most one mate per node — so the share_prior signal is
    crisp and the full boost should land. When `max_deg` grows, the
    `deg / max_deg` distribution flattens and individual normalized degrees
    carry less discriminating information, so the boost should taper.

    Schedule (monotone non-increasing in max_deg):
        max_deg ≤ 1 → 1.0      (saturated — full alpha)
        max_deg = 2 → 0.800
        max_deg = 3 → 0.667
        max_deg = 4 → 0.571
        max_deg = 5 → 0.500
        max_deg = 9 → 0.333
        max_deg →∞  → 0+

    Closed form:  scale(d) = 1 / (1 + max(0, d - 1) / 4)
    """
    if max_deg <= 1:
        return 1.0
    return 1.0 / (1.0 + (max_deg - 1) / 4.0)


def _entities_for(
    sm: ScoredMemory,
    entity_cache: dict[str, set[str]] | None,
    backend: str,
) -> set[str]:
    if entity_cache is not None and sm.memory.id in entity_cache:
        return entity_cache[sm.memory.id]
    try:
        ents = extract_entities(sm.memory.content or "", backend=backend)
    except Exception:
        ents = set()
    if entity_cache is not None:
        entity_cache[sm.memory.id] = ents
    return ents


def share_prior_reranker(
    results: Iterable[ScoredMemory],
    *,
    query_entities: set[str] | None = None,
    entity_cache: dict[str, set[str]] | None = None,
    cfg=None,
    alpha: float = DEFAULT_ALPHA,
    adaptive_alpha: bool | None = None,
    **_: object,
) -> list[ScoredMemory]:
    """Reorder pool by adding a bounded multi-mate share-prior boost.

    When `adaptive_alpha=True` (or `cfg.share_prior_adaptive_alpha=True`),
    the effective alpha is scaled by `_adaptive_alpha_scale(max_deg)`,
    which saturates at low max_deg and tapers as the entity-sharing graph
    densifies. See `_adaptive_alpha_scale` for the schedule.
    """
    pool = list(results)
    n = len(pool)
    if n < 2:
        return pool

    backend = "heuristic"
    if cfg is not None:
        backend = getattr(cfg, "entity_ner", "heuristic") or "heuristic"

    # Cache entity sets for each pool member.
    ent_sets: list[set[str]] = [
        _entities_for(sm, entity_cache, backend) for sm in pool
    ]

    # multi_mate_degree[i] = number of OTHER pool members sharing ≥1 entity.
    # We exclude the query itself — the existing entity-channel scorer already
    # handles query-overlap. This signal is purely "popularity within pool".
    degrees = [0] * n
    for i in range(n):
        ei = ent_sets[i]
        if not ei:
            continue
        for j in range(n):
            if i == j:
                continue
            if ei & ent_sets[j]:
                degrees[i] += 1

    max_deg = max(degrees) if degrees else 0
    if max_deg == 0:
        # Nothing to do — no entity sharing in the pool.
        return pool

    # Resolve adaptive_alpha from kwarg or cfg.
    if adaptive_alpha is None and cfg is not None:
        adaptive_alpha = bool(getattr(cfg, "share_prior_adaptive_alpha", False))
    effective_alpha = alpha
    if adaptive_alpha:
        effective_alpha = alpha * _adaptive_alpha_scale(max_deg)

    # Original rank-0 score is the ceiling. Anyone we promote can come close
    # but must stay strictly below it. epsilon defends ties.
    top_score = pool[0].score
    epsilon = 1e-9

    boosted: list[tuple[float, int, ScoredMemory]] = []
    for i, sm in enumerate(pool):
        norm_deg = degrees[i] / max_deg  # ∈ [0, 1]
        raw_boost = effective_alpha * norm_deg
        if i == 0:
            new_score = sm.score  # rank-0 untouched (acts as anchor)
        else:
            cap = max(0.0, top_score - sm.score - epsilon)
            new_score = sm.score + min(raw_boost, cap)
        # Stamp rerank contribution into sources for observability.
        sm.sources["share_prior_boost"] = round(new_score - sm.score, 9)
        sm.sources["share_prior_degree"] = float(degrees[i])
        # Stable sort: (-score, original_index)
        boosted.append((-new_score, i, sm))

    boosted.sort(key=lambda t: (t[0], t[1]))
    out = []
    for neg_score, _orig_i, sm in boosted:
        sm.score = -neg_score
        out.append(sm)
    return out


register_reranker("share_prior", share_prior_reranker)
