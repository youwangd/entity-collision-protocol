"""Cluster-output → sibling-evidence adapter (Personize §8 prior-sharing plumbing).

Bridges the two pure §8 primitives:

  1. `schema_family.cluster()` / `cluster_by_cooccurrence()` produce a flat
     partition of schema_ids into cluster-mates.
  2. `schema_family_decision.decide_with_family()` consumes, per owner,
     an iterable of sibling `EvidenceWindow`s.

What was missing was the lookup that turns (clusters, per-schema evidence
in this window) into the per-owner sibling-evidence list. This module is
that lookup. Pure function, no I/O, no clocks.

**Why a separate module.** Keeps `schema_family` (clustering) and
`schema_family_decision` (policy) free of any dict-shape assumptions about
how the pipeline materializes evidence. The pipeline owns the messy job of
collecting `(schema_id, supports, contradictions)` tuples for the current
window; this adapter is the one place that knows the contract.

**Contract.**

  * Input clusters are a partition of a subset of schema_ids; we tolerate
    schemas in the cluster output that have no evidence in this window
    (they contribute nothing) and schemas with evidence that are absent
    from any cluster (they get an empty siblings list — defensible because
    a schema with no clustering signal has no peers to share priors with).
  * Owner is excluded from its own siblings list (no double-counting).
  * Sibling evidence is yielded in deterministic schema_id-sorted order;
    `decide_with_family` only needs the sum, but determinism makes test
    failures readable and protects future callers that might care.

Invariants (locked in `tests/property/test_schema_family_evidence.py`):

  E1. Owner exclusion: for any owner `o`, `siblings_for(o, ...)` never
      yields the EvidenceWindow keyed at `o`.
  E2. Singletons → empty siblings: if `o`'s cluster is `{o}`, the result
      is `[]` regardless of evidence_by_schema content.
  E3. Determinism: result is sorted by sibling schema_id.
  E4. Round-trip with decide_with_family at share=0 is identity-safe:
      bare `decide(state, own)` == `decide_with_family(state, own,
      siblings_for(o, clusters, evidence), share=0.0)`.
  E5. Cluster partition is honored: a sibling of `o` must be in the
      same cluster as `o`.
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence, Tuple

from engram.consolidation.schema_decision import EvidenceWindow


def _find_cluster(
    schema_id: str,
    clusters: Sequence[frozenset[str]],
) -> frozenset[str] | None:
    """Return the cluster containing `schema_id`, or None if unclustered."""
    for c in clusters:
        if schema_id in c:
            return c
    return None


def siblings_for(
    owner: str,
    clusters: Sequence[frozenset[str]],
    evidence_by_schema: Mapping[str, EvidenceWindow],
) -> Tuple[EvidenceWindow, ...]:
    """Collect sibling EvidenceWindows for `owner` from its cluster.

    Args:
      owner: schema_id whose siblings we want.
      clusters: output of `schema_family.cluster()` (or
        `cluster_by_cooccurrence()`). A flat partition of schema_ids.
      evidence_by_schema: schema_id → EvidenceWindow for the current
        consolidation window. Schemas without recorded evidence in this
        window are simply absent from the mapping (treated as no
        contribution, NOT zero-evidence — those would dilute share=1.0
        cumulative semantics).

    Returns:
      Tuple of EvidenceWindow entries for cluster-mates of `owner`,
      excluding `owner` itself, sorted by sibling schema_id for
      determinism. Empty when `owner` is a singleton, unclustered, or
      its siblings have no evidence this window.
    """
    cluster = _find_cluster(owner, clusters)
    if cluster is None or len(cluster) <= 1:
        return ()
    out: list[tuple[str, EvidenceWindow]] = []
    for sid in cluster:
        if sid == owner:
            continue
        ev = evidence_by_schema.get(sid)
        if ev is None:
            continue
        out.append((sid, ev))
    out.sort(key=lambda kv: kv[0])
    return tuple(ev for _, ev in out)


def all_owner_siblings(
    clusters: Sequence[frozenset[str]],
    evidence_by_schema: Mapping[str, EvidenceWindow],
) -> dict[str, Tuple[EvidenceWindow, ...]]:
    """Bulk-build the owner→siblings map for every schema in `evidence_by_schema`.

    Convenience wrapper for callers that want to iterate decision-making
    over an entire window in one pass. Pure.
    """
    return {
        owner: siblings_for(owner, clusters, evidence_by_schema)
        for owner in evidence_by_schema
    }


__all__: Iterable[str] = ("siblings_for", "all_owner_siblings")
