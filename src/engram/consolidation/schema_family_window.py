"""Window-batch §8 decision driver (Personize §8 plumbing).

Final glue between the §8 primitives and the consolidation pipeline:

    schema_family.cluster*()                  →  Sequence[frozenset[str]]
    schema_family_evidence.all_owner_siblings →  owner → siblings tuple
    schema_family_decision.decide_with_family →  one decision per owner
    -----------------------------------------------------------------
    schema_family_window.decide_window         →  owner → EventKind | None

`decide_window()` is the single call the pipeline makes per consolidation
cycle once the §8 sharing knob is wired into stage 6: pass in the
per-schema `SchemaState` snapshot, the per-schema `EvidenceWindow` map
collected during the cycle, the clustering, thresholds, and a `share`
fraction; receive a deterministic owner→event-kind dict.

**Why a separate module.**
  * `schema_family_evidence` is a pure lookup; it must not depend on the
    decision policy. `schema_family_decision` is a single-owner wrapper;
    it must not know about clusters of states. This module is the only
    place that combines all three, and it is dependency-free of the
    pipeline (no I/O, no clocks, no SQL).
  * Lets the §8 sharing knob be A/B-tested at the window level (one
    side-by-side `decide_window(..., share=0.0)` vs
    `decide_window(..., share=0.75)` call) without re-running the entire
    consolidation pipeline.

**Contract.**
  * Output keys are exactly `set(states_by_schema) ∩ set(evidence_by_schema)`.
    A schema with state but no evidence in this window contributes
    nothing (no decision); a schema with evidence but no state is a
    caller bug — we raise `KeyError` immediately so the pipeline cannot
    silently drop a CREATE.
  * Sibling lookup uses `schema_family_evidence.siblings_for`, so
    owner-exclusion (E1), partition-honoring (E5), and determinism (E3)
    are inherited.
  * `share=0.0` is byte-identical to a loop of bare `decide(state, ev)`
    calls (regression-safety; matches `decide_with_family`'s G1).

Invariants (locked in `tests/property/test_schema_family_window.py`):

  W1. share=0 ⇒ decide_window result is identical, key-by-key, to the
      bare-`decide()` mapping for every owner.
  W2. Empty clusters ⇒ same as W1 for any share (no siblings to share).
  W3. Determinism: result is invariant under input dict-iteration order.
  W4. Result keys are exactly evidence_by_schema's keys (states-only or
      evidence-less schemas contribute neither a key nor a side-effect).
  W5. Missing state for an evidenced schema raises KeyError(schema_id).
  W6. share outside [0, 1] raises ValueError (delegated to decide_with_family).
"""
from __future__ import annotations

from typing import Iterable, Mapping, Sequence

from engram.consolidation.schema_decision import EvidenceWindow, Thresholds
from engram.consolidation.schema_family_decision import decide_with_family
from engram.consolidation.schema_family_evidence import all_owner_siblings
from engram.consolidation.schema_lifecycle import EventKind, SchemaState


def decide_window(
    states_by_schema: Mapping[str, SchemaState],
    evidence_by_schema: Mapping[str, EvidenceWindow],
    clusters: Sequence[frozenset[str]] = (),
    thresholds: Thresholds = Thresholds(),
    share: float = 0.0,
) -> dict[str, EventKind | None]:
    """Apply §8 family-aware decisions across an entire window.

    Args:
      states_by_schema: schema_id → current SchemaState. Must contain a
        state for every schema_id present in `evidence_by_schema`.
      evidence_by_schema: schema_id → EvidenceWindow recorded in this
        consolidation cycle. Drives the output key set.
      clusters: output of `schema_family.cluster*()`; pass `()` (or any
        partition where every schema is a singleton) to disable
        prior-sharing.
      thresholds: promote/deprecate/recover thresholds.
      share: fraction of sibling evidence to credit, in [0, 1].

    Returns:
      schema_id → EventKind | None, one entry per evidenced schema.
      Iteration order matches sorted(evidence_by_schema) for determinism.

    Raises:
      KeyError: if any evidenced schema has no entry in `states_by_schema`.
      ValueError: if `share` is outside [0.0, 1.0] (delegated).
    """
    if not (0.0 <= share <= 1.0):
        # Surface ValueError before any work, matching decide_with_family.G6.
        raise ValueError(f"share must be in [0.0, 1.0], got {share}")

    siblings_map = all_owner_siblings(clusters, evidence_by_schema)
    out: dict[str, EventKind | None] = {}
    # Sorted for W3 determinism (dict insertion order = output iteration order).
    for owner in sorted(evidence_by_schema):
        if owner not in states_by_schema:
            raise KeyError(owner)
        own = evidence_by_schema[owner]
        sibs = siblings_map.get(owner, ())
        out[owner] = decide_with_family(
            states_by_schema[owner],
            own,
            siblings=sibs,
            thresholds=thresholds,
            share=share,
        )
    return out


__all__: Iterable[str] = ("decide_window",)
