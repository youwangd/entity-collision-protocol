"""Pure schema-lifecycle decision policy.

Maps an evidence summary for a single schema in a single rolling window to
a lifecycle event kind (or `None` if no transition is warranted).

This is the policy half of the lifecycle pipeline; the reducer in
`schema_lifecycle.py` is the mechanics half. Splitting them lets us:

  * unit-test the policy against synthetic evidence with no event-log
    plumbing in the way; and
  * swap policies later (e.g. a learned threshold per schema family)
    without touching the reducer or the DAG invariants.

Policy summary (TODO-RESEARCH §B closed 2026-05-24, paraphrase of Personize.ai
"Governed Memory" §6 promote/deprecate/recover):

    INFERRED  + supports >= K_promote                         → PROMOTE
    INFERRED  + contradictions >= K_deprecate                 → DEPRECATE
    PROMOTED  + contradictions >= K_deprecate                 → DEPRECATE
    DEPRECATED + supports >= K_recover AND fresh window_id    → RECOVER
    otherwise                                                 → None

Determinism: pure function of `(state, evidence, thresholds)`. No RNG,
no clocks. Same inputs → same EventKind.

Invariants (locked here, fuzzed in
`tests/property/test_schema_decision.py`):

  P1. PROMOTE is only ever emitted from INFERRED.
  P2. DEPRECATE is only ever emitted from INFERRED or PROMOTED.
  P3. RECOVER is only ever emitted from DEPRECATED, and only when
      `evidence.window_id != state.last_window_id` (matches reducer
      invariant #5: a fresh window is required to leave deprecation).
  P4. With supports==contradictions==0 the decision is `None` for all
      states (no evidence → no action, never spurious BUMP_VERSION).
  P5. Thresholds are non-negative ints and the decision is monotone in
      evidence: adding more supports cannot turn a PROMOTE into None,
      adding more contradictions cannot turn a DEPRECATE into None.
"""
from __future__ import annotations

from dataclasses import dataclass

from engram.consolidation.schema_lifecycle import EventKind, SchemaState, SchemaStatus


@dataclass(frozen=True)
class EvidenceWindow:
    """Rolling-window evidence summary for one schema.

    Attributes:
      window_id: stable identifier for this evidence window. RECOVER
        decisions require this to differ from `SchemaState.last_window_id`.
      supports: count of *distinct* supporting facts observed in this
        window. (Distinctness is the caller's job — the policy treats
        the integer as ground truth.)
      contradictions: count of distinct contradicting facts observed
        in this window.
    """
    window_id: str
    supports: int = 0
    contradictions: int = 0

    def __post_init__(self) -> None:  # pragma: no cover - trivial guard
        if self.supports < 0 or self.contradictions < 0:
            raise ValueError("evidence counts must be non-negative")


@dataclass(frozen=True)
class Thresholds:
    """Promote/deprecate/recover thresholds.

    All counts; no rates. Rate-based policies can be layered on by the
    caller (compute rate → derive a synthetic supports/contradictions
    pair → pass in). Keeping the policy integer-valued makes property
    tests trivially decidable.
    """
    promote: int = 3
    deprecate: int = 2
    recover: int = 3

    def __post_init__(self) -> None:  # pragma: no cover - trivial guard
        if self.promote <= 0 or self.deprecate <= 0 or self.recover <= 0:
            raise ValueError("thresholds must be positive")


def decide(
    state: SchemaState,
    evidence: EvidenceWindow,
    thresholds: Thresholds = Thresholds(),
) -> EventKind | None:
    """Return the lifecycle event kind warranted by `evidence`, or None.

    Contradictions take precedence over supports for INFERRED schemas:
    a single window that simultaneously crosses both thresholds means
    the schema is unstable enough that promotion would be premature.
    This is conservative-by-design and keeps the DAG simple (we never
    have to break a tie between PROMOTE and DEPRECATE in the same step).
    """
    # P3 short-circuit: deprecated schemas can ONLY be moved by a
    # fresh-window RECOVER. Any other evidence is logged-but-ignored
    # at this layer (the caller may still record the supports for
    # downstream telemetry).
    if state.status == SchemaStatus.DEPRECATED:
        if (
            evidence.supports >= thresholds.recover
            and evidence.window_id != state.last_window_id
        ):
            return EventKind.RECOVER
        return None

    # Both INFERRED and PROMOTED can be DEPRECATED on contradiction.
    if evidence.contradictions >= thresholds.deprecate:
        return EventKind.DEPRECATE

    # PROMOTE only from INFERRED (P1). PROMOTED schemas with more
    # supports stay promoted — there is no `promoted → promoted_more`
    # transition in the DAG.
    if (
        state.status == SchemaStatus.INFERRED
        and evidence.supports >= thresholds.promote
    ):
        return EventKind.PROMOTE

    return None


__all__ = ["EvidenceWindow", "Thresholds", "decide"]
