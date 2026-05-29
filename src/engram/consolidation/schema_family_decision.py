"""Schema-family-aware lifecycle decision (Personize §8 prior-sharing).

Layered wrapper over `schema_decision.decide()`: aggregates evidence from a
schema's cluster-mates into a single effective `EvidenceWindow`, then
delegates to the existing single-schema policy. Cluster membership comes
from `schema_family.cluster()`.

**Sharing model.** A scalar `share ∈ [0, 1]` controls how much sibling
evidence the owner schema "borrows":

    eff_supports        = own.supports        + floor(share * Σ siblings.supports)
    eff_contradictions  = own.contradictions  + floor(share * Σ siblings.contradictions)
    eff_window_id       = own.window_id   (RECOVER freshness is owner-anchored)

Floors keep the policy integer-valued (matches `Thresholds`). The owner's
window_id is preserved so RECOVER's "fresh window" check (P3) is judged
against the owner's history, not a sibling's.

**Why a wrapper, not an edit to `decide()`.** Regression safety. With
`share=0.0` or no siblings, this function is **byte-identical** in
behavior to `decide()`. The §8 sharing knob can be A/B-rolled out behind
config without touching the well-tested single-schema policy.

Invariants (locked in `tests/property/test_schema_family_decision.py`):

  G1. share=0 ⇒ decide_with_family == decide(state, own, thresholds)
      for any sibling input.
  G2. Empty siblings ⇒ same identity as G1 for any share.
  G3. Monotonicity in share (supports): for fixed nonneg sibling supports
      and zero sibling contradictions, increasing `share` cannot turn a
      PROMOTE into None or a None-due-to-low-supports into DEPRECATE.
  G4. Monotonicity in share (contradictions): for fixed nonneg sibling
      contradictions and zero sibling supports from INFERRED/PROMOTED
      states, increasing `share` cannot turn a DEPRECATE into None.
  G5. share=1.0 with siblings == own counts is equivalent to doubling
      own counts (modulo floor, which is exact when share=1.0).
  G6. share outside [0, 1] raises ValueError.
"""
from __future__ import annotations

import math
import threading
from contextlib import contextmanager
from typing import Callable, Iterable, Iterator

from engram.consolidation.schema_decision import (
    EvidenceWindow,
    Thresholds,
    decide,
)
from engram.consolidation.schema_lifecycle import EventKind, SchemaState


# §94b-internal — optional decision trace hook.
#
# decide_with_family is purposefully simple, but §94b found the share knob
# operationally inert on the cross-session corpus. To distinguish
#   (a) decisions are identical across share values (math is moot at scale)
# from
#   (b) decisions differ but downstream retrieval is below corpus granularity,
# we expose a thread-local list-of-callbacks that observes every decision
# call. Recorders are pure no-ops by default; nothing in production code
# pushes a recorder. Tests / drivers use `family_decision_trace()`.
#
# Invariants (locked in tests/property/test_decision_trace.py):
#   T1. With no recorder installed, behavior is byte-identical to pre-trace.
#   T2. Recorder receives one record per decide_with_family call, in order.
#   T3. Recorder is thread-local; threads do not see each other's traces.
#   T4. Exceptions from recorder do not affect the return value (best-effort).
_TraceRecord = dict
_Recorder = Callable[[_TraceRecord], None]
_TLS = threading.local()


def _get_recorders() -> list[_Recorder]:
    rs = getattr(_TLS, "recorders", None)
    if rs is None:
        rs = []
        _TLS.recorders = rs
    return rs


@contextmanager
def family_decision_trace(
    sink: list[_TraceRecord] | None = None,
) -> Iterator[list[_TraceRecord]]:
    """Capture every decide_with_family call in the current thread.

    Yields a list that the contextmanager appends one dict per call to.
    The dict has keys: state, share, own_s, own_c, sib_s, sib_c, eff_s,
    eff_c, decision (str|None), borrowed_via_share (bool).
    """
    if sink is None:
        sink = []
    recorders = _get_recorders()

    def _rec(rec: _TraceRecord) -> None:
        sink.append(rec)

    recorders.append(_rec)
    try:
        yield sink
    finally:
        # Safe even if list was mutated; remove the exact callback.
        try:
            recorders.remove(_rec)
        except ValueError:
            pass


def _emit_trace(rec: _TraceRecord) -> None:
    rs = getattr(_TLS, "recorders", None)
    if not rs:
        return
    for r in tuple(rs):
        try:
            r(rec)
        except Exception:
            # T4: never let a recorder bug change a decision.
            pass


def _aggregate_siblings(
    siblings: Iterable[EvidenceWindow],
) -> tuple[int, int]:
    """Sum supports/contradictions across a sibling iterable. Pure."""
    s_sum = 0
    c_sum = 0
    for ev in siblings:
        s_sum += ev.supports
        c_sum += ev.contradictions
    return s_sum, c_sum


def decide_with_family(
    state: SchemaState,
    own: EvidenceWindow,
    siblings: Iterable[EvidenceWindow] = (),
    thresholds: Thresholds = Thresholds(),
    share: float = 0.0,
) -> EventKind | None:
    """Decide a lifecycle event using own + (share-weighted) sibling evidence.

    Args:
      state: owner schema's current lifecycle state.
      own: owner schema's evidence in the current window.
      siblings: evidence windows from cluster-mates (same window).
      thresholds: promote/deprecate/recover thresholds.
      share: fraction of sibling evidence to credit to owner, in [0, 1].
        `share=0.0` is regression-safe (identical to bare `decide()`).

    Returns:
      Event kind warranted by the effective evidence, or None.

    Raises:
      ValueError: if `share` is outside [0.0, 1.0].
    """
    if not (0.0 <= share <= 1.0):
        raise ValueError(f"share must be in [0.0, 1.0], got {share}")

    has_recorders = bool(getattr(_TLS, "recorders", None))

    if share == 0.0:
        # G1 / regression safety fast path.
        result = decide(state, own, thresholds)
        if has_recorders:
            _emit_trace({
                "state": state.name if hasattr(state, "name") else str(state),
                "share": 0.0,
                "own_s": own.supports,
                "own_c": own.contradictions,
                "sib_s": 0,
                "sib_c": 0,
                "eff_s": own.supports,
                "eff_c": own.contradictions,
                "decision": result.name if result is not None else None,
                "borrowed_via_share": False,
            })
        return result

    sib_s, sib_c = _aggregate_siblings(siblings)
    if sib_s == 0 and sib_c == 0:
        # G2: nothing to borrow.
        result = decide(state, own, thresholds)
        if has_recorders:
            _emit_trace({
                "state": state.name if hasattr(state, "name") else str(state),
                "share": share,
                "own_s": own.supports,
                "own_c": own.contradictions,
                "sib_s": 0,
                "sib_c": 0,
                "eff_s": own.supports,
                "eff_c": own.contradictions,
                "decision": result.name if result is not None else None,
                "borrowed_via_share": False,
            })
        return result

    eff = EvidenceWindow(
        window_id=own.window_id,
        supports=own.supports + math.floor(share * sib_s),
        contradictions=own.contradictions + math.floor(share * sib_c),
    )
    result = decide(state, eff, thresholds)
    if has_recorders:
        _emit_trace({
            "state": state.name if hasattr(state, "name") else str(state),
            "share": share,
            "own_s": own.supports,
            "own_c": own.contradictions,
            "sib_s": sib_s,
            "sib_c": sib_c,
            "eff_s": eff.supports,
            "eff_c": eff.contradictions,
            "decision": result.name if result is not None else None,
            "borrowed_via_share": (
                eff.supports != own.supports
                or eff.contradictions != own.contradictions
            ),
        })
    return result


__all__ = ["decide_with_family", "family_decision_trace"]
