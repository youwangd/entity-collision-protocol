"""Multi-window evidence decay for schema-lifecycle decisions.

The single-window policy in `schema_decision.py` makes promote/deprecate/
recover decisions from one rolling `EvidenceWindow` at a time. Real
agents accumulate evidence over many windows; treating them in isolation
forces an awkward choice of window size and discards the temporal
gradient (recent contradictions should weigh more than ancient supports).

This module is the multi-window aggregator: it collapses a chronological
sequence of `EvidenceWindow`s into a single *effective* window suitable
for `decide()`, with an exponential decay that re-weights older windows
toward zero.

Pure function: same input → same output. No clocks, no RNG. Integer
output (we floor at the boundary so the integer-decidable downstream
policy stays integer-decidable, and property tests stay sound).

Design choices, lock in property tests:

  D1. **factor=1.0 ⇒ pure cumulative sum** over the last `horizon` windows.
      This is the additive multi-window mode: history is treated as one
      big window. Useful for schemas where evidence is rare.

  D2. **factor=0.0 ⇒ only newest window matters.** This recovers the
      current single-window behavior exactly. Useful as a regression
      guard: callers who haven't opted into decay see no change.

  D3. **Monotonicity.** Increasing any single window's `supports`
      cannot decrease the effective `supports`. Same for contradictions.
      (This is what makes the existing `decide()` P5 monotonicity
      compose with the aggregator.)

  D4. **Horizon truncation is suffix-only.** Only the last `horizon`
      windows count; older windows are dropped before weighting. This
      bounds memory and gives a clean test target ("adding ancient
      windows beyond the horizon is a no-op").

  D5. **Newest window's `window_id` is preserved** in the aggregate.
      This is what `decide()` uses for the RECOVER fresh-window check;
      we must not silently re-use a stale window_id from history.

  D6. **Zero windows ⇒ ValueError.** The caller must hand us at least
      one window. Empty history is a programming error, not an
      "everything is fine" state.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

from engram.consolidation.schema_decision import EvidenceWindow


@dataclass(frozen=True)
class DecayPolicy:
    """Exponential decay across evidence windows.

    Weight of the i-th most-recent window in [0, horizon) is
    `factor ** i` (so i=0 → 1.0, the newest, has full weight).

    Attributes:
      factor: per-window decay multiplier in [0.0, 1.0]. 1.0 = no decay
        (cumulative). 0.0 = full decay (only newest window matters).
      horizon: number of trailing windows to consider. Older ones are
        dropped from the suffix. Must be >= 1.
    """
    factor: float = 1.0
    horizon: int = 8

    def __post_init__(self) -> None:
        if not (0.0 <= self.factor <= 1.0):
            raise ValueError(f"decay factor must be in [0.0, 1.0], got {self.factor}")
        if self.horizon < 1:
            raise ValueError(f"horizon must be >= 1, got {self.horizon}")


def aggregate(
    history: Sequence[EvidenceWindow],
    policy: DecayPolicy = DecayPolicy(),
) -> EvidenceWindow:
    """Collapse chronologically ordered evidence into one effective window.

    `history[0]` is the OLDEST, `history[-1]` is the NEWEST.

    Counts are `floor(sum_i count_i * factor ** (newest_idx - i))`
    over the last `policy.horizon` windows. Floor (not round) keeps
    the result conservative: it only ever fires `decide()` on
    evidence that actually accumulated past an integer threshold.

    The returned window's `window_id` is the newest window's id, so
    `decide()`'s fresh-window check still works against the prior
    `SchemaState.last_window_id`.

    Raises:
      ValueError: if `history` is empty.
    """
    if not history:
        raise ValueError("aggregate() requires at least one EvidenceWindow")

    suffix = history[-policy.horizon:]
    newest_idx = len(suffix) - 1

    # factor=0 needs special handling because 0**0 = 1 but we want only
    # the newest window to count. Equivalent to slicing to [-1:].
    if policy.factor == 0.0:
        suffix = suffix[-1:]
        newest_idx = 0

    weighted_supports = 0.0
    weighted_contras = 0.0
    for i, ev in enumerate(suffix):
        age = newest_idx - i
        w = policy.factor ** age if age > 0 else 1.0
        weighted_supports += ev.supports * w
        weighted_contras += ev.contradictions * w

    newest = suffix[-1]
    return EvidenceWindow(
        window_id=newest.window_id,
        supports=int(math.floor(weighted_supports)),
        contradictions=int(math.floor(weighted_contras)),
    )


__all__ = ["DecayPolicy", "aggregate"]
