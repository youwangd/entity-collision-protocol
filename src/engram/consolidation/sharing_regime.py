"""§81 — Sharing-operability regime classifier.

Given a per-tau fragmentation curve `frag(true_c)` produced by the
§76/§79/§80 calibration drivers, classify each tau row by *operational
mechanism* with respect to the §75 fragmentation gate.

Four mutually-exclusive labels:

* ``COLLAPSED`` — the meter is dead at c=0 *and* at the c=0.10 §69
  safety frontier (both fragmentations ≤ 0.05). Single-link clustering
  glues outsiders into existing clusters; the gate cannot fire because
  the signal it measures never moves. Observed in §78 at tau ≤ 0.10.
* ``SINGLETON_CLIFF`` — fragmentation is already saturated at c=0
  (≥ 0.90). cluster() returns near-all-singletons even with zero
  outsider injection, so the gate trips identically and is equivalent
  to ``share=0.0``. Observed in §77/§78 at tau ≥ 0.25 on LoCoMo.
* ``NATURALLY_FRAGMENTED`` — the meter is well-defined and the curve
  is monotone, but the baseline frag(c=0) already exceeds 0.05. The
  gate trips out-of-the-box on a clean corpus, so it is *operable* but
  not *gateable* in the §79 sense — share collapse is unconditional,
  not contamination-driven.
* ``GATEABLE`` — frag(c=0) ≤ 0.05 AND frag(c=0.10) − frag(c=0) ≥ 0.05.
  The §79 sweet spot: a calibrated per-tau ``fragmentation_max`` separates
  clean from contaminated regimes. The recommended `fmax` is `frag(c=0.10)`.

This module is pure: it consumes a list of `{tau, frag_at_c0,
frag_at_c10, ...}` dicts (the row schema produced by both
``evals.schema_fragmentation_per_tau_calibration`` and
``evals.locomo_fragmentation_per_tau_calibration``) and emits a
labelled, summary-aware report. No I/O, no clocks, no RNG.

Threshold defaults match the §76/§79/§80 calibration conventions and
are exposed as keyword args for sweep work.

See SCALE_REPORT §81 for the LoCoMo + synthetic re-classification.
"""

from __future__ import annotations

from typing import Mapping, Sequence

# Public label constants.
COLLAPSED = "COLLAPSED"
SINGLETON_CLIFF = "SINGLETON_CLIFF"
NATURALLY_FRAGMENTED = "NATURALLY_FRAGMENTED"
GATEABLE = "GATEABLE"

REGIME_LABELS = (COLLAPSED, SINGLETON_CLIFF, NATURALLY_FRAGMENTED, GATEABLE)


def classify_row(
    row: Mapping,
    *,
    clean_eps: float = 0.05,
    saturate_eps: float = 0.90,
    contam_lift: float = 0.05,
    debiased: bool = False,
) -> str:
    """Classify a single per-tau fragmentation row.

    Required keys on `row`: ``frag_at_c0``, ``frag_at_c10`` (either may
    be ``None``, which routes the row to ``COLLAPSED``).

    Decision tree (in order):

    1. saturated at c0 (`frag_at_c0 ≥ saturate_eps`) → SINGLETON_CLIFF.
    2. clean-and-collapsed (`frag_at_c0 ≤ clean_eps` AND
       `frag_at_c10 ≤ clean_eps`) → COLLAPSED.
    3. clean baseline (`frag_at_c0 ≤ clean_eps`) AND lift
       (`frag_at_c10 - frag_at_c0 ≥ contam_lift`) → GATEABLE.
    4. else → NATURALLY_FRAGMENTED.

    The (saturated, clean) combination is impossible by construction
    (`saturate_eps > clean_eps`), so the order between cases 1 and 2 is
    immaterial; we keep saturated-first for readability.

    ``debiased=True`` switches to baseline-debiased semantics
    (§82): the gate measures *added* fragmentation `Δ := f10 − f0`
    against `contam_lift`, ignoring the baseline `f0` for the clean-
    baseline check. SINGLETON_CLIFF (saturation at c0) is preserved.
    A row is GATEABLE-debiased iff (a) not saturated at c0 AND (b)
    Δ ≥ contam_lift. Recovers two LoCoMo tau bins (0.10, 0.15)
    flagged NATURALLY_FRAGMENTED under default semantics: the gate
    *can* fire on contamination, it just rides on top of a non-zero
    baseline. Predicted by §81 NEXT.md and confirmed in SCALE_REPORT
    §82.
    """
    if not (0.0 < clean_eps < saturate_eps < 1.0):
        raise ValueError(
            "clean_eps must be in (0, saturate_eps) and saturate_eps in "
            f"(clean_eps, 1.0); got clean_eps={clean_eps}, "
            f"saturate_eps={saturate_eps}"
        )
    if contam_lift < 0:
        raise ValueError(f"contam_lift must be >=0; got {contam_lift}")

    frag_c0 = row.get("frag_at_c0")
    frag_c10 = row.get("frag_at_c10")

    if frag_c0 is None or frag_c10 is None:
        return COLLAPSED

    if frag_c0 >= saturate_eps:
        return SINGLETON_CLIFF

    delta = frag_c10 - frag_c0

    if debiased:
        # Baseline-debiased semantics: the gate fires on *added*
        # fragmentation. f0 ≤ clean_eps + Δ ≤ clean_eps ⇒ COLLAPSED
        # (signal dead in both axes); else if Δ ≥ contam_lift the
        # row is gateable regardless of the f0 baseline.
        if frag_c0 <= clean_eps and delta < contam_lift:
            return COLLAPSED
        if delta >= contam_lift:
            return GATEABLE
        return NATURALLY_FRAGMENTED

    if frag_c0 <= clean_eps and frag_c10 <= clean_eps:
        return COLLAPSED

    if frag_c0 <= clean_eps and delta >= contam_lift:
        return GATEABLE

    return NATURALLY_FRAGMENTED


def recommended_fmax(
    row: Mapping,
    *,
    debiased: bool = False,
    max_margin: bool = False,
    calibration_table: str = "LOCOMO",
) -> float | None:
    """Return per-tau `fragmentation_max` for GATEABLE rows, else None.

    Default semantics: ``fmax := frag_at_c10`` (rounded to 4 decimals).
    Debiased semantics (§82): ``fmax := frag_at_c0 + (f10 − f0) / 2`` —
    halfway between the clean baseline and the c=0.10 contamination
    point, the operating point that maximizes c=0/c=0.10 separability
    on a noisy real corpus where f0 is non-zero.

    ``max_margin=True`` (§84/§86): when the row's tau matches a
    calibrated key in ``engram.consolidation.calibration`` (default
    ``LOCOMO`` table), return the pre-computed max-margin fmax
    (`argmax_x P(f0 < x) · P(f10 ≥ x)` over a paired bootstrap
    distribution). Strictly-better-or-equal to the midpoint by
    construction. Requires ``debiased=True`` — max-margin is only
    defined under baseline-debiased semantics. Falls back to the
    midpoint when the tau is not in the calibration table.
    """
    if max_margin and not debiased:
        raise ValueError("max_margin=True requires debiased=True")
    if classify_row(row, debiased=debiased) != GATEABLE:
        return None
    if max_margin:
        from engram.consolidation.calibration import lookup_max_margin_fmax

        tau = row.get("tau")
        cal = lookup_max_margin_fmax(tau, table=calibration_table) if tau is not None else None
        if cal is not None:
            return round(float(cal), 4)
        # fall through to midpoint when uncalibrated
    if debiased:
        f0 = row["frag_at_c0"]
        f10 = row["frag_at_c10"]
        return round((f0 + f10) / 2.0, 4)
    return round(row["frag_at_c10"], 4)


def classify_curve(
    rows: Sequence[Mapping],
    *,
    clean_eps: float = 0.05,
    saturate_eps: float = 0.90,
    contam_lift: float = 0.05,
    debiased: bool = False,
    max_margin: bool = False,
    calibration_table: str = "LOCOMO",
) -> dict:
    """Classify a per-tau fragmentation curve and emit a summary.

    Returns a dict with::

        {
          "rows": [
            {"tau": float, "regime": str, "recommended_fmax": float|None,
             "frag_at_c0": float|None, "frag_at_c10": float|None},
            ...
          ],
          "summary": {
            "n_taus": int,
            "by_regime": {COLLAPSED: int, SINGLETON_CLIFF: int,
                          NATURALLY_FRAGMENTED: int, GATEABLE: int},
            "gateable_taus": [float, ...],
            "median_recommended_fmax": float | None,
            "operable": bool,        # any GATEABLE
          },
        }

    Determinism: row order is preserved; ``gateable_taus`` is in input
    order (which the upstream drivers emit ascending).
    """
    out_rows: list[dict] = []
    by_regime: dict[str, int] = {label: 0 for label in REGIME_LABELS}
    gateable_taus: list[float] = []
    fmax_values: list[float] = []

    for row in rows:
        regime = classify_row(
            row,
            clean_eps=clean_eps,
            saturate_eps=saturate_eps,
            contam_lift=contam_lift,
            debiased=debiased,
        )
        fmax = (
            recommended_fmax(
                row,
                debiased=debiased,
                max_margin=max_margin,
                calibration_table=calibration_table,
            )
            if regime == GATEABLE
            else None
        )
        out_rows.append(
            {
                "tau": row.get("tau"),
                "regime": regime,
                "recommended_fmax": fmax,
                "frag_at_c0": row.get("frag_at_c0"),
                "frag_at_c10": row.get("frag_at_c10"),
            }
        )
        by_regime[regime] += 1
        if regime == GATEABLE:
            gateable_taus.append(row.get("tau"))
            if fmax is not None:
                fmax_values.append(fmax)

    median_fmax: float | None
    if fmax_values:
        s = sorted(fmax_values)
        n = len(s)
        median_fmax = (
            s[n // 2] if n % 2 == 1 else (s[n // 2 - 1] + s[n // 2]) / 2
        )
        median_fmax = round(median_fmax, 4)
    else:
        median_fmax = None

    return {
        "rows": out_rows,
        "summary": {
            "n_taus": len(out_rows),
            "by_regime": by_regime,
            "gateable_taus": gateable_taus,
            "median_recommended_fmax": median_fmax,
            "operable": by_regime[GATEABLE] > 0,
        },
    }


__all__ = [
    "COLLAPSED",
    "SINGLETON_CLIFF",
    "NATURALLY_FRAGMENTED",
    "GATEABLE",
    "REGIME_LABELS",
    "classify_row",
    "recommended_fmax",
    "classify_curve",
]
