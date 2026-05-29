"""§84 — Maximum-margin ``fmax`` rule (replaces midpoint).

`recommended_fmax(debiased=True)` in `engram.consolidation.sharing_regime`
currently returns the midpoint ``(f0 + f10) / 2`` of the clean baseline
and the c=0.10 contamination point. That picks the geometric center
without weighting for the *shape* of the (f0, f10) joint distribution —
on a noisy real corpus the empirical clouds for f0 and f10 can be
asymmetric, and a midpoint can land inside the f0 mass (gating clean
schemas) or outside the f10 mass (failing to gate contaminated ones).

This driver replaces that with the **maximum-margin** rule

    fmax* := argmax_x  P(f0 < x) · P(f10 ≥ x)

over the §83 paired bootstrap distribution of (f0_b, f10_b). That's
the operating point that maximizes the joint probability of correctly
not-gating clean rows AND correctly gating contaminated rows under
the empirical bootstrap clouds — a 1-D Youden-J / max-margin cut on
the bootstrap as classifier-CDF estimate.

Pure given the §83 JSON. Reports per-tau:
  - fmax_midpoint (§82 baseline, mean of f0_all + f10_all means / 2)
  - fmax_max_margin (this rule)
  - margin (the maximized product)
  - lift over midpoint (margin(max_margin) − margin(midpoint))
  - which f0_b / f10_b grid was searched

Entry: `python -m evals.fmax_max_margin --in <§83-json> --out <§84-json>`
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from evals.io_utils import atomic_write_json


def _margin(x: float, f0s: list[float], f10s: list[float]) -> float:
    """P(f0 < x) · P(f10 ≥ x) on the empirical paired distributions."""
    n0 = len(f0s)
    n1 = len(f10s)
    if n0 == 0 or n1 == 0:
        return 0.0
    p_clean = sum(1 for v in f0s if v < x) / n0
    p_contam = sum(1 for v in f10s if v >= x) / n1
    return p_clean * p_contam


def _search_max_margin(f0s: list[float], f10s: list[float]) -> tuple[float, float]:
    """Grid-search over the union of {f0s, f10s} sorted candidates;
    return (fmax*, margin*). Ties broken toward the smaller x (favors
    not-gating, i.e. precision)."""
    candidates = sorted(set(f0s) | set(f10s))
    if not candidates:
        return float("nan"), 0.0
    # Insert midpoints between consecutive candidates so the optimum
    # can land between observed values (avoids piecewise-constant ties).
    extended: list[float] = []
    for i, c in enumerate(candidates):
        extended.append(c)
        if i + 1 < len(candidates):
            extended.append((c + candidates[i + 1]) / 2.0)
    best_x = extended[0]
    best_m = -1.0
    for x in extended:
        m = _margin(x, f0s, f10s)
        if m > best_m:
            best_m = m
            best_x = x
    return best_x, best_m


def evaluate_tau(row: dict) -> dict:
    """One §83 per-tau row → §84 fmax+margin summary."""
    f0s = list(row.get("f0_all", []))
    f10s = list(row.get("f10_all", []))
    if not f0s or not f10s:
        return {
            "tau": row.get("tau"),
            "n_boot": row.get("n_boot"),
            "fmax_midpoint": None,
            "fmax_max_margin": None,
            "margin_midpoint": 0.0,
            "margin_max_margin": 0.0,
            "lift": 0.0,
            "skipped": True,
            "skip_reason": "missing paired f0_all/f10_all (older §83 output)",
        }
    f0_mean = sum(f0s) / len(f0s)
    f10_mean = sum(f10s) / len(f10s)
    midpoint = (f0_mean + f10_mean) / 2.0
    m_mid = _margin(midpoint, f0s, f10s)
    fmax_star, m_star = _search_max_margin(f0s, f10s)
    return {
        "tau": row["tau"],
        "n_boot": row["n_boot"],
        "f0_mean": f0_mean,
        "f10_mean": f10_mean,
        "fmax_midpoint": round(midpoint, 4),
        "fmax_max_margin": round(fmax_star, 4),
        "margin_midpoint": round(m_mid, 4),
        "margin_max_margin": round(m_star, 4),
        "lift": round(m_star - m_mid, 4),
        "skipped": False,
    }


def run(in_path: str | Path) -> dict:
    src = json.loads(Path(in_path).read_text())
    rows = [evaluate_tau(r) for r in src.get("by_tau", [])]
    return {
        "source": str(in_path),
        "n_boot": src.get("n_boot"),
        "n_schemas": src.get("n_schemas"),
        "by_tau": rows,
        "summary": {
            "any_lift_positive": any(r.get("lift", 0.0) > 0 for r in rows),
            "max_lift_tau": (
                max(rows, key=lambda r: r.get("lift", 0.0))["tau"]
                if rows
                else None
            ),
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="in_path",
        default="bench/results/locomo_fragmentation_per_tau_bootstrap.json",
    )
    p.add_argument(
        "--out",
        dest="out_path",
        default="bench/results/fmax_max_margin.json",
    )
    args = p.parse_args()
    res = run(args.in_path)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, res)
    print(json.dumps(res["summary"], indent=2))
    print()
    print(
        f"{'tau':>6} {'f0μ':>7} {'f10μ':>7} "
        f"{'fmax_mid':>9} {'fmax_mm':>9} "
        f"{'m_mid':>7} {'m_mm':>7} {'lift':>7}"
    )
    for r in res["by_tau"]:
        if r.get("skipped"):
            print(f"{r['tau']:>6.2f}  (skipped: {r.get('skip_reason')})")
            continue
        print(
            f"{r['tau']:>6.2f} "
            f"{r['f0_mean']:>7.4f} "
            f"{r['f10_mean']:>7.4f} "
            f"{r['fmax_midpoint']:>9.4f} "
            f"{r['fmax_max_margin']:>9.4f} "
            f"{r['margin_midpoint']:>7.4f} "
            f"{r['margin_max_margin']:>7.4f} "
            f"{r['lift']:>7.4f}"
        )


if __name__ == "__main__":
    main()
