"""Share × contamination sweep for §8 prior-sharing — deployment lookup.

Where §70's joint κ × c grid pinned share=0.75 and showed contamination
*dominates* tightness (∂FD/∂c ≫ ∂FD/∂κ), this driver pins tightness=1.0
and walks the *share × contamination* plane. The deployment question it
answers:

    given measured (or estimated) contamination c, what is the highest
    share value that still keeps (FP ≤ 1%, FD ≤ 5%)?

Equivalently: as cluster contamination rises, how aggressively must we
back off prior-sharing to stay inside the §8 safe envelope? §69 proved
share=0.75 collapses past c≈0.10. The expected shape is a monotone
upper-bound curve `share_max(c)`: for c=0.0 we recover the §62 sweet
spot (share≈0.75 safe, share=1.0 regresses); for large c every share>0
is unsafe and the only defensible setting is share=0.0.

Pure simulation, deterministic. Reuses `evals.schema_share_sweep.run_share`
so the (share=0, c=0) cell is byte-identical to §62 baseline and the
(share=0.75, c=*) column is byte-identical to §69.

Usage
-----
  python -m evals.schema_share_contam_sweep \\
      --out bench/results/share_contam_grid.json \\
      --n-clusters 400

Default grid: share ∈ {0.0, 0.25, 0.5, 0.75, 1.0} ×
              contamination ∈ {0.0, 0.05, 0.1, 0.25, 0.5, 0.75, 1.0}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evals.schema_share_sweep import SimConfig, run_share
from evals.io_utils import atomic_write_json


def run_grid(
    cfg_base: SimConfig,
    shares: list[float],
    contaminations: list[float],
) -> dict:
    """Evaluate every (share, contamination) cell at fixed tightness=1.0."""
    cells = []
    for s in shares:
        for c in contaminations:
            cfg = SimConfig(
                n_clusters=cfg_base.n_clusters,
                cluster_size=cfg_base.cluster_size,
                n_per_window=cfg_base.n_per_window,
                n_windows=cfg_base.n_windows,
                beta_a=cfg_base.beta_a,
                beta_b=cfg_base.beta_b,
                promote_thresh=cfg_base.promote_thresh,
                deprecate_thresh=cfg_base.deprecate_thresh,
                recover_thresh=cfg_base.recover_thresh,
                seed=cfg_base.seed,
                tightness=1.0,
                contamination=c,
            )
            m = run_share(cfg, s).to_dict()
            m["share"] = s
            m["contamination"] = c
            cells.append(m)
    return {
        "config": {
            "tightness": 1.0,
            "n_clusters": cfg_base.n_clusters,
            "cluster_size": cfg_base.cluster_size,
            "n_per_window": cfg_base.n_per_window,
            "n_windows": cfg_base.n_windows,
            "beta_a": cfg_base.beta_a,
            "beta_b": cfg_base.beta_b,
            "seed": cfg_base.seed,
        },
        "shares": shares,
        "contaminations": contaminations,
        "cells": cells,
    }


def safe_envelope(
    grid: dict,
    fp_max: float = 0.01,
    fd_max: float = 0.05,
    progress_min: float = 0.0,
) -> dict[float, float | None]:
    """For each contamination, the largest share with (FP≤fp_max, FD≤fd_max,
    promote_rate_high≥progress_min).

    The progress floor is critical: in this generative model with sparse
    evidence, share≤0.5 has promote_rate_high=0 — every schema sits in
    INFERRED forever. The error rates are trivially 0 because no decisions
    are made. Without a progress floor the envelope reports share_max=0.0
    as "safe" everywhere, which is technically true and operationally
    useless.

    Returns {c: share_max or None}; None means no share value satisfies all
    three constraints at that contamination level.
    """
    out: dict[float, float | None] = {}
    for c in grid["contaminations"]:
        best: float | None = None
        for s in grid["shares"]:
            cell = next(
                x for x in grid["cells"]
                if x["share"] == s and x["contamination"] == c
            )
            if (
                cell["false_promote_low_q"] <= fp_max
                and cell["false_deprecate_high_q"] <= fd_max
                and cell["promote_rate_high"] >= progress_min
            ):
                if best is None or s > best:
                    best = s
        out[c] = best
    return out


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-clusters", type=int, default=400)
    p.add_argument("--cluster-size", type=int, default=4)
    p.add_argument("--n-per-window", type=int, default=1)
    p.add_argument("--n-windows", type=int, default=30)
    p.add_argument("--beta-a", type=float, default=0.4)
    p.add_argument("--beta-b", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0xE17A11)
    p.add_argument("--shares", type=str, default="0.0,0.25,0.5,0.75,1.0")
    p.add_argument(
        "--contaminations", type=str, default="0.0,0.05,0.1,0.25,0.5,0.75,1.0",
    )
    p.add_argument("--fp-max", type=float, default=0.01)
    p.add_argument("--fd-max", type=float, default=0.05)
    p.add_argument(
        "--progress-min", type=float, default=0.5,
        help="minimum promote_rate_high for a cell to count as 'safe' "
             "(default 0.5: at least half of high-q schemas must promote)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    ss = [float(x) for x in ns.shares.split(",") if x.strip()]
    cs = [float(x) for x in ns.contaminations.split(",") if x.strip()]
    for s in ss:
        if not (0.0 <= s <= 1.0):
            raise SystemExit(f"share out of range: {s}")
    for c in cs:
        if not (0.0 <= c <= 1.0):
            raise SystemExit(f"contamination out of range: {c}")
    cfg_base = SimConfig(
        n_clusters=ns.n_clusters,
        cluster_size=ns.cluster_size,
        n_per_window=ns.n_per_window,
        n_windows=ns.n_windows,
        beta_a=ns.beta_a,
        beta_b=ns.beta_b,
        seed=ns.seed,
    )
    out = run_grid(cfg_base, ss, cs)
    out["safe_envelope"] = {
        f"{c}": v for c, v in
        safe_envelope(
            out, fp_max=ns.fp_max, fd_max=ns.fd_max,
            progress_min=ns.progress_min,
        ).items()
    }
    out["envelope_thresholds"] = {
        "fp_max": ns.fp_max, "fd_max": ns.fd_max,
        "progress_min": ns.progress_min,
    }
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ns.out, out)
    print(f"wrote {ns.out}  ({ns.out.stat().st_size} bytes)")

    def cell(s: float, c: float, key: str) -> float | None:
        for x in out["cells"]:
            if x["share"] == s and x["contamination"] == c:
                return x[key]
        return None
    for label, key in [
        ("false_promote_low_q", "false_promote_low_q"),
        ("false_deprecate_high_q", "false_deprecate_high_q"),
    ]:
        print(f"\n{label} (rows=share, cols=contamination):")
        print("       " + " ".join(f"c={c:>4}" for c in cs))
        for s in ss:
            row = " ".join(
                ("     -" if (v := cell(s, c, key)) is None else f"{v:>6.3f}")
                for c in cs
            )
            print(f"s={s:>4} {row}")

    print(
        f"\nsafe envelope (FP≤{ns.fp_max}, FD≤{ns.fd_max}, "
        f"promote_rate_high≥{ns.progress_min}) "
        "→ max safe share per contamination:"
    )
    for c in cs:
        v = out["safe_envelope"][f"{c}"]
        print(f"  c={c:>4}  share_max = {'none' if v is None else f'{v}'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
