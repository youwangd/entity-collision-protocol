"""Joint κ (tightness) × contamination sweep for §8 prior-sharing.

Closes the open question carried by runs #68 and #69: each axis was
characterized in isolation, but the §8 deployment rule

    share=0.75 is safe iff κ ≥ 21 AND contamination ≤ 10%

was an *AND* over two univariate sweeps. Are the axes independent
(damage = max of the two), super-additive, or sub-additive when both
degrade together? This driver answers it on one fixed-share grid.

Pure simulation, no I/O beyond the JSON dump. Reuses
`evals.schema_share_sweep.run_share` so the generative model and RNG
streams are byte-identical to §62/§68/§69 on the diagonal cells
(tightness=1.0, contamination=0.0) and on each one-axis edge.

Usage
-----
  python -m evals.schema_share_joint_sweep \\
      --out bench/results/share_joint_grid.json \\
      --share 0.75 --n-clusters 400

Default grid: tightness ∈ {1.0, 0.5, 0.1, 0.0} ×
              contamination ∈ {0.0, 0.1, 0.25, 0.5, 0.75, 1.0}.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from evals.schema_share_sweep import SimConfig, run_share
from evals.io_utils import atomic_write_json


def run_grid(
    cfg_base: SimConfig,
    share: float,
    tightnesses: list[float],
    contaminations: list[float],
) -> dict:
    """Evaluate every (tightness, contamination) cell at one share."""
    cells = []
    for t in tightnesses:
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
                tightness=t,
                contamination=c,
            )
            m = run_share(cfg, share).to_dict()
            m["tightness"] = t
            m["contamination"] = c
            cells.append(m)
    return {
        "config": {
            "share": share,
            "n_clusters": cfg_base.n_clusters,
            "cluster_size": cfg_base.cluster_size,
            "n_per_window": cfg_base.n_per_window,
            "n_windows": cfg_base.n_windows,
            "beta_a": cfg_base.beta_a,
            "beta_b": cfg_base.beta_b,
            "seed": cfg_base.seed,
        },
        "tightnesses": tightnesses,
        "contaminations": contaminations,
        "cells": cells,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--share", type=float, default=0.75)
    p.add_argument("--n-clusters", type=int, default=400)
    p.add_argument("--cluster-size", type=int, default=4)
    p.add_argument("--n-per-window", type=int, default=1)
    p.add_argument("--n-windows", type=int, default=30)
    p.add_argument("--beta-a", type=float, default=0.4)
    p.add_argument("--beta-b", type=float, default=0.4)
    p.add_argument("--seed", type=int, default=0xE17A11)
    p.add_argument(
        "--tightnesses", type=str, default="1.0,0.5,0.1,0.0",
    )
    p.add_argument(
        "--contaminations", type=str, default="0.0,0.1,0.25,0.5,0.75,1.0",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    if not (0.0 <= ns.share <= 1.0):
        raise SystemExit(f"share out of range: {ns.share}")
    ts = [float(x) for x in ns.tightnesses.split(",") if x.strip()]
    cs = [float(x) for x in ns.contaminations.split(",") if x.strip()]
    for t in ts:
        if not (0.0 <= t <= 1.0):
            raise SystemExit(f"tightness out of range: {t}")
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
    out = run_grid(cfg_base, ns.share, ts, cs)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ns.out, out)
    print(f"wrote {ns.out}  ({ns.out.stat().st_size} bytes)")

    def cell(t: float, c: float, key: str) -> float | None:
        for x in out["cells"]:
            if x["tightness"] == t and x["contamination"] == c:
                return x[key]
        return None
    for label, key in [
        ("false_promote_low_q", "false_promote_low_q"),
        ("false_deprecate_high_q", "false_deprecate_high_q"),
    ]:
        print(f"\n{label} (share={ns.share}, rows=tightness, cols=contamination):")
        header = "       " + " ".join(f"c={c:>4}" for c in cs)
        print(header)
        for t in ts:
            row = " ".join(
                ("     -" if (v := cell(t, c, key)) is None else f"{v:>6.3f}")
                for c in cs
            )
            print(f"t={t:>4} {row}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
