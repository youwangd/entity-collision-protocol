"""evals.run_arm — unified arm driver for matched baseline-vs-treatment runs.

Eliminates the per-script bootstrap re-implementation by taking
(dataset, arm, n, seed) and emitting a self-describing bundle:

    out_dir/
      baseline.json      # full adapter output for arm=baseline
      treatment.json     # full adapter output for the treatment arm
      paired_ci.json     # paired bootstrap CIs on Δhit@1 / Δhit@k / Δmrr

Pairing is positional and we assert ID alignment (question_id for LME,
sample_id+category for LoCoMo) before computing diffs.

Usage:
    python -m evals.run_arm \\
        --dataset lme --dataset-path "$LONGMEMEVAL_PATH/longmemeval_s.json" \\
        --arm both --n 100 --seed 42 \\
        --embed st --vector-weight 0.3 \\
        --out-dir evals/results/lme_n100_st_vw03_both

Both LME and LoCoMo are supported via --dataset {lme,locomo}.

The script is intentionally a thin orchestrator over the existing
adapter functions — it does not duplicate their config logic. We
construct the same Config object the adapters' `_build_config` /
`_build_arm_config` would have, and we reuse `evals.bootstrap_ci`
helpers for the paired CI math.

Exit codes:
    0  success
    2  unknown --dataset
    3  pairing failed (length mismatch or ID divergence)
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any

from evals.bootstrap_ci import _bootstrap_mean_ci, _paired_diff_ci
from evals.io_utils import atomic_write_json


# ------------------------------------------------------------------ adapters


def _run_lme(
    dataset_path: str,
    arm: str,
    n: int,
    *,
    embed: str | None,
    vector_weight: float | None,
    qe_dominance: float,
    qe_type_purity_min: float | None,
    qe_backend: str,
    sp_alpha: float,
    sp_pool: int,
    k: int,
) -> dict[str, Any]:
    from evals.longmemeval_adapter import _build_config, run_lme
    from engram.core.config import Config

    cfg = (
        _build_config(
            arm,
            qe_dominance,
            sp_alpha,
            sp_pool,
            qe_type_purity_min=qe_type_purity_min,
            qe_backend=qe_backend,
        )
        if arm != "baseline"
        else None
    )
    if vector_weight is not None:
        if cfg is None:
            cfg = Config()
        cfg.retrieval.vector_weight = float(vector_weight)
    return run_lme(
        dataset_path, max_instances=n, k=k, config=cfg, embedder_name=embed
    )


def _run_locomo(
    dataset_path: str,
    arm: str,
    n: int,
    *,
    embed: str | None,
    vector_weight: float | None,
    qe_dominance: float,
    qe_type_purity_min: float | None,
    qe_backend: str,
    sp_alpha: float,
    sp_pool: int,
    k: int,
) -> dict[str, Any]:
    from evals.locomo_adapter import _build_arm_config, run_locomo
    from engram.core.config import Config

    arm_cfg = _build_arm_config(
        arm,
        qe_dominance,
        sp_alpha,
        sp_pool,
        qe_type_purity_min=qe_type_purity_min,
        qe_backend=qe_backend,
    )
    if vector_weight is not None:
        if arm_cfg is None:
            arm_cfg = Config()
        arm_cfg.retrieval.vector_weight = float(vector_weight)
    return run_locomo(
        dataset_path,
        max_instances=n,
        k=k,
        config=arm_cfg,
        embedder=embed,
    )


# ---------------------------------------------------------------- pairing


def _extract_pairs(
    dataset: str, baseline: dict, treatment: dict
) -> tuple[list[float], list[float], list[float], list[float], list[float], list[float], int]:
    """Return aligned (b_h1, t_h1, b_hk, t_hk, b_rr, t_rr, n)."""
    if dataset == "lme":
        b = baseline.get("per_instance", [])
        t = treatment.get("per_instance", [])
        key = "question_id"
    else:
        b = baseline.get("per_query", [])
        t = treatment.get("per_query", [])
        key = "sample_id"

    if len(b) != len(t):
        raise SystemExit(
            f"[run_arm] paired length mismatch: baseline n={len(b)} vs treatment n={len(t)}"
        )

    b_h1: list[float] = []
    t_h1: list[float] = []
    b_hk: list[float] = []
    t_hk: list[float] = []
    b_rr: list[float] = []
    t_rr: list[float] = []
    misaligned = 0
    for br, tr in zip(b, t):
        if br.get(key) != tr.get(key):
            misaligned += 1
            continue
        b_h1.append(float(br.get("hit_at_1", 0)))
        t_h1.append(float(tr.get("hit_at_1", 0)))
        b_hk.append(float(br.get("hit_at_k", 0)))
        t_hk.append(float(tr.get("hit_at_k", 0)))
        b_rr.append(float(br.get("reciprocal_rank", 0.0)))
        t_rr.append(float(tr.get("reciprocal_rank", 0.0)))
    if misaligned > 0:
        raise SystemExit(
            f"[run_arm] paired ID divergence: {misaligned} rows with mismatched {key}"
        )
    return b_h1, t_h1, b_hk, t_hk, b_rr, t_rr, len(b_h1)


def _paired_summary(
    b_h1, t_h1, b_hk, t_hk, b_rr, t_rr, *, resamples: int, seed: int
) -> dict[str, Any]:
    out: dict[str, Any] = {"n": len(b_h1)}
    for name, b, t in (
        ("hit_at_1", b_h1, t_h1),
        ("hit_at_k", b_hk, t_hk),
        ("mrr", b_rr, t_rr),
    ):
        bm, blo, bhi = _bootstrap_mean_ci(b, resamples, seed)
        tm, tlo, thi = _bootstrap_mean_ci(t, resamples, seed + 1)
        dm, dlo, dhi = _paired_diff_ci(t, b, resamples, seed + 2)
        out[name] = {
            "baseline": {"mean": round(bm, 4), "ci_lo": round(blo, 4), "ci_hi": round(bhi, 4)},
            "treatment": {"mean": round(tm, 4), "ci_lo": round(tlo, 4), "ci_hi": round(thi, 4)},
            "delta": {"mean": round(dm, 4), "ci_lo": round(dlo, 4), "ci_hi": round(dhi, 4)},
            "significant_at_05": (dlo > 0) or (dhi < 0),
        }
    return out


# -------------------------------------------------------------------- main


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["lme", "locomo"], required=True)
    p.add_argument("--dataset-path", type=str, default=None,
                   help="Path to dataset (defaults to $LONGMEMEVAL_PATH/longmemeval_s.json or $LOCOMO_PATH)")
    p.add_argument("--arm", choices=["prf", "share_prior", "both"], required=True,
                   help="Treatment arm to compare against baseline")
    p.add_argument("--n", type=int, required=True, help="Number of instances")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--embed", type=str, default=None)
    p.add_argument("--vector-weight", type=float, default=None)
    p.add_argument("--qe-dominance", type=float, default=0.3)
    p.add_argument("--qe-type-purity-min", type=float, default=None)
    p.add_argument("--qe-backend", type=str, default="heuristic")
    p.add_argument("--sp-alpha", type=float, default=0.10)
    p.add_argument("--sp-pool", type=int, default=20)
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--out-dir", type=str, required=True)
    args = p.parse_args()

    # Default dataset path
    if not args.dataset_path:
        if args.dataset == "lme":
            base = os.environ.get("LONGMEMEVAL_PATH")
            if base:
                args.dataset_path = os.path.join(base, "longmemeval_s.json")
        else:
            args.dataset_path = os.environ.get("LOCOMO_PATH")
    if not args.dataset_path:
        raise SystemExit("--dataset-path required (or set $LONGMEMEVAL_PATH / $LOCOMO_PATH)")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runner = _run_lme if args.dataset == "lme" else _run_locomo
    common = dict(
        embed=args.embed,
        vector_weight=args.vector_weight,
        qe_dominance=args.qe_dominance,
        qe_type_purity_min=args.qe_type_purity_min,
        qe_backend=args.qe_backend,
        sp_alpha=args.sp_alpha,
        sp_pool=args.sp_pool,
        k=args.k,
    )

    random.seed(args.seed)

    print(f"[run_arm] baseline: {args.dataset} n={args.n} ...", flush=True)
    t0 = time.monotonic()
    baseline = runner(args.dataset_path, "baseline", args.n, **common)
    baseline["wall_s"] = round(time.monotonic() - t0, 2)
    atomic_write_json(out_dir / "baseline.json", baseline)

    print(f"[run_arm] treatment ({args.arm}): {args.dataset} n={args.n} ...", flush=True)
    t0 = time.monotonic()
    treatment = runner(args.dataset_path, args.arm, args.n, **common)
    treatment["wall_s"] = round(time.monotonic() - t0, 2)
    atomic_write_json(out_dir / "treatment.json", treatment)

    b_h1, t_h1, b_hk, t_hk, b_rr, t_rr, n_paired = _extract_pairs(
        args.dataset, baseline, treatment
    )
    paired = _paired_summary(
        b_h1, t_h1, b_hk, t_hk, b_rr, t_rr,
        resamples=args.resamples, seed=args.seed,
    )
    paired["arm"] = args.arm
    paired["dataset"] = args.dataset
    paired["n_paired"] = n_paired
    paired["seed"] = args.seed
    atomic_write_json(out_dir / "paired_ci.json", paired)

    print(json.dumps(paired, indent=2))


if __name__ == "__main__":
    main()
