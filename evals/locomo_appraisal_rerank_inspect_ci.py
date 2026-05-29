"""§94c-appraisal-inspect-CI — bootstrap CI + permutation test on the
appraisal-rerank inspector artifact.

Inputs the JSON artifact written by `evals.locomo_appraisal_rerank_inspect`
and emits:

  1. Percentile bootstrap 95% CI on the **salience gap**
     (displacing − displaced_gold) — the headline mean +0.1404 (n=136).
  2. Percentile bootstrap 95% CI on the **Scherer relevance gap**
     (n=106).
  3. Permutation test on the asymmetry **lost_rank1 vs gained_rank1**
     under H0 = arm-label exchangeable per question. Under the null,
     for each question we randomly relabel which arm is "anchor" vs
     "anchor+appraisal" with prob 0.5 (which flips the sign of the
     rank movement, so `lost_rank1` ↔ `gained_rank1` and
     `improved_within_topk` ↔ `worsened_within_topk`). The observed
     statistic is `lost_rank1 − gained_rank1` (= 5 − 1 = 4 on the
     fixture). We report the two-sided permutation p-value.

This pins the `appraisal` re-rank effect: if the salience-gap CI
excludes 0, displacing items are *materially* more salient than the
gold they displace. If the permutation p is large, the 5:1 lost:gained
asymmetry is consistent with chance.

Usage:
    python -m evals.locomo_appraisal_rerank_inspect_ci \\
        --in bench/results/locomo_appraisal_rerank_inspect.json \\
        --resamples 10000 --permutations 10000 --seed 42 \\
        --out bench/results/locomo_appraisal_rerank_inspect_ci.json \\
        --md-out bench/results/locomo_appraisal_rerank_inspect_ci.md
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from evals.bootstrap_ci import _bootstrap_mean_ci
from evals.io_utils import atomic_write_json, atomic_write_text


def _two_sided_p_from_diffs(diffs: list[float]) -> float:
    """Two-sided percentile p-value for H0: mean(diffs)=0."""
    n = len(diffs)
    if n == 0:
        return 1.0
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    p_one = min(pos, neg) / n
    return min(1.0, 2 * p_one)


def _bootstrap_p(values: list[float], resamples: int, seed: int) -> float:
    """Bootstrap two-sided p-value for H0: mean(values)=0.

    Counts the fraction of resamples whose mean has the opposite sign
    of the observed mean (or equal to zero), times 2.
    """
    n = len(values)
    if n == 0:
        return 1.0
    obs = statistics.fmean(values)
    if obs == 0.0:
        return 1.0
    rng = random.Random(seed)
    extreme = 0
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += values[rng.randrange(n)]
        m = s / n
        # H0: true mean is 0. We approximate the null by re-centering:
        # under bootstrap, the centered mean (m - obs) has mean 0.
        # p ≈ Pr(|m - obs| >= |obs|) under null.
        if abs(m - obs) >= abs(obs):
            extreme += 1
    return min(1.0, extreme / resamples)


def _permutation_lost_minus_gained(
    per_q: list[dict], permutations: int, seed: int
) -> dict:
    """Two-sided permutation test on (lost_rank1 - gained_rank1).

    Per question, with prob 0.5 swap arm labels (which flips the bin
    `lost_rank1` ↔ `gained_rank1` and `improved` ↔ `worsened`). All
    other bins are symmetric under the swap.
    """
    rng = random.Random(seed)

    swap_map = {
        "lost_rank1": "gained_rank1",
        "gained_rank1": "lost_rank1",
        "improved_within_topk": "worsened_within_topk",
        "worsened_within_topk": "improved_within_topk",
    }
    bins = [q["movement_bin"] for q in per_q]

    def _stat_from(bs: list[str]) -> int:
        c = {}
        for b in bs:
            c[b] = c.get(b, 0) + 1
        return c.get("lost_rank1", 0) - c.get("gained_rank1", 0)

    obs = _stat_from(bins)
    n_extreme = 0
    null_dist: list[int] = []
    for _ in range(permutations):
        flipped = [
            (swap_map.get(b, b) if rng.random() < 0.5 else b) for b in bins
        ]
        s = _stat_from(flipped)
        null_dist.append(s)
        if abs(s) >= abs(obs):
            n_extreme += 1
    p = (n_extreme + 1) / (permutations + 1)  # +1 smoothing (Phipson&Smyth)
    return {
        "observed_lost_minus_gained": obs,
        "permutations": permutations,
        "p_value_two_sided": round(p, 4),
        "null_mean": round(statistics.fmean(null_dist), 4),
        "null_stdev": round(statistics.pstdev(null_dist), 4),
    }


def run_ci(
    inspect_json_path: str,
    *,
    resamples: int = 10_000,
    permutations: int = 10_000,
    seed: int = 42,
) -> dict:
    raw = json.loads(Path(inspect_json_path).read_text())
    per_q = raw.get("per_question") or []

    # Salience gap and rel gap (only on rows with both displacing and
    # displaced_gold_in_a so the gap is defined).
    sg: list[float] = []
    rg: list[float] = []
    for q in per_q:
        d = q.get("displacing")
        if not d:
            continue
        a_gold = d.get("displaced_gold_in_a")
        if not a_gold:
            continue
        disp = d["displacing"]
        sg.append(float(disp["salience"]) - float(a_gold["salience"]))
        rg.append(float(disp["rel"]) - float(a_gold["rel"]))

    def _ci(values: list[float]) -> dict:
        if not values:
            return {"n": 0}
        m, lo, hi = _bootstrap_mean_ci(values, resamples, seed)
        p = _bootstrap_p(values, resamples, seed + 1)
        return {
            "n": len(values),
            "mean": round(m, 6),
            "ci_lo": round(lo, 6),
            "ci_hi": round(hi, 6),
            "p_two_sided": round(p, 4),
            "ci_excludes_zero": bool(lo > 0 or hi < 0),
        }

    salience_ci = _ci(sg)
    rel_ci = _ci(rg)

    perm = _permutation_lost_minus_gained(per_q, permutations, seed + 2)

    return {
        "source": str(inspect_json_path),
        "anchor": raw.get("anchor"),
        "probe_stage": raw.get("probe_stage"),
        "embedder": raw.get("embedder"),
        "max_instances": raw.get("max_instances"),
        "n_questions": raw.get("n_questions"),
        "k": raw.get("k"),
        "ci_config": {
            "resamples": resamples,
            "permutations": permutations,
            "seed": seed,
            "alpha": 0.05,
        },
        "salience_gap_ci": salience_ci,
        "rel_gap_ci": rel_ci,
        "permutation_lost_minus_gained": perm,
    }


def render_markdown(rep: dict) -> str:
    lines: list[str] = []
    lines.append(
        f"### §94c-appraisal-inspect-CI — bootstrap CI on the "
        f"salience-gap (anchor=`{rep['anchor']}`, probe=`{rep['probe_stage']}`, "
        f"embedder={rep['embedder']}, n_questions={rep['n_questions']})"
    )
    lines.append("")
    sg = rep["salience_gap_ci"]
    rg = rep["rel_gap_ci"]
    perm = rep["permutation_lost_minus_gained"]
    if sg.get("n"):
        flag = " ★" if sg["ci_excludes_zero"] else ""
        lines.append(
            f"- **Salience gap (displacing − displaced_gold).** "
            f"n={sg['n']}  mean={sg['mean']:+.4f}  "
            f"95% CI [{sg['ci_lo']:+.4f}, {sg['ci_hi']:+.4f}]  "
            f"p={sg['p_two_sided']:.3f}{flag}"
        )
    if rg.get("n"):
        flag = " ★" if rg["ci_excludes_zero"] else ""
        lines.append(
            f"- **Scherer relevance gap (displacing − gold).** "
            f"n={rg['n']}  mean={rg['mean']:+.4f}  "
            f"95% CI [{rg['ci_lo']:+.4f}, {rg['ci_hi']:+.4f}]  "
            f"p={rg['p_two_sided']:.3f}{flag}"
        )
    lines.append(
        f"- **Permutation (lost_rank1 − gained_rank1).** "
        f"observed={perm['observed_lost_minus_gained']}, "
        f"permutations={perm['permutations']}, "
        f"p={perm['p_value_two_sided']:.3f}, "
        f"null mean={perm['null_mean']:+.3f} ± {perm['null_stdev']:.3f}"
    )
    lines.append("")
    return "\n".join(lines) + "\n"


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--in",
        dest="inp",
        default="bench/results/locomo_appraisal_rerank_inspect.json",
    )
    p.add_argument("--resamples", type=int, default=10_000)
    p.add_argument("--permutations", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    p.add_argument("--md-out", default=None)
    args = p.parse_args()

    rep = run_ci(
        args.inp,
        resamples=args.resamples,
        permutations=args.permutations,
        seed=args.seed,
    )
    sg = rep["salience_gap_ci"]
    rg = rep["rel_gap_ci"]
    perm = rep["permutation_lost_minus_gained"]
    if sg.get("n"):
        print(
            f"salience_gap n={sg['n']} mean={sg['mean']:+.4f} "
            f"CI[{sg['ci_lo']:+.4f},{sg['ci_hi']:+.4f}] "
            f"p={sg['p_two_sided']:.3f} "
            f"excludes_zero={sg['ci_excludes_zero']}"
        )
    if rg.get("n"):
        print(
            f"rel_gap      n={rg['n']} mean={rg['mean']:+.4f} "
            f"CI[{rg['ci_lo']:+.4f},{rg['ci_hi']:+.4f}] "
            f"p={rg['p_two_sided']:.3f} "
            f"excludes_zero={rg['ci_excludes_zero']}"
        )
    print(
        f"perm(lost−gained) obs={perm['observed_lost_minus_gained']} "
        f"p={perm['p_value_two_sided']:.3f}"
    )
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep)
        print(f"[appraisal-inspect-ci] wrote {args.out}")
    if args.md_out:
        Path(args.md_out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(args.md_out, render_markdown(rep))
        print(f"[appraisal-inspect-ci] wrote {args.md_out}")


if __name__ == "__main__":
    main()
