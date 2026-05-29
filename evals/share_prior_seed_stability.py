"""§96 stacking sweep — seed-stability check.

Prior run (seed=42) reported Δpair@10 = +0.050 for share_prior α=0.10
stacked with any entity_weight ∈ {0.10, 0.20, 0.30} at pool_size=20. The
NEXT.md flagged that single-seed result as the obvious follow-up: re-run
across multiple seeds to see whether the lift survives sampling noise.

This driver runs a *trimmed* grid (the recommended pool=20 row only,
ew ∈ {0.0, 0.10, 0.20, 0.30}, α=0.10) across N seeds and reports
mean ± std of pair@5 / pair@10 / Δpair@10 vs. the per-seed baseline.

Output
------
- evals/results/share_prior_seed_stability.json
- Markdown table on stdout
- if --update-report, appends to SHARE_PRIOR_REPORT.md
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from statistics import mean, pstdev

from evals.share_prior_stack_sweep import _eval_arm
from evals.share_prior_sweep import generate_bridge_corpus
from evals.io_utils import atomic_write_json, atomic_write_text


def _stats(xs):
    if not xs:
        return {"mean": 0.0, "std": 0.0, "n": 0}
    return {
        "mean": round(mean(xs), 6),
        "std": round(pstdev(xs) if len(xs) > 1 else 0.0, 6),
        "n": len(xs),
        "min": round(min(xs), 6),
        "max": round(max(xs), 6),
    }


def run(*, alpha, entity_weights, pool_size, n_pairs, plain_distractors, seeds):
    t0 = time.monotonic()
    per_seed = []
    for seed in seeds:
        ds = generate_bridge_corpus(
            n_pairs=n_pairs, plain_distractors=plain_distractors, seed=seed
        )
        base = _eval_arm(ds, reranker=None, alpha=0.0,
                         entity_weight=0.0, pool_size=pool_size)
        arms = [{"label": "baseline", **base}]
        for ew in entity_weights:
            arm = _eval_arm(ds, reranker="share_prior", alpha=alpha,
                            entity_weight=ew, pool_size=pool_size)
            arm["d_pair@5"] = round(arm["pair_recall@5"]
                                    - base["pair_recall@5"], 6)
            arm["d_pair@10"] = round(arm["pair_recall@10"]
                                     - base["pair_recall@10"], 6)
            arms.append({"label": f"share_prior+ew={ew}", **arm})
        per_seed.append({"seed": seed, "arms": arms})

    # aggregate by label across seeds
    by_label = {}
    for record in per_seed:
        for arm in record["arms"]:
            by_label.setdefault(arm["label"], []).append(arm)

    aggregates = {}
    for label, arms in by_label.items():
        aggregates[label] = {
            "pair@5": _stats([a["pair_recall@5"] for a in arms]),
            "pair@10": _stats([a["pair_recall@10"] for a in arms]),
            "d_pair@5": _stats([a.get("d_pair@5", 0.0) for a in arms]),
            "d_pair@10": _stats([a.get("d_pair@10", 0.0) for a in arms]),
        }

    return {
        "alpha": alpha,
        "entity_weights": entity_weights,
        "pool_size": pool_size,
        "corpus": {"n_pairs": n_pairs,
                   "plain_distractors": plain_distractors,
                   "seeds": seeds},
        "per_seed": per_seed,
        "aggregates": aggregates,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }


def _md(rep):
    seeds = rep["corpus"]["seeds"]
    lines = [
        f"Wall: {rep['wall_seconds']}s  "
        f"seeds={seeds}  "
        f"α={rep['alpha']}  "
        f"pool={rep['pool_size']}  "
        f"n_pairs={rep['corpus']['n_pairs']}  "
        f"distractors={rep['corpus']['plain_distractors']}",
        "",
        "| arm | pair@5 (μ±σ) | pair@10 (μ±σ) | "
        "Δpair@5 (μ±σ, [min,max]) | Δpair@10 (μ±σ, [min,max]) |",
        "|:---|:---|:---|:---|:---|",
    ]
    # stable order: baseline first, then share_prior arms in input order
    label_order = ["baseline"] + [
        f"share_prior+ew={ew}" for ew in rep["entity_weights"]
    ]
    for label in label_order:
        if label not in rep["aggregates"]:
            continue
        a = rep["aggregates"][label]
        lines.append(
            f"| {label} "
            f"| {a['pair@5']['mean']:.3f} ± {a['pair@5']['std']:.3f} "
            f"| {a['pair@10']['mean']:.3f} ± {a['pair@10']['std']:.3f} "
            f"| {a['d_pair@5']['mean']:+.3f} ± {a['d_pair@5']['std']:.3f} "
            f"[{a['d_pair@5']['min']:+.3f},{a['d_pair@5']['max']:+.3f}] "
            f"| {a['d_pair@10']['mean']:+.3f} ± {a['d_pair@10']['std']:.3f} "
            f"[{a['d_pair@10']['min']:+.3f},{a['d_pair@10']['max']:+.3f}] |"
        )
    # verdict line
    sp_arms = [
        rep["aggregates"][f"share_prior+ew={ew}"]
        for ew in rep["entity_weights"]
        if f"share_prior+ew={ew}" in rep["aggregates"]
    ]
    if sp_arms:
        best = max(sp_arms, key=lambda x: x["d_pair@10"]["mean"])
        ci_lo = best["d_pair@10"]["mean"] - best["d_pair@10"]["std"]
        ci_hi = best["d_pair@10"]["mean"] + best["d_pair@10"]["std"]
        verdict = (
            f"\n**Verdict:** best Δpair@10 across share_prior arms "
            f"= {best['d_pair@10']['mean']:+.3f} ± {best['d_pair@10']['std']:.3f} "
            f"(±1σ band: [{ci_lo:+.3f}, {ci_hi:+.3f}]). "
        )
        if ci_lo > 0 and best["d_pair@10"]["min"] >= 0:
            verdict += "Signal **survives seed shuffling** with no negative seed."
        elif ci_lo > 0:
            verdict += ("Mean is positive at ±1σ but at least one seed "
                        "regressed; treat as noisy lift.")
        else:
            verdict += ("±1σ band straddles 0 — single-seed +0.050 result "
                        "**does not generalize**, downgrade recommendation.")
        lines.append(verdict)
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--alpha", type=float, default=0.10)
    p.add_argument("--entity-weights", type=str, default="0.0,0.1,0.2,0.3")
    p.add_argument("--pool-size", type=int, default=20)
    p.add_argument("--n-pairs", type=int, default=60)
    p.add_argument("--plain-distractors", type=int, default=50)
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--out",
                   default="evals/results/share_prior_seed_stability.json")
    p.add_argument("--update-report", action="store_true")
    args = p.parse_args()

    ews = [float(x) for x in args.entity_weights.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]

    rep = run(alpha=args.alpha, entity_weights=ews,
              pool_size=args.pool_size,
              n_pairs=args.n_pairs,
              plain_distractors=args.plain_distractors,
              seeds=seeds)

    md = _md(rep)
    print("§96 share_prior — seed-stability (bridge recipe)")
    print(md)

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        atomic_write_json(args.out, rep, default=str)
        print(f"[seed-stability] wrote {args.out}")

    if args.update_report:
        report = Path("SHARE_PRIOR_REPORT.md")
        header = (f"\n## Seed-stability check "
                  f"(α={rep['alpha']}, "
                  f"pool={rep['pool_size']}, "
                  f"seeds={rep['corpus']['seeds']})\n\n")
        text = header + md + "\n"
        if report.exists():
            atomic_write_text(report, report.read_text() + text)
        else:
            atomic_write_text(report, "# §96 share_prior — Stacking Report\n" + text)
        print(f"[seed-stability] appended to {report}")


if __name__ == "__main__":
    main()
