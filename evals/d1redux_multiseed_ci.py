"""§D9 multi-seed bootstrap CI: heuristic vs spacy_sm on the synthK6 collider.

Re-runs the §D9 fixture (n_entities=12, K=6, distractors_per_entity=4,
hash256 embedder) across multiple seeds × {heuristic, spacy_sm} backends,
captures per-query rows for entity_weight ∈ {0.0, 0.10}, and computes
paired bootstrap CIs on:

  Δ_within(ew=0.10 − ew=0.00, per backend)              — channel lift
  Δ_between(spacy_sm Δ@ew=0.10 − heuristic Δ@ew=0.10)   — backend gap

Pairing is by (seed, query_index) so each draw resamples query *positions*
that are valid across all four arms.

Usage (default 3 seeds, ~5 min wall on hash256 hardware):
    python -m evals.d1redux_multiseed_ci --seeds 42,7,11 \
        --out evals/results/d1redux_multiseed_ci.json
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import time
from pathlib import Path

from evals.entity_channel_sweep import run_sweep
from evals.io_utils import atomic_write_json


WEIGHTS = [0.0, 0.10]


def _bootstrap_paired(diffs: list[float], resamples: int, seed: int,
                      alpha: float = 0.05) -> dict:
    n = len(diffs)
    if n == 0:
        return {"mean": 0.0, "ci_lo": 0.0, "ci_hi": 0.0, "n": 0}
    rng = random.Random(seed)
    means: list[float] = []
    for _ in range(resamples):
        s = 0.0
        for _ in range(n):
            s += diffs[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int(math.floor((alpha / 2) * resamples))]
    hi = means[min(int(math.ceil((1 - alpha / 2) * resamples)) - 1, resamples - 1)]
    return {
        "mean": round(statistics.fmean(diffs), 6),
        "ci_lo": round(lo, 6),
        "ci_hi": round(hi, 6),
        "n": n,
    }


def _run(seed: int, ner: str, *, embed_name: str = "hash256") -> dict:
    from evals.ablation import _make_embedder
    from evals._embed_cache import CachingEmbeddingProvider
    emb = _make_embedder(embed_name)
    if emb is not None:
        emb = CachingEmbeddingProvider(emb)
    rep = run_sweep(
        weights=WEIGHTS,
        seed=seed,
        entity_ner=ner,
        fixture="synth_entity",
        synth_n_entities=12,
        synth_collision_degree=6,
        synth_distractors_per_entity=4,
        embedder=emb,
        embed_name=embed_name,
        save_per_query=True,
    )
    return rep


def _per_query_at(rep: dict, ew: float) -> list[dict]:
    for a in rep["arms"]:
        if abs(a["entity_weight"] - ew) < 1e-9:
            return a.get("rows", [])
    raise KeyError(f"no arm with entity_weight={ew}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", default="42,7,11",
                   help="Comma-separated seeds.")
    p.add_argument("--embed", default="hash256")
    p.add_argument("--ners", default="heuristic,spacy_sm",
                   help="Comma-separated NER backends to compare. The first "
                        "is treated as the 'baseline' for between-backend "
                        "gap CIs against each subsequent backend.")
    p.add_argument("--resamples", type=int, default=5000)
    p.add_argument("--bootstrap-seed", type=int, default=20260523)
    p.add_argument("--out", default="evals/results/d1redux_multiseed_ci.json")
    args = p.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    ners = tuple(s.strip() for s in args.ners.split(",") if s.strip())
    assert len(ners) >= 1
    t0 = time.monotonic()

    runs: dict[tuple[int, str], dict] = {}
    for s in seeds:
        for ner in ners:
            print(f"[d1redux-ms] seed={s} ner={ner} embed={args.embed} ...", flush=True)
            tic = time.monotonic()
            runs[(s, ner)] = _run(s, ner, embed_name=args.embed)
            print(f"            wall={time.monotonic() - tic:.1f}s", flush=True)

    # --- Within-backend channel lift (paired across all seeds, all queries) ---
    within: dict[str, dict] = {}
    for ner in ners:
        diff_h1, diff_h5 = [], []
        for s in seeds:
            r0 = _per_query_at(runs[(s, ner)], 0.0)
            r1 = _per_query_at(runs[(s, ner)], 0.10)
            assert len(r0) == len(r1), f"len mismatch {len(r0)} vs {len(r1)}"
            for q0, q1 in zip(r0, r1):
                diff_h1.append(q1["hit@1"] - q0["hit@1"])
                diff_h5.append(q1["hit@5"] - q0["hit@5"])
        within[ner] = {
            "d_hit@1": _bootstrap_paired(diff_h1, args.resamples, args.bootstrap_seed),
            "d_hit@5": _bootstrap_paired(diff_h5, args.resamples, args.bootstrap_seed + 1),
        }

    # --- Between-backend: each non-baseline ner Δ − baseline Δ at ew=0.10 ---
    baseline_ner = ners[0]
    between: dict[str, dict] = {}
    for ner in ners[1:]:
        between_h1, between_h5 = [], []
        for s in seeds:
            sp0 = _per_query_at(runs[(s, ner)], 0.0)
            sp1 = _per_query_at(runs[(s, ner)], 0.10)
            he0 = _per_query_at(runs[(s, baseline_ner)], 0.0)
            he1 = _per_query_at(runs[(s, baseline_ner)], 0.10)
            m = min(len(sp0), len(sp1), len(he0), len(he1))
            for i in range(m):
                sp_d1 = sp1[i]["hit@1"] - sp0[i]["hit@1"]
                he_d1 = he1[i]["hit@1"] - he0[i]["hit@1"]
                sp_d5 = sp1[i]["hit@5"] - sp0[i]["hit@5"]
                he_d5 = he1[i]["hit@5"] - he0[i]["hit@5"]
                between_h1.append(sp_d1 - he_d1)
                between_h5.append(sp_d5 - he_d5)
        between[f"{ner}_minus_{baseline_ner}"] = {
            "d_hit@1": _bootstrap_paired(between_h1, args.resamples, args.bootstrap_seed + 100),
            "d_hit@5": _bootstrap_paired(between_h5, args.resamples, args.bootstrap_seed + 101),
        }

    # Per-seed point estimates (no CI) for inspection.
    per_seed = []
    for s in seeds:
        row = {"seed": s}
        for ner in ners:
            arm0 = next(a for a in runs[(s, ner)]["arms"] if a["entity_weight"] == 0.0)
            arm1 = next(a for a in runs[(s, ner)]["arms"] if abs(a["entity_weight"] - 0.10) < 1e-9)
            row[f"{ner}_h1@0"] = arm0["hit@1"]
            row[f"{ner}_h1@.1"] = arm1["hit@1"]
            row[f"{ner}_d_h1"] = round(arm1["hit@1"] - arm0["hit@1"], 6)
            row[f"{ner}_d_h5"] = round(arm1["hit@5"] - arm0["hit@5"], 6)
        per_seed.append(row)

    summary = {
        "seeds": seeds,
        "embed": args.embed,
        "ners": list(ners),
        "fixture": "synth_entity n_entities=12 K=6 distractors_per_entity=4",
        "weights_compared": [0.0, 0.10],
        "resamples": args.resamples,
        "alpha": 0.05,
        "within_backend_lift_paired_by_query": within,
        "between_backend_gap_paired_by_query": between,
        "per_seed_point": per_seed,
        "wall_seconds": round(time.monotonic() - t0, 2),
    }

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, summary)
    print(f"\n[d1redux-ms] wrote {args.out}")
    print(json.dumps({
        "within": within,
        "between": between,
        "per_seed": per_seed,
    }, indent=2))


if __name__ == "__main__":
    main()
