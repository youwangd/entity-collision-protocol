"""Calibrate the runtime contamination meter against a known true-c knob.

SCALE_REPORT §73 wired ``contamination_rate(...)`` into the stage 6 gate
behind ``schema_family_contamination_max=0.10``. That ``0.10`` was lifted
from the §69 deployment rule, but §69 measures ``c`` as the
*outsider-injection probability* into the sibling-evidence stream — not
as a reading of this meter on a real ``cluster()`` partition. Without a
calibration curve linking the two, the operational threshold is
uninterpretable.

This driver provides that curve.

Generative model
----------------
* Global vocabulary ``V`` of size ``vocab_size``.
* ``n_clusters`` latent clusters. Each cluster c picks a "core vocab"
  ``V_c ⊂ V`` of size ``core_size`` (disjoint per cluster when feasible).
* Each cluster has ``k = cluster_size`` schemas. With probability
  ``1 - p`` a schema is an *insider*: features sampled from ``V_c``.
  With probability ``p`` it is an *outsider*: features sampled
  uniformly from ``V``, independent of any cluster's core.
* All schemas (insiders + outsiders) are pooled and passed through
  ``schema_family.cluster(features, tau)``. The clustering routine
  has no access to the latent labels.

Then for each ``p`` we report the meter reading
``contamination_rate(features, clusters, tau)``, the mean per-cluster
contamination among non-singleton clusters, the fragmentation rate
(#singletons / #schemas), and cluster-size diagnostics.

Why this is non-trivial
-----------------------
Single-link clustering on outsider-rich corpora doesn't necessarily
*report* the contamination — outsiders frequently get split off into
their own singletons, deflating the meter. Predicted shape: meter
rises with ``p`` for small ``p`` (outsiders that happen to share
tokens with a core get glued in), flattens or drops for large ``p``
(outsiders fragment into singletons that contribute zero pair weight).

The meter's defensibility as a runtime gate hinges on whether the
calibration curve is monotone over the operating regime
``p ∈ [0, 0.25]``. If yes: the §69 rule ``cmax = 0.10`` translates
to a meter threshold readable directly off the curve. If no: the
gate's ``0.10`` default has to be re-derived.

Usage
-----
  python -m evals.schema_contamination_calibration \\
      --out bench/results/contamination_calibration.json \\
      --n-clusters 200
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path

from evals.io_utils import atomic_write_json

from engram.consolidation.schema_family import cluster as cluster_fn
from engram.consolidation.schema_family_contamination import (
    cluster_contamination,
    contamination_rate,
)


@dataclass
class CalibConfig:
    n_clusters: int = 200
    cluster_size: int = 4
    vocab_size: int = 2000
    core_size: int = 8
    schema_size: int = 6
    tau: float = 0.5
    seed: int = 0xCA11B


def _build_corpus(
    cfg: CalibConfig, p: float
) -> dict[str, frozenset[str]]:
    """Build the schema feature corpus with outsider-injection rate p."""
    seed = (cfg.seed * 1_000_003) ^ int(round(p * 1e9))
    rng = random.Random(seed & 0xFFFFFFFF)
    vocab = [f"t{i}" for i in range(cfg.vocab_size)]

    # Disjoint cores when feasible; otherwise sample with replacement.
    cores: list[list[str]] = []
    if cfg.n_clusters * cfg.core_size <= cfg.vocab_size:
        pool = list(vocab)
        rng.shuffle(pool)
        for i in range(cfg.n_clusters):
            cores.append(pool[i * cfg.core_size : (i + 1) * cfg.core_size])
    else:
        for _ in range(cfg.n_clusters):
            cores.append(rng.sample(vocab, cfg.core_size))

    features: dict[str, frozenset[str]] = {}
    for ci, core in enumerate(cores):
        for ki in range(cfg.cluster_size):
            sid = f"c{ci:04d}_s{ki}"
            if rng.random() < p:
                feats = rng.sample(vocab, cfg.schema_size)
            else:
                feats = rng.sample(core, min(cfg.schema_size, len(core)))
            features[sid] = frozenset(feats)
    return features


def _evaluate(cfg: CalibConfig, p: float) -> dict:
    features = _build_corpus(cfg, p)
    clusters = cluster_fn(features, tau=cfg.tau)
    rate = contamination_rate(features, clusters, cfg.tau)

    sizes = [len(c) for c in clusters]
    n_singletons = sum(1 for s in sizes if s == 1)
    nonsingleton = [c for c in clusters if len(c) >= 2]
    if nonsingleton:
        per = [cluster_contamination(features, c, cfg.tau) for c in nonsingleton]
        mean_nonsingleton = sum(per) / len(per)
    else:
        mean_nonsingleton = 0.0

    n_schemas = sum(sizes)
    return {
        "true_c": p,
        "meter_rate": rate,
        "mean_per_nonsingleton": mean_nonsingleton,
        "n_clusters": len(clusters),
        "n_singletons": n_singletons,
        "n_nonsingleton_clusters": len(nonsingleton),
        "fragmentation": n_singletons / n_schemas if n_schemas else 0.0,
        "max_cluster_size": max(sizes) if sizes else 0,
        "mean_cluster_size": sum(sizes) / len(sizes) if sizes else 0.0,
    }


def run_grid(cfg: CalibConfig, contaminations: list[float]) -> dict:
    cells = [_evaluate(cfg, p) for p in contaminations]
    return {
        "config": {
            "n_clusters": cfg.n_clusters,
            "cluster_size": cfg.cluster_size,
            "vocab_size": cfg.vocab_size,
            "core_size": cfg.core_size,
            "schema_size": cfg.schema_size,
            "tau": cfg.tau,
            "seed": cfg.seed,
        },
        "cells": cells,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-clusters", type=int, default=200)
    ap.add_argument("--cluster-size", type=int, default=4)
    ap.add_argument("--tau", type=float, default=0.5)
    ap.add_argument(
        "--contaminations",
        type=str,
        default="0.0,0.025,0.05,0.075,0.1,0.15,0.2,0.25,0.5,0.75,1.0",
    )
    ap.add_argument("--seed", type=int, default=0xCA11B)
    args = ap.parse_args()
    cs = [float(x) for x in args.contaminations.split(",")]
    cfg = CalibConfig(
        n_clusters=args.n_clusters,
        cluster_size=args.cluster_size,
        tau=args.tau,
        seed=args.seed,
    )
    result = run_grid(cfg, cs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, result)
    print(f"wrote {args.out}")
    for cell in result["cells"]:
        print(
            f"  c={cell['true_c']:.3f}  meter={cell['meter_rate']:.3f}  "
            f"meanNS={cell['mean_per_nonsingleton']:.3f}  "
            f"frag={cell['fragmentation']:.3f}  "
            f"nClus={cell['n_clusters']:>4}  "
            f"maxSize={cell['max_cluster_size']}"
        )


if __name__ == "__main__":
    main()
