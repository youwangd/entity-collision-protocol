"""Calibrate the §74 fragmentation gate's `fragmentation_max` default.

Companion to ``evals/schema_contamination_calibration.py``. SCALE_REPORT §74
established that in the realistic (disjoint-core, large-vocab) clustering
regime, the §73 contamination meter reads ≈0.0 across true-c ∈ [0, 1] —
single-link expels outsiders as singletons rather than gluing them in —
and **fragmentation tracks true-c almost 1:1**. The §75 commit shipped the
gate but kept ``fragmentation_max=None`` (disabled) pending a real-corpus-
defensible default. This driver provides that calibration.

Goal
----
The §69 deployment rule says ``share=0.75`` is safe iff true contamination
``c ≤ 0.10``. We need a fragmentation reading ``frag*`` that corresponds
to ``c=0.10`` so the gate trips precisely at the safety frontier.

Design
------
For each ``(vocab_size, tau, schema_size, core_size)`` regime cell,
sweep ``true_c ∈ {0.0, 0.05, 0.10, 0.15, 0.25, 0.50, 1.0}`` and record
``fragmentation_rate`` after ``cluster()``. Report:

* ``frag_at_c10``  — fragmentation reading exactly at the §69 frontier
                      ``c=0.10``. This is the candidate ``fragmentation_max``.
* ``monotone``     — whether fragmentation is weakly non-decreasing in
                      true-c on this cell (sanity).
* ``frag_at_c0``   — baseline fragmentation with no outsiders. Should
                      be ≈ 0 in the realistic regime; if it isn't,
                      the regime is outside the meter's domain.

Recommendation logic
--------------------
* If ``frag_at_c10`` is stable across realistic regimes (low variance),
  use the median as the default ``fragmentation_max``.
* If ``frag_at_c0`` is non-trivial in some regimes (e.g. small vocab),
  document those as out-of-domain — fragmentation alone can't gate
  there; caller needs both meters.

Usage
-----
  python -m evals.schema_fragmentation_calibration \\
      --out bench/results/fragmentation_calibration.json
"""
from __future__ import annotations

import argparse
import random
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

from engram.consolidation.schema_family import cluster as cluster_fn
from engram.consolidation.schema_family_contamination import fragmentation_rate
from evals.io_utils import atomic_write_json


@dataclass
class RegimeCell:
    n_clusters: int
    cluster_size: int
    vocab_size: int
    core_size: int
    schema_size: int
    tau: float
    seed: int = 0xCA11B


def _build_corpus(cell: RegimeCell, p: float) -> dict[str, frozenset[str]]:
    seed = (cell.seed * 1_000_003) ^ int(round(p * 1e9))
    rng = random.Random(seed & 0xFFFFFFFF)
    vocab = [f"t{i}" for i in range(cell.vocab_size)]
    cores: list[list[str]] = []
    if cell.n_clusters * cell.core_size <= cell.vocab_size:
        pool = list(vocab)
        rng.shuffle(pool)
        for i in range(cell.n_clusters):
            cores.append(pool[i * cell.core_size : (i + 1) * cell.core_size])
    else:
        for _ in range(cell.n_clusters):
            cores.append(rng.sample(vocab, cell.core_size))
    features: dict[str, frozenset[str]] = {}
    for ci, core in enumerate(cores):
        for ki in range(cell.cluster_size):
            sid = f"c{ci:04d}_s{ki}"
            if rng.random() < p:
                feats = rng.sample(vocab, cell.schema_size)
            else:
                feats = rng.sample(core, min(cell.schema_size, len(core)))
            features[sid] = frozenset(feats)
    return features


def evaluate_cell(cell: RegimeCell, contaminations: list[float]) -> dict:
    points = []
    for p in contaminations:
        feats = _build_corpus(cell, p)
        clusters = cluster_fn(feats, tau=cell.tau)
        frag = fragmentation_rate(feats, clusters)
        points.append({"true_c": p, "fragmentation": frag})
    frags = [pt["fragmentation"] for pt in points]
    monotone = all(b + 1e-9 >= a for a, b in zip(frags, frags[1:]))
    # Locate the c=0.10 reading (or the closest <=0.10 frontier point).
    frag_at_c10 = next(
        (pt["fragmentation"] for pt in points if abs(pt["true_c"] - 0.10) < 1e-9),
        None,
    )
    frag_at_c0 = next(
        (pt["fragmentation"] for pt in points if pt["true_c"] == 0.0),
        None,
    )
    return {
        "regime": asdict(cell),
        "points": points,
        "frag_at_c0": frag_at_c0,
        "frag_at_c10": frag_at_c10,
        "monotone_in_true_c": monotone,
    }


def run_calibration(
    cells: list[RegimeCell], contaminations: list[float]
) -> dict:
    results = [evaluate_cell(c, contaminations) for c in cells]
    realistic = [r["frag_at_c10"] for r in results if r["frag_at_c0"] is not None and r["frag_at_c0"] <= 0.05]
    summary: dict = {
        "n_cells": len(results),
        "n_realistic_cells": len(realistic),
    }
    if realistic:
        summary["median_frag_at_c10"] = statistics.median(realistic)
        summary["min_frag_at_c10"] = min(realistic)
        summary["max_frag_at_c10"] = max(realistic)
        summary["mean_frag_at_c10"] = statistics.fmean(realistic)
        if len(realistic) > 1:
            summary["stdev_frag_at_c10"] = statistics.stdev(realistic)
    return {
        "contaminations": contaminations,
        "cells": results,
        "summary": summary,
    }


def default_regime_grid() -> list[RegimeCell]:
    """A small grid spanning the realistic operating envelope.

    Crosses tau ∈ {0.4, 0.5, 0.6}, vocab_size ∈ {1000, 2000, 4000},
    schema_size ∈ {4, 6}, core_size fixed at 8. n_clusters scaled to
    keep cores disjoint when feasible.
    """
    cells: list[RegimeCell] = []
    for tau in (0.4, 0.5, 0.6):
        for vocab in (1000, 2000, 4000):
            for sz in (4, 6):
                cells.append(RegimeCell(
                    n_clusters=min(200, vocab // 8),
                    cluster_size=4,
                    vocab_size=vocab,
                    core_size=8,
                    schema_size=sz,
                    tau=tau,
                ))
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--contaminations", type=str,
        default="0.0,0.05,0.1,0.15,0.25,0.5,1.0",
    )
    args = ap.parse_args()
    cs = [float(x) for x in args.contaminations.split(",")]
    cells = default_regime_grid()
    result = run_calibration(cells, cs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, result)
    print(f"wrote {args.out}")
    s = result["summary"]
    print(
        f"\nrealistic cells: {s['n_realistic_cells']}/{s['n_cells']}"
    )
    if "median_frag_at_c10" in s:
        print(
            f"frag_at_c=0.10 across realistic cells: "
            f"min={s['min_frag_at_c10']:.4f} "
            f"median={s['median_frag_at_c10']:.4f} "
            f"max={s['max_frag_at_c10']:.4f} "
            f"mean={s['mean_frag_at_c10']:.4f}"
        )
        if "stdev_frag_at_c10" in s:
            print(f"  stdev={s['stdev_frag_at_c10']:.4f}")
    print("\nper-cell c=0.10 readings:")
    for r in result["cells"]:
        rg = r["regime"]
        print(
            f"  tau={rg['tau']} vocab={rg['vocab_size']:>4} "
            f"sz={rg['schema_size']} core={rg['core_size']}: "
            f"frag(c=0)={r['frag_at_c0']:.4f}  "
            f"frag(c=0.10)={r['frag_at_c10']:.4f}  "
            f"mono={r['monotone_in_true_c']}"
        )


if __name__ == "__main__":
    main()
