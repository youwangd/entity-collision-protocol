"""Per-tau calibration of `fragmentation_max` against the §69 c=0.10 frontier.

Direct follow-up to SCALE_REPORT §78 (LoCoMo low-tau regime sweep). §78
established two regime cliffs in the cluster() output as tau swept down
from 0.5 to 0.05:

* **collapse cliff** at tau≲0.10 — single-link glues everything together,
  fragmentation→0 even with substantial outsider injection;
* **singleton cliff** at tau≳0.25 — outsiders fragment off as singletons,
  fragmentation→1 with even small c.

§76 calibrated `fragmentation_max=0.10` at the §69 frontier `c=0.10` but
*only* on tau ∈ {0.4, 0.5, 0.6}. §78's implication is unambiguous: that
single threshold is *tau-conditional*. Any future adaptive-tau heuristic
that varies tau across windows must re-calibrate fmax per-tau or the gate
becomes load-bearing for the wrong reason.

This driver provides the per-tau calibration table.

For each tau in the sweep, sweep true-c at fine resolution near the §69
frontier (c=0.10) and record `fragmentation_rate(features, cluster())`.
For each tau row report:

* ``frag_at_c10``      — candidate per-tau ``fragmentation_max`` default.
* ``frag_at_c0``       — baseline (should be ~0 in the realistic regime;
                         large baseline ⇒ tau outside meter's domain).
* ``frag_at_c25``      — fragmentation at the §69 ``c=0.25`` "no setting
                         is safe" cliff. Useful upper-frontier reading.
* ``monotone``         — whether fragmentation is weakly non-decreasing
                         in true-c on this tau (sanity).
* ``saturated_at_c0``  — boolean: ``frag_at_c0 ≥ 0.9`` (singleton cliff).
* ``collapsed_at_c0``  — boolean: ``frag_at_c0 ≤ 0.05`` AND
                         ``frag_at_c10 ≤ 0.05`` (collapse cliff:
                         meter is dead).

Recommendation logic
--------------------
A tau row is *gateable* iff ``frag_at_c0 < 0.05`` AND
``frag_at_c10 - frag_at_c0 ≥ 0.05`` (the meter has dynamic range across
the safety frontier). On gateable rows, ``frag_at_c10`` is the
recommended per-tau ``fragmentation_max``. Non-gateable rows must be
documented as out-of-domain — the §74 gate can't carry weight there and
the operator either picks a different tau or relies on an alternative
gate.

Usage
-----
  python -m evals.schema_fragmentation_per_tau_calibration \
      --out bench/results/fragmentation_per_tau_calibration.json
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
class Cell:
    n_clusters: int
    cluster_size: int
    vocab_size: int
    core_size: int
    schema_size: int
    tau: float
    seed: int = 0xCA11B


def _build_corpus(cell: Cell, p: float) -> dict[str, frozenset[str]]:
    """Disjoint-core synthetic corpus with outsider-injection rate p.

    Byte-identical generative model to ``schema_fragmentation_calibration``
    (and §69) so this driver's c=0 / c=0.10 readings are directly
    comparable to those tables.
    """
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


def _frag_at(points: list[dict], target: float) -> float | None:
    return next(
        (pt["fragmentation"] for pt in points if abs(pt["true_c"] - target) < 1e-9),
        None,
    )


def evaluate_tau_row(cell: Cell, contaminations: list[float]) -> dict:
    points: list[dict] = []
    for p in contaminations:
        feats = _build_corpus(cell, p)
        clusters = cluster_fn(feats, tau=cell.tau)
        frag = fragmentation_rate(feats, clusters)
        points.append({"true_c": p, "fragmentation": frag})
    frags = [pt["fragmentation"] for pt in points]
    monotone = all(b + 1e-9 >= a for a, b in zip(frags, frags[1:]))
    frag_at_c0 = _frag_at(points, 0.0)
    frag_at_c10 = _frag_at(points, 0.10)
    frag_at_c25 = _frag_at(points, 0.25)
    saturated_at_c0 = frag_at_c0 is not None and frag_at_c0 >= 0.9
    collapsed_at_c0 = (
        frag_at_c0 is not None
        and frag_at_c10 is not None
        and frag_at_c0 <= 0.05
        and frag_at_c10 <= 0.05
    )
    gateable = (
        frag_at_c0 is not None
        and frag_at_c10 is not None
        and frag_at_c0 < 0.05
        and (frag_at_c10 - frag_at_c0) >= 0.05
    )
    return {
        "regime": asdict(cell),
        "points": points,
        "frag_at_c0": frag_at_c0,
        "frag_at_c10": frag_at_c10,
        "frag_at_c25": frag_at_c25,
        "monotone_in_true_c": monotone,
        "saturated_at_c0": saturated_at_c0,
        "collapsed_at_c0": collapsed_at_c0,
        "gateable": gateable,
        "recommended_fmax": frag_at_c10 if gateable else None,
    }


def run_per_tau(taus: list[float], contaminations: list[float]) -> dict:
    cells = [
        Cell(
            n_clusters=200,
            cluster_size=4,
            vocab_size=2000,
            core_size=8,
            schema_size=6,
            tau=t,
        )
        for t in taus
    ]
    rows = [evaluate_tau_row(c, contaminations) for c in cells]
    gateable = [r for r in rows if r["gateable"]]
    summary: dict = {
        "n_taus": len(rows),
        "n_gateable": len(gateable),
        "n_collapsed": sum(1 for r in rows if r["collapsed_at_c0"]),
        "n_saturated": sum(1 for r in rows if r["saturated_at_c0"]),
        "gateable_taus": [r["regime"]["tau"] for r in gateable],
    }
    if gateable:
        rec = [r["recommended_fmax"] for r in gateable]
        summary["min_recommended_fmax"] = min(rec)
        summary["max_recommended_fmax"] = max(rec)
        summary["median_recommended_fmax"] = statistics.median(rec)
        summary["mean_recommended_fmax"] = statistics.fmean(rec)
        if len(rec) > 1:
            summary["stdev_recommended_fmax"] = statistics.stdev(rec)
    return {
        "taus": taus,
        "contaminations": contaminations,
        "rows": rows,
        "summary": summary,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument(
        "--taus",
        type=str,
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50",
    )
    ap.add_argument(
        "--contaminations",
        type=str,
        default="0.0,0.05,0.10,0.15,0.25,0.50,1.0",
    )
    args = ap.parse_args()
    taus = [float(x) for x in args.taus.split(",")]
    cs = [float(x) for x in args.contaminations.split(",")]
    result = run_per_tau(taus, cs)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out, result)
    print(f"wrote {args.out}")
    s = result["summary"]
    print(
        f"\ntaus: gateable={s['n_gateable']}/{s['n_taus']}  "
        f"collapsed={s['n_collapsed']}  saturated={s['n_saturated']}"
    )
    if s["n_gateable"]:
        print(
            f"recommended_fmax across gateable taus: "
            f"min={s['min_recommended_fmax']:.4f} "
            f"median={s['median_recommended_fmax']:.4f} "
            f"max={s['max_recommended_fmax']:.4f}"
        )
    print("\nper-tau readings:")
    for r in result["rows"]:
        rg = r["regime"]
        flags = []
        if r["collapsed_at_c0"]:
            flags.append("COLLAPSED")
        if r["saturated_at_c0"]:
            flags.append("SATURATED")
        if r["gateable"]:
            flags.append("gateable")
        flag_str = (" [" + ",".join(flags) + "]") if flags else ""
        rec = r["recommended_fmax"]
        rec_str = f"{rec:.4f}" if rec is not None else "  --  "
        print(
            f"  tau={rg['tau']:.2f}  "
            f"frag(c=0)={r['frag_at_c0']:.4f}  "
            f"frag(c=0.10)={r['frag_at_c10']:.4f}  "
            f"frag(c=0.25)={r['frag_at_c25']:.4f}  "
            f"rec_fmax={rec_str}  "
            f"mono={r['monotone_in_true_c']}{flag_str}"
        )


if __name__ == "__main__":
    main()
