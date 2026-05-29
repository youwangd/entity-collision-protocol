"""§80 LoCoMo real-corpus per-tau ``fragmentation_max`` calibration.

Direct follow-up to SCALE_REPORT §79 (synthetic per-tau calibration). §79
recommended a bimodal per-tau lookup
``fmax(tau) = 0.0925 if tau<0.25 else 0.1013`` derived on the disjoint-core
synthetic regime. The deferred follow-up §79 carried explicitly:

    "Real-corpus per-tau calibration on LoCoMo still pending."

This driver is that calibration (closes the §79 follow-up).

Method
------
Take the 543 real LoCoMo per-session-per-speaker schema fingerprints
(via ``evals.locomo_fragmentation_replication._extract_schemas``) as the
*natural-c=0 baseline*. Inject outsiders at rate ``p`` ∈ a fine sweep
near the §69 frontier: each schema is, with probability ``p``, replaced
with a fingerprint built from random tokens drawn from the corpus's own
vocabulary (matching the §76/§79 generative recipe so the c-axis remains
comparable). Cluster with ``schema_family.cluster()`` at
tau ∈ {0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50}. Report
``fragmentation_rate(fps, clusters)`` per (tau, c) cell.

For each tau row report:

* ``frag_at_c0``      — fragmentation on the unperturbed real corpus.
* ``frag_at_c10``     — fragmentation at the §69 ``c=0.10`` frontier.
* ``frag_at_c25``     — fragmentation at the §69 ``c=0.25`` cliff.
* ``monotone``        — whether fragmentation is weakly non-decreasing
                        in true-c on this tau (sanity).
* ``recommended``     — proposed real-corpus per-tau ``fragmentation_max``
                        default. ``None`` for non-gateable rows
                        (collapsed/saturated meter).

A tau row is *gateable* iff ``frag_at_c0 ≤ 0.05`` AND the meter has
≥ 0.05 dynamic range across c=0 → c=0.10 (matches §79's gateability
test). Saturation (``frag_at_c0 ≥ 0.9``) marks the singleton cliff:
the gate would always trip at zero outsider injection — useless as a
guardrail.

Why this is *not* a re-run of §79
---------------------------------
§79's synthetic regime:

* per-cluster size-6 disjoint cores carved out of a 2000-token vocab;
* outsider = uniform-vocab random size-6 fingerprint.

LoCoMo is naturally size-skewed (fp_size ∈ [5, 76], median 35) and
its 3716-token vocab is *not* evenly partitioned — most pairs already
have Jaccard ≪ 0.10 from the start (§78 documents this: at tau=0.50
every schema is a singleton with no injection at all). So we expect
the singleton cliff (saturation) to land at lower tau on LoCoMo than
on §79's synthetic. The interesting part is whether *any* tau on
LoCoMo is gateable in the §79 sense; if yes, the per-tau lookup
table updates with real-corpus thresholds.

Headline expectation (§78 informs the prior)
--------------------------------------------
* tau ≥ 0.25: saturated cliff — singletons at c=0 already.
* tau ≤ 0.10: collapse cliff — single mega-cluster at c=0.
* tau ∈ {0.15, 0.20}: candidate gateable interval.

If §80 confirms this, the deployed gate ``fragmentation_max=0.10``
inherits a documented *real-corpus* domain of applicability rather
than the §79 synthetic-only one.

Pure: deterministic given (locomo_path, p, seed, tau). No clocks.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from evals.locomo_fragmentation_replication import _extract_schemas
from engram.consolidation.schema_family import cluster as cluster_fn
from engram.consolidation.schema_family_contamination import (
    contamination_rate,
    fragmentation_rate,
)
from evals.io_utils import atomic_write_json


def _inject_outsiders(
    fps: dict[str, frozenset[str]],
    p: float,
    seed: int,
) -> dict[str, frozenset[str]]:
    """Return a c-perturbed copy of ``fps`` with outsider rate ``p``.

    Each schema is, with probability ``p``, replaced by a fingerprint
    of the same size drawn uniformly without replacement from the
    union vocabulary of ``fps``. ``p=0`` is byte-identical to ``fps``
    (the lazy-RNG path is the §69/§79 contract).
    """
    if p <= 0.0:
        return dict(fps)
    rng = random.Random((seed * 1_000_003) ^ int(round(p * 1e9)))
    vocab = sorted(set().union(*fps.values()))
    out: dict[str, frozenset[str]] = {}
    for sid, fp in fps.items():
        if rng.random() < p:
            size = max(1, len(fp))
            out[sid] = frozenset(rng.sample(vocab, min(size, len(vocab))))
        else:
            out[sid] = fp
    return out


def _frag_at(points: list[dict], target: float) -> float | None:
    return next(
        (pt["fragmentation"] for pt in points if abs(pt["true_c"] - target) < 1e-9),
        None,
    )


def evaluate_tau_row(
    fps: dict[str, frozenset[str]],
    tau: float,
    contaminations: list[float],
    seed: int,
) -> dict:
    points: list[dict] = []
    for p in contaminations:
        feats = _inject_outsiders(fps, p, seed)
        clusters = cluster_fn(feats, tau=tau)
        frag = fragmentation_rate(feats, clusters)
        contam = contamination_rate(feats, clusters, tau)
        points.append(
            {"true_c": p, "fragmentation": frag, "contamination": contam}
        )
    frags = [pt["fragmentation"] for pt in points]
    monotone = all(b >= a - 1e-12 for a, b in zip(frags, frags[1:]))
    frag_c0 = _frag_at(points, 0.0)
    frag_c10 = _frag_at(points, 0.10)
    frag_c25 = _frag_at(points, 0.25)
    saturated = (frag_c0 is not None and frag_c0 >= 0.9)
    collapsed = (
        frag_c0 is not None
        and frag_c10 is not None
        and frag_c0 <= 0.05
        and frag_c10 <= 0.05
    )
    gateable = (
        frag_c0 is not None
        and frag_c10 is not None
        and frag_c0 <= 0.05
        and (frag_c10 - frag_c0) >= 0.05
    )
    recommended = round(frag_c10, 4) if gateable else None
    return {
        "tau": tau,
        "frag_at_c0": frag_c0,
        "frag_at_c10": frag_c10,
        "frag_at_c25": frag_c25,
        "monotone": monotone,
        "saturated_at_c0": saturated,
        "collapsed_at_c0": collapsed,
        "gateable": gateable,
        "recommended_fmax": recommended,
        "points": points,
    }


def run(
    locomo_path: str | Path,
    taus: tuple[float, ...] = (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
    contaminations: tuple[float, ...] = (
        0.00, 0.025, 0.05, 0.075, 0.10, 0.15, 0.20, 0.25,
    ),
    seed: int = 0xC0FFEE,
) -> dict:
    """Per-tau fmax calibration on LoCoMo. Pure given inputs."""
    data = json.loads(Path(locomo_path).read_text())
    fps, _ = _extract_schemas(data)
    rows = [evaluate_tau_row(fps, tau, list(contaminations), seed) for tau in taus]
    gateable = [r for r in rows if r["gateable"]]
    summary = {
        "n_taus": len(taus),
        "n_gateable": len(gateable),
        "median_recommended_fmax": (
            statistics.median(r["recommended_fmax"] for r in gateable)
            if gateable else None
        ),
    }
    return {
        "n_samples": len(data),
        "n_schemas": len(fps),
        "seed": seed,
        "contaminations": list(contaminations),
        "by_tau": rows,
        "summary": summary,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="bench/data/locomo10.json")
    p.add_argument(
        "--out",
        dest="out_path",
        default="bench/results/locomo_fragmentation_per_tau_calibration.json",
    )
    p.add_argument("--taus", default="0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    p.add_argument(
        "--contaminations",
        default="0.0,0.025,0.05,0.075,0.10,0.15,0.20,0.25",
    )
    p.add_argument("--seed", type=int, default=0xC0FFEE)
    args = p.parse_args()
    taus = tuple(float(x) for x in args.taus.split(","))
    cs = tuple(float(x) for x in args.contaminations.split(","))
    res = run(args.in_path, taus=taus, contaminations=cs, seed=args.seed)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, res)
    # Print compact table to stdout
    print(json.dumps(res["summary"], indent=2))
    print()
    print(f"{'tau':>6} {'frag@0':>8} {'frag@10':>8} {'frag@25':>8} "
          f"{'mono':>5} {'sat':>4} {'coll':>5} {'gate':>5} {'fmax':>8}")
    for r in res["by_tau"]:
        rec = "-" if r["recommended_fmax"] is None else f"{r['recommended_fmax']:.4f}"
        print(
            f"{r['tau']:>6.2f} "
            f"{r['frag_at_c0']:>8.4f} "
            f"{r['frag_at_c10']:>8.4f} "
            f"{r['frag_at_c25']:>8.4f} "
            f"{str(r['monotone']):>5} "
            f"{str(r['saturated_at_c0']):>4} "
            f"{str(r['collapsed_at_c0']):>5} "
            f"{str(r['gateable']):>5} "
            f"{rec:>8}"
        )


if __name__ == "__main__":
    main()
