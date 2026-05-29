"""§83 — Bootstrap CI on Δ for the LoCoMo per-tau fragmentation curve.

Promotes §82's debiased classifier from a point-estimate (per-tau Δ
above/below ``contam_lift=0.05``) to an interval estimate. For each
tau the statistic of interest is::

    Δ(tau) := frag(c=0.10) − frag(c=0)

§82 found, on the 543-schema LoCoMo corpus,

* tau=0.10 → Δ = 0.105 (GATEABLE-debiased, well above 0.05)
* tau=0.15 → Δ = 0.074 (GATEABLE-debiased, marginal above 0.05)
* tau=0.20 → Δ = 0.020 (NATURALLY_FRAGMENTED, below 0.05 noise band)
* tau ≥ 0.25 → SINGLETON_CLIFF (separate question, not bootstrapped here)

Whether each Δ is CI-positive against zero, and whether the two
GATEABLE Δs are CI-above ``contam_lift``, decides whether §82's
2/7 GATEABLE ruling is data or noise. If tau=0.20 turns out to be
CI-positive too, ``contam_lift`` should drop; if tau=0.15 is
borderline, GATEABLE-debiased tightens to tau=0.10 only.

Method
------
Schema-level **m-out-of-n subsample bootstrap, without replacement**.
Naive with-replacement resampling glues exact-duplicate fps under
single-link clustering at any tau (Jaccard=1), which artificially
depresses frag(c=0) and inflates Δ — a real methodological hazard
flagged on the first run. Subsampling without replacement preserves
the structural \"no exact duplicates\" property of the source corpus.

For ``B`` draws of size ``m = floor(0.8 · n_schemas)`` (default 80%
subsample):

* recompute fragmentation at c=0 (the subsample itself);
* recompute fragmentation at c=0.10 (apply the §80 ``_inject_outsiders``
  recipe to the subsample, with a per-bootstrap-deterministic seed
  tied to the bootstrap index so no shared RNG state leaks).

Returns the bootstrap distribution of Δ(tau) and reports
percentile-style CI summary stats. Pure given
(locomo_path, taus, n_boot, seed, m_frac); no clocks.

Compute budget
--------------
``cluster_fn`` on 543 fps at tau=0.15 ≈ 0.4 s; the bootstrap path
runs cluster *twice* per draw (c=0 and c=0.10) — so B=200 × 3 taus
≈ 8 min. The driver supports ``--n-boot`` and ``--taus`` to keep
runs inside cron budget; the default keeps the borderline trio.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

from evals.locomo_fragmentation_per_tau_calibration import _inject_outsiders
from evals.locomo_fragmentation_replication import _extract_schemas
from engram.consolidation.schema_family import cluster as cluster_fn
from engram.consolidation.schema_family_contamination import fragmentation_rate
from evals.io_utils import atomic_write_json


def _resample(
    fps: dict[str, frozenset[str]],
    rng: random.Random,
    m_frac: float = 0.8,
) -> dict[str, frozenset[str]]:
    """m-out-of-n subsample, without replacement.

    With-replacement bootstrap was rejected: it produces exact-duplicate
    fps that single-link cluster at Jaccard=1 regardless of tau,
    artificially depressing frag(c=0) and biasing Δ upward (observed:
    tau=0.20 Δ point estimate 0.020 vs with-replacement bootstrap mean
    0.099). Subsampling without replacement preserves the corpus's
    \"no exact duplicates\" property.
    """
    if not (0.0 < m_frac <= 1.0):
        raise ValueError(f"m_frac must be in (0, 1]; got {m_frac}")
    keys = list(fps.keys())
    n = len(keys)
    m = max(1, int(round(m_frac * n)))
    sids = rng.sample(keys, m)
    return {sid: fps[sid] for sid in sids}


def _delta_for_resample(
    sample: dict[str, frozenset[str]],
    tau: float,
    seed: int,
) -> tuple[float, float, float]:
    """(f0, f10, Δ) on a single bootstrap resample at one tau."""
    cls0 = cluster_fn(sample, tau=tau)
    f0 = fragmentation_rate(sample, cls0)
    perturbed = _inject_outsiders(sample, 0.10, seed)
    cls10 = cluster_fn(perturbed, tau=tau)
    f10 = fragmentation_rate(perturbed, cls10)
    return f0, f10, f10 - f0


def _percentile(xs: list[float], q: float) -> float:
    """Linear-interp percentile (q in [0, 1]). Pure, no numpy dep."""
    if not xs:
        return float("nan")
    s = sorted(xs)
    n = len(s)
    if n == 1:
        return s[0]
    pos = q * (n - 1)
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return s[int(pos)]
    frac = pos - lo
    return s[lo] * (1 - frac) + s[hi] * frac


def evaluate_tau(
    fps: dict[str, frozenset[str]],
    tau: float,
    n_boot: int,
    seed: int,
    m_frac: float = 0.8,
) -> dict:
    """Per-tau Δ bootstrap. Returns CI stats + the empirical
    distribution sample (truncated for storage)."""
    rng = random.Random((seed * 1_000_003) ^ int(round(tau * 1e9)))
    deltas: list[float] = []
    f0s: list[float] = []
    f10s: list[float] = []
    for b in range(n_boot):
        sample = _resample(fps, rng, m_frac=m_frac)
        f0, f10, d = _delta_for_resample(sample, tau, seed=(seed ^ (b * 2654435761)))
        f0s.append(f0)
        f10s.append(f10)
        deltas.append(d)
    mean = statistics.fmean(deltas)
    sd = statistics.pstdev(deltas) if len(deltas) > 1 else 0.0
    p025 = _percentile(deltas, 0.025)
    p975 = _percentile(deltas, 0.975)
    p_below_zero = sum(1 for d in deltas if d <= 0) / len(deltas)
    p_below_lift = sum(1 for d in deltas if d < 0.05) / len(deltas)
    # CI-positive against zero ⇔ 2.5 percentile > 0.
    ci_positive = p025 > 0.0
    # CI above 0.05 contam_lift ⇔ 2.5 percentile > 0.05.
    ci_above_lift = p025 > 0.05
    return {
        "tau": tau,
        "n_boot": n_boot,
        "m_frac": m_frac,
        "mean": mean,
        "sd": sd,
        "ci95_lo": p025,
        "ci95_hi": p975,
        "p_below_zero": p_below_zero,
        "p_below_lift": p_below_lift,
        "ci_positive": ci_positive,
        "ci_above_lift": ci_above_lift,
        # Keep first 50 draws for downstream plotting / regression sanity.
        "deltas_head": deltas[:50],
        # Paired (f0, f10) head — feeds maximum-margin fmax driver (§84).
        "f0_head": f0s[:50],
        "f10_head": f10s[:50],
        # Full paired bootstrap (kept under separate key for §84 driver
        # which needs all B draws, not just the head). Cheap to store:
        # 2 × n_boot floats per tau.
        "f0_all": f0s,
        "f10_all": f10s,
    }


def run(
    locomo_path: str | Path,
    taus: tuple[float, ...] = (0.10, 0.15, 0.20),
    n_boot: int = 200,
    seed: int = 0xB07ADA,
    m_frac: float = 0.8,
) -> dict:
    """Pure given (locomo_path, taus, n_boot, seed, m_frac)."""
    data = json.loads(Path(locomo_path).read_text())
    fps, _ = _extract_schemas(data)
    rows = [evaluate_tau(fps, tau, n_boot, seed, m_frac=m_frac) for tau in taus]
    return {
        "n_samples": len(data),
        "n_schemas": len(fps),
        "n_boot": n_boot,
        "seed": seed,
        "m_frac": m_frac,
        "by_tau": rows,
        "summary": {
            "ci_positive_taus": [r["tau"] for r in rows if r["ci_positive"]],
            "ci_above_lift_taus": [r["tau"] for r in rows if r["ci_above_lift"]],
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="bench/data/locomo10.json")
    p.add_argument(
        "--out",
        dest="out_path",
        default="bench/results/locomo_fragmentation_per_tau_bootstrap.json",
    )
    p.add_argument("--taus", default="0.10,0.15,0.20")
    p.add_argument("--n-boot", type=int, default=200)
    p.add_argument("--m-frac", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=0xB07ADA)
    args = p.parse_args()
    taus = tuple(float(x) for x in args.taus.split(","))
    res = run(args.in_path, taus=taus, n_boot=args.n_boot, seed=args.seed, m_frac=args.m_frac)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, res)
    print(json.dumps(res["summary"], indent=2))
    print()
    print(
        f"{'tau':>6} {'mean':>8} {'sd':>8} {'ci95_lo':>9} {'ci95_hi':>9} "
        f"{'p≤0':>6} {'p<.05':>6} {'>0?':>5} {'>.05?':>6}"
    )
    for r in res["by_tau"]:
        print(
            f"{r['tau']:>6.2f} "
            f"{r['mean']:>8.4f} "
            f"{r['sd']:>8.4f} "
            f"{r['ci95_lo']:>9.4f} "
            f"{r['ci95_hi']:>9.4f} "
            f"{r['p_below_zero']:>6.3f} "
            f"{r['p_below_lift']:>6.3f} "
            f"{str(r['ci_positive']):>5} "
            f"{str(r['ci_above_lift']):>6}"
        )


if __name__ == "__main__":
    main()
