"""§78 LoCoMo low-tau regime characterization for `cluster()` on real corpus.

Context
-------
§77 measured `cluster()` on LoCoMo schemas at tau ∈ {0.3, 0.4, 0.5} only,
landing at fragmentation ≈ 1.0 everywhere — interpreted as "no structural
sibling signal at operationally-relevant tau". §74 had separately concluded
that `contamination_rate` reads identically 0.0 in the realistic regime,
making the §73 contamination gate a no-op.

This driver fills the dynamic-range gap by sweeping
tau ∈ {0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50} on the same 543
LoCoMo schemas. It exposes the *full* regime structure of the metric on
a real corpus.

Headline finding (run 2026-05-21)
---------------------------------
   tau   n_clusters   frag    contam
   0.05    1          0.000   0.879
   0.10   44          0.077   0.987
   0.15  374          0.584   0.832
   0.20  506          0.873   0.140
   0.25  536          0.974   0.000
   0.30  542          0.996   0.000
   0.40  543          1.000   0.000
   0.50  543          1.000   0.000

Two regime transitions, both sharp:

1. **Cluster-collapse cliff at tau ≈ 0.10**: below 0.15, single-link
   single-link Jaccard glues everything via low-Jaccard chains
   (transitive closure dominates); fragmentation drops, contamination
   *spikes* near 1.0. Mega-clusters are inevitable.

2. **Singletons cliff at tau ≈ 0.25**: above this, fragmentation is
   already > 0.97 and contamination has flushed to 0.0 because
   nothing transitively chains anymore.

The §74 conclusion that "contamination_rate reads identically 0 in
realistic regime" is a **high-tau-only artifact**. On real LoCoMo,
contamination is the dominant gate signal at tau ∈ [0.10, 0.20];
fragmentation is the dominant signal at tau ≥ 0.20. They do **not**
both gate the same regime — §74's recommendation to use fragmentation
as the "real meter" is right at default tau=0.5, but at lower tau the
two gates measure complementary regimes.

Operational consequence
-----------------------
- Default `schema_family_tau=0.5` (current) is in the inert regime.
  All schemas are singletons; both gates are no-ops.
- If a future heuristic ever lowers tau (e.g. for property-sparse
  early schemas), the contamination gate becomes load-bearing first
  (around tau≈0.15-0.20), then both gates trip simultaneously below 0.15.
- Calibrating `schema_family_contamination_max` against §69's c=0.10
  threshold is **not** safe across the full tau range: at tau=0.10
  the natural baseline contamination is ~0.99 — the gate would always
  trip even with no outsider injection.

The §73 contamination gate's `cmax=0.10` is therefore tau-conditional:
defensible at tau ≥ 0.25, undefined at tau ≤ 0.20.

Pure: no clocks, no RNG, deterministic given the input json.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from evals.locomo_fragmentation_replication import _extract_schemas
from engram.consolidation.schema_family import cluster
from engram.consolidation.schema_family_contamination import (
    contamination_rate,
    fragmentation_rate,
)
from evals.io_utils import atomic_write_json


def run(
    locomo_path: str | Path,
    taus: tuple[float, ...] = (0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50),
) -> dict:
    """Sweep tau and report fragmentation+contamination for each."""
    data = json.loads(Path(locomo_path).read_text())
    fps, _ = _extract_schemas(data)
    n = len(fps)
    out: dict = {
        "n_samples": len(data),
        "n_schemas": n,
        "by_tau": [],
    }
    for tau in taus:
        clusters = cluster(fps, tau=tau)
        sizes = sorted((len(c) for c in clusters), reverse=True)
        out["by_tau"].append(
            {
                "tau": tau,
                "n_clusters": len(clusters),
                "largest_cluster": sizes[0] if sizes else 0,
                "fragmentation_rate": fragmentation_rate(fps, clusters),
                "contamination_rate": contamination_rate(fps, clusters, tau),
            }
        )
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="bench/data/locomo10.json")
    p.add_argument(
        "--out",
        dest="out_path",
        default="bench/results/locomo_fragmentation_tau_sweep.json",
    )
    p.add_argument("--taus", default="0.05,0.10,0.15,0.20,0.25,0.30,0.40,0.50")
    args = p.parse_args()
    taus = tuple(float(x) for x in args.taus.split(","))
    res = run(args.in_path, taus=taus)
    Path(args.out_path).parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(args.out_path, res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
