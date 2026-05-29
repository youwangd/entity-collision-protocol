"""§87 — End-to-end gate replication on LoCoMo at the §86 calibrated fmax.

Closes NEXT pickup #1: with §83 + §84 + §86 in hand, the GATEABLE-debiased
operating point at ``tau=0.10`` has a defensible runtime ``fmax=0.1486``
(max-margin) shipped in ``engram.consolidation.calibration``. This driver
quantifies what the gate *actually does* on the real LoCoMo schema set
at that operating point, so the default-flip
``schema_family_share=0.0 → 0.75`` paired with
``schema_family_fragmentation_max=0.1486`` can be decided on data.

What we measure
---------------
For each tau in the calibrated grid (0.10, 0.15, 0.20):
  * fragmentation_rate on the real corpus (replication of §76/§80).
  * the §86 calibrated max-margin fmax (via ``lookup_max_margin_fmax``).
  * gate verdict: ``frag <= fmax`` → share active; else share collapses.
  * cluster topology *if* the gate passes:
    - n_clusters, n_singletons, max_cluster_size, mean_non_singleton_size.
    - n_owners_with_siblings: how many pattern owners would receive any
      sibling-prior credit (this is the *operationally useful* number).
    - share_credit_rows: total (owner, sibling_window) pairs the
      ``decide_with_family`` call would see.

Decision rule for default-flip
------------------------------
A flip from ``schema_family_share=0.0 → 0.75`` paired with the §86
calibrated fmax is **defensible** iff at least one calibrated tau on
LoCoMo:
  (a) passes the gate (frag <= fmax), AND
  (b) yields ≥ 1 non-singleton cluster (otherwise share is a no-op
      because ``all_owner_siblings`` returns ``()`` for singletons).

Without (b), even a passing gate gives zero operational lift — the
flip is cosmetic. With (b), there is at least one window where
``decide_with_family`` is doing different work than bare ``decide``.

Pure: deterministic given the input json. No clocks, no RNG, no
network. Closes the loop from §76 → §80 → §82 → §83 → §84 → §86 to a
single defensible runtime number per tau.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from engram.consolidation.calibration import lookup_max_margin_fmax
from engram.consolidation.schema_family import cluster
from engram.consolidation.schema_family_contamination import (
    contamination_rate,
    fragmentation_rate,
)
from evals.locomo_fragmentation_replication import _extract_schemas
from evals.io_utils import atomic_write_json


def _cluster_topology(clusters):
    sizes = [len(c) for c in clusters]
    if not sizes:
        return {
            "n_clusters": 0,
            "n_singletons": 0,
            "n_non_singleton_clusters": 0,
            "max_cluster_size": 0,
            "mean_non_singleton_size": 0.0,
            "n_owners_with_siblings": 0,
            "share_credit_rows": 0,
        }
    non_sing = [s for s in sizes if s > 1]
    n_owners_with_sibs = sum(non_sing)  # every non-singleton member has ≥1 sibling
    # share_credit_rows = sum_{c: |c|>1} |c| * (|c|-1)
    rows = sum(s * (s - 1) for s in non_sing)
    return {
        "n_clusters": len(clusters),
        "n_singletons": sizes.count(1),
        "n_non_singleton_clusters": len(non_sing),
        "max_cluster_size": max(sizes),
        "mean_non_singleton_size": (
            round(sum(non_sing) / len(non_sing), 4) if non_sing else 0.0
        ),
        "n_owners_with_siblings": n_owners_with_sibs,
        "share_credit_rows": rows,
    }


def run(
    locomo_path: str | Path,
    taus: tuple[float, ...] = (0.10, 0.15, 0.20),
    table: str = "LOCOMO",
) -> dict:
    """Execute the gate replication. Pure, deterministic."""
    data = json.loads(Path(locomo_path).read_text())
    fps, _ = _extract_schemas(data)
    n = len(fps)

    by_tau = []
    any_passes_with_lift = False
    for tau in taus:
        fmax = lookup_max_margin_fmax(tau, table=table)
        clusters = cluster(fps, tau=tau)
        frag = fragmentation_rate(fps, clusters)
        contam = contamination_rate(fps, clusters, tau)
        passes = (fmax is not None) and (frag <= float(fmax))
        topo = _cluster_topology(clusters)
        share_active_and_useful = passes and topo["n_non_singleton_clusters"] > 0
        if share_active_and_useful:
            any_passes_with_lift = True
        by_tau.append(
            {
                "tau": tau,
                "calibrated_fmax_max_margin": fmax,
                "fragmentation_rate": frag,
                "contamination_rate": contam,
                "gate_passes": passes,
                "share_active_and_useful": share_active_and_useful,
                **topo,
            }
        )

    return {
        "n_samples": len(data),
        "n_schemas": n,
        "calibration_table": table,
        "by_tau": by_tau,
        "summary": {
            "any_tau_share_active_and_useful": any_passes_with_lift,
            "default_flip_defensible": any_passes_with_lift,
        },
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="in_path", default="bench/data/locomo10.json")
    p.add_argument(
        "--out",
        dest="out_path",
        default="bench/results/locomo_gate_replication.json",
    )
    p.add_argument("--taus", default="0.10,0.15,0.20")
    p.add_argument("--table", default="LOCOMO")
    args = p.parse_args()
    taus = tuple(float(x) for x in args.taus.split(","))
    res = run(args.in_path, taus=taus, table=args.table)
    out = Path(args.out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(out, res)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
