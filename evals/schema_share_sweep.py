"""Schema-family prior-sharing sweep — Personize §8 paper figure.

Quantifies what the `share ∈ [0, 1]` knob in
`schema_family_decision.decide_with_family` buys when own-window evidence
is sparse but a cluster of related schemas (siblings) shares latent
quality. Pure simulation, no I/O, deterministic under a seed.

Generative model
----------------
* Each cluster has a latent prior `q ∈ [0, 1]` drawn from Beta(a, b).
  `q` is the per-fact probability that the schema is "real" (i.e.
  any single observed fact in the window is a *support*; with
  probability `1 - q` it is a *contradiction*).
* A cluster has `k` schemas. With `tightness=1.0` all schemas share
  the cluster `q` exactly. With `tightness ∈ (0, 1)` each schema
  draws its own `q_i ~ Beta(α_i, β_i)` calibrated so its mean is
  `q_cluster` and its concentration is `tightness * BIG + (1-tightness)
  * SMALL`; tightness=0.0 reduces to per-schema independent draws
  with no cluster structure. The looser-tightness sweep
  (`--tightness 0.0,0.5,1.0`) is the §8 robustness probe — does
  prior-sharing still pay when siblings are only weakly correlated?
* In each window, every schema independently observes `n_per_window`
  facts. supports = Σ Bernoulli(q), contradictions = n_per_window - supports.

Decision experiment
-------------------
For each cluster:
  1. Initialise every schema as INFERRED (status=INFERRED, version=1).
  2. For each of `T` windows:
       a. Draw evidence for every schema.
       b. For each schema (the "owner"), aggregate its k-1 siblings'
          evidence in the same window.
       c. Call `decide_with_family(state, own, siblings, share=s)`.
       d. Apply the returned EventKind to the owner's state via the
          mechanical reducer in `schema_lifecycle`.
  3. Stop when the owner first leaves INFERRED.

Metrics (per share value, averaged over all owner schemas)
----------------------------------------------------------
  * `time_to_promote_high_q`: median windows-to-PROMOTE among schemas
    with `q >= 0.7`. Lower is better (fast promotion of good schemas).
  * `time_to_deprecate_low_q`: median windows-to-DEPRECATE among schemas
    with `q <= 0.3`. Lower is better (fast culling of bad schemas).
  * `false_promote_rate`: fraction of schemas with `q <= 0.3` that hit
    PROMOTE before DEPRECATE. Lower is better.
  * `false_deprecate_rate`: fraction of schemas with `q >= 0.7` that
    hit DEPRECATE before PROMOTE. Lower is better.
  * `agree_rate_oracle`: fraction of decisions identical to the
    "infinite-evidence" oracle decision (i.e. the decision that would
    be made if `n_per_window = 1000` were drawn from the true `q`
    every window).

The expected paper claim: increasing `share` should reduce
time-to-promote and time-to-deprecate without inflating false rates
(when sibling clusters genuinely share `q`); at `share=1.0` the speedup
saturates because we already get k× the effective sample size.

Usage
-----
  python -m evals.schema_share_sweep --out bench/results/share_sweep.json
"""
from __future__ import annotations

import argparse
import random
from dataclasses import dataclass, field
from pathlib import Path

from engram.consolidation.schema_decision import EvidenceWindow, Thresholds
from engram.consolidation.schema_family_decision import decide_with_family
from engram.consolidation.schema_lifecycle import (
    EventKind,
    SchemaState,
    SchemaStatus,
)
from evals.io_utils import atomic_write_json


# --- Generative model ---


@dataclass
class SimConfig:
    """Generative + experimental parameters."""
    n_clusters: int = 200
    cluster_size: int = 4  # schemas per cluster (k); siblings = k - 1
    n_per_window: int = 4  # facts observed per schema per window
    n_windows: int = 20
    beta_a: float = 1.0  # Beta(a, b) prior over q
    beta_b: float = 1.0
    promote_thresh: int = 3
    deprecate_thresh: int = 2
    recover_thresh: int = 3
    seed: int = 0xE17A11
    tightness: float = 1.0  # 1.0 = all siblings share cluster q exactly;
                            # 0.0 = independent per-schema draws
    contamination: float = 0.0  # P(any given sibling slot is an "outsider"
                                # whose q is drawn fresh from Beta(a,b)
                                # instead of from the cluster). Models
                                # cluster() mis-grouping unrelated schemas
                                # into the sibling set. Owner is never an
                                # outsider — only the k-1 sibling slots are
                                # at risk, evaluated independently.


_BIG_KAPPA = 200.0  # concentration when tightness=1.0 (siblings ≈ cluster q)
_SMALL_KAPPA = 2.0  # concentration when tightness=0.0 (siblings ~ Beta(a,b))


def _draw_sibling_q(
    rng: random.Random, q_cluster: float, tightness: float,
) -> float:
    """Draw an individual schema's q given the cluster q and tightness.

    Mean of the sibling distribution is exactly `q_cluster`; concentration
    `kappa = α + β` is interpolated linearly between SMALL and BIG. At
    `tightness=1.0` returns `q_cluster` deterministically (kappa=∞).
    """
    if tightness >= 1.0:
        return q_cluster
    kappa = tightness * _BIG_KAPPA + (1.0 - tightness) * _SMALL_KAPPA
    # Mode-preserving Beta with mean=q_cluster, α+β=kappa.
    eps = 1e-6
    qc = max(eps, min(1.0 - eps, q_cluster))
    a = qc * kappa
    b = (1.0 - qc) * kappa
    return rng.betavariate(a, b)


def _draw_q(rng: random.Random, cfg: SimConfig) -> float:
    """Sample a cluster latent quality from Beta(a, b)."""
    return rng.betavariate(cfg.beta_a, cfg.beta_b)


def _draw_supports(rng: random.Random, q: float, n: int) -> int:
    """Binomial(n, q)."""
    return sum(1 for _ in range(n) if rng.random() < q)


# --- Apply decision to state (mechanical, mirrors `lifecycle_projection`) ---


def _apply(state: SchemaState, kind: EventKind | None, window_id: str) -> SchemaState:
    """Mechanically advance a schema state given a decision (or None).

    Mirrors the reducer rules in `schema_lifecycle` without the buffer
    plumbing; used here only to drive the per-owner per-window loop.
    """
    if kind is None:
        return SchemaState(
            schema_id=state.schema_id,
            status=state.status,
            version=state.version,
            last_window_id=window_id,
        )
    if kind == EventKind.PROMOTE:
        return SchemaState(
            schema_id=state.schema_id,
            status=SchemaStatus.PROMOTED,
            version=state.version,
            last_window_id=window_id,
        )
    if kind == EventKind.DEPRECATE:
        return SchemaState(
            schema_id=state.schema_id,
            status=SchemaStatus.DEPRECATED,
            version=state.version,
            last_window_id=window_id,
        )
    if kind == EventKind.RECOVER:
        return SchemaState(
            schema_id=state.schema_id,
            status=SchemaStatus.INFERRED,
            version=state.version,
            last_window_id=window_id,
        )
    if kind == EventKind.BUMP_VERSION:
        return SchemaState(
            schema_id=state.schema_id,
            status=state.status,
            version=state.version + 1,
            last_window_id=window_id,
        )
    return state  # CREATE or unknown — noop in the sim loop


# --- Per-share metric collection ---


@dataclass
class ShareMetrics:
    """Aggregated per-share metrics over all owner schemas."""
    share: float
    time_to_promote_high: list[int] = field(default_factory=list)
    time_to_deprecate_low: list[int] = field(default_factory=list)
    false_promote_low: int = 0
    false_deprecate_high: int = 0
    n_high: int = 0
    n_low: int = 0

    def to_dict(self) -> dict:
        def pct(lst: list[int], p: float) -> float | None:
            if not lst:
                return None
            xs = sorted(lst)
            i = max(0, min(len(xs) - 1, int(p * (len(xs) - 1))))
            return float(xs[i])

        return {
            "share": self.share,
            "n_high_q": self.n_high,
            "n_low_q": self.n_low,
            "ttp_high_q_p50": pct(self.time_to_promote_high, 0.50),
            "ttp_high_q_p95": pct(self.time_to_promote_high, 0.95),
            "ttd_low_q_p50": pct(self.time_to_deprecate_low, 0.50),
            "ttd_low_q_p95": pct(self.time_to_deprecate_low, 0.95),
            "promote_rate_high": (
                len(self.time_to_promote_high) / self.n_high
                if self.n_high else None
            ),
            "deprecate_rate_low": (
                len(self.time_to_deprecate_low) / self.n_low
                if self.n_low else None
            ),
            "false_promote_low_q": (
                self.false_promote_low / self.n_low if self.n_low else None
            ),
            "false_deprecate_high_q": (
                self.false_deprecate_high / self.n_high if self.n_high else None
            ),
        }


def run_share(cfg: SimConfig, share: float) -> ShareMetrics:
    """Simulate the lifecycle for every owner schema under one `share`."""
    metrics = ShareMetrics(share=share)
    rng = random.Random(cfg.seed)
    th = Thresholds(
        promote=cfg.promote_thresh,
        deprecate=cfg.deprecate_thresh,
        recover=cfg.recover_thresh,
    )
    for c_idx in range(cfg.n_clusters):
        q_cluster = _draw_q(rng, cfg)
        # Per-schema q (siblings deviate from cluster mean by `tightness`).
        q_per_schema = [
            _draw_sibling_q(rng, q_cluster, cfg.tightness)
            for _ in range(cfg.cluster_size)
        ]
        # Pre-draw evidence for every (schema, window) using each schema's
        # individual q. Siblings still share the same window_id, so cluster-
        # level evidence aggregation in `decide_with_family` still applies.
        evidence_per_window: list[list[EvidenceWindow]] = []
        # Also pre-draw an "outsider" evidence row per window: a synthetic
        # schema whose q is drawn fresh from Beta(a,b) (i.e. NOT correlated
        # with the cluster). Used to simulate cluster() mis-grouping
        # unrelated schemas into the sibling set when contamination > 0.
        # Lazy: skip the draw entirely at contamination=0.0 to preserve
        # byte-identity with the legacy (pre-contamination) RNG stream.
        outsider_evidence_per_window: list[EvidenceWindow] = []
        if cfg.contamination > 0.0:
            q_outsider_per_window = [_draw_q(rng, cfg) for _ in range(cfg.n_windows)]
        else:
            q_outsider_per_window = []
        for w in range(cfg.n_windows):
            wid = f"c{c_idx}-w{w}"
            row = []
            for sch_idx in range(cfg.cluster_size):
                s = _draw_supports(rng, q_per_schema[sch_idx], cfg.n_per_window)
                c = cfg.n_per_window - s
                row.append(EvidenceWindow(
                    window_id=wid, supports=s, contradictions=c,
                ))
            evidence_per_window.append(row)
            if cfg.contamination > 0.0:
                s_out = _draw_supports(rng, q_outsider_per_window[w], cfg.n_per_window)
                c_out = cfg.n_per_window - s_out
                outsider_evidence_per_window.append(EvidenceWindow(
                    window_id=wid, supports=s_out, contradictions=c_out,
                ))

        for owner_idx in range(cfg.cluster_size):
            q_own = q_per_schema[owner_idx]
            is_high = q_own >= 0.7
            is_low = q_own <= 0.3
            if is_high:
                metrics.n_high += 1
            if is_low:
                metrics.n_low += 1
            # Pre-decide which sibling slots are "contaminated" (outsiders
            # mis-clustered into this owner's sibling set). Cluster-level
            # mis-grouping: if a sibling is an outsider, it's an outsider
            # in every window. Owner itself is never an outsider. Lazy:
            # skip the per-slot RNG draws at contamination=0.0 for byte-
            # identity with the legacy stream.
            if cfg.contamination > 0.0:
                sibling_is_outsider = [
                    (i != owner_idx and rng.random() < cfg.contamination)
                    for i in range(cfg.cluster_size)
                ]
            else:
                sibling_is_outsider = [False] * cfg.cluster_size
            state = SchemaState(
                schema_id=f"c{c_idx}-s{owner_idx}",
                status=SchemaStatus.INFERRED,
                version=1,
                last_window_id="init",
            )
            outcome = None  # "promote" | "deprecate" | None
            t_outcome = None
            for w in range(cfg.n_windows):
                row = evidence_per_window[w]
                own = row[owner_idx]
                siblings = [
                    (outsider_evidence_per_window[w]
                     if sibling_is_outsider[i] else r)
                    for i, r in enumerate(row)
                    if i != owner_idx
                ]
                kind = decide_with_family(
                    state, own, siblings, thresholds=th, share=share,
                )
                state = _apply(state, kind, own.window_id)
                if kind == EventKind.PROMOTE and outcome is None:
                    outcome = "promote"
                    t_outcome = w + 1
                    break
                if kind == EventKind.DEPRECATE and outcome is None:
                    outcome = "deprecate"
                    t_outcome = w + 1
                    break

            if is_high:
                if outcome == "promote":
                    metrics.time_to_promote_high.append(t_outcome)  # type: ignore[arg-type]
                elif outcome == "deprecate":
                    metrics.false_deprecate_high += 1
            if is_low:
                if outcome == "deprecate":
                    metrics.time_to_deprecate_low.append(t_outcome)  # type: ignore[arg-type]
                elif outcome == "promote":
                    metrics.false_promote_low += 1
    return metrics


def run_sweep(cfg: SimConfig, shares: list[float]) -> dict:
    """Run the full share sweep and return a JSON-ready dict."""
    cells = [run_share(cfg, s).to_dict() for s in shares]
    return {
        "config": {
            "n_clusters": cfg.n_clusters,
            "cluster_size": cfg.cluster_size,
            "n_per_window": cfg.n_per_window,
            "n_windows": cfg.n_windows,
            "beta_a": cfg.beta_a,
            "beta_b": cfg.beta_b,
            "thresholds": {
                "promote": cfg.promote_thresh,
                "deprecate": cfg.deprecate_thresh,
                "recover": cfg.recover_thresh,
            },
            "seed": cfg.seed,
            "tightness": cfg.tightness,
            "contamination": cfg.contamination,
        },
        "shares": shares,
        "cells": cells,
    }


# --- CLI ---


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-clusters", type=int, default=200)
    p.add_argument("--cluster-size", type=int, default=4)
    p.add_argument("--n-per-window", type=int, default=4)
    p.add_argument("--n-windows", type=int, default=20)
    p.add_argument("--beta-a", type=float, default=1.0)
    p.add_argument("--beta-b", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0xE17A11)
    p.add_argument(
        "--shares", type=str, default="0.0,0.1,0.25,0.5,0.75,1.0",
        help="comma-separated share values in [0, 1]",
    )
    p.add_argument(
        "--tightness", type=float, default=1.0,
        help="sibling correlation in [0, 1]: 1.0=identical q (default), "
             "0.0=independent draws around cluster mean",
    )
    p.add_argument(
        "--contamination", type=float, default=0.0,
        help="P(sibling slot is an outsider with cluster-uncorrelated q) "
             "in [0, 1]. Models cluster() mis-grouping. Default 0.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = _parse_args(argv)
    cfg = SimConfig(
        n_clusters=ns.n_clusters,
        cluster_size=ns.cluster_size,
        n_per_window=ns.n_per_window,
        n_windows=ns.n_windows,
        beta_a=ns.beta_a,
        beta_b=ns.beta_b,
        seed=ns.seed,
        tightness=ns.tightness,
        contamination=ns.contamination,
    )
    shares = [float(x) for x in ns.shares.split(",") if x.strip()]
    for s in shares:
        if not (0.0 <= s <= 1.0):
            raise SystemExit(f"share out of range: {s}")
    if not (0.0 <= ns.tightness <= 1.0):
        raise SystemExit(f"tightness out of range: {ns.tightness}")
    if not (0.0 <= ns.contamination <= 1.0):
        raise SystemExit(f"contamination out of range: {ns.contamination}")
    out = run_sweep(cfg, shares)
    ns.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(ns.out, out)
    print(f"wrote {ns.out}  ({ns.out.stat().st_size} bytes)")
    # Quick stdout summary table.
    print(f"\nshare sweep (n_clusters={cfg.n_clusters}, k={cfg.cluster_size}):")
    print(f"{'share':>6} {'ttp_p50':>8} {'ttd_p50':>8} {'fp_low':>8} {'fd_high':>8}")
    for c in out["cells"]:
        ttp = c["ttp_high_q_p50"]
        ttd = c["ttd_low_q_p50"]
        fp = c["false_promote_low_q"]
        fd = c["false_deprecate_high_q"]
        print(
            f"{c['share']:>6.2f} "
            f"{('-' if ttp is None else f'{ttp:>8.0f}'):>8} "
            f"{('-' if ttd is None else f'{ttd:>8.0f}'):>8} "
            f"{('-' if fp is None else f'{fp:>8.3f}'):>8} "
            f"{('-' if fd is None else f'{fd:>8.3f}'):>8}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
