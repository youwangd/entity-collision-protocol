"""Runtime cluster-contamination diagnostic for schema-family prior-sharing.

Closes the §8 deployment-rule gap left open by SCALE_REPORT §69-71. Those
synthetic sweeps established that `schema_family_share=0.75` is safe iff
cluster contamination ≤10%, but offered no way to *measure* contamination
on a real (unlabeled) cluster output. This module ships that meter.

Background
----------
`schema_family.cluster*()` runs single-link agglomerative clustering on
Jaccard-of-feature-sets at threshold `tau`. Single-link is permissive:
two schemas can land in the same cluster connected only by a transitive
chain of pairwise-similar intermediates, not direct similarity. The §69
synthetic contamination knob simulates exactly this: an "outsider"
schema whose features were drawn from an uncorrelated distribution can
still be union-found into the cluster if it happens to share enough
features with one neighbor.

A contaminated cluster therefore exhibits a characteristic signature:
its **direct pairwise-Jaccard floor** (min over all within-cluster
pairs) collapses below `tau`, even though the cluster is connected at
single-link level. A clean (high-tightness) cluster has every pair
above `tau` directly.

Estimators
----------
We expose three intentionally simple, fully observable scalars (all
pure functions of `(features, clusters, tau)`):

1. ``cluster_contamination(features, cluster, tau)`` — per cluster:
   fraction of within-cluster pairs with direct Jaccard < `tau`.
   Singletons return 0.0 (no pairs ⇒ no contamination signal). A
   tight (uncontaminated) cluster scores 0.0; a fully-transitive
   chain scores ≈ 1 - 1/comb(n,2) at the limit.

2. ``contamination_rate(features, clusters, tau)`` — corpus-level:
   weighted mean of per-cluster contamination, weights = pair counts.
   Singletons contribute zero pairs → don't move the score. Range
   [0.0, 1.0]. This is the scalar to compare against the §69 rule
   ("safe iff ≤ 10%").

3. ``min_within_jaccard(features, cluster)`` — diagnostic floor for
   debugging a single suspicious cluster. Returns 1.0 for singletons
   by convention (vacuously tight).

Design invariants (locked in
``tests/property/test_schema_family_contamination.py``):

  K1. Singletons contribute zero contamination weight.
  K2. ``contamination_rate`` is in [0.0, 1.0] always.
  K3. Identical-feature clusters score 0.0 contamination at any tau≤1.0.
  K4. Output is invariant to dict iteration order (determinism).
  K5. Partition consistency: if `clusters` doesn't partition the keys
      of `features`, raise ValueError. We refuse to silently impute.
  K6. Compositional: a contamination-free cluster passed alongside a
      contaminated one yields a rate strictly less than the
      contaminated cluster's own rate (the clean pairs dilute it).
  K7. ``cluster_contamination`` on the empty cluster raises
      ValueError (we never expect that from `cluster*()`, and silently
      returning 0 would mask upstream bugs).

What we explicitly don't do
---------------------------
- We don't try to *fix* contamination (drop outliers, retighten tau).
  This is a meter, not an actuator. Caller decides what to do with
  the number — typical usage is to gate `schema_family_share`:

      rate = contamination_rate(features, clusters, tau)
      effective_share = cfg.schema_family_share if rate <= 0.10 else 0.0

  This is the operational form of the §69 deployment rule.

- We don't infer the canonical "outsider" schema. Single-link offers
  no unique decomposition into core+outsider; the user can call
  ``cluster_contamination`` per-cluster and inspect.

- We don't IDF-weight features. Same `tau` and feature semantics as
  `schema_family.cluster*()`; this module is a passive observer of
  whatever the caller already clustered.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Mapping, Sequence

from .schema_family import jaccard


def _validate_partition(
    features: Mapping[str, frozenset[str] | set[str]],
    clusters: Sequence[frozenset[str]],
) -> None:
    """K5: refuse anything but a true partition of features.keys()."""
    seen: set[str] = set()
    for c in clusters:
        if not c:
            raise ValueError("empty cluster in partition")
        for sid in c:
            if sid in seen:
                raise ValueError(f"schema {sid!r} appears in >1 cluster")
            seen.add(sid)
    feat_keys = set(features.keys())
    if seen != feat_keys:
        missing = feat_keys - seen
        extra = seen - feat_keys
        raise ValueError(
            f"clusters do not partition features: "
            f"missing={sorted(missing)!r} extra={sorted(extra)!r}"
        )


def min_within_jaccard(
    features: Mapping[str, frozenset[str] | set[str]],
    cluster: Iterable[str],
) -> float:
    """Min pairwise Jaccard inside `cluster` (1.0 for singletons).

    Diagnostic helper: collapse below `tau` ⇒ the cluster is held
    together only by transitive single-link chains, not direct
    similarity ⇒ contamination suspect.
    """
    members = sorted(cluster)
    if not members:
        raise ValueError("min_within_jaccard called on empty cluster")
    if len(members) == 1:
        return 1.0  # vacuously tight; convention matches K3 spirit
    floor = 1.0
    for a, b in combinations(members, 2):
        s = jaccard(frozenset(features[a]), frozenset(features[b]))
        if s < floor:
            floor = s
    return floor


def cluster_contamination(
    features: Mapping[str, frozenset[str] | set[str]],
    cluster: Iterable[str],
    tau: float,
) -> float:
    """Fraction of within-cluster pairs with direct Jaccard < `tau`.

    Singletons return 0.0 (no pairs to assess). A perfectly tight
    cluster — every pair directly above tau — also returns 0.0. A
    fully transitive chain (only adjacent pairs above tau) approaches
    1 - (n-1)/comb(n,2).
    """
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0.0, 1.0], got {tau}")
    members = sorted(cluster)
    if not members:
        raise ValueError("cluster_contamination called on empty cluster")
    if len(members) == 1:
        return 0.0  # K1
    pairs = 0
    below = 0
    for a, b in combinations(members, 2):
        pairs += 1
        if jaccard(frozenset(features[a]), frozenset(features[b])) < tau:
            below += 1
    return below / pairs


def contamination_rate(
    features: Mapping[str, frozenset[str] | set[str]],
    clusters: Sequence[frozenset[str]],
    tau: float,
) -> float:
    """Corpus-level weighted contamination ∈ [0.0, 1.0].

    Weighted mean of per-cluster contamination, weights = pair counts
    `comb(|c|, 2)`. Singletons contribute zero weight (K1). If every
    cluster is a singleton, returns 0.0 (no pairs ⇒ no signal ⇒
    nothing to be contaminated).

    This is the scalar to compare against the §69 deployment rule:

        rate = contamination_rate(features, clusters, tau)
        if rate <= 0.10: enable share=0.75 safely.
    """
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0.0, 1.0], got {tau}")
    _validate_partition(features, clusters)

    total_pairs = 0
    total_below = 0
    for c in clusters:
        members = sorted(c)
        n = len(members)
        if n < 2:
            continue
        for a, b in combinations(members, 2):
            total_pairs += 1
            if jaccard(frozenset(features[a]), frozenset(features[b])) < tau:
                total_below += 1
    if total_pairs == 0:
        return 0.0
    return total_below / total_pairs


def fragmentation_rate(
    features: Mapping[str, frozenset[str] | set[str]],
    clusters: Sequence[frozenset[str]],
) -> float:
    """Fraction of `features` that landed as singletons in `clusters`.

    Companion to ``contamination_rate``. SCALE_REPORT §74 calibrated
    the contamination meter against the §69 outsider-injection rate `c`
    and found the meter reads ≈0.0 in the realistic regime — outsiders
    are *expelled* as singletons by ``cluster()`` rather than glued
    in, which the pairwise-Jaccard meter cannot see (singletons
    contribute zero pair weight by K1).
    
    Fragmentation tracks `c` almost 1:1 in that regime and is the
    monotone deployment-rule signal we actually want. Combined with
    the contamination meter, a tripped fragmentation gate captures
    the "the cluster() output is too noisy to trust prior-sharing"
    case that contamination misses by construction.

    Range [0.0, 1.0]. Empty `features` → 0.0 by convention (vacuously
    not fragmented). K5-style partition consistency enforced.

    Invariants (locked in
    ``tests/property/test_schema_family_contamination.py``):

      F1. All-singleton clusters → 1.0.
      F2. One big cluster (no singletons) → 0.0.
      F3. Range [0.0, 1.0] always.
      F4. Partition mismatch raises ValueError (matches K5).
      F5. Output independent of cluster ordering.
    """
    _validate_partition(features, clusters)
    n_schemas = len(features)
    if n_schemas == 0:
        return 0.0
    singletons = sum(1 for c in clusters if len(c) == 1)
    return singletons / n_schemas


__all__ = [
    "min_within_jaccard",
    "cluster_contamination",
    "contamination_rate",
    "fragmentation_rate",
]
