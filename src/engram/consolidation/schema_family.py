"""Schema-family clustering (Personize.ai 'Governed Memory' §8).

Buckets sibling schemas so a promote/deprecate/recover decision on one can
inform priors on its cluster-mates. The paper argues for this without
specifying the distance metric; we ship two defensible choices and let the
caller pick:

  * `cluster()` — Jaccard on **property-name sets**. Cheap, structural;
    suitable when schemas have rich typed-property scaffolding.
  * `cluster_by_cooccurrence()` — Jaccard on **evidence-window-id sets**.
    Two schemas that fire in the same windows are presumed to share a
    generative theme, regardless of their declared properties. Suitable
    when property scaffolding is sparse (e.g. early in a schema's life)
    but evidence ledgers are populated.

Both share the same single-link agglomerative core (`_cluster_by_jaccard`).

**Distance.** Jaccard similarity on the chosen feature set:

    sim(A, B) = |A.feat ∩ B.feat| / |A.feat ∪ B.feat|

with `sim = 0` when either side has an empty feature set (an empty set
carries no shared structure; a singleton is the only defensible cluster
for it).

**Clustering.** Single-link agglomerative via union-find: two schemas
are in the same cluster iff there exists a chain of pairwise
similarities each ≥ `tau`. This is intentionally permissive — `decide()`
is the gatekeeper, not the cluster.

**Purity.** Pure functions. No clocks, no RNG. Output is canonicalized:
clusters are frozensets, the returned tuple is sorted by min-element of
each cluster, so same-input ⇒ same-output regardless of dict iteration
order.

Design invariants (locked in `tests/property/test_schema_family.py`,
hold for **both** `cluster()` and `cluster_by_cooccurrence()` since they
share the same core):

  F1. Partition: every input schema appears in exactly one output cluster.
  F2. tau=1.0 ⇒ clusters group only schemas with **identical** non-empty
      feature sets (and singletons for everything else).
  F3. Empty-feature schemas are always singletons (defensible: nothing
      to share). This is the only place the metric is conventionally
      extended; not a property of Jaccard.
  F4. Symmetry / determinism: permuting the input dict's insertion order
      yields the same output tuple (canonical sort).
  F5. Adding an unrelated schema (Jaccard < tau against every existing
      schema, including transitively) leaves prior clusters intact.
  F6. Reflexivity: a schema is always in its own cluster (trivially true
      under union-find init).

Cross-metric invariant (`cluster_by_cooccurrence` only):

  C1. If every schema fires in exactly one distinct window (no shared
      windows), all schemas are singletons regardless of `tau < 1.0`.

What we *don't* do here (deferred):

  * Hierarchical / dendrogram output. We only need a flat partition for
    the §8 prior-sharing use case.
  * Online / incremental clustering. Re-run on the full schema table;
    cheap (n^2 with tiny n in practice).
  * Weighted features (idf-style downweighting of common windows /
    properties). v0.3 deferred — see TODO-RESEARCH §D reading queue.
    (The §B schema-lifecycle thread that originally tracked this is
    closed as of 2026-05-24; weighting is a separate retrieval-side
    follow-up.)
"""
from __future__ import annotations

from typing import Mapping, Tuple


def jaccard(a: frozenset[str] | set[str], b: frozenset[str] | set[str]) -> float:
    """Jaccard similarity with the empty-set convention sim(∅, *) = 0.0.

    Pure. No side effects. The empty-set convention is deliberate: an
    empty feature set carries no structural information to share with
    a cluster. Standard math defines 0/0 as undefined; we pick 0 because
    F3 (empty schemas are singletons) is the behavior we want from
    `cluster()` / `cluster_by_cooccurrence()`.
    """
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    union = len(a | b)
    return inter / union


def _cluster_by_jaccard(
    feature_sets: Mapping[str, frozenset[str] | set[str]],
    tau: float,
) -> Tuple[frozenset[str], ...]:
    """Single-link agglomerative cluster on Jaccard ≥ `tau`.

    Private core shared by `cluster()` (property-name features) and
    `cluster_by_cooccurrence()` (evidence-window-id features). Pure.
    """
    if not (0.0 <= tau <= 1.0):
        raise ValueError(f"tau must be in [0.0, 1.0], got {tau}")

    # Sort ids up front so the union-find walk is deterministic.
    ids = sorted(feature_sets.keys())
    parent = {sid: sid for sid in ids}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: str, y: str) -> None:
        rx, ry = find(x), find(y)
        if rx == ry:
            return
        # Always point larger id at smaller id for canonical roots.
        if rx < ry:
            parent[ry] = rx
        else:
            parent[rx] = ry

    # O(n^2) pairwise; n is tiny in practice (#schemas << #facts).
    for i, a in enumerate(ids):
        fa = frozenset(feature_sets[a])
        if not fa:
            continue  # F3: empty stays singleton
        for b in ids[i + 1:]:
            fb = frozenset(feature_sets[b])
            if not fb:
                continue
            if jaccard(fa, fb) >= tau:
                union(a, b)

    buckets: dict[str, set[str]] = {}
    for sid in ids:
        root = find(sid)
        buckets.setdefault(root, set()).add(sid)

    clusters = [frozenset(b) for b in buckets.values()]
    clusters.sort(key=lambda c: min(c))
    return tuple(clusters)


def cluster(
    schemas: Mapping[str, frozenset[str] | set[str]],
    tau: float = 0.5,
) -> Tuple[frozenset[str], ...]:
    """Single-link agglomerative cluster of schemas by Jaccard on **property names**.

    Args:
      schemas: schema_id → set of property names.
      tau: similarity threshold in [0.0, 1.0]. tau=1.0 only collapses
        schemas with identical property sets. tau=0.0 collapses any
        pair with at least one shared property (transitively).

    Returns:
      Tuple of frozensets, one per cluster. Output is canonicalized:
      each cluster is a frozenset of schema_ids; the tuple is sorted by
      `min(cluster)` so the result is deterministic across input dict
      orderings.

    Raises:
      ValueError: if `tau` is outside [0.0, 1.0].
    """
    return _cluster_by_jaccard(schemas, tau)


def cluster_by_cooccurrence(
    window_membership: Mapping[str, frozenset[str] | set[str]],
    tau: float = 0.5,
) -> Tuple[frozenset[str], ...]:
    """Single-link agglomerative cluster of schemas by Jaccard on **shared evidence windows**.

    Two schemas that have fired (via supports or contradictions) in
    largely the same evidence windows are presumed to share a
    generative theme, even when their declared property sets diverge.
    This is the §8 \"co-occurrence-in-evidence-window\" distance the
    paper argues for but doesn't formalize.

    Args:
      window_membership: schema_id → set of window_ids in which the
        schema has accumulated **any** evidence (supports or
        contradictions). The caller is responsible for materializing
        this from the lifecycle ledger / projection. Empty set means
        the schema has no recorded evidence yet → singleton (F3).
      tau: similarity threshold in [0.0, 1.0]. Same semantics as
        `cluster()`.

    Returns:
      Tuple of frozensets, canonicalized identically to `cluster()`.

    Raises:
      ValueError: if `tau` is outside [0.0, 1.0].

    Notes:
      - For idempotency under window-id renaming: this metric is
        invariant to any bijective relabeling of window ids.
      - For sparsity: schemas that have only ever fired in unique
        windows are forced singletons (C1).
      - Confounder hazard: a noisy \"this window fired everything\"
        bug in the upstream pipeline would over-cluster. We do not
        idf-downweight here; that's a deliberate v0 choice. Caller
        should pre-filter / cap window membership if abuse is feared.
    """
    return _cluster_by_jaccard(window_membership, tau)


__all__ = ["jaccard", "cluster", "cluster_by_cooccurrence"]
