# A7. Extended Methods

This appendix expands §3's main-body protocol description with the
supporting method detail that the ACL 2-column 6-page Industry
Track body cannot accommodate. Each subsection is referenced by a
one-sentence pointer at the end of §3.3; the full content lives
here.

## A7.1 LoCoMo and adaptive-vw experiment

For the adaptive-vw null result, we use the LoCoMo adapter
(`evals/locomo_adapter.py`) over n=1978 questions and compute (a)
the per-query oracle hit@1 (best vw per query), (b) static-best
vw, and (c) tau-thresholded policies on `bm25_top1 - bm25_top2`,
normalized gap, and crowdedness@0.95.

## A7.2 share_prior reranker (Personize §96 adaptation)

We adapt the *shared prior* signal from Personize (§96) into a
candidate-pool reranker. Rather than scoring each candidate against
the query alone, we build an undirected entity-sharing graph over
the top-`pool_size` fused candidates: an edge connects two
candidates whenever their extracted named-entity sets intersect.
Each candidate's *multi-mate degree* — the number of pool members
it shares at least one entity with — is a within-pool popularity
signal that, on multi-hop bridge queries, fires for the bridge fact
even when the bridge fact does not lexically match the query
string. We add a bounded boost `α · deg / max_deg` to each
candidate's fused score and re-sort.

**Rank-0 preservation invariant.** We cap the per-candidate boost
so that no non-rank-0 candidate's post-boost score can equal or
exceed the original rank-0 score:
$\text{score}'_i = \text{score}_i + \min(\alpha \cdot \text{deg}_i / \text{max\_deg},\, \text{score}_0 - \text{score}_i - \varepsilon)$.

By construction $\text{score}'_i < \text{score}_0$ for all $i \geq 1$, so `hit@1`
is mathematically — and, as we verify in §4, empirically — never
regressed. The reranker is therefore safe to ship default-off
behind a flag and accept-only at higher k. Verified across
`5 seeds × 5 α values × 2 recipes × 3 pool sizes = 150 arms`:
`Δhit@1 ≡ 0.000` everywhere (`SHARE_PRIOR_REPORT.md`, all tables).

**Protocol.** We evaluate on two synthetic recipes that probe
orthogonal failure modes: **Unique-entity** (do-no-harm; each gold
has a distinct anchor entity, no graph structure; share_prior
expected inert at hit@1 and at most mildly noisy at hit@k); and
**Bridge** (target signal; each query is a pair of facts joined
by a shared anchor entity; gold pair facts share a degree-2 hub
in the graph and should be promoted).

For each recipe we sweep `α ∈ [0.02, 0.30]`, `pool_size ∈
{20, 40, 80}`, and `entity_weight ∈ {0.0, 0.1, 0.2, 0.3}` (the
channel that also lives in the fused scorer), across seeds
`{42..46}` and corpus scales from 60 to 120 query pairs with up
to 200 distractor facts. We report paired `Δhit@k` and
`Δpair_recall@k` versus the same fused-scorer baseline (no rerank)
on identical seeds, with n=10 paired-bootstrap 95% CIs (§A.4.7).

**Pool-size dilution.** As `pool_size` grows, the `deg / max_deg`
distribution becomes less peaked; the relative ordering between
the bridge fact and the rest of the pool flattens. We confirm this
empirically (`pool_size ∈ {20, 40, 80}` at `α = 0.10`, `ew = 0.10`
on the corrected small bridge corpus:
`Δpair@10 = {+0.050, −0.025, +0.025}`; §A.4.7).

**Adaptive α (opt-in regularizer).** A constant α boosts identically
whether the candidate pool's most-shared entity has degree 1 or
degree 9, even though the rank-0 cap already saturates the "graph
is a near-matching" case. We schedule a tapered multiplier on
`max_deg`,
$\alpha_{\text{eff}}(\text{max\_deg}) = \alpha \cdot 1 / (1 + \max(0, \text{max\_deg} - 1) / 4)$,

i.e. $\alpha_{\text{eff}} = \alpha$ at `max_deg ∈ {0, 1}` and decays monotonically
toward 0 as the pool densifies (`max_deg = 5 → 0.5α`, `max_deg = 9
→ α/3`). This lives behind
`RetrievalConfig.share_prior_adaptive_alpha` (default `False`)
and ships as a hedge against α over-shoot rather than an
unconditional improvement; §A.4.7 reports the empirical regime
where it pays off.

## A7.3 PRF entity expansion (dominance-gated)

Reranking cannot promote what the first-pass top-K never admitted,
so we close the upstream gap with pseudo-relevance feedback (PRF)
restricted to entity tokens. Given a first-pass top-K, we extract
entities from each of the top-`top_k_for_prf` results and rank
them by document frequency within the pool. We then expand the
query with the most-dominant entity *only if* it appears in at
least `min_dominance · top_k_for_prf` of the first-pass results;
otherwise we issue the original query unchanged. The dominance
gate is the do-no-harm guarantee: when no single entity dominates
the first-pass pool (the typical unique-entity workload), we do
not expand at all, so unique-recipe `hit@1` is not regressed by
classic PRF over-expansion.

The expansion is wired into `RetrievalEngine.search` behind
`RetrievalConfig.query_expansion_min_dominance: float | None`
(default `None` = off). At runtime, when a non-`None` value is
configured, the engine performs a first-pass retrieval, computes
the dominance-gated entity, and (if the gate fires) re-issues a
single expanded query whose results replace the first-pass results
for the user-visible top-k. The operating point we recommend,
defended in §A.4.7, is `min_dominance = 0.3` with
`top_k_for_prf = 20`.

## A7.4 Governed Memory integration (secondary systems note)

> **Scope of this section.** §A7.4 documents the four write-path
> primitives (dual extraction, cosine dedup, mechanical merge,
> schema lifecycle) that make our paired-bootstrap CIs replayable.
> Reviewers focused exclusively on the entity-collision protocol
> may skip this subsection: nothing in §4 depends on schema-
> lifecycle behaviour beyond the testbed-determinism requirement
> we surface here. The lifecycle's *own* retrieval behaviour (and
> its non-contribution to recall) is the subject of §A.4.6's
> bisection and §A4.2's discussion — also orthogonal to the
> headline two-axis claim.

Engram's write/consolidation path adopts four primitives from the
Personize Governed Memory proposal \citep{taheri2026gm}. This
subsystem is a v0.2 systems contribution; the paper's headline
claims do not depend on it.

### A7.4.1 Dual extraction with per-fact confidence

`FactExtraction` returns each fact with `extraction_confidence ∈
[0, 1]`; `RetrievalConfig.use_extraction_confidence` (default
`True`) multiplies the fused score by this factor and records it
in `ScoredMemory.sources`.

### A7.4.2 Write-side cosine dedup at 0.92

`StorageConfig.write_dedup_threshold = 0.92` gates `store.upsert`
against the in-process vector index. This is a pure write-path
filter — not a merge — orthogonal to extraction confidence. The
independence is formalized as two invariants (I1/I2): the
deduplication outcome is invariant under keeper-side confidence,
and a deduped write does not mutate the keeper's confidence. Both
invariants are verified by randomized property tests over the
joint state space; case counts and traceability in §A6.

### A7.4.3 Mechanical merge (no-LLM)

`MechanicalMerge` (Stage 12) walks `store.all_active()` and, for
any pair above `merge_threshold = 0.95`, suppresses the lower-
salience member. No LLM call; order-stable; idempotent under fixed
embeddings.

### A7.4.4 Schema lifecycle as a pure reducer

`schema_lifecycle.py` folds an immutable `SchemaLifecycleEvent`
log into a `{schema_id: SchemaState}` snapshot. The transition DAG
is `INFERRED → PROMOTED → DEPRECATED`, `INFERRED → DEPRECATED`,
`DEPRECATED → INFERRED` (recovery only, fresh `window_id`). Five
invariants are enforced: transition closure, CREATE-once, no-op on
unknown id, version monotone under status changes, and recovery
freshness. `RetrievalConfig.respect_schema_lifecycle` (default
`True`) filters DEPRECATED candidates before scoring. Each
invariant's Hypothesis property test and the bug-class it catches
are consolidated in §A.4.17.

### A7.4.5 What we did not adopt

We omit Personize's LLM-mediated merge (replaced by §A7.4.3's
mechanical merge) and their governance-review stage (no analogue
in single-user deployments).

## A7.5 Discriminator tags — full schema

| tag | example query | example answer set | vocabulary |
|---|---|---|---|
| `preference` | "what does Alice prefer for X?" | dark mode, light mode, … | open phrasal |
| `project`    | "what is Alice working on?"     | varied open phrases       | open phrasal |
| `technical`  | "what does Alice use for X?"    | varied open phrases       | open phrasal |
| `service`    | "what service does Alice use?"  | aws, gcp, azure, …        | closed lexical |
| `tool`       | "what tool does Alice use?"     | git, docker, postgres, …  | closed lexical |

Tag selection follows the construction rule (closed enum vs free
phrasal slot, §3.3, §75 Limitations Author-as-annotator). The
vocabulary column is what determines the lexical/intent split that
the two-axis result of §4.3 isolates.
