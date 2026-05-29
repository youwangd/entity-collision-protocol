# Appendix B. Security audits of Engram retrieval and storage primitives

This appendix collects security-side threat analyses of Engram's retrieval
pipeline and storage primitives. These are systems-side audits of
ACL/access-control side-channels in PRF query expansion, share_prior
reranking, schema-lifecycle caches, BM25/vector candidate pools, mechanical
merge, FactExtraction, write-side cosine dedup, governed-memory primitives,
and cross-channel coupling. They are *not* measurement-side threats to the
retrieval claims of §4 — those are addressed in §6 of the main paper.

We retain them in the appendix because they document the security-audit
methodology that supports the Engram artifact release; readers interested
only in retrieval results can skip this appendix.

Section numbers preserve the original §6.X numbering as §A.6.X for
cross-reference stability.

## A.6.6 PRF query-expansion ACL side-channel (closed)

Pseudo-relevance-feedback (PRF, §A7.3 / §A.4.7) runs a first-pass
retrieval and mines dominant entities from the top-K texts to
construct an expanded query. In the original wire-up the first
pass executed at the `RetrievalEngine` layer — strictly upstream
of the outer per-result ACL filter in `Engram.recall()` — so the
entity-mining pool could include memories the actor lacked READ
scope for. The expanded query, and therefore the actor's final
ranking over its *own* memories, then depended on the cross-agent
corpus. This is a side-channel oracle: an actor with `scope='own'`
could detect the presence of specific tokens in another agent's
private memories by observing rank-perturbations on its own
queries, even though no cross-agent content ever reached the
user-visible output.

The repro is small: Alice owns four `notes …` memories, two with
disambiguating entities (Apollo, Beta). Bob privately owns 20
docs of the form `notes Apollo Apollo item N`. Without the fix,
PRF mines `apollo` from Bob's docs, rewrites Alice's query to
`notes apollo`, and Alice's Apollo doc rises one rank. Flip Bob's
corpus to Beta and Alice's Beta doc rises instead. The ACL
*recall* filter still strips Bob's results from the output — the
leak is purely in Alice's own ranking.

Closure (commit `07d5c35`, branch
`feat/prf-acl-side-channel-fix`). `RetrievalEngine.search()`
accepts an optional `_acl_filter: (Memory) -> bool` that
`_search_with_prf` applies to the first-pass results *before*
the PRF expander is called. `Engram.recall()` binds the filter to
`self._acl_allows_read(actor, mem.agent_id)` for the current
actor. The filter is inert when ACL is disabled, and federated
actors with `scope='*'` (e.g. an auditor / reviewer grant) still
mine the full pool — the escape hatch is preserved by the same
permission model that governs federated reads. Pinned by
`tests/adversarial/test_prf_acl_side_channel.py` (7 cases
covering rank invariance across four Bob corpora, expander pool
isolation, the federated escape hatch, and ACL-disabled
no-regression). Pre-fix: 4 of the 7 cases fail with diagnostic
diffs; post-fix: 7/7 pass. Full suite remains green at 1581
passing.

## A.6.7 PRF IDF-rarity ACL side-channel — closed

The PRF expander has a second cross-agent signal beyond the mining
pool addressed in §A.6.6: the §A.4.15g *IDF-rarity gate* drops candidate
entities whose corpus rarity falls below a threshold (default
`idf_min_rarity = 0.5`). Pre-fix, the rarity score was
`1 − df/N` where both df and N were computed against the **global**
FTS index — including memories the actor cannot READ. The keep/drop
decision for an Alice-pool entity therefore depended on Bob's
private corpus.

The leak is detectable end-to-end in numbers, not just in principle.
Alice owns four `notes …` memories; Apollo appears in one of them
(df=1 in Alice's slice, N=4 ⇒ rarity=0.75, gate keeps Apollo). Bob
privately writes 50 `notes Apollo Apollo …` memories. Pre-fix
rarity collapses to `df/N ≈ 51/54`, giving rarity ≈ 0.06 — below
0.5 — and the gate **drops** Apollo from Alice's PRF expansion,
silently flipping her ranking on her own corpus. Switching Bob's
corpus to Beta-dense flips the gated entity from one to the other:
the post-expansion ranks vary with Bob's vocabulary even though no
cross-agent content reaches Alice's output. This is the classic
shape of a side-channel oracle.

Closure (commit `e0aa422`). `RetrievalEngine._build_prf_rarity_lookup`
gains an `allowed_agents: set[str] | None` parameter that scopes
**both** the df numerator and the N denominator via
`m.agent_id IN (?, …)`. `Engram._prf_rarity_allowed_agents(actor)`
computes the allow-list from the ACL grant for the current actor:
`None` when ACL is disabled, when the actor has `scope='*'`, or when
the actor holds `Permission.FEDERATED` (preserving the federated
escape hatch); `{actor, ''}` for `scope='own'`, mirroring
`Grant.can_access`. `recall()` threads it through the new
`_rarity_allowed_agents` private kwarg on `search()`. Inert in all
single-actor and federated configurations.

Pinned by `tests/adversarial/test_prf_idf_acl_side_channel.py`
(7 properties): four rank-invariance arms under the IDF gate × Bob
signal, a direct rarity-lookup numeric assertion (`rarity(Apollo)
≥ 0.5` post-fix vs. ≈ 0.029 pre-fix in the same corpus), the
federated reader keeping global df, and ACL-disabled no-regression.
Pre-fix: 3 of the 7 cases fail (the direct lookup canaries); the
behavioural rank tests do not perturb in this corpus geometry
because the expander-pool fix from §A.6.6 already strips Bob's docs
from the entity-mining input — but the gate's numeric decision is
still demonstrably wrong, and an adversary with control over Alice's
pool composition could still pivot the gate. Post-fix: 7/7 pass.
Full suite: 1581 → 1588 passing, 3 skipped, 181 deselected, 226 s.


## A.6.8 share_prior reranker ACL side-channel (§D-share-prior-acl)

A third sibling of the §A.6.6 / §A.6.7 leaks: the §96 *share_prior*
reranker (`src/engram/retrieval/rerankers/share_prior.py`) builds an
undirected entity-sharing graph over the post-fusion candidate pool
and adds a bounded `α · deg / max_deg` boost to each candidate's
score. The reranker runs at the `RetrievalEngine` layer, which sits
*upstream* of `Engram.recall()`'s outer ACL filter. Without an
ACL-aware filter on the reranker pool, the entity-sharing graph
spans cross-agent docs, so each Alice doc's `degrees[i]` counts
edges into Bob's private corpus and the `max_deg` normaliser is
global. The visible output is still all Alice's (the outer filter
strips cross-agent rows), but their *ranking* — and even their
absolute scores — are now a function of Bob's private content.

Closure: `RetrievalEngine.search()` now drops cross-agent docs from
`results` *before* slicing the rerank pool, gated on the same
`_acl_filter` already threaded through for §A.6.6. None preserves
federated / single-actor behaviour (no over-correction). Inert when
no reranker is configured.

Pinned by `tests/adversarial/test_share_prior_acl_side_channel.py`
(8 tests / properties): five rank-and-score-invariance arms under
diversifying / dense / orthogonal Bob corpora, a federated
`scope='*'` reviewer who must still see Bob's docs, the ACL-off
no-regression path, and a direct reranker-pool isolation canary
(spy on `apply_reranker`'s pool, assert no `Zorbax` from Bob).
Pre-fix: the canary fails immediately (15 Zorbax docs in the pool);
the rank-invariance arms happen to hold because share_prior's
rank-0 preservation cap absorbs the boost differences in this
corpus geometry — but the leak is unambiguous at the pool layer
and would surface as score perturbation on a slightly larger
`α` or denser sharing graph. Post-fix: 8/8 pass.
Full suite: 1588 → 1596 passing, 3 skipped, 181 deselected, 226 s.

## A.6.9 Schema-lifecycle cache ACL audit — clean

The §A7.4.4 schema-lifecycle gate replays the buffer's
`CONSOLIDATION_SCHEMA_LIFECYCLE` event stream through a mtime-keyed
`CachedLifecycleSnapshot` and uses the resulting `{schema_id:
SchemaState}` map to drop DEPRECATED `MemoryType.SCHEMA` candidates
from a recall's result set. Three siblings of §A.6.6–§A.6.8 (PRF×ACL,
PRF IDF-rarity, share_prior reranker) all turned out to be silent
cross-agent ranking channels; we audited the lifecycle cache for the
same shape of leak.

Audit closure (3 invariants + 1 positive control, pinned by
`tests/adversarial/test_lifecycle_cache_acl_side_channel.py`):

- **ACL-LC-1:** Alice's recall ranking over FACT memories is
  bit-identical (content + score equality) under three Bob-emitted
  lifecycle traffic patterns: empty, single CREATE+DEPRECATE on
  `bob_schema_x`, and a six-event churn (CREATE/PROMOTE/DEPRECATE/
  BUMP_VERSION across three Bob schema ids). The lifecycle filter
  only fires on candidates of type SCHEMA, so cross-agent FACT
  ranking cannot depend on Bob's schema status. Pinned for all three
  arms.
- **ACL-LC-2:** When a SCHEMA candidate's `schema_id` collides with a
  DEPRECATED entry in the snapshot, it is suppressed *regardless of
  which actor emitted the lifecycle event*. This is the **intended**
  behaviour — schemas are `agent_id=''` (system-wide patterns), and
  the lifecycle DAG is global by design — but we pin it explicitly so
  any future "scope lifecycle by emitter" change has to refresh this
  test rather than silently change semantics.
- **ACL-LC-3:** With `respect_schema_lifecycle=False`, lifecycle
  traffic is fully inert — even a CREATE+DEPRECATE pair on a present
  SCHEMA leaves it reachable. Confirms the cached snapshot is not
  silently feeding any other recall signal (no second consumer).
- **Positive control:** With the gate on, an explicit DEPRECATE
  *does* suppress a present SCHEMA. Paired with ACL-LC-3, the two
  show the cache is reachable when on and inert when off, so ACL-LC-3
  is a real null and not a silent skip.

Result: no leak. The lifecycle cache passes the cross-agent audit
without code changes. Closes the last open NEXT-list audit item
("confirm no other recall paths feed cross-agent texts into a
learned signal"). Full suite: 1596 → 1602 passing, 3 skipped, 181
deselected, 226 s.

## A.6.10 Cross-actor schema-id targeted suppression — pinned threat

§A.6.9 named, in passing, that DEPRECATE on a colliding `schema_id` is
"global by design." Pinned here as an explicit, named threat-model
statement instead of a quiet implementation detail.

**Attack.** A malicious actor that learns a victim schema's
`schema_id` can append `CREATE+DEPRECATE` lifecycle events on that
id and silently suppress the schema from every other actor's recall.
The lifecycle DAG is intentionally global (schemas are
`agent_id=''`, system-wide patterns), so the suppression applies to
everyone uniformly. There is no per-emitter ACL on lifecycle events.

**Observability.** Schema ids are exposed through normal recall:
any actor whose query surfaces a schema candidate sees its
`memory.id` in the result row. Schema ids are
`mem-sc-<uuid4().hex[:12]>` — 48 bits of entropy, so blind guessing
costs ~2^47 writes per landed suppression on average, but observed
ids are free.

**Pinned tests** (`tests/adversarial/test_schema_id_targeted_suppression.py`,
+7 tests):

- **SC-ID-1:** A `CREATE+DEPRECATE` pair on a known schema id
  globally suppresses the schema. Worst-case attack surface, pinned.
- **SC-ID-2:** A DEPRECATE on a non-existent / mis-guessed id is a
  full no-op (5 parametric arms over different guess shapes).
  Confirms the entropy floor — guessing alone is infeasible.
- **SC-ID-3:** End-to-end chain. An actor recalls a system-wide
  schema, observes its id from `result.memory.id`, then writes
  `CREATE+DEPRECATE`; subsequent recalls miss the schema. This is
  the realistic attack path.

**Mitigation surface (deferred — not implemented in v0.2):**

- **Scope lifecycle by emitter (per-agent DAGs).** Cleanest fix; but
  it breaks the intended global-schema semantics, so it would need
  a parallel "shared-pattern" type and an explicit promotion
  pathway from per-agent → shared.
- **Quorum gate on DEPRECATE.** Require ≥k independent emitters or an
  audit-trail signature before a deprecation takes effect. Adds
  latency to legitimate deprecations.
- **Redact schema ids in cross-actor recall result rows.** Closes
  observability but breaks debuggability and any actor's ability to
  reference its own schemas by id.

We document this as a known, accepted limitation of the global
lifecycle DAG. The system is single-tenant by intent; multi-tenant
deployments must either trust all actors with respect to schema
deprecation or pick one of the mitigations above. Future work item.


## A.6.11 BM25/vector candidate-pool ACL side-channel (closed)

A fifth sibling of the §A.6.6–§A.6.9 channels — and the most direct one,
because it fires on the *primary* candidate-generation path rather
than on an opt-in retrieval feature. The retrieval engine builds the
hybrid candidate pool by calling `store.search_text(query, limit=k*5)`
for BM25 and `vector.search(query_vec, limit=k*5)` for sqlite-vec.
Neither call was ACL-scoped: both scanned all agents' memories, then
each candidate received a `bm25_rank` (and `vector_rank`) reflecting
its position in the *global* pool. The outer ACL filter at
`engine.recall()` (engine.py:408) stripped cross-agent docs *after*
RRF fusion, so the visible output contained only Alice's docs but
their fused scores — and therefore their relative order — depended on
where Bob's private docs landed in the global pool.

This is a presence oracle that fires even when the reranker, PRF,
and lifecycle gate are all off. In the worst case (Bob's private
corpus saturates the top-`k*5` BM25 hits), Alice's recall returned
the empty set despite her own corpus matching the query: a
denial-of-recall as a function of Bob's content.

**Closure (`engine.py:_search_with_prf`):** drop cross-agent
candidates from the pool *before* RRF fusion, then re-number
`bm25_rank` and `vector_rank` so they are contiguous over the
actor-visible candidates. We reuse the `_acl_filter` callable
already threaded through for §A.6.6–§A.6.8. None preserves federated /
single-actor behaviour exactly. Test pin:
`tests/adversarial/test_vector_channel_acl_side_channel.py` —
3 BM25 parametric arms (orthogonal / Apollo-overlap / Beta-overlap)
verify Alice's order *and* fused scores are bit-identical with and
without Bob's corpus, plus a positive control that confirms the
seed actually does perturb retrieval when ACL is off (so the
invariance is real, not a harness artifact).

**Residual.** FTS5's BM25 score itself is computed over global
corpus statistics (avgdl, df, N). Re-numbering closes the
*rank-position* channel, which is the only BM25-derived signal that
enters fused scoring (RRF uses the rank, not the raw BM25 score).
The score-magnitude channel does not enter recall scoring at all in
v0.2 and so is not exploitable through the public API. We note this
explicitly so future work that begins consuming raw BM25 scores
re-opens the audit.

## A.6.12 MechanicalMerge ACL side-channel (closed)

The Stage 12 `MechanicalMerge` consolidator is the write-side analogue
of the §A.6.11 read-side channel. It iterates `store.all_active()`
(global, no ACL scope), embeds each memory, asks the vector store for
near-duplicates above its cosine threshold (default 0.95 in
production, 0.90 in this test), and unconditionally moves the
lower-salience side of each matching pair into `MemoryState.SUPPRESSED`.

The pre-fix implementation never consulted `agent_id` on the matched
pair, which fused two distinct vulnerabilities into one stage:

  1. **Existence oracle.** When Bob writes content semantically
     near-identical to Alice's, mechanical merge silently moves Bob's
     row into SUPPRESSED. Alice cannot read Bob's row directly under
     ACL — but she can observe a *state transition* on it via
     lifecycle metadata (`memories_suppressed` in the consolidation
     report, `state` column queries on the audit path). This separates
     "Bob never wrote this" from "Bob wrote a near-duplicate of my
     content," which the threat model forbids.

  2. **Cross-tenant denial-of-recall.** In the asymmetric case where
     one tenant runs at higher salience than another, the louder
     tenant's writes systematically suppress the quieter tenant's
     near-duplicates. Multi-tenant deployments cannot tolerate this:
     a single noisy tenant could erase a quiet tenant's memories
     simply by writing similar content at higher salience.

**Closure (`pipeline.py:MechanicalMerge.run`):** over-fetch the
candidate pool from `limit=5` to `limit=20` and skip any pair whose
`agent_id` strings differ. System-owned memories (`agent_id=''`,
e.g. SCHEMA prototypes) remain mergeable globally — that is the
intended behaviour for shared system patterns and matches the §A.6.9
lifecycle-cache audit's treatment of schemas as agent-less.

Test pin: `tests/adversarial/test_mechanical_merge_acl_side_channel.py`
(5 invariants).

  - **MM-ACL-1** Cross-agent near-duplicates are *never* suppressed,
    in either salience-ordering direction (two parametric arms:
    Alice high / Bob low, and the symmetric reversal).
  - **MM-ACL-2** Same-agent near-duplicates *are* still suppressed.
    This is a positive control — the fix is a scope filter, not a
    stage disable.
  - **MM-ACL-3.** System memories with empty
    `agent_id` remain globally
    mergeable.
  - **MM-ACL-4** A system memory does *not* suppress an agent-owned
    near-duplicate — `agent_id=''` is not a wildcard owner.

Pre-fix, three of five tests fail (MM-ACL-1 both arms + MM-ACL-4),
confirming the channel is real and the patch is the minimum scope
filter that closes it without breaking the same-agent merge path.

**Residual.** The merge stage's candidate fan-out is now bounded by
`limit=20` over actor-visible duplicates. In a pathological corpus
where one agent has ≥20 near-duplicates within the threshold, the
21st-onwards may go unmerged in a single pass; the next consolidation
cycle catches them. We accept this — the alternative (full
per-agent scan) is O(n²) in agent-owned content, and 20 is two
orders of magnitude above the empirical near-duplicate density we
see in the §4 corpora.

## A.6.13 FactExtraction ACL inheritance bug (closed)

The most direct ACL leak in the v0.2 surface, and the worst one to
have shipped — though it was caught before any external user was
exposed. The Stage 4c `FactExtraction` consolidator distils EPISODE
memories into FACT rows via an LLM call. The synthesised
`CONSOLIDATION_EXTRACT` event carries no actor context, so the
default `Memory.agent_id` for the resulting fact was `''` (empty —
"system memory" in the v0.2 ACL model).

Pre-fix flow:

  1. Alice (`agent_id='alice'`) writes "I am meeting Mallory at 3pm
     at the Whitebridge."
  2. Consolidation extracts the fact "Alice meets Mallory at the
     Whitebridge" — distilled, structured, often more searchable
     than the source.
  3. The fact is stored with `agent_id=''`. Under
     `Grant.can_access`, *any* actor's grant matches `agent_id=''`
     because system-shared content is intentionally readable to all.
  4. Bob's `recall("Mallory")` surfaces the distilled fact. Alice's
     ACL does not protect her — her own consolidation pipeline
     promoted her content into the system pool.

This is strictly worse than §A.6.11 / §A.6.12: those leak a *signal*
(rank position, suppression state) about Alice's content. This one
leaks the **distilled content itself**, including any facts the LLM
extracted with structured `properties` — which in the dual-extraction
schema (Governed Memory paper) include explicit entity bindings.

**Closure (`pipeline.py:FactExtraction.run`):** one line —
`fact_memory.agent_id = memory.agent_id`. The source episode's owner
is the only correct attribution; the consolidation event has no
actor context to fall back on. Schemas keep `agent_id=''`
intentionally (system-shared patterns, matching the §A.6.9 lifecycle
audit).

Test pin: `tests/adversarial/test_fact_extraction_acl_inheritance.py`
(4 invariants):

  - **FE-ACL-1** Facts from Alice's episodes inherit `agent_id='alice'`.
  - **FE-ACL-2** Facts from Bob's episodes inherit `agent_id='bob'`.
  - **FE-ACL-3** Facts from system episodes stay `agent_id=''`. The
    fix forbids silent *promotion* of agent-owned content into the
    system pool, not legitimate system-to-system extraction.
  - **FE-ACL-4** Mixed batch (Alice + Bob episodes in one stage
    invocation) → per-fact attribution is correct, no
    cross-contamination.

This is the third write-side ACL bug found in the audit pass
(§A.6.12 mechanical merge, §A.6.13 fact inheritance, plus the Stage
9 schema extraction explicitly retains `agent_id=''` by design).
The pattern is consistent: any stage that synthesises a new event
without an actor context defaults the resulting memory to system
scope, and any `Memory.from_event` call site needs an explicit
attribution rule. Future stages must be audited against this
invariant before merge.

## A.6.14 Write-side cosine dedup ACL side-channel (closed)

**Threat (§D-write-dedup-acl):** the Governed Memory write-side cosine
dedup (§A7.4, threshold 0.92) calls `vector_store.search()` over the
*global* vector index. A writer Bob whose content embedding sits within
the dedup threshold of one of Alice's stored memories has his write
silently suppressed: `engine.remember()` returns the deduped event id,
the projection is not inserted, and the audit log records
`remember_deduped status=skipped`.

This produces three observable presence oracles for Bob:

  1. **Audit channel:** `remember_deduped` fires only when a near-cosine
     neighbour exists *somewhere* in the store. Mining the audit log
     reveals which probes intersect Alice's content.
  2. **Recall asymmetry:** Bob's `remember()` returns a non-empty event
     id, but his subsequent `recall()` over his own scope returns 0
     hits. The gap is observable without any access to Alice's data.
  3. **Storage delta:** the JSONL event buffer grows but Bob's
     projection-row count does not. Any monitor watching the
     event-log/projection delta sees the leak.

This is *write-side* and survives every read-side ACL fix landed
through §A.6.13 — the previously-closed channels (PRF mining pool,
IDF-rarity, share_prior reranker, lifecycle cache, BM25/vector
candidate pool, mechanical merge, fact-extraction inheritance) all
operate after the writer's content has either landed or been deduped.

**Closure (`store/memory.py:upsert`, `engine.py:remember`):** an
`acl_filter` callable plumbed into the dedup loop. When ACL is
enabled, the writer's `_acl_allows_read` is consulted on each
candidate neighbour's `agent_id` before the cosine comparison, and
candidates outside the writer's visible scope are skipped. The
candidate pool is widened from K=5 → K=32 when filtering is active so
top-K cross-agent neighbours don't crowd out a same-agent duplicate.
Behaviour with ACL disabled is identical to the prior implementation
(global dedup), preserving the legacy single-actor semantics.

Test pin: `tests/adversarial/test_write_dedup_acl_side_channel.py`
(3 invariants):

  - **WD-1** Bob writes content matching Alice's memory under ACL →
    Bob's projection lands and is recallable in his own scope. The
    failing pre-fix observation is exactly the leak.
  - **WD-2** ACL disabled → cross-actor dedup still fires (regression
    guard for legacy single-tenant deployments).
  - **WD-3** Same agent writes the same content twice → dedup fires
    within the writer's own scope (the legitimate behaviour we
    preserve; only cross-agent dedup is the leak).

**Race extension (write-write contention).** The §A.6.14 fix is verified
under single-threaded conditions; we re-test it under contention because
two `Engram` instances on the same store hold *separate* `_dedup_lock`s,
so ACL filtering must be correct in the absence of cross-instance
serialisation. The order in `Engine.remember()` is `store.upsert()` →
`vector.upsert()`, so a vector becomes visible to other writers only
after its row is committed; this happens to make the race window
*safer* than the sequential case for the leak direction, but two
hazards still need pinning:

  - **R1 (stale-row hazard).** A vector hit may resolve to `cand is None`
    if the row is not yet visible to the second writer's connection.
    The current code path treats this as "skip neighbour, do not
    suppress" — the safe direction. A future refactor that flipped the
    None-policy to "suppress" would silently reopen §A.6.14 whenever a
    row materialised after its vector index entry. We pin the policy
    with a monkey-patched `store.get → None` test that asserts Bob's
    write still lands.

  - **R2 (Alice×Bob identical-payload storm).** 20-thread
    `ThreadPoolExecutor` (10 alice + 10 bob) all writing the same
    payload behind a `Barrier`. Invariant: each actor retains ≥1 row
    (cross-actor cannot suppress) AND same-actor dedup still bounds
    each actor's count to ≤50% of writes (regression bound — naive
    code with no dedup serialisation would let all 10 land per actor).

  - **R3 (presence-oracle under contention).** After 5+5 concurrent
    same-payload writes, Bob's own-scope `recall()` must return his
    payload regardless of Alice's interleaving. Verified to *fail*
    when ACL filter is forcibly disabled — the test is real, not
    vacuous.

Test pin: `tests/adversarial/test_write_dedup_acl_race.py` (3
invariants R1/R2/R3, marker = `concurrency`).

This is the *fourth* write-side ACL bug found in the audit pass
(§A.6.12, §A.6.13, §A.6.14, plus the design-intentional schema sharing). The
pattern continues to be consistent: any stage that consults global
state (vector index, mechanical-merge cluster, fact-extraction
attribution) without an explicit ACL gate becomes a side-channel.
After §A.6.14 the v0.2 audit set is: read-side closed (§§A.6.6–A.6.11),
write-side closed (§§A.6.12–A.6.14), with §A.6.10 (cross-actor schema-id
targeted suppression) pinned as accepted behaviour of the global
lifecycle DAG.

## A.6.15 Governed-Memory `extraction_confidence` ACL audit — clean

**Threat (§D-extraction-confidence-acl):** the Governed Memory paper
\citep{taheri2026gm} introduces a per-fact `extraction_confidence` (EC)
multiplier set at consolidation time and applied at recall time as a
score downweight (`engine.py:680–686`). EC enters fused scoring as
`final *= clamp(ec, 0, 1)` whenever
`RetrievalConfig.use_extraction_confidence=True` (the default).

The audit risk: would EC ever consume cross-agent state, either on the
write side (extractor reading another actor's memories) or the read
side (recall scoring weighting Alice's candidates by Bob's EC
distribution — e.g. a "calibrate against corpus mean" normaliser)?

**Result: clean, no closure required.** Reading the code paths:

  - **Write path** (`consolidation/pipeline.py:362–406`): `FactExtraction.run`
    iterates over `ctx.memories_created` (the current run's episodes),
    invokes the LLM on a single episode's content, and stores the
    parsed `confidence` field clamped to [0, 1] on the resulting fact.
    No cross-agent memory is consulted; agent_id is inherited from the
    source episode (closed in §A.6.13).
  - **Read path** (`retrieval/engine.py:680–686`): `_final_score`
    multiplies in only the *candidate's own* `memory.extraction_confidence`.
    There is no aggregate, no peer lookup, no normalisation against
    other candidates' EC distributions.
  - **Storage path** (`store/memory.py:294, 781`): EC is a per-row
    column read from the candidate's own row; no cross-row JOIN.

**Pin (`tests/adversarial/test_extraction_confidence_acl_side_channel.py`,
7 tests, 4 parametric arms in EC-ACL-1):**

  - **EC-ACL-1** Alice's FACT-only ranking (content + score) is
    bit-identical regardless of Bob's EC distribution. Four arms:
    no Bob facts (control), all-1.0, all-0.0, 5-quantile spread.
    Pins write-side and read-side together: a future change that
    introduces a peer-aware EC normaliser (e.g. corpus-mean
    calibration) has to refresh this test.
  - **EC-ACL-2** EC is a strict per-candidate multiplier: across
    `c ∈ {0.1, 0.25, 0.5, 0.75, 0.9}`, the score of a fixed Alice
    candidate at EC=c equals `c × score(EC=1.0)` to `rel_tol=1e-6`.
    Confirms the EC channel has no nonlinear / cross-candidate
    coupling.
  - **EC-ACL-3.** With
    `use_extraction_confidence=False`, no Alice
    candidate's score depends on any EC value (own or Bob's).
    Confirms the gate has no second consumer.
  - **EC-ACL-4** Positive control: with the gate ON, an Alice
    candidate at EC=0.05 scores < 0.5× the same candidate at EC=0.95.
    Pins EC-ACL-3 as a real null rather than a dead gate.

§A.6.15 closes the last open audit thread on the Governed Memory feature
surface. Combined with §§A.6.6–A.6.14, the v0.2 cross-actor audit set is
fully accounted for: read-side closed (§§A.6.6–A.6.11), write-side closed
(§§A.6.12–A.6.14), §A.6.10 pinned as accepted behaviour, §A.6.15 audited clean.


## A.6.16 Quorum-gated DEPRECATE (mitigation prototype for §A.6.10)

§A.6.10 pinned, as accepted behaviour, the fact that any actor that
learns a `schema_id` can append `CREATE+DEPRECATE` and globally
suppress the schema. We deferred a fix in v0.2 because every available
mitigation traded against either the global-schema semantics or the
debuggability of recall result rows. In this section we land the
*reducer-level* prototype of a quorum gate — opt-in, defaults-off — so
the production wiring can follow without further reducer churn and so
the threat model now has a concrete, defended mitigation rather than
just a gap.

**Mechanism.** `SchemaLifecycleEvent` gains an optional `emitter_id`
field, and `reduce_events` gains a `deprecate_quorum_k: int = 1`
parameter. When `k > 1`, a DEPRECATE event no longer fires the
INFERRED|PROMOTED → DEPRECATED transition on its own; instead the
reducer accumulates `pending_deprecate_emitters: frozenset[str]` on
the schema state and the transition fires only once the set has at
least `k` distinct entries. Any non-DEPRECATE transition (CREATE,
PROMOTE, RECOVER) clears the ballot — promotion or recovery is
positive evidence the schema is alive, so partial dissent collected
beforehand should not be reusable later. The default `k=1` preserves
the legacy single-emitter path byte-for-byte; this is intentional, the
mitigation is *available*, not *mandatory*, because the system is
single-tenant by design.

**Properties pinned**
(`tests/adversarial/test_schema_deprecate_quorum.py`, +14 tests):

- **Q-1, Q-9** Default `k=1` is bit-identical to the pre-quorum
  reducer. Q-9 is the positive control that confirms the §A.6.10 attack
  still works at `k=1` — its job is to fail loudly if we accidentally
  flip the default.
- **Q-2** Under `k=2`, a single attacker's DEPRECATE leaves the
  schema INFERRED with a one-element ballot. The §A.6.10 attack stops
  working at `k=2`.
- **Q-3** Two distinct emitters fire the transition exactly once;
  ballot clears.
- **Q-4** A single emitter re-voting up to k times does *not* satisfy
  quorum — distinctness is on `emitter_id`, not on event count. Sybil
  resistance proper is out of scope (the reducer trusts the emitter
  id), but at minimum repeated self-voting cannot bootstrap quorum.
- **Q-5** PROMOTE clears a partial DEPRECATE ballot. Otherwise an
  attacker who voted long ago could collude with one fresh emitter to
  suppress a now-promoted schema. This is the "stale dissent" bug.
- **Q-6** RECOVER (deprecated → inferred) likewise clears the ballot;
  the next quorum attempt restarts from zero votes.
- **Q-7, Q-8** A DEPRECATE event with `emitter_id=None` under `k>1`
  raises `LifecycleViolation` in strict mode and is dropped silently
  in non-strict mode — i.e. malformed events cannot bypass the gate
  by omitting the emitter.
- **Q-10** Parametric over `k ∈ {2, 3, 5}`: the reducer fires
  *exactly* at the k-th distinct vote, never earlier and never later;
  intermediate ballots have the expected cardinality.
- **Q-11** Invalid `k=0` raises `ValueError` at the API boundary.
- **Q-12** The quorum fold is a pure function: same input ⇒ same
  output across two calls.

**Residual.** The prototype lives at the reducer layer. The
production wiring — which consolidator is allowed to attach an
`emitter_id`, how `emitter_id`s are bound to actor identities, and
which deployment configurations should run with `k > 1` by default —
is left for the integration step. The point of pinning the prototype
now is that the reducer's invariants are non-negotiable from here
forward; downstream code can land without further property churn.

§A.6.16 also gives §A.6.10 a concrete defended path: multi-tenant
deployments that cannot tolerate the single-actor suppression
channel can opt into `deprecate_quorum_k >= 2` and receive a
quorum-gated lifecycle DAG with the full property suite above as the
contract.

### 6.16 phase B: call-site wiring and cache identity

The phase A landing (commit `4ea3589`) threaded `emitter_id` through
the wire format and the reducer; phase B (commit `71eefe9`) closed
the plumbing on both ends:

- **Write side.** `Config.consolidator_id: str = ""` is the per-process
  identity of the consolidator emitting lifecycle events; it is
  threaded through `ConsolidationPipeline.run` into
  `StageContext.consolidator_id` and from there into every
  `make_lifecycle_event(emitter_id=...)` call site under
  `SchemaUpdate`. The empty default preserves byte-stable legacy
  emission (the `emitter_id` key is omitted from `metadata`
  entirely; pinned by test `B-10`). Production deployments set this
  to a stable per-node id (e.g. `socket.gethostname()`).
- **Read side.** `RetrievalConfig.deprecate_quorum_k: int = 1` is
  plumbed through `RetrievalEngine.search` into
  `CachedLifecycleSnapshot.get(deprecate_quorum_k=...)`, which forwards
  it to `reduce_events`. The cache treats `k` as part of the
  snapshot's *identity*: a stream that produced `DEPRECATED` under
  $k=1$ may yield `INFERRED` (with one pending ballot) under $k=2$
  for the *same* event log, so changing $k$ between calls forces a
  full rebuild. This is pinned in both directions by tests `B-6`
  and `B-7`; `B-9` exercises the engine→cache→reducer composition
  end-to-end on a real `RetrievalEngine.search` call.

The two knobs (`Config.consolidator_id` and
`RetrievalConfig.deprecate_quorum_k`) are independent and both
default to bit-identical legacy behaviour, so phase B is
regression-safe for single-tenant single-node deployments while
multi-tenant deployments can adopt either or both as their threat
model demands.

## A.6.17 Cross-channel coupling audit

The per-channel ACL audits (vector pool, BM25/FTS5, mechanical merge,
fact extraction, write-side dedup, extraction confidence,
lifecycle/cache+quorum) each pin one channel's behaviour against an
adversarial peer. They are necessary but not sufficient: a side
channel can hide in a *pair* of channels even when each is
individually clean. We therefore catalogued every off-diagonal pair
$X{\times}Y$ on the channel set and classified each as **I**
(independent by construction — no shared cross-actor state, no shared
score channel), **C+covered** (composed and pinned by an existing
single test path), or **C+gap** (composed and *not* pinned by a single
existing test → write a dedicated combinatorial test). The catalog
and the per-pair reasoning live in
`research_notes/cross_channel_coupling_audit.md`.

The catalog has been audited at two channel counts. At $n=7$
(vector pool, BM25/FTS5, mechanical merge, fact extraction, write-side
dedup, extraction confidence, lifecycle/cache+quorum) the surface is
$\binom{7}{2}=21$ pairs with split **11 I / 8 C+covered / 2 C+gap**.
The 2026-05-24 expansion adds two read-time score channels — H (PRF
entity expansion, dominance-gated) and I (share_prior reranker,
rank-0-capped boost) — following the v0.2 wiring of both as
runtime-toggleable knobs in `RetrievalConfig`. At $n=9$ the surface
grows to $\binom{9}{2}=36$ pairs (+15 new off-diagonals); the new
verdict split is **17 I / 19 C+covered / 0 C+gap**. All six
read-time × write-time corners (C×H, D×H, E×H, C×I, D×I, E×I) are
**I** by the read/write decoupling argument: H and I read only from
the post-write actor-scoped candidate pool and have no path to the
write-time channels' state. The remaining nine new pairs are all
**C+covered** by the diagonal ACL audits (which already exercise
hybrid recall under PRF and share_prior) plus the §A.4.7 PRF×SP×EC
panel which exhaustively pins F×I and H×I across six sweep axes at
$n{=}10$ seeds with paired bootstrap CIs. The two gaps catalogued at
$n=7$ were both write-time mech-merge couplings, since closed:

- **C×E (mech-merge × write-dedup).** Adversarial scenario: Bob-write
  → Alice-merge interleavings could in principle let dedup-derived
  state from a peer leak into Alice's merge candidate set. Closed by
  `tests/adversarial/test_cxe_mech_merge_x_write_dedup_compose.py`
  with three deterministic two-actor invariants (CXE-1..CXE-3): under
  any Bob near-dup history, Alice's intra-actor merge invariant
  holds, Bob's rows survive composition, and the intra-Alice merge
  still fires.
- **C×F (mech-merge × extraction-confidence).** Adversarial scenario:
  could a peer's EC distribution influence which of Alice's two
  same-actor candidates survives a merge, or perturb the surviving
  EC? Closed by
  `tests/adversarial/test_cxf_mech_merge_x_extraction_confidence.py`
  with three invariants (CXF-1..CXF-3) sweeping Bob's EC over the four
  cells $\{$low,high$\}\times\{$low,high$\}$, asserting Alice's
  survivor identity and the bit-identity of its EC, plus a
  salience-only-survivor positive control to catch any future
  EC-weighted refactor.

Both gap tests were green on first run. The catalog also fixes a
forward decision rule for future channels: a new channel $X$ adds $n$
pairs, and $X{\times}Y$ is declared **I** with one-sentence reasoning
iff $X$ shares no per-actor state and no multiplicative/additive score
contribution with $Y$ — otherwise it is **C+gap** until a single test
path exercises both. **Read/write decoupling refinement (n=9 update):**
read-time score channels (F, G, H, I) cannot couple to write-time
channels (C, D, E) because their input is the post-write actor-scoped
candidate slice; this collapses the $X{\times}Y$ surface for every new
read-time channel from $n-1$ pairs to at most the prior count of
read-time channels. The resulting audit-cost ceiling continues to
scale linearly: at $n=9$ the coupled subset is 19/36 (53%, all
covered), and the jump from 8/21 (38%) at $n=7$ comes entirely from
H and I being read-time score channels which couple with every prior
read-time channel by construction.

