# A4. Extended Discussion

This appendix expands §5.1's main-body operational rule with the
four supporting analyses that the ACL 2-column 6-page Industry
Track body cannot accommodate. Each is referenced by a one-sentence
pointer in §5.1; the full content lives here.

## A4.1 Why does adaptive vector-weight routing fail?

The 11.7pp oracle gap on LoCoMo is real — there exists a per-query
optimal `vw`, and switching at query time would close the gap. But
the gap/crowdedness signals we tested do **not** localize that
optimal. Two hypotheses:

1. **Signal coarseness.** `bm25_top1 - bm25_top2` only sees the
   top-2 distance; it misses the broader candidate distribution.
2. **Confounding with hardness.** A small gap may indicate "BM25 is
   uncertain" (good signal) **or** "all candidates are semantically
   near the gold" (bad signal — vector won't help either). The
   signals collapse both regimes.

We additionally trained a `GradientBoostingClassifier` over the full
BM25 feature panel + category one-hot under leave-one-conversation-
out CV across all 10 LoCoMo samples (§4.4,
`evals/locomo_learned_router.py`). The learned router is also null
(Δhit@1 = −0.0005 [−0.0030, +0.0015] on HT; ST identical within
Monte-Carlo noise). The router degenerates to vw=0 because 1801/1978
queries already prefer vw=0 in the oracle. The 11.7 pp headroom is
real but unrecoverable from any pre-routing signal we can observe;
gold position is required, which by definition cannot exist at
decision time. We accordingly *re-frame* vector fusion as a
**paraphrase-robustness** mechanism rather than a per-query precision
lever, and validate that framing on LongMemEval (§A.4.8.2).

**Verdict.** Adaptive vw routing is a measured null on LoCoMo. The
correct operational role for vector fusion is paraphrase-robustness,
not per-query precision routing.

## A4.2 Schema lifecycle as a research artifact

> **Why this section exists.** A reviewer may reasonably ask why a
> paper about retrieval-evaluation methodology spends discussion
> on schema-lifecycle invariants. The answer is the bridge to §1's
> "deterministic governance is a prerequisite, not a contribution":
> the cross-encoder paired-bootstrap inversions in §A.4.16.3
> (n=100 → n=500) only make sense if re-running the same ingest
> against the same dataset produces the same memory store byte-for-
> byte. The three properties below are what give us that guarantee.
> They are *machinery for the methodology*, not a retrieval claim
> — §A.4.6 falsified the retrieval interpretation directly, and
> §A4.3 below restates the consolidation claim accordingly.

§A.4.6's bisection landed the operational claim that the lifecycle is
not a retrieval mechanism. What it *is* a mechanism for is
**deterministic governance under replay**. We formalize three
invariants on the lifecycle reducer, each verified by exhaustive
execution traces over the property-based testing substrate; the
specific test file paths and per-property case counts are catalogued
in the implementation traceability index (§A6).

1. **Lifecycle decisions are events, not in-place mutations.** A
   schema's state at time *t* is the fold of its decision log up to
   *t*. This is the same discipline event-sourced ledgers borrow
   from accounting; in a memory system it gives bit-identical audit
   replay across re-runs of the same ingest stream.
2. **Family clustering is decision-stable under permutation.** For
   any permutation of the input fact stream, the *family assignment*
   a fact lands in is invariant; only the order of decisions within
   a family is permuted. This is what makes the lifecycle safe to
   run concurrently: writers don't race for cluster identity.
3. **Decay is monotone in real time, not in arrival time.** A fact's
   confidence trajectory depends only on wall-clock spacing, not on
   whether other facts were observed between ticks; the property is
   verified under fuzzed interleaving of `tick()` and `update()`
   calls. This rules out a class of write-amplification bugs where a
   chatty witness inadvertently extends an unrelated fact's
   half-life.

These properties generalize beyond Engram: any memory system that
wants deterministic replay — audit trails, regression-debug
reproduction, cross-machine rehydration — needs all three. The
lifecycle's value is not retrieval lift but the substrate it
provides for every other claim in the paper: §A.4.7's PRF×SP
operating point is only defensible because the ingest state is
replayable from the decision log.

### A4.2.1 Lifecycled schemas vs Letta-style memory blocks

Letta's `human`/`persona` blocks \citep{packer2023memgpt} and Engram's
SCHEMA memories are adjacent in design space — both are named,
mutable agent state — but differ on three axes that matter for
governed deployment.

1. **Mutation discipline.** A Letta block is a string the LLM
   rewrites in place via tool calls; prior content is reachable only
   through external chat history. An Engram schema is a fold over an
   append-only decision log (§A7.4.4): every state change is a typed
   event with a reason field, and previous state is recoverable by
   replaying any prefix. Letta optimises for prompt-window
   compactness; Engram for audit-grade replay.
2. **Identity stability under adversary.** A Letta block is
   identified by name; whoever can issue a tool call can rewrite it.
   Engram schemas are content-addressed (cluster centroid + family
   key, §A7.4.4) with a quorum-gated DEPRECATE primitive (§A.4.6,
   §A.6.16) requiring *k* independent emitters over a *w*-event
   window. A single compromised emitter cannot take a schema down.
3. **Recovery semantics.** Letta has no first-class undo: a
   corrupted block must be reconstructed from chat history by the
   same LLM that may be the corruption source. Engram's lifecycle
   DAG includes a RECOVER edge — verified under randomized event
   interleaving (see §A6) — that re-promotes a DEPRECATED schema
   once subsequent evidence reaches the same quorum. RECOVER is
   path-dependent on the decision log, not on current LLM
   judgment.

The trade is real: Letta is cheaper to operate (no append-only log,
no quorum bookkeeping) and for chat-assistant workloads the savings
dominate. Engram's bet is that the substrate cost is recovered the
first time a regulator, debugger, or post-incident reviewer asks
"what did this agent know and when" — a question Letta's named-block
design cannot answer without external instrumentation.

## A4.3 The honest version of "consolidation lifts retrieval"

§A.4.6 forces a re-statement of the consolidation claim. A naive
reading of the §87 pipeline says "more stages, more lift." The
bisection says otherwise: on the Mem0-shaped LoCoMo10 fixture, **one
stage (episode extraction) carries the full retrieval delta, and
seven of the eleven downstream stages emit identically-zero per-pair
diffs against any prefix that already includes extraction.** Two of
the four non-trivially-moving downstream stages (`schema_update`,
`appraisal`) are *negative* on Δhit@1 at point-estimate, and neither
survives a paired bootstrap.

This is stronger than "the rest of the pipeline doesn't help": it
falsifies stage-additive lift attribution. Stage ablations that
report a single end-to-end metric and treat a positive delta as
evidence-of-mechanism are under-specified — the lift was already
present before the ablated stage ran. Our v0.2 minimum:
**leave-one-out necessity** + **paired bootstrap on the per-question
diff**, with multiple-comparison cost paid explicitly.

The architectural implication: the lifecycle stages are **not
retrieval mechanisms**. They are governance — dedup at write,
schema as a writable cluster, appraisal as a salience signal for
downstream policy. They earn their place on §A4.2's grounds, not §4's.

**Verdict.** Lifecycle consolidation does not improve retrieval; it
provides governance. Future ablations of "consolidation" features
must report leave-one-out + paired-bootstrap with multiple-comparison
cost paid explicitly.

## A4.4 The PRF latency myth

A controlled cProfile re-measurement (§A.4.15-profile, n=30 k, 200
paired queries) falsifies the staged claim that PRF "doubles recall
p50." PRF-only p50 is *0.86 ms below* baseline (40.08 vs 40.94) and
PRF-only p95 is *4.7 ms below*; share_prior-only matches baseline
within noise. Only the combined PRF×share_prior arm shows real
overhead (+14 % p50, +24 % p95). The dominant cost across all arms
is `sqlite3.Connection.execute` (~73 % of cumulative recall time);
PRF doubles `engine.search` call count but adds only ~6 % to its
cumulative cost because the second pass hits warm pages.

Generalizing: single-shot microbenches with small n on a hot data
path are noise-dominated at the ms scale. Latency claims for a
retrieval lever must be paired (same query stream, warm cache) and
reported as a percentile distribution, not a mean. v0.2 standing
rule: no latency claim ships without n ≥ 200 paired queries and
matched ingest. The candidate-pool prune for v0.3 is justified for
the `both` arm only.

**Verdict.** Single-shot ms-scale microbenches are noise-dominated.
Standing rule: paired n ≥ 200 + percentile distribution, or no
latency claim.
