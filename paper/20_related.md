# 2. Related Work

<!-- Industry Track triage 2026-05-27: trimmed body to ~600w; the
     2.1.1 / 2.2.1 / 2.3.1 / 2.3.2 / 2.3.3 / 2.4 expansions live in
     paper/A3_extended_related.md. Anchor citations + direct
     comparison kept here. Citation pass 2026-05-24 + GPTZero
     pre-flight 2026-05-26 verifications still apply to all cited
     arXiv IDs. -->

## 2.1 Agent memory systems

Three contemporary agent-memory designs span the architectural
spectrum from LLM-driven to mechanical governance.

**Letta / MemGPT** \citep{packer2023memgpt}
implements OS-style virtual context management: the LLM itself acts
as memory manager via tool calls, paging content between a small
main context and an unbounded archival store. Consolidation is
agent-driven; the OSS path exposes no write-side dedup primitive.

**Mem0** \citep{chhikara2025mem0} is a dynamic
extract→consolidate→retrieve loop. The v3 (April 2026) flip to
single-pass ADD-only extraction with cross-memory entity linking
reports LoCoMo 91.6 / LongMemEval 94.8 under an LLM-as-judge metric.
Engram does not currently ship an LLM-judge mode; direct numerical
comparison to Mem0's reported scores is therefore out of scope, and
we report session-level hit@k against the same benchmarks instead.

Personize.ai's "Governed Memory" line
\citep{taheri2026gm} is the closest
prior in design space. Engram implements the §1-6 stack (dual
extraction with per-fact confidence, write-side cosine dedup at
0.92, mechanical merge, schema lifecycle) and extends §7-8 with
calibrated contamination/fragmentation meters (§A7.2, §A.4.7) and a
quorum-gated DEPRECATE primitive (§A.4.6) that hardens schema
lifecycle against single-emitter takedown attacks.

**Where Engram sits.** Mem0 v3 sidesteps governance with ADD-only
writes; Letta delegates governance to the LLM-as-memory-manager.
Engram's bet is that **mechanical, replayable governance** —
extraction confidence per fact, schema transitions through an
explicit DAG, prior-sharing gated by calibrated meters — beats
LLM-in-the-loop judgment for memory workloads where audit trails
and replay determinism are first-class requirements. The empirical
question this paper engages is whether such governance costs recall
(§4.5, §4.6) and whether the dense-retrieval lift it preserves is
uniform across query types (§4.1-4.3). Additional contemporary
designs (A-MEM, HippoRAG, Cognee, Zep/Graphiti) are surveyed in
§A3.1.

## 2.2 Benchmarks and evaluation

We evaluate on two community-standard agent-memory benchmarks:

- **LongMemEval** \citep{wu2025longmemeval} — 5 question
  categories over multi-session chats. Headline numbers in §4.6 use
  the `longmemeval_s` split (n=500), SHA-256-pinned in
  `paper/REPRODUCIBILITY.md` §0.

- **LoCoMo** \citep{maharana2024locomo} — 10
  multi-session conversations, 1978 questions across 5 categories.
  We report per-category 95% paired-bootstrap CIs across
  `vector_weight ∈ {0.0, 0.3, 0.5, 0.7, 1.0}` for both
  HashTrigram-256 and ST MiniLM-384 embedders.

The broader long-context retrieval-evaluation suite from which we
draw stratification methodology (RULER, NIAH, ∞Bench, LongBench-v2,
LV-Eval/LooGLE/L-Eval) is detailed in §A3.2.

## 2.3 Retrieval baselines

**BM25 as a strong baseline** \citep{thakur2021beir} anchors the entity-collision protocol:
distractor-shared entity tokens fix the BM25 lexical floor by
construction so any lift over BM25 is attributable to the embedder
(§3, §4.1).

**Hash-trigram / sketched embeddings** \citep{weinberger2009hashing} provide a model-load-free alternative to dense embedders.
The two-axis result of §4.3 quantifies exactly when this
trade-off is acceptable: lexical-discriminator queries at deep
collision recover ~50% of dense-embedder lift; intent-style
queries do not.

The dense-retrieval evaluation ecosystem (BEIR, MTEB, MS MARCO,
TREC DL), our explicit non-inclusion of late-interaction (ColBERT)
and learned-sparse (SPLADE) retrievers, the pseudo-relevance
feedback line (RM3, Rocchio), and the event-sourcing /
bi-temporal / CRDT lineage that informs §A7.4.4 schema lifecycle
are surveyed in §A3.3-§A3.4.
