# 3. Methods

<!-- Industry Track triage 2026-05-27: §3.4 (LoCoMo adaptive-vw
     protocol), §3.5 (share_prior reranker — full Personize §96
     adaptation), §3.6 (PRF entity expansion), and §3.7 (governed-
     memory primitives) moved to paper/A7_extended_methods.md.
     §3.8 ablation TOC trimmed; full table preserved in
     paper/A1_appendix_ablations.md §A.4.18. Body retains §3.1
     (retrieval cell + encoder grid), §3.1.1 (latency-cost), §3.2
     (entity-collision protocol), §3.3 (discriminator tags). -->

## 3.1 Engram retrieval cell

The unit-under-test is a single retrieval call,
`engine.recall(query, k, vector_weight=vw)`, with the BM25 score
(FTS5 over a tokenized text column) and an embedder-cosine score
combined as `score = (1−vw)·bm25_norm + vw·cos_sim`, where
`bm25_norm` is min-max normalized over the candidate set. We sweep
`vw ∈ {0.0, 0.3, 0.5, 0.7, 1.0}`, with `vw=0.0` being the
"BM25-only" floor.

Three embedders are compared: **HashTrigram-256** (character-trigram
hashing, 256-dim signed bag-of-trigrams, L2-normalized; zero model
load, ~9.6 ms p50 write at 10k); **ST MiniLM-384**
(`sentence-transformers/all-MiniLM-L6-v2`, 384-dim, normalized;
~17–21 ms p50 write at 1k); **BGE-large-1024** (`BAAI/bge-large-
en-v1.5`, 1024-dim, normalized; 2.7× MiniLM's parameter count;
~21.5 s/instance LongMemEval ingest on Apple Silicon MPS, ~150.8
s/instance on commodity CPU).

### 3.1.1 Encoder latency-cost trade-off

The three encoders span ~1000× in per-instance ingest cost on
commodity CPU and ~80× on Apple Silicon MPS. The headline-recall
column is the K=16 lexical-tag entity-collision lift on `service`
(the strongest hash-trigram cell) and the LongMemEval-S n=500
paired Δhit@1 vs MiniLM (the strongest BGE cell); the deployment-
cost column folds in model-load (one-time, amortized) and per-
instance ingest (paid per write). Numbers from the artifacts cited
in §4 and §A.4.16; full artifact registry in §A.4.18.

| encoder           | dim  | model load (s) | ingest p50 / inst (CPU) | ingest p50 / inst (MPS) | recall p50 / q | headline lift |
|-------------------|-----:|---------------:|------------------------:|------------------------:|---------------:|---------------|
| HashTrigram-256   |  256 |          ~0.0  |                ~525 ms  |                       — |       ~9.6 ms  | +5.7 pp Δhit@1 (K=16, `service`) |
| ST MiniLM-384     |  384 |          ~1.5  |               ~17–21 ms |                       — |      ~10.3 ms  | CI-positive on all 5 tags at K≥4 |
| BGE-large-1024    | 1024 |          ~5.0  |               ~150.8 s  |               ~21.5 s   |       ~120 ms  | +5.8 pp Δhit@1 LME n=500 |
| RM3 (BM25 + PRF)  | n/a  |          ~0.0  |               ~1.5 ms/d |                       — |       ~259 ms  | SIG-NEG on `single-session-user` (−7.1 pp) |

*Reading the table.* HashTrigram is essentially free at write time
(no model, no GPU, FTS5-only). MiniLM is the v0.2 default-embedder
choice: ~17–21 ms ingest is acceptable on commodity CPU and lift
is universal (CI-positive on all 5 tags at K≥4). BGE-large is
**only** defensible on accelerator hardware — at 150.8 s/instance
LME ingest on CPU, a +5.8 pp Δhit@1 gain costs ~720× the per-
instance write budget of MiniLM; on MPS the ratio collapses to ~7×
and the trade-off becomes workload-dependent. RM3 occupies a
fourth operating point: zero model-load, sub-millisecond ingest,
~25× the per-query latency of HashTrigram, but SIG-regresses
LongMemEval `single-session-user` by 7.1 pp via query drift and
fails to recover the `single-session-preference` cliff (§A.4.16.4).

The "encoder capacity is not the binding constraint" claim (§4.1)
is therefore **also** a deployment-cost claim: the 2.7×-parameter
BGE upgrade does not uniformly improve recall, and even on cells
it does help the cost differential is steep enough that v0.2
ships MiniLM as default and BGE as a workload-targeted opt-in.

## 3.2 Entity-collision protocol

For each tag `t ∈ {preference, project, technical, service, tool}`
and collision degree `K ∈ {1, 2, 4, 8, 16}`:

1. Generate `n_entities = 32` distinct entities, each with one
   gold answer document of the form
   `"<entity> uses <answer> for <tag>."`
2. For each entity, generate `K-1` distractor documents that
   **share the entity tokens but flip the answer**:
   `"<entity> uses <other_answer> for <tag>."` so a BM25 retriever
   sees identical query-entity overlap on all K candidates per
   query.
3. Issue a query `"what does <entity> use for <tag>?"` and measure
   `hit@1` (gold = top-1 result).

This fixes the BM25 floor at `1/K` in expectation per query (random
tie-breaking among the K candidates), so any lift over `1/K` is
attributable to the embedder distinguishing the answer slot.

We report **paired Δhit@1** (per-entity matched pairs of `vw=0.5`
vs `vw=0.0`) with 95% bootstrap CIs (10k resamples).

## 3.3 Discriminator tags

We label `service` (closed-vocabulary proper-noun answers: aws,
gcp, azure) and `tool` (closed-vocabulary: git, docker, postgres)
as **lexical discriminators**, and `preference` (open phrasal: dark
mode, light mode), `project`, `technical` as **intent-style
discriminators**. The full per-tag query/answer schema is in §A7.5.

The supporting methods detail — LoCoMo adaptive-vw experiment
protocol (§A7.1), the share_prior reranker derivation (§A7.2), PRF
entity expansion (§A7.3), and the four governed-memory primitives
(§A7.4) — live in the extended-methods appendix. The full
claim → section → artifact registry (26 result tables, 37
reproduce scripts) is in §A.4.18 and verified by
`scripts/verify_repro_artifacts.sh`.
