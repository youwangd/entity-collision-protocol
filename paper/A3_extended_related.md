# A3. Extended Related Work

This appendix expands §2's main-body summary into the full
related-work coverage that an ACL 2-column 6-page Industry Track
body cannot accommodate. Anchor citations + direct comparisons live
in §2; the expansions below cover (1) other contemporary agent-
memory designs, (2) the long-context retrieval-evaluation suite,
(3) the dense-retrieval evaluation ecosystem, (4) late-interaction
and learned-sparse non-inclusion, (5) pseudo-relevance feedback,
and (6) the event-sourcing / bi-temporal / CRDT lineage.

## A3.1 Other contemporary memory designs

Beyond the three direct-comparison systems in §2.1, four further
agent-memory designs warrant mention:

**A-MEM** \citep{xu2025amem} builds an *agentic
memory* substrate as a Zettelkasten-style graph: each note is an
LLM-generated atomic unit linked to others through LLM-suggested
associative edges, with periodic LLM-mediated "consolidation"
passes that rewrite link structure. The contrast with Engram is
governance: A-MEM's link graph is rewritten whenever the
consolidation LLM judges the substrate is stale, and there is no
replay or audit trail across rewrites. Engram's schema lifecycle
(§A7.4.4) explicitly trades off this flexibility for deterministic
replay.

**HippoRAG** \citep{gutierrez2024hipporag, gutierrez2025hipporag2} routes retrieval through a knowledge graph
constructed at write time and uses Personalized PageRank at query
time to surface bridge entities for multi-hop questions. Where
Engram's `share_prior` reranker (§A7.2) computes within-pool entity
co-occurrence on the *post-retrieval* candidate set, HippoRAG
operates pre-retrieval over the full graph. The two are
complementary: HippoRAG addresses recall-side bridge promotion,
share_prior addresses precision-side rank-0 preservation.

**Cognee** (cognee-ai/cognee, 2024–) is an open-source semantic
memory layer combining vector retrieval with a knowledge-graph
index synthesized via LLM extraction. Like A-MEM and HippoRAG,
Cognee centers the graph as the substrate; like Mem0 v3, it relies
on add-only writes with periodic re-consolidation. We do not
benchmark against Cognee directly because it does not expose a
session-level hit@k metric reproducible from raw input chats.

**Zep / Graphiti** \citep{rasmussen2025zep}
provides a temporal knowledge-graph memory with bi-temporal edges
and reified relationships, evaluated on Deep Memory Retrieval (DMR).
Zep emphasises temporal coherence — what was true when — which
Engram approximates through schema-lifecycle decay (§A7.4.4) but
does not store as first-class temporal edges. The two designs
answer different "what did the agent know" questions: Zep's is
graph-shaped, Engram's is event-log-shaped.

The §2.1 "where Engram sits" framing extends to these systems:
A-MEM, HippoRAG, Cognee, and Zep each center a graph substrate
that is rewritten under LLM mediation. Engram's mechanical-
governance bet differs categorically.

## A3.2 Long-context retrieval evaluation suite

The two benchmarks named in §2.2 are the agent-memory community's
conversational anchors; the broader long-context retrieval-
evaluation ecosystem we draw methodology from includes:

- **RULER** \citep{hsieh2024ruler} — 13 synthetic
  tasks (NIAH variants, multi-key/value retrieval, variable
  tracking, frequent-words extraction, long-document QA) parametric
  in context length up to 1M tokens. RULER showed that nominally
  long-context models often degrade well below their advertised
  window; we adopt its **stratified-by-task** discipline at the
  retriever level rather than the model level, which is the
  methodological seed of our entity-collision per-tag stratification.

- **∞Bench** \citep{zhang2024infbench} — 12 tasks
  averaging >100k tokens across math, code, novels, and dialogue.
  Used in the long-context evaluation literature as a length-stress
  complement to RULER's task-stress; not directly applicable to
  agent memory because tasks are document-centric, not session-
  centric.

- **LongBench-v2** \citep{bai2024longbenchv2} — 503
  multiple-choice questions over 8k–2M-token contexts.
  Methodologically closer to multi-doc QA than agent memory.

- **NIAH / Needle-in-a-Haystack** \citep{kamradt2023niah} — single-fact
  retrieval at controlled depth. The closest one-axis ancestor of
  entity-collision; entity-collision generalises by stratifying on
  *discriminator type*, which NIAH does not.

- **LV-Eval / LooGLE / L-Eval** \citep{an2024leval, li2024loogle, yuan2024lveval} — long-context QA suites that all report a
  single hit@k or LLM-judge accuracy per model, exhibiting the
  tag-mixing problem §1 motivates the entity-collision protocol
  to address.

## A3.3 Dense-retrieval evaluation ecosystem

- **BEIR** \citep{thakur2021beir} — 18-task
  zero-shot retrieval benchmark. Established the "BM25 is hard to
  beat zero-shot" finding that anchors our entity-collision
  protocol's BM25-floor design.

- **MTEB** \citep{muennighoff2023mteb} — Massive
  Text Embedding Benchmark, 56 datasets across 8 task families.
  Source for our embedder choices: BGE-large-en-v1.5 sat near the
  top of MTEB's retrieval leaderboard at the time of our encoder
  grid freeze, which is why we extended the protocol to it as the
  encoder-capacity falsification test.

- **MS MARCO** \citep{bajaj2016msmarco} and
  **TREC Deep Learning** \citep{craswell2020trecdl19, craswell2021trecdl20} — passage-retrieval
  staples. Useful as a sanity prior for relative encoder ordering
  but tangential to the agent-memory setting because queries are
  short and corpora are static.

## A3.4 Late-interaction and learned-sparse baselines

The v0.2 measurement grid covers three **single-vector dense
encoders** (HashTrigram-256, MiniLM-384, BGE-large-1024). We
deliberately do not include late-interaction (ColBERT, ColBERTv2; \citealp{khattab2020colbert, santhanam2022colbertv2}) or learned-sparse retrievers (SPLADE, SPLADE++; \citealp{formal2021splade, formal2022splade2}) — both are documented to outperform single-vector
dense encoders on BEIR — because the headline question this paper
engages is specifically **whether per-query semantic capacity in
the single-vector regime is the binding constraint on agent-memory
retrieval**. The two-axis result of §4.3 and the encoder-capacity
falsification of §A.4.16 are claims about that regime: a
2.7×-parameter increase within the single-vector family does not
collapse the lexical-vs-intent split, which is the methodological
finding that motivates the protocol. ColBERT and SPLADE answer a
different question — whether *interaction structure* (token-level
late interaction) or *index sparsity* (learned sparse projections)
recovers cells the single-vector family cannot — and a clean
answer to that question requires its own protocol design, not a
4th column on this grid. We accordingly flag late-interaction and
learned-sparse retrievers as a **v0.3 follow-up** with its own
protocol freeze, not a v0.2 omission.

A secondary, deployment-side consideration reinforces this scope
boundary: ColBERT's per-token storage and SPLADE's per-query
inference cost both invert the deployment trade-off table in
§3.1.1 on commodity-CPU hosts (the v0.2 testbed and the v0.2
default-embedder choice). A reviewer interested in the
accelerator-only deployment regime should read §A.4.16 alongside
this section: the same trade-off applies to BGE-large there, and
ColBERT/SPLADE land further along the same axis.

## A3.5 Pseudo-relevance feedback

- **RM3** \citep{lavrenko2001rm} — the canonical relevance-model PRF expansion
  that mixes a discriminative term distribution via learned λ. §5.1
  details why our heuristic-PRF falsification (§A.4.15j-o) does not
  extrapolate to RM3. AUDIT-D ships an RM3 arm across the entity-
  collision grid, BEIR FiQA, and LongMemEval n=500 (§A.4.16.4);
  the headline finding is that RM3 does not rescue PRF on
  intent-style queries, sharpening §4.3's two-axis claim.

- **Rocchio relevance feedback** \citep{rocchio1971feedback} — original vector-
  space PRF. Cited for completeness; supplanted by RM3 in modern
  retrieval-evaluation practice.

## A3.6 Schema-lifecycle and event-sourced memory

The schema-lifecycle invariant set we discuss in §A7.4.4 and §A4.2
draws on three threads:

- **Event sourcing** \citep{fowler2005es, vernon2013ddd} —
  pattern from domain-driven design where state is the fold of an
  immutable event log rather than a mutable record. Our schema
  lifecycle is an event-sourced reducer
  (`tests/property/test_schema_lifecycle.py`).

- **Bi-temporal data modelling** \citep{snodgrass1999tsql, date2002temporal} — the discipline of separating "what was
  true" from "when we knew it was true." Engram approximates the
  latter through write timestamps on the decision log; we do not
  claim full bi-temporal correctness.

- **CRDT / monotone reducer literature** \citep{shapiro2011crdt} — informs our
  "decay is monotone in real time" invariant
  (`test_schema_decay.py`).
