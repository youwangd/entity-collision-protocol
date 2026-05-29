# Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory

## Abstract

End-to-end agent-memory benchmarks report a single hit@k per
retriever, confounding lexical leakage (uncontrolled
query/gold/distractor entity overlap) with tag-mixing
(preferences, services, tools averaged together). We propose
**entity-collision**, a system-agnostic protocol that pins the
BM25 floor by construction — every distractor shares the answer's
entity tokens — and stratifies queries by discriminator tag, so any
lift over BM25 is attributable to the embedder. Applied to an
open-source agent-memory testbed across 5 tags × 3 embedders × 5
collision degrees with paired-bootstrap 95% CIs, the protocol
reveals a **two-axis pattern**: a 256-d hash trigram helps only
on closed-vocabulary lexical tags at deep collision; MiniLM-384
dominates both axes; and a 2.7×-parameter BGE-large does not
uniformly improve on MiniLM — it wins on intent-style queries but
loses on lexical ones. Encoder capacity alone is not the binding
constraint. The synthetic intent-tag null replicates on LongMemEval
(n=500) as a single-session-preference recall cliff. Adaptive
vector-weight routing on LoCoMo is a measured null: 11.7 pp of
oracle headroom exists, but no signal we tested recovers it.
All 26 result tables and 37 reproduce scripts are version-controlled
and verified by a public registry; the protocol is exercised on a
**deterministically governed** memory testbed (event-sourced
decision log, DAG-state-machine schema lifecycle) so every reported
CI is reproducible byte-for-byte from the ingest stream.

<!-- FRAMING FROZEN 2026-05-23, refined to measurement-first 2026-05-24,
     extended to 3-embedder grid (BGE-large) 2026-05-24,
     re-titled to entity-collision-led + abstract trimmed 150-180w 2026-05-26.
     Headline: two-axis result (lexical vs intent discriminator)
     survives a 2.7×-parameter encoder swap; synthetic→natural bridge
     replication on LongMemEval. Protocol = contribution; Engram = testbed. -->
