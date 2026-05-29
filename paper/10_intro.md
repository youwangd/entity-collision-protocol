# 1. Introduction

Agent-memory systems are increasingly deployed as the long-context
substrate for LLM assistants — Letta, Mem0, MemGPT, and the
"governed memory" line all argue that an external memory store
beats unbounded context windows on cost, latency, and recall.
A recurring open question across these systems is what retrieval
machinery is actually earning its keep: is BM25 enough? Does
dense embedding help? When? For a team deciding which embedder
to ship behind their agent memory, the answer determines model-load
cost, ingest latency, and recall headroom — but the field's
end-to-end benchmarks confound these decisions with lexical-leakage
and tag-mixing artifacts.

This paper contributes a measurement-first methodology for
agent-memory retrieval evaluation: a stratified, lexical-floor-pinned
protocol that lets deployment teams characterize per-tag retrieval
behavior under operational constraints — embedder cost, ingest
latency, recall headroom — rather than chase end-to-end averages
that conflate orthogonal failure modes.

End-to-end benchmarks (LongMemEval, LoCoMo) report a single hit@k
per retriever, which is insufficient for two reasons:

1. **Lexical leakage.** End-to-end benchmark queries usually share
   the answer's entity tokens with the gold passage but not with
   distractors. BM25 trivially wins, and any embedder "lift" is
   confounded with lexical anchor strength.

2. **Tag-mixing.** Benchmarks blend categories (preferences,
   projects, technical facts, services, tools) into one number. A
   retriever may be systematically better on some tags and worse
   on others; the average obscures this.

We address both with an **entity-collision protocol**: synthetic
queries where every distractor shares the answer's entity tokens
(BM25 floor fixed by construction), stratified by discriminator
tag. Results across 5 tags, 3 embedders, 5 collision degrees with
paired bootstrap 95% CIs (Figure 1, §4.1) reveal a two-axis
pattern that **replicates on natural data**: LongMemEval (n=500)
shows the same intent-tag weakness as a single-session-preference
recall cliff, and LoCoMo quantifies an 11.7-pp residual oracle
headroom that no signal we tested recovers.

The work was done on **Engram**, an open-source agent-memory system
we built as a controlled testbed. Engram is the artifact through
which we run the experiments and which we release for
reproducibility ([repo URL anonymized for review]); it is not
itself the contribution of this paper. A measurable agent-memory
system presupposes deterministic write/merge mechanics so that
re-running a configuration produces a bit-identical store — we
exercise the protocol on a governed-memory testbed (event-sourced
decision log, DAG schema lifecycle); §A7.4, §A.4.6, §A4.2 detail
the substrate. The state-machine and linear-scale evidence are
testbed sanity, not retrieval claims.

## Contributions

1. **Entity-collision evaluation protocol** that fixes the BM25
   floor and isolates the embedder-attributable retrieval lift,
   open-sourced under `evals/entity_collision_*`. The protocol is
   system-agnostic and applies to any retriever exposing a
   per-document score.

2. **Two-axis empirical finding with synthetic→natural replication
   and an encoder-capacity falsification.** On the synthetic grid,
   hash trigrams help on lexical-discriminator tags at deep
   collision (K≥4 on `tool`, K=16 on `service`) but are null or
   negative on intent-style tags; MiniLM-384 dominates both axes.
   Extending to BGE-large-en-v1.5 (1024-d, 2.7× MiniLM's parameter
   count) does **not** collapse this two-axis structure: BGE wins
   on intent-style `project` (+8 to +14 pp BGE−MiniLM at
   K∈{2,4,8,16}, all CI-positive) but loses on lexical `tool`/
   `technical` (−2.7 to −11.7 pp at K∈{4,8,16}, all CI-significant).
   Encoder capacity alone is not the binding constraint.
   The same per-tag pattern reproduces on LongMemEval (n=500): the
   intent-tag null predicts the single-session-preference cliff
   (hit@1 = 0.367 vs 0.81 overall). The decision-shaped corollary
   — closed-vocabulary queries can ride a 256-d hash trigram at
   zero model-load cost, while intent-style queries require a dense
   encoder — is what a deployment team needs from the protocol that
   a single hit@k average obscures.

A secondary, sidebar measurement — adaptive vector-weight routing
on LoCoMo as a measured null with 11.7 pp of unrecoverable oracle
headroom — appears in §4.4 and motivates the re-framing of vector
fusion as a paraphrase-robustness mechanism. Schema-lifecycle
invariants and a 1M-memory write-latency characterization are
testbed sanity rather than primary claims; see §A.4.14, §A4.2, §A6.

All 26 result tables and 37 reproduce scripts cited in the paper
are version-controlled and verified by `verify_repro_artifacts.sh`;
see `paper/REPRODUCIBILITY.md`.
