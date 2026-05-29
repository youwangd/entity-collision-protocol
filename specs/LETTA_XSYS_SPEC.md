# LETTA-XSYS-REPLICATION — Cross-system replication spec for AUDIT-A

> **Closes:** AUDIT-A (Letta or Mem0 cross-system replication on ≥1
> entity-collision tag pair). Riskiest open gap as of audit pass v2.
> Goal: demonstrate the two-axis result (lexical vs intent
> discriminator) is **not** a Engram-specific BM25 implementation
> artifact.
>
> **Status:** SPEC — not yet executed. Owner: the lead author. Estimate: 3-5
> days end-to-end (1 day for Letta-side ingest harness, 1 day for the
> protocol port, 1 day for paired-CI computation, 1-2 days for write-up
> + reproduce script + paper integration).
>
> **Why Letta over Mem0.** Letta exposes a clean `agent.send_message`
> + `archival_memory_search` REST API; Mem0's API is more deeply
> entangled with their LLM-backed merge step which would confound the
> measurement. Letta's archival memory is a vector + BM25 hybrid we
> can probe at the same level we probe Engram's.

## 1. Scope

We replicate **two tag pairs** from the entity-collision protocol on
Letta's archival memory:

- **Lexical tag:** `tool` (closed-vocabulary, proper-noun answers
  like git, docker, postgres) — where Engram's hash-trigram recovers
  ~50% of dense lift at K∈{4,8}.
- **Intent tag:** `preference` (open-vocabulary intent answers) —
  where Engram's hash-trigram is null at every K.

Both tags at K∈{2, 4, 8, 16} with n=32 entities. Only the **dense**
arm runs (Letta's archival default). We do not need to replicate
HashTrigram-256; that arm is Engram-specific.

The cross-system claim is narrower than Engram's claim. We assert:

> "On Letta's archival memory with its default dense embedder, the
> lexical-vs-intent discriminator stratification reproduces: dense
> lift over Letta-internal-BM25 is CI-positive at K≥4 on `tool`
> *and* on `preference`, with the same monotonic-in-K shape Engram
> reports."

We do **not** assert Letta beats / matches / loses to Engram in
absolute terms — that's not what cross-system replication is for.

## 2. Letta integration interface

Required Letta endpoints (verified against
`github.com/letta-ai/letta` HEAD as of audit pre-flight; pin SHA at
experiment-start time):

```
POST /v1/agents          # create scratch agent per cell
POST /v1/agents/{id}/archival_memory   # ingest a memory string
POST /v1/agents/{id}/archival_memory/search  # query, returns ranked passages
DELETE /v1/agents/{id}   # tear down
```

The harness reuses `evals/entity_collision.generate_dataset()`
verbatim — corpus generation is system-agnostic. Only the
ingest/query loop differs.

## 3. New file: `evals/letta_entity_collision.py`

```python
"""Cross-system replication of the entity-collision protocol on Letta.

Pinned Letta SHA: <to be filled at experiment-start time>.
Pinned Letta default embedder: text-embedding-ada-002 or
letta-default-* (record exact name + dim in the artifact).
"""
import os
import time
from letta_client import Letta  # or whatever the official SDK is named at pin time
from evals.entity_collision import generate_dataset
from evals.metrics import find_match_rank, hit_at_k

LETTA_BASE_URL = os.environ["LETTA_BASE_URL"]   # e.g. http://localhost:8283
LETTA_API_KEY  = os.environ.get("LETTA_API_KEY")  # may be None for self-host

def run_cell(tag: str, K: int, n_entities: int = 32, seed: int = 42) -> dict:
    ds = generate_dataset(tags=[tag], collision_degrees=[K],
                          n_entities=n_entities, seed=seed)
    client = Letta(base_url=LETTA_BASE_URL, api_key=LETTA_API_KEY)
    # one fresh agent per cell, tear down at the end (no cross-cell state)
    agent = client.agents.create(name=f"engram-xsys-{tag}-K{K}-{seed}")
    try:
        for text, meta in ds.memories:
            client.agents.archival_memory.create(agent_id=agent.id, text=text)
            time.sleep(0.05)  # polite to local Letta; real API has rate limits

        records = []
        for q in ds.queries:
            hits = client.agents.archival_memory.search(
                agent_id=agent.id, query=q.text, top_k=10)
            ranked_texts = [h.text for h in hits]
            rank = find_match_rank(ranked_texts, q.expected_substrings)
            records.append({"q": q.text, "rank": rank, "K": K, "tag": tag})
        return {"tag": tag, "K": K, "n": len(records),
                "hit_at_1": hit_at_k(records, 1),
                "hit_at_10": hit_at_k(records, 10),
                "per_query": records}
    finally:
        client.agents.delete(agent_id=agent.id)
```

## 4. Driver: `evals/letta_entity_collision_sweep.py`

```python
"""Sweep K∈{2,4,8,16} × tag∈{tool, preference} on Letta.

Outputs:
  bench/results/letta_xsys/{tag}_K{K}.json     # raw per-cell records
  bench/results/letta_xsys/{tag}_K{K}_ci.json  # paired-bootstrap vs Engram
"""
import json
from pathlib import Path
from evals.letta_entity_collision import run_cell

OUT = Path("bench/results/letta_xsys")
OUT.mkdir(parents=True, exist_ok=True)

for tag in ["tool", "preference"]:
    for K in [2, 4, 8, 16]:
        result = run_cell(tag, K)
        (OUT / f"{tag}_K{K}.json").write_text(json.dumps(result, indent=2))
```

## 5. Paired-CI vs Engram

Same protocol, same query set, paired by `(entity, discriminator)`
key. We re-use `evals/entity_collision_ci.py` with a `--system letta`
flag that points it at the Letta artifact paths. The output is a
**within-Letta** per-cell hit@1 + 95% CI (the Letta-vs-Engram
absolute comparison is **not** the claim and is not made).

The cross-system claim's CI is on **Letta's lexical-vs-intent
delta**: at K∈{4,8,16}, is `tool` × Letta hit@1 > `preference` ×
Letta hit@1 with non-overlapping CIs? If yes, the two-axis result
replicates.

## 6. Deliverable: §A.4.17 in `paper/A1_appendix_ablations.md`

Mirror the §A.4.16 (BGE) structure:

```markdown
## A.4.17 Cross-system replication on Letta archival memory

### A.4.17.1 Setup
- Letta SHA, default embedder name + dim, host
- n=32 entities, K∈{2,4,8,16}, paired by (entity, disc)
- One fresh agent per cell, no cross-cell state

### A.4.17.2 Per-tag hit@1 + 95% CI
Two tables (tool, preference) × 4 K values.

### A.4.17.3 Lexical-vs-intent delta on Letta
The cross-system claim: at K=4/8/16, Letta's tool > preference at
non-overlapping CIs. Pass / fail per K.

### A.4.17.4 Verdict
- Pass: "the two-axis stratification holds across Engram and Letta —
  the lexical/intent split is a property of the protocol, not the
  system."
- Fail: "Letta's archival memory does not stratify as Engram's does,
  scoping the two-axis claim to Engram-class hybrid retrievers."

Either result is publishable; honest reporting either way.
```

## 7. Pre-experiment checklist (before running)

- [ ] Self-host Letta on the M4 Pro (or Linux x86_64 host with disk for
      the SQLite/PG backing store).
- [ ] Pin Letta's SHA at experiment start; record in §A.4.17.1.
- [ ] Confirm Letta's default embedder fits in the M4's memory
      (BGE-large + a Letta agent state may be tight on 32GB).
- [ ] Add `letta-client` to `requirements-eval.txt`; do NOT add to
      `engram/`'s install — Letta is an evaluation-only dependency.
- [ ] Smoke-test ingest/query of a 10-memory toy corpus end-to-end
      before the full sweep.
- [ ] Set `LETTA_BASE_URL` / `LETTA_API_KEY` in the M4 environment.

## 8. Estimated wall time on M4 Pro

| Step                                 | Cells | Per-cell wall | Total            |
|--------------------------------------|-------|---------------|------------------|
| 32 entities × K memories per cell    | 8     | 30-90 s ingest| ≈10 min          |
| Per-cell query loop (32 queries)     | 8     | 30-60 s       | ≈8 min           |
| Per-cell teardown                    | 8     | <5 s          | <1 min           |
| **Sweep total (8 cells)**            | —     | —             | **≈20 min**      |
| Paired-CI computation                | —     | —             | <2 min           |
| Cross-system delta CI                | —     | —             | <2 min           |

8 cells is small because we're only running 2 tags × 4 K values on
the dense arm. Letta's archival API is the rate-limit, not compute.

## 9. Failure modes to expect

1. **Letta's BM25 ≠ Engram's BM25.** Letta uses pgvector + bm25
   ranking from `tantivy` or similar. The BM25-only equivalent is
   not directly exposable. **Mitigation:** report Letta's
   archival-default scoring as one number per cell; the cross-system
   claim is on the *stratification* (tool > preference), not on
   Engram-vs-Letta absolute alignment.
2. **Letta's default embedder is OpenAI-hosted.** If
   `text-embedding-ada-002`, that is a 1536-d encoder we have not
   characterized. **Mitigation:** record exactly which embedder
   Letta defaults to in §A.4.17.1; our claim is "across systems",
   not "across encoders" — the encoder swap was already covered in
   §A.4.16 (BGE).
3. **Letta has its own deduplication / merge.** This will compress
   K identical-entity memories. **Mitigation:** disable Letta's
   archival deduplication if the API exposes a flag; otherwise add
   a unique salt token to each memory's `meta` field so dedup
   doesn't collapse them.

## 10. Stop signals for the experiment

- **STOP if** Letta's API changes / SHA does not match the pin during
  the run. Re-pin and re-run from cell 1; partial results are not
  poolable across SHA changes.
- **STOP if** any cell returns `n_paired < 24` (i.e. ≥8 queries lost
  to API errors). Investigate Letta logs; do not paper over.
- **STOP if** the per-cell hit@1 is exactly 0 or exactly 1 across
  all queries — that is a probe-side bug, not a result.
- **PROCEED if** the per-tag-per-K cell yields a non-degenerate
  hit@1 distribution and CI bounds are sensible.

## 11. Disclosure obligations (per ACL §c/§d)

The harness code in `evals/letta_entity_collision*.py` is
author-written, not assistant-written. The §A.4.17 narrative in the
paper will be author-drafted from artifact JSON and **will not** be
covered by the §A6 / §A.4.8.1 assistant-drafted disclosure block.
Re-confirm at write-up time and update
`paper/80_acknowledgements_cameraready.md` accordingly.
