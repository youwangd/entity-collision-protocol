# (Repository anonymized for review)

Source code, benchmarks, and reproduce scripts for the EMNLP 2026 Industry Track submission:

> **Entity-Collision: A Stratified Protocol for Attributing Retrieval Lift in Agent Memory**
> Anonymous Authors. Under submission to EMNLP 2026 Industry Track.
> See `paper/dist/engram_v0.2_emnlp.pdf` for the review PDF.

The paper's contribution is the entity-collision evaluation protocol and the two-axis empirical finding it surfaces (§1, §3.2-3.3). Engram is the testbed on which that protocol is exercised, released here for reproducibility.

## Reproducing the paper

```bash
pip install -e .
bash paper/REPRODUCIBILITY.md  # see for the full reproduce sequence
```

## Repository layout

| Path | Contents |
|---|---|
| `paper/` | Markdown sources for body + 7 appendices, plus `paper/dist/engram_v0.2_emnlp.pdf` |
| `evals/` | Entity-collision protocol implementation, LongMemEval/LoCoMo/BEIR adapters |
| `engram/` | Agent-memory testbed source |
| `scripts/` | Build scripts, sweeps, anonymization checks |
| `tests/` | Unit + property tests |

## License

Apache 2.0 (anonymized for review).
