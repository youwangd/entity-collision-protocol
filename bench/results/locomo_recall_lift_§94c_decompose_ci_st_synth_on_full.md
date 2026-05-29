# §94c-decompose-CI under MiniLM-384, synthesis on, FULL LoCoMo10

`evals.locomo_recall_lift_decompose_ci --dataset bench/data/locomo10.json
--max-instances 10 --resamples 10000 --seed 42 --embedder st --synthesis`

n_paired = 1978 · wall = 1018.65s · embedder = SentenceTransformer-MiniLM-384

S1 = `extraction` only · S7 = full default pipeline (synthesis on, gate off)

## Paired bootstrap on (S1 − S7)

| metric    | mean    | 95% CI               | p         |
|-----------|---------|----------------------|-----------|
| Δh@1      | +0.0010 | [-0.0066, +0.0086]   | 0.831     |
| Δh@k      | -0.0046 | [-0.0096, +0.0005]   | 0.087     |
| ΔMRR      | -0.0027 | [-0.0072, +0.0019]   | 0.252     |
| Δprk      | -0.0046 | [-0.0096, +0.0005]   | 0.089     |
| Δgrk      | -0.0055 | [-0.0102, -0.0008]   | **0.019** |

## Verdict

**Closes the §94c-decompose CI matrix on full LoCoMo10 under MiniLM-384.**
With ~6.6× the sample of #94c-decompose-CI-st (1978 vs 301), four of the
five metrics now bracket zero. Only Δgrk (gold_recall@k) survives
significance — and at -0.55pp, it's a tiny structural slip.

Compare to small fixture (n=301): Δh@k p=0.034 → p=0.087 (attenuated),
Δgrk p=0.038 → p=0.019 (held, slightly tightened), Δh@1 p=0.196 → p=0.831
(definitively null). The "downstream costs ~1pp set-recall" finding is
the only robust signal.

## Comparison: §94c-decompose-CI matrix (closed)

| metric | HT-256 small (n=301, synth=False) | HT-256 small (synth=True)¹ | ST-384 small (n=301, synth=True) | **ST-384 FULL (n=1978, synth=True)** |
|--------|-----------------------------------|----------------------------|-----------------------------------|--------------------------------------|
| Δh@1   | +0.0066 [-0.013,+0.027] p=0.638 | (above) | +0.0133 [-0.003,+0.033] p=0.196 | **+0.0010 [-0.007,+0.009] p=0.831** |
| Δh@k   | (above) | (above) | -0.0133 [-0.027,-0.003] p=**0.034** | -0.0046 [-0.010,+0.001] p=0.087 |
| ΔMRR   | (above) | (above) | +0.0052 [-0.006,+0.018] p=0.379 | -0.0027 [-0.007,+0.002] p=0.252 |
| Δgrk   | (above) | (above) | -0.0119 [-0.026,-0.001] p=**0.038** | -0.0055 [-0.010,-0.001] p=**0.019** |

¹Reference baseline: §94c-decompose-CI hashtrigram-256.

## Paper §5.3 wording (proposed)

> "Decomposing the consolidation pipeline into S1=extraction-only and
> S7=full default at LoCoMo10 scale (n=1978 paired) under MiniLM-384,
> only Δgold_recall@k rejects null (-0.55pp, p=0.019, paired-bootstrap
> 95% CI [-1.0, -0.1]). All four other metrics — Δh@1, Δh@k, ΔMRR, Δprk
> — bracket zero. The h@k slip observed at small fixture (#94c-st,
> n=301, p=0.034) attenuates to p=0.087 at full corpus. Conclusion:
> downstream consolidation stages are pareto-neutral on top-1, MRR, and
> hit@k headlines and cost a structural ~0.5pp on gold-set-recall@k —
> embedder-independent (same sign and magnitude as hashtrigram-256)."
