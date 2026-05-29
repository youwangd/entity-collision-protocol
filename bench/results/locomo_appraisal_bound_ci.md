# §94c-appraisal-bound-CI — paired bootstrap (cap=0.30 − cap=None)

Dataset: bench/data/locomo10.json (max_instances=2, k=10, embedder=hashtrigram).
n_paired=301 | resamples=10000 | seed=42 | wall=67.63s.

## Headline (point estimates)

| arm | Δh@1 | Δh@k | Δprk | Δgrk | ΔMRR |
|---|---|---|---|---|---|
| a (cap=0.30) | +0.0864 | +0.1528 | +0.1429 | +0.1466 | +0.0970 |
| b (cap=None) | +0.0764 | +0.1528 | +0.1395 | +0.1461 | +0.0899 |

## Paired bootstrap CI on per-pair (Δ_a − Δ_b)

| metric | mean | 95% CI | p (two-sided) |
|---|---|---|---|
| delta_h1 | +0.0100 | [-0.0100, +0.0299] | 0.3992 |
| delta_hk | +0.0000 | [-0.0133, +0.0133] | 1.0000 |
| delta_rr | +0.0071 | [-0.0030, +0.0178] | 0.1758 |
| delta_prk | +0.0033 | [-0.0100, +0.0166] | 0.8232 |
| delta_grk | +0.0006 | [-0.0116, +0.0138] | 0.9800 |

