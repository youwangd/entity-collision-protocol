# §94d-tau-CI — paired bootstrap (tau=0.30 − tau=0.05) @ full pipeline, synthesis=True

Dataset: bench/data/locomo10.json (max_instances=2, k=10, embedder=hashtrigram, min_supports=2).
n_paired=301 | resamples=10000 | seed=42 | wall=70.65s.

## Headline (point estimates)

| arm | Δh@1 | Δh@k | Δprk | Δgrk | ΔMRR |
|---|---|---|---|---|---|
| a (tau=0.30) | +0.0731 | +0.1462 | +0.1329 | +0.1394 | +0.0874 |
| b (tau=0.05) | +0.0731 | +0.1528 | +0.1395 | +0.1461 | +0.0884 |

## Paired bootstrap CI on per-pair (Δ_a − Δ_b)

| metric | mean | 95% CI | p (two-sided) |
|---|---|---|---|
| delta_h1 | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| delta_hk | -0.0066 | [-0.0166, +0.0000] | 0.2664 |
| delta_rr | -0.0010 | [-0.0032, +0.0011] | 0.3524 |
| delta_prk | -0.0066 | [-0.0166, +0.0000] | 0.2664 |
| delta_grk | -0.0066 | [-0.0166, +0.0000] | 0.2664 |

