# §94d-tau-CI — paired bootstrap (tau=0.30 − tau=0.05) @ full pipeline, synthesis=True

Dataset: bench/data/locomo10.json (max_instances=2, k=10, embedder=st, min_supports=2).
n_paired=301 | resamples=10000 | seed=42 | wall=156.45s.

## Headline (point estimates)

| arm | Δh@1 | Δh@k | Δprk | Δgrk | ΔMRR |
|---|---|---|---|---|---|
| a (tau=0.30) | +0.0764 | +0.1362 | +0.1296 | +0.1300 | +0.0886 |
| b (tau=0.05) | +0.0864 | +0.1362 | +0.1329 | +0.1322 | +0.0965 |

## Paired bootstrap CI on per-pair (Δ_a − Δ_b)

| metric | mean | 95% CI | p (two-sided) |
|---|---|---|---|
| delta_h1 | -0.0100 | [-0.0233, +0.0000] | 0.0984 |
| delta_hk | +0.0000 | [-0.0100, +0.0100] | 1.0000 |
| delta_rr | -0.0079 | [-0.0155, -0.0022] | 0.0002 |
| delta_prk | -0.0033 | [-0.0166, +0.0066] | 0.7820 |
| delta_grk | -0.0022 | [-0.0116, +0.0050] | 0.6758 |

