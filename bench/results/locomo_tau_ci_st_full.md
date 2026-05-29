# §94d-tau-CI — paired bootstrap (tau=0.30 − tau=0.05) @ full pipeline, synthesis=True

Dataset: bench/data/locomo10.json (max_instances=10, k=10, embedder=st, min_supports=2).
n_paired=1978 | resamples=10000 | seed=42 | wall=1105.99s.

## Headline (point estimates)

| arm | Δh@1 | Δh@k | Δprk | Δgrk | ΔMRR |
|---|---|---|---|---|---|
| a (tau=0.30) | +0.0768 | +0.0829 | +0.0799 | +0.0778 | +0.0694 |
| b (tau=0.05) | +0.0768 | +0.0819 | +0.0794 | +0.0772 | +0.0696 |

## Paired bootstrap CI on per-pair (Δ_a − Δ_b)

| metric | mean | 95% CI | p (two-sided) |
|---|---|---|---|
| delta_h1 | +0.0000 | [-0.0025, +0.0025] | 1.0000 |
| delta_hk | +0.0010 | [-0.0010, +0.0030] | 0.4416 |
| delta_rr | -0.0002 | [-0.0017, +0.0012] | 0.7464 |
| delta_prk | +0.0005 | [-0.0015, +0.0025] | 0.8234 |
| delta_grk | +0.0007 | [-0.0011, +0.0027] | 0.5212 |

