# §94c-appraisal-bound-multihop-CI — paired bootstrap (cap=0.30 − cap=None) on n_gold≥2

Dataset: bench/data/locomo10.json (max_instances=2, k=10, embedder=hashtrigram).
n_paired_total=301 | n_paired_multihop=41 | resamples=10000 | seed=42 | wall=63.15s.

## Multi-hop headline (point estimates per arm)

| arm | n | Δprk | Δgrk |
|---|---|---|---|
| a (cap=0.30) | 41 | -0.0976 | -0.0699 |
| b (cap=None) | 41 | -0.0976 | -0.0496 |

## Paired bootstrap CI on per-pair (Δ_a − Δ_b), n_gold≥2

| metric | mean | 95% CI | p (two-sided) |
|---|---|---|---|
| delta_h1 | +0.0000 | [+0.0000, +0.0000] | 1.0000 |
| delta_hk | -0.0244 | [-0.0732, +0.0000] | 0.7396 |
| delta_rr | +0.0039 | [-0.0072, +0.0145] | 0.4700 |
| delta_prk | +0.0000 | [-0.0732, +0.0732] | 1.0000 |
| delta_grk | -0.0203 | [-0.0650, +0.0244] | 0.4396 |

