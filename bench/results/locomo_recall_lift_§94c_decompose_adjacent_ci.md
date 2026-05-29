### §94c-decompose-adjacent-CI — per-transition paired bootstrap CI (max_instances=2, k=10, embedder=hashtrigram, synthesis=False, resamples=10000)

| transition | added | n_paired | Δh@1 mean (CI) p | Δh@k mean (CI) p | ΔMRR mean (CI) p | Δprk mean (CI) p | Δgrk mean (CI) p |
| --- | --- | ---:| --- | --- | --- | --- | --- |
| `S1_extraction_only -> S2_+fact` | `fact_extraction` | 301 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S2_+fact -> S3_+interference` | `interference` | 301 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S3_+interference -> S4_+schema_update` | `schema_update` | 301 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S4_+schema_update -> S5_+somatic` | `somatic_marking` | 301 | -0.0033 [-0.0166, +0.0066] p=0.788 | +0.0000 [+0.0000, +0.0000] p=1.000 | -0.0031 [-0.0111, +0.0036] p=0.406 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S5_+somatic -> S6_+merge_persist` | `mechanical_merge` | 301 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S6_+merge_persist -> S7_full_default` | `(implicit_full_default)` | 301 | +0.0100 [-0.0066, +0.0266] p=0.331 | -0.0100 [-0.0233, +0.0000] p=0.100 | +0.0039 [-0.0047, +0.0131] p=0.383 | -0.0066 [-0.0166, +0.0000] p=0.269 | -0.0075 [-0.0166, -0.0008] p=0.038★ |

★ = 95% CI excludes zero. Pairing key = (sample_id, question, category). Method = percentile bootstrap on per-pair (Δ_{S_a} − Δ_{S_b}).
