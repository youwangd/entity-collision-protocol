### §94c-decompose-positive-control-CI — paired bootstrap on (S4 − S3), tau=0.30, min_supports=2, max_instances=2, k=10, embedder=hashtrigram, synthesis=True, resamples=10000

S3 stages = `extraction,fact_extraction,interference`  S4 stages = `extraction,fact_extraction,interference,schema_update`  n_paired = 301

| metric | mean (S4−S3) | 95% CI | p_two_sided | n | flag |
| --- | ---:| --- | ---:| ---:| ---:|
| `delta_h1` | -0.0033 | [-0.0100, +0.0000] | 0.742 | 301 |  |
| `delta_hk` | -0.0100 | [-0.0233, +0.0000] | 0.096 | 301 |  |
| `delta_rr` | -0.0031 | [-0.0078, +0.0006] | 0.112 | 301 |  |
| `delta_prk` | -0.0100 | [-0.0233, +0.0000] | 0.096 | 301 |  |
| `delta_grk` | -0.0100 | [-0.0233, +0.0000] | 0.096 | 301 |  |

**Reading.** Sign convention is S4 − S3, so positive means `schema_update` *helps* and negative means it *hurts*. ★ = 95% CI excludes zero. If every metric brackets zero, the lone non-trivial signal from the positive-control sweep is within noise — file `schema_update` as formally inert and recommend default-disabling the stage.
