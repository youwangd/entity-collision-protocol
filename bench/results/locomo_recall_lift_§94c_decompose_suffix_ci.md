### §94c-decompose-suffix-CI — localize S6→S7 Δgrk bite (max_instances=2, k=10, embedder=hashtrigram, synthesis=False, resamples=10000)

Anchor: **S6** = [extraction, fact_extraction, interference, schema_update, somatic_marking, mechanical_merge]. Each row probes S6 + one of the 7 stages bundled into S6→S7. The bottom 'bundle' row reports the full S6→S7 CI for reference.

| probe | added | n | Δh@1 mean (CI) p | Δh@k mean (CI) p | ΔMRR mean (CI) p | Δprk mean (CI) p | Δgrk mean (CI) p |
| --- | --- | ---:| --- | --- | --- | --- | --- |
| `S6_+merge_persist -> S6+deduplication` | `deduplication` | 301 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S6_+merge_persist -> S6+appraisal` | `appraisal` | 301 | +0.0133 [+0.0000, +0.0299] p=0.132 | -0.0100 [-0.0233, +0.0000] p=0.100 | +0.0053 [-0.0024, +0.0140] p=0.193 | -0.0066 [-0.0166, +0.0000] p=0.269 | -0.0075 [-0.0166, -0.0008] p=0.038★ |
| `S6_+merge_persist -> S6+emotion_tagging` | `emotion_tagging` | 301 | -0.0033 [-0.0100, +0.0000] p=0.738 | +0.0000 [+0.0000, +0.0000] p=1.000 | -0.0016 [-0.0054, +0.0005] p=0.374 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S6_+merge_persist -> S6+decay` | `decay` | 301 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S6_+merge_persist -> S6+suppression` | `suppression` | 301 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S6_+merge_persist -> S6+temperament_drift` | `temperament_drift` | 301 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S6_+merge_persist -> S6+mood_update` | `mood_update` | 301 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 | +0.0000 [+0.0000, +0.0000] p=1.000 |
| `S6_+merge_persist -> S7_full_default` | `(bundle)` | 301 | +0.0100 [-0.0066, +0.0266] p=0.331 | -0.0100 [-0.0233, +0.0000] p=0.100 | +0.0039 [-0.0047, +0.0131] p=0.383 | -0.0066 [-0.0166, +0.0000] p=0.269 | -0.0075 [-0.0166, -0.0008] p=0.038★ |

★ = 95% CI excludes zero. Pairing key = (sample_id, question, category). Method = percentile bootstrap on per-pair (Δ_S6 − Δ_{S6+x}).
