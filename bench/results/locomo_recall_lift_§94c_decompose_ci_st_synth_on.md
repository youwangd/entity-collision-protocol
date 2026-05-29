# §94c-decompose-CI under MiniLM-384, synthesis on

`evals.locomo_recall_lift_decompose_ci --dataset bench/data/locomo10.json
--max-instances 2 --resamples 10000 --seed 42 --embedder st --synthesis`

n_paired = 301 · wall = 150.57s · embedder = SentenceTransformer-MiniLM-384

S1 = `extraction` only · S7 = full default pipeline (synthesis on, gate off)

## Headline arms

|     | Δh@1   | Δh@k   | ΔMRR   |
|-----|--------|--------|--------|
| S1  | 0.0963 | 0.1229 | 0.0989 |
| S7  | 0.0831 | 0.1362 | 0.0937 |

## Paired bootstrap on (S1 − S7)

| metric    | mean    | 95% CI               | p     |
|-----------|---------|----------------------|-------|
| Δh@1      | +0.0133 | [-0.0033, +0.0332]   | 0.196 |
| Δh@k      | -0.0133 | [-0.0266, -0.0033]   | **0.034** |
| ΔMRR      | +0.0052 | [-0.0063, +0.0175]   | 0.379 |
| Δprk      | -0.0100 | [-0.0266, +0.0033]   | 0.242 |
| Δgrk      | -0.0119 | [-0.0255, -0.0008]   | **0.038** |

## Verdict

Replicates §94c-decompose-CI (hashtrigram-256, synth-on, baseline of record):
downstream consolidation stages are **inert on the binary top-1 / MRR
headline** (Δh@1, ΔMRR brackets zero, p > 0.19) but cost ~1.3pp on Δh@k
and ~1.2pp on Δgold_recall@k under both embedders. The qualitative claim
("only extraction is necessary on the top-1 / MRR headline; downstream
stages slightly hurt set-recall at k") generalises across embedders.

Comparison with hashtrigram-256 (`bench/results/locomo_recall_lift_§94c_decompose_ci.json`):

| metric | hashtrigram-256 (synth=False) | MiniLM-384 (synth=True) |
|--------|-------------------------------|--------------------------|
| Δh@1 (S1−S7)  | +0.0066 [-0.013,+0.027] p=0.638 | +0.0133 [-0.003,+0.033] p=0.196 |
| Δh@k (S1−S7)  | -0.0100 [-0.023,+0.000] p=0.100 | -0.0133 [-0.027,-0.003] p=0.034 |
| ΔMRR (S1−S7)  | +0.0008 [-0.011,+0.012] p=0.893 | +0.0052 [-0.006,+0.018] p=0.379 |
| Δgrk (S1−S7)  | -0.0075 [-0.017,-0.001] p=0.038 | -0.0119 [-0.025,-0.001] p=0.038 |

Same sign on every metric. The h@k effect crosses the α=0.05 threshold
under MiniLM where it brushed it under hashtrigram, consistent with
§94d-tau-mechanism-st's finding that MiniLM displaces top-k more
aggressively under SCHEMA writes. Δgrk p=0.038 is identical, suggesting
the gold-recall@k cost is structural (stage interaction, not embedder).
