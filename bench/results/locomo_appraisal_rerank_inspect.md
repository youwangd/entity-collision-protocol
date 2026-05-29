### §94c-decompose-suffix-CI follow-up — `appraisal` re-rank inspector (max_instances=2, k=10, embedder=HashTrigram-256, n_questions=301)

Anchor: **S6_+merge_persist** vs `S6_+merge_persist + appraisal`. Each row is a per-question rank movement bin (A=anchor, B=anchor+probe). Salience gap = displacing rank-1 item's salience minus the displaced gold's salience under A.

| movement_bin | count |
| --- | ---: |
| `stable_rank1` | 159 |
| `stable_within_topk` | 79 |
| `absent_both` | 32 |
| `worsened_within_topk` | 14 |
| `improved_within_topk` | 8 |
| `lost_rank1` | 5 |
| `entered_topk` | 3 |
| `gained_rank1` | 1 |

**Salience gap (displacing − displaced_gold)** — if positive, appraisal rewards the wrong item.
  n=136  mean=+0.1404  median=+0.0000  min=+0.0000  p25=+0.0000  p75=+0.3450  max=+0.7000

**Scherer relevance gap (displacing − gold).** n=106  mean=+0.0226  median=+0.0000

**Movement bins by category** — does any bin (esp. `lost_rank1`) cluster in one category?

| category | total | `stable_rank1` | `stable_within_topk` | `absent_both` | `worsened_within_topk` | `improved_within_topk` | `lost_rank1` | `entered_topk` | `gained_rank1` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `4` | 114 | 66 | 29 | 13 | 2 | 3 | 1 | 0 | 0 |
| `5` | 71 | 44 | 16 | 4 | 2 | 2 | 3 | 0 | 0 |
| `2` | 63 | 33 | 19 | 2 | 5 | 1 | 1 | 1 | 1 |
| `1` | 42 | 15 | 11 | 9 | 4 | 1 | 0 | 2 | 0 |
| `3` | 11 | 1 | 4 | 4 | 1 | 1 | 0 | 0 | 0 |

**Lost vs gained rank-1 by category** (asymmetry = displacing − surfacing):

| category | lost_rank1 | gained_rank1 | net |
| --- | ---: | ---: | ---: |
| `5` | 3 | 0 | -3 |
| `2` | 1 | 1 | +0 |
| `4` | 1 | 0 | -1 |

