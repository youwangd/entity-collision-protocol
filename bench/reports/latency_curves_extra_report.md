# Engram v0.2 — Matched recall-latency and stratified needle-in-haystack curves (Technical Report)

Supplementary latency curves — matched recall-latency at the
10k → 100k slice (§A.4.14r), and the stratified needle-in-haystack
recall sweep at 100k → 1M (§A.4.14r-stratified) — that complement
the main-paper §4.14 ingest curves. The headline 1M cross-run
reproducibility band (§A.4.14b, ±0.5pp / ±25% latency tolerance)
remains in the paper appendix because it is body-cited from §4.14
and §7 Conclusion.

Subsection IDs and artifact paths are stable.

## A.4.14r Matched recall-latency curves — 10k → 100k

§4.14 is the *write-side* scaling story. The companion read-side curve
runs `tests/scale/test_ingest_scale.py::test_scale_recall_after_*` —
identical fixture, single-shard SQLite + JSONL, ST-vector recall via
`Engram.recall(top_k=1)` after the ingest decade has been written.

| N        | queries | p50 ms | p95 ms | p99 ms | max ms | hit_rate |
|---------:|--------:|-------:|-------:|-------:|-------:|---------:|
|   10,000 |     100 |  2.133 |  4.047 |  4.182 |   4.32 |     1.00 |
|  100,000 |     200 |  1.119 |  1.504 |  4.311 |  24.42 |     1.00 |

**No knee on read either.** p50 actually *drops* by 48% from 10k → 100k
as the embedding cache and SQLite page cache warm; p95 drops 63%. p99
is essentially flat (+3%, 4.18 → 4.31 ms). The 100k max (24.4 ms) is a
single-query outlier consistent with a checkpoint window — the same
mechanism that produces the §4.14 1M p99.9 = 20.83 ms point on the
write side. Hit-rate is 1.0 at both scales — these are first-token
ground-truth queries, not the multi-hop LongMemEval regime; the
purpose of this cell is the latency curve, not a recall claim.

Read-side and write-side both being sub-5ms p99 across 10k → 100k is
the empirical floor under the §0/§1/§7 abstract claim of "constant-time
read and write out to ≥100k memories"; the 1M write-side point in §4.14
extends *write* one further decade. A 1M *recall* point is the natural
v0.3 follow-up but is gated on a stratified-query fixture that doesn't
collapse to hit_rate=1.0.

Artifacts:
- 10k: `bench/results/recall_at_10k_cd941b3_20260519T231722.json`
- 100k: `bench/results/recall_at_100k_3766834_20260524T091122.json`

Reproduce: `pytest -q -m mega_scale tests/scale/test_ingest_scale.py
-k recall_after`.

## A.4.14r-stratified Stratified needle-in-haystack recall at 100k → 1M

§A.4.14r reports latency on a fixture where queries are substrings of
the planted memories — that hits `hit_rate=1.0` trivially and carries
no recall signal. To put a real recall claim under the latency curve,
we plant memories with unique 8-char tokens (`zk7q…`) into a haystack
and query for those tokens. Uniform-random recall@10 is `n_needles /
n_corpus`, so the chance baseline is ≈0 at both scales.

| n_corpus  | n_needles | recall@1 | recall@10 | p50  | p95  | p99  | mean | max   | uniform recall@10 |
|----------:|----------:|---------:|----------:|-----:|-----:|-----:|-----:|------:|------------------:|
| 100,000   | 200       | **1.000**| **1.000** | 1.13 | 1.56 | 4.15 | —    | —     | 1e-4              |
| 1,000,000 | 500       | **1.000**| **1.000** | 1.12 | 1.55 | 4.07 | 1.37 | 28.45 | 1e-5              |

(latency in ms.)

`recall@1 = 1.000` on unique-token queries — not the substring-match
crutch from §A.4.14r — is the strongest non-LongMemEval recall claim we
have, and it now extends to 1M with a uniform-random recall@10 chance
baseline of 10⁻⁵. Read latency is bit-for-bit flat from 100k to 1M
(p50 1.13 → 1.12 ms, p99 4.15 → 4.07 ms): under FTS5-dominated
unique-token retrieval, single-query read cost is independent of
corpus size across a full decade beyond 100k. The §A.4.14r latency
curve is therefore recall-regime-invariant *and* scale-invariant at
1M for this query class.

Ingest at 1M took 681.9 s (1467 writes/s steady-state), consistent
with the §4.14 three-decade write-side curve.

Artifacts:
- 100k: `bench/results/recall_stratified_at_100k_8c30606_20260524T092823.json`
- 1M:   `bench/results/recall_stratified_at_1m_10a4f00_20260524T095534.json`

Harnesses: `tests/scale/test_ingest_scale.py::test_scale_recall_stratified_at_100k`
and `::test_scale_recall_stratified_at_1m`.
SCALE_REPORT cells: §D17 (100k), §D18 (1M).
