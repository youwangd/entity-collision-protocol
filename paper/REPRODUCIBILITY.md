# Reproducibility — Engram v0.3

This document gives the exact recipe for regenerating every headline number
in §4 of the paper. The single entry point is `scripts/reproduce.sh`; this
file documents the seeds, CLI flags, expected wall-clock, and acceptance
tolerance for each table.

**Acceptance criterion (paper-wide):** every headline confidence-interval
midpoint reproduces to within ±0.5 percentage points across runs on the
pinned environment. Wider drifts indicate either a non-determinism bug or
an environment skew — both must be tracked down before a camera-ready.

---

## 0. Environment

- **Python:** 3.11.9 (pinned in `Dockerfile`)
- **Pip extras:** `engram[all,dev]` + `hypothesis`, `pytest-xdist`
- **spaCy model:** `en_core_web_sm` (pulled at image build)
- **OS:** Debian bookworm slim base; reproducible on Linux x86_64.
  macOS/arm64 has been spot-checked but is not the canonical target.

```bash
docker build -t engram-repro .
docker run --rm -it -v "$PWD/out:/engram/out" engram-repro
```

Or natively:

```bash
git clone https://example.com/anonymous/repo && cd engram
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e '.[all,dev]' hypothesis pytest-xdist
python -m spacy download en_core_web_sm
./scripts/reproduce.sh
```

---

## 1. Headline tables

Each table is gated by a `--tables Tn` flag in `reproduce.sh`. The table
ID, source paper section, expected runtime, and seed are listed below.

| ID | Paper section          | What it produces                                           | Wall   | Seed |
|----|------------------------|------------------------------------------------------------|--------|------|
| T0 | venue/anonymization     | EMNLP review-PDF anonymization gate (paper/00..70_*.md)    | <1 s   | n/a  |
| T1 | §4 entity-collision     | K∈{1,2,4,8} colliding-fact sweep (BM25 vs vector vs hybrid) | ~5 min | 0    |
| T2 | §A.4.15k v0.3 defaults    | Regression guard pinning the v0.3 operating point          | <1 min | n/a  |
| T3 | §D5 1M-ingest           | p50/p95/p99 ingest latency, head/tail drift                | 60–90m | 0    |
| T4 | §4.5 LongMemEval head   | n=500 baseline vs PRF×SP @ v0.3 defaults                   | ~3.5 h | n/a* |
| T5 | figures                 | Vector PDF regen for `paper/figures/`                      | ~1 min | n/a  |

The wired modules / tests are:

- **T0:** `python scripts/check_anon.py` — fails build (exit 1) if any
  of (internal-identifier-list-redacted)/
  internal hostnames/12-digit AWS account IDs/`github.com/anonymous`
  appears in the paper body. HTML comments are pandoc-stripped; pass
  `--strict` to scan inside them too. Mandatory under `paper/VENUE.md`
  before review-PDF build.
- **T1:** `python -m evals.entity_collision_sweep`
- **T2:** `pytest tests/unit/test_v0_3_defaults_locked.py`
- **T3:** `pytest -m mega_scale tests/scale/test_ingest_1m.py`
- **T4:** `python -m evals.longmemeval_adapter`
- **T5:** `scripts/regen_figures.sh` (regenerates `ec_paper_figure.png`,
  `ingest_1m_latency.png`, `lme_per_type.png`)

\* LME runs are deterministic given a fixed dataset slice; the dataset
itself is pinned by the SHA-256 below.

### T4 — LongMemEval dataset

LongMemEval is fetched from the official HuggingFace mirror:

```
HF dataset:  xiaowu0162/longmemeval
File:        longmemeval_s.json
Size:        278,025,796 bytes
SHA-256:     08d8dad4be43ee2049a22ff5674eb86725d0ce5ff434cde2627e5e8e7e117894
```

Set `LONGMEMEVAL_PATH=/path/to/longmemeval/dir` (the dir containing
`longmemeval_s.json`); `reproduce.sh T4` will skip cleanly if unset.

---

## 2. CLI knobs at the v0.3 operating point

These are the values defended in §4.5 with paired bootstrap CIs at α=0.05,
d=0.3, pool=20. Any number in §4 of the paper assumes them unless the
table explicitly varies one knob.

```
--vector-weight 0.3
--qe-dominance 0.3
--sp-alpha 0.05
--sp-pool 20
--qe-anchor-share-max 0.5
```

The same operating point is encoded as the default in `RetrievalConfig`
(commit e428687) and pinned by `tests/unit/test_v0_3_defaults_locked.py`
(commit 2a71014).

---

## 3. Determinism notes

- **Embeddings (sentence-transformers):** deterministic on CPU at fp32.
  We do not use GPU for any headline number to avoid CUDA non-determinism.
- **FAISS / sqlite-vec:** deterministic given fixed insertion order; we
  fix insertion order via a sorted iteration over input keys.
- **Hypothesis:** every property test pins a `derandomize=True` profile
  in `tests/conftest.py`. Failure shrinking is reproducible.
- **Bootstrap CIs:** all bootstrap resampling uses `numpy.random.default_rng(0)`.

Any test that is non-deterministic on rerun is a bug — file an issue.

---

## 4. Diffing a fresh run against the canonical refs

Each invocation of `reproduce.sh` writes to a tagged file:
`<basename>_<sha>_<UTC-timestamp>.json`. The committed canonical refs
live alongside (e.g. `lme_n500_st_vw03_baseline.json`,
`bench/results/ingest_1m_<sha>_<ts>.json` from prior runs).

`scripts/diff_results.py` is the acceptance gate. Default tolerances:

- rate metrics (`session_hit_at_1`, `session_hit_at_k`): **±0.5pp absolute**
- latency metrics (`recall_ms.*`, `ingest_ms.*`, `latency_ms.*`): **±25% relative**

Tolerances are tunable via `--rate-tol-abs` / `--latency-tol-rel`. Exit
code is 0 on pass, 1 on fail, 2 on usage error — wire it into CI.

```bash
# T4 LongMemEval acceptance check
python scripts/diff_results.py \
    evals/results/lme_n500_st_vw03_baseline.json \
    evals/results/lme_n500_v03defaults_<sha>_<ts>.json

# T3 1M-ingest acceptance check — diff a fresh run against the most recent
# committed canonical (currently bench/results/ingest_1m_7b5b578_20260524T071106.json,
# the 7-rep no-cliff series; pick the latest committed *.json under bench/results/
# matching `ingest_1m_*.json` as the reference).
python scripts/diff_results.py \
    bench/results/ingest_1m_7b5b578_20260524T071106.json \
    bench/results/ingest_1m_<sha>_<ts>.json
```

---

## 4b. BGE-large encoder-capacity falsification artifacts (§A.4.16)

The §A.4.16 encoder-capacity falsification arm (BGE-large vs MiniLM)
is reproduced by:

```bash
# Per-tag entity-collision sweeps (5 tags × {raw, _ci})
bash scripts/run_bge_sweeps.sh

# Paired bootstrap CI between BGE and MiniLM at K=16
python scripts/ec_bge_vs_minilm_ci.py

# §A.4.16.3 — LongMemEval n=500 (full-panel) BGE-vs-default paired CI on natural data
python scripts/lme_bge_vs_minilm_n500_paired_ci.py
# (n=100 preliminary, superseded but retained for audit:
#  python scripts/lme_bge_vs_minilm_n100_paired_ci.py)
```

Artifacts (committed under `bench/results/`):

| File | Purpose |
|------|---------|
| `ec_bge_large_{preference,project,service,technical,tool}_n32_K16.json` | Per-tag BGE entity-collision raw counts |
| `ec_bge_large_{preference,project,service,technical,tool}_n32_K16_ci.json` | Per-tag BGE bootstrap CIs |
| `ec_bge_vs_minilm_ci.json` | Paired BGE-vs-MiniLM bootstrap CI (§A.4.16 headline) |
| `lme_n500_bge_large_baseline.json` | LongMemEval **full n=500** BGE baseline (§A.4.16.3 current headline) |
| `lme_n500_bge_vs_default_ci.json` | Paired BGE-vs-default-embedder bootstrap CI on the full n=500 panel (§A.4.16.3 headline; B=10000, seed=42) |
| `lme_n100_k10_baseline_bge.json` | LongMemEval n=100 BGE baseline on the 70-SSU+30-MS stratified subsample (§A.4.16.3 preliminary, superseded) |
| `lme_n100_bge_vs_default_ci.json` | Paired BGE-vs-default-embedder bootstrap CI on the n=100 subsample (§A.4.16.3 preliminary, superseded) |

Wall-clock: ~25 min for the full sweep (5 tags × CI) on a single CPU
core; ~2 h 57 min for the LongMemEval n=500 BGE arm on M4 Pro / MPS
(`ENGRAM_ST_DEVICE=mps`, `ENGRAM_ST_BATCH=256`, fp32; 21.5 s/instance
ingest, 7× faster than CPU); ~4 h 12 min for the earlier n=100 CPU
preliminary (retained for audit). Paired bootstrap CIs are
deterministic given seed=42.

---

## 5. Known caveats

- The 1M-ingest wall-clock varies ±15% across runs depending on host CPU
  thermal headroom (m6i.large noticeably faster than t3.medium). Latency
  *percentiles* are stable; *throughput* is not. Report both.
- spaCy model versions drift quietly across pip resolutions. The
  `Dockerfile` pin is the source of truth; native installs may see ±1 entity
  per document on edge cases.
