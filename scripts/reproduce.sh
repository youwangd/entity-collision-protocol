#!/usr/bin/env bash
# scripts/reproduce.sh — single entry point for regenerating every headline
# number in paper/40_results.md. Idempotent; safe to re-run.
#
# Usage:
#   ./scripts/reproduce.sh             # all headline tables
#   ./scripts/reproduce.sh --quick     # skip 1M-ingest + n=500 LME (long)
#   ./scripts/reproduce.sh --tables T1 T2 ...   # specific tables only
#
# Each table writes a versioned JSON under bench/results/ or evals/results/
# tagged with the current git sha + a UTC timestamp. Diffing against the
# committed reference is the responsibility of paper/REPRODUCIBILITY.md.
#
# This script intentionally does NOT mutate the working tree. It only writes
# under bench/results/ and evals/results/.
set -euo pipefail

# ---- env ---------------------------------------------------------------------
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

SHA="$(git rev-parse --short HEAD 2>/dev/null || echo nogit)"
STAMP="$(date -u +%Y%m%dT%H%M%S)"
TAG="${SHA}_${STAMP}"

mkdir -p bench/results evals/results

# ---- args --------------------------------------------------------------------
QUICK=0
WANTED=()
while (("$#")); do
    case "$1" in
        --quick) QUICK=1; shift ;;
        --tables) shift; while (("$#")) && [[ "$1" != --* ]]; do WANTED+=("$1"); shift; done ;;
        -h|--help) sed -n '1,18p' "$0"; exit 0 ;;
        *) echo "unknown arg: $1" >&2; exit 2 ;;
    esac
done

want() {
    [[ ${#WANTED[@]} -eq 0 ]] && return 0
    local t
    for t in "${WANTED[@]}"; do [[ "$t" == "$1" ]] && return 0; done
    return 1
}

log() { printf '\n[reproduce %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ---- T0: anonymization linter (always run; cheap) ----------------------------
# Mandatory gate from paper/VENUE.md — the review PDF must not contain any
# self-identifying string. Keep this fast and unconditional.
if want T0; then
    log "T0 — anonymization linter (paper/00..70_*.md)"
    python scripts/check_anon.py
fi

# ---- T1: entity-collision discriminator sweep --------------------------------
# Generator: evals/entity_collision_sweep.py — sweeps K ∈ {1,2,4,8} colliding
# facts per entity, reports BM25-only vs vector-only vs hybrid hit@1.
if want T1; then
    log "T1 — entity-collision discriminator sweep (K∈{1,2,4,8})"
    python -m evals.entity_collision_sweep \
        --out "evals/results/entity_collision_sweep_${TAG}.json" \
        --seed 0
fi

# ---- T2: §4.15k v0.3-defaults regression guard -------------------------------
if want T2; then
    log "T2 — v0.3-defaults regression guard (LME-derived operating point)"
    python -m pytest -q tests/unit/test_v0_3_defaults_locked.py --tb=short
fi

# ---- T3: D5 1M-ingest write-path latency curve (~60–90 min) ------------------
if want T3; then
    if [[ $QUICK -eq 1 ]]; then
        log "T3 — SKIPPED in --quick mode (1M-ingest is ~60–90 min)"
    else
        log "T3 — D5 1M-ingest @ v0.3 defaults (mega_scale marker)"
        python -m pytest -q tests/scale/test_ingest_1m.py \
            -m mega_scale --tb=short \
            --junitxml="bench/results/ingest_1m_${TAG}.junit.xml"
    fi
fi

# ---- T4: LongMemEval headline (n=500, both arms, ~3.5h) ----------------------
if want T4; then
    if [[ $QUICK -eq 1 ]]; then
        log "T4 — SKIPPED in --quick mode (LME n=500 both arms is ~3.5h)"
    elif [[ -z "${LONGMEMEVAL_PATH:-}" ]]; then
        log "T4 — SKIPPED (LONGMEMEVAL_PATH not set; see paper/REPRODUCIBILITY.md §0)"
    else
        log "T4 — LongMemEval n=500 @ v0.3 defaults (baseline + PRF×SP)"
        python -m evals.longmemeval_adapter \
            --dataset "${LONGMEMEVAL_PATH}/longmemeval_s.json" \
            --max-instances 500 \
            --k 10 \
            --arm both \
            --embed st \
            --vector-weight 0.3 \
            --qe-dominance 0.3 \
            --sp-alpha 0.05 \
            --sp-pool 20 \
            --qe-anchor-share-max 0.5 \
            --out "evals/results/lme_n500_v03defaults_${TAG}.json"
    fi
fi

# ---- T5: figures regen -------------------------------------------------------
if want T5; then
    if [[ -x scripts/regen_figures.sh ]]; then
        log "T5 — regenerating paper/figures/"
        scripts/regen_figures.sh
    else
        log "T5 — SKIPPED (scripts/regen_figures.sh not present yet)"
    fi
fi

log "done. tag=${TAG}"
log "Diff against canonical refs: scripts/diff_results.py REF NEW"
log "  e.g. python scripts/diff_results.py evals/results/lme_n500_st_vw03_baseline.json evals/results/lme_n500_v03defaults_${TAG}.json"
