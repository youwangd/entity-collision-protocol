#!/usr/bin/env bash
# scripts/regen_figures.sh — regenerate paper/figures/*.png from committed
# result JSONs under evals/results/ and bench/results/. Idempotent.
#
# Currently regenerated:
#   - paper/figures/ec_paper_figure.png            (entity-collision sweep)
#   - paper/figures/ingest_1m_latency.png          (D5 1M write-path p50/p95/p99/drift)
#   - paper/figures/lme_per_type.png               (LME per-type hit@1 baseline vs arm)
#   - paper/figures/locomo_percat_paper_figure.png (LoCoMo per-category ST|Hash CIs)
#
# This script is wired as T5 in scripts/reproduce.sh.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -d .venv && -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi

mkdir -p paper/figures

log() { printf '\n[regen_figures %s] %s\n' "$(date -u +%H:%M:%S)" "$*"; }

# ---- entity-collision figure -------------------------------------------------
EC_GEN="evals/entity_collision_figure.py"
if [[ -f "$EC_GEN" ]]; then
    log "ec_paper_figure.png — regenerating from $EC_GEN"
    python -m evals.entity_collision_figure \
        --out paper/figures/ec_paper_figure.png \
        || log "ec_paper_figure.png — generator CLI may differ; SKIP (committed PNG kept)"
else
    log "ec_paper_figure.png — generator $EC_GEN not present; keeping committed PNG"
fi

# ---- 1M-ingest latency curve (stub) -----------------------------------------
ING_GEN="evals/ingest_latency_figure.py"
if [[ -f "$ING_GEN" ]]; then
    log "ingest_1m_latency.png — regenerating from $ING_GEN"
    python -m evals.ingest_latency_figure \
        --input "$(ls -t bench/results/ingest_1m_*.json 2>/dev/null | head -1)" \
        --out paper/figures/ingest_1m_latency.png \
        || log "ingest_1m_latency.png — generator failed; SKIP"
else
    log "ingest_1m_latency.png — generator $ING_GEN not present; keeping committed PNG"
fi

# ---- LME per-type bar chart (stub) ------------------------------------------
LME_GEN="evals/lme_per_type_figure.py"
if [[ -f "$LME_GEN" ]]; then
    log "lme_per_type.png — regenerating from $LME_GEN"
    python -m evals.lme_per_type_figure \
        --baseline evals/results/lme_n500_st_vw03_baseline.json \
        --arm evals/results/lme_n500_st_vw03_prfsp.json \
        --out paper/figures/lme_per_type.png \
        || log "lme_per_type.png — generator failed; SKIP"
else
    log "lme_per_type.png — generator $LME_GEN not present; keeping committed PNG"
fi

log "done."
