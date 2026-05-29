#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
LOG=bench/results/bge_sweep_runner.log
: > "$LOG"
echo "[$(date -u +%H:%M:%S)] BGE-large sweep runner started" >> "$LOG"
for TAG in service tool preference project technical; do
    OUT="bench/results/ec_bge_large_${TAG}_n32_K16.json"
    if [ -f "$OUT" ]; then
        echo "[$(date -u +%H:%M:%S)] $TAG: SKIP (exists)" >> "$LOG"
        continue
    fi
    echo "[$(date -u +%H:%M:%S)] $TAG: START" >> "$LOG"
    python -m evals.entity_collision_sweep \
        --tag "$TAG" \
        --n-entities 32 \
        --degrees 1,2,4,8,16 \
        --seed 42 \
        --embed bge_large \
        --out "$OUT" \
        >> "$LOG" 2>&1
    echo "[$(date -u +%H:%M:%S)] $TAG: DONE -> $OUT" >> "$LOG"
done
echo "[$(date -u +%H:%M:%S)] BGE-large sweep runner finished" >> "$LOG"
