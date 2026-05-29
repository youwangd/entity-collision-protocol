#!/usr/bin/env bash
# Run real LoCoMo with ST embedder across the four arms.
# Mirrors the BM25-only sweep in bench/results/locomo_real_n10_*.json
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export LOCOMO_PATH="${LOCOMO_PATH:-$PWD/data/locomo/locomo10.json}"
mkdir -p bench/results
for ARM in baseline prf share_prior both; do
  OUT="bench/results/locomo_real_n10_st_${ARM}.json"
  if [ -f "$OUT" ]; then
    echo "[skip] $OUT exists"
    continue
  fi
  echo "[run] arm=$ARM -> $OUT"
  python -m evals.locomo_adapter \
    --embedder st \
    --arm "$ARM" \
    --out "$OUT" \
    2>&1 | tail -5
done
echo "[done] all arms"
ls -la bench/results/locomo_real_n10_st_*.json
