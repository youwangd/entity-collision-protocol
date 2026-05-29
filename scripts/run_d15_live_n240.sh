#!/usr/bin/env bash
# §D15 live re-run at n=240 stratified, shuffle-seed=42, ST embedder, vw=0.3.
# Three arms: baseline (reuse if present), gated_pref, gated_kuPref.
# Output JSONs land under bench/results/.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export LONGMEMEVAL_PATH="${LONGMEMEVAL_PATH:-$(pwd)/data/longmemeval/longmemeval_s.json}"

mkdir -p bench/results evals/logs
N=240
SEED=42
EMBED=st
VW=0.3

baseline_out=bench/results/lme_d15_baseline_n240.json
gated_pref_out=bench/results/lme_d15_gated_pref_n240.json
gated_kuPref_out=bench/results/lme_d15_gated_kuPref_n240.json

# Sanity: copy D14 baseline if it matches the same params (same dataset/embedder/seed/n).
if [[ -f bench/results/lme_d14_baseline_n240.json && ! -f "$baseline_out" ]]; then
  cp bench/results/lme_d14_baseline_n240.json "$baseline_out"
  echo "[d15] reused D14 baseline → $baseline_out"
fi

run_arm() {
  local name="$1"; local out="$2"; shift 2
  if [[ -f "$out" ]]; then
    echo "[d15] $name already present → $out (skip)"
    return 0
  fi
  echo "[d15] $name → $out @ $(date)"
  python -m evals.longmemeval_adapter \
    --max-instances "$N" --stratify --shuffle-seed "$SEED" \
    --embed "$EMBED" --vector-weight "$VW" \
    "$@" --out "$out" 2>&1 | tail -50 \
    | tee evals/logs/lme_d15_${name}.log
}

# Baseline (no PRF, no share_prior)
run_arm baseline "$baseline_out" --arm baseline
# Gated PRF only, allow={single-session-preference}
run_arm gated_pref "$gated_pref_out" --arm prf --type-allow single-session-preference
# Gated PRF only, allow={knowledge-update, single-session-preference}
run_arm gated_kuPref "$gated_kuPref_out" --arm prf --type-allow knowledge-update,single-session-preference

echo "[d15] all arms done @ $(date)"
ls -la bench/results/lme_d15_*.json
