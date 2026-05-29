#!/usr/bin/env bash
# Purity threshold sweep — 3-seed paired-bootstrap CI per purity point.
# Runs purity_min ∈ {0.5, 0.6, 0.7, 0.8, 0.9} on multi-entity-hard.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
mkdir -p bench/results/purity_sweep
for p in 0.5 0.6 0.7 0.8 0.9; do
  echo "=== purity_min=${p} ==="
  PYTHONPATH=. python evals/multi_entity_hard_typed_arms.py \
    --n-facts 500 --n-sessions 25 --seeds 1 2 3 \
    --type-purity-min "${p}" \
    --out "bench/results/purity_sweep/typed_3seed_p${p}.json" 2>&1 | tail -15
done
echo "=== sweep complete ==="
