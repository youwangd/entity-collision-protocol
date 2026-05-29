#!/usr/bin/env bash
# §D15d — re-run LongMemEval PRF arm with anchor-share gate
# at thresholds {0.7, 0.5, 0.4} to confirm inertness on the +3.33 pp lift.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate
export LONGMEMEVAL_PATH="$PWD/data/longmemeval/longmemeval_s.json"
mkdir -p bench/results

for thresh in 0.7 0.5 0.4; do
  out="bench/results/lme_full500_k10_prf_anchor${thresh}.json"
  echo "=== anchor_share_max=${thresh} -> ${out} ==="
  time python -m evals.longmemeval_adapter \
    --arm prf --max-instances 500 --k 10 \
    --qe-anchor-share-max "${thresh}" \
    --out "${out}" \
  | tail -5
done

echo "=== compare arms vs baseline ==="
python -m evals.lme_compare_arms \
  --baseline bench/results/lme_full500_k10.json \
  --arms     bench/results/lme_full500_k10_prf.json \
             bench/results/lme_full500_k10_prf_anchor0.7.json \
             bench/results/lme_full500_k10_prf_anchor0.5.json \
             bench/results/lme_full500_k10_prf_anchor0.4.json \
  --out      bench/results/lme_full500_d15d_anchor_sweep.json \
  | tail -40
echo "DONE"
