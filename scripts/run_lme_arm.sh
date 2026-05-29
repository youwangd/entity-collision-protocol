#!/usr/bin/env bash
# run_lme_arm.sh — duplicate-safe wrapper around evals/longmemeval_adapter
#
# Why: cron ticks have, in the past, raced two identical adapter invocations
# on the same --out path, wasting CPU-hours and producing torn JSON. This
# wrapper:
#   (1) refuses to start if any other process is already running with the
#       same --out basename in its argv (pgrep match);
#   (2) takes an flock on the output path so even simultaneous launches
#       under pgrep's blind spot (~50 ms race) are serialized.
#
# Usage:
#   scripts/run_lme_arm.sh --out evals/results/lme_n500_st_vw03_baseline.json \
#                          --max-instances 500 --k 10 --arm baseline \
#                          --embed st --vector-weight 0.3 \
#                          --dataset $LONGMEMEVAL_PATH/longmemeval_s.json
#
# All flags are passed through to `python -m evals.longmemeval_adapter`.
# Activates .venv automatically. Exit codes:
#   0  success (or duplicate-detected, treated as no-op)
#   2  --out missing from argv
#   3  duplicate run already in flight (informational; not an error)

set -euo pipefail

cd "$(dirname "$0")/.."

# Extract --out path from argv without consuming it.
OUT=""
args=("$@")
for ((i = 0; i < ${#args[@]}; i++)); do
    if [[ "${args[$i]}" == "--out" ]]; then
        OUT="${args[$((i + 1))]}"
        break
    fi
done

if [[ -z "$OUT" ]]; then
    echo "[run_lme_arm] ERROR: --out <path> is required in argv" >&2
    exit 2
fi

OUT_BASE="$(basename "$OUT")"

# pgrep guard: case-sensitive substring match against full command line.
if pgrep -fa "longmemeval_adapter.*${OUT_BASE}" >/dev/null 2>&1; then
    echo "[run_lme_arm] DUPLICATE: an adapter run targeting ${OUT_BASE} is already in flight" >&2
    pgrep -fa "longmemeval_adapter.*${OUT_BASE}" >&2 || true
    exit 3
fi

# Activate venv.
# shellcheck disable=SC1091
source .venv/bin/activate

# flock the output path to serialize even tighter races. Lock file lives
# next to the output and is cleaned up on graceful exit.
LOCK="${OUT}.lock"
mkdir -p "$(dirname "$LOCK")"

exec 9>"$LOCK"
if ! flock -n 9; then
    echo "[run_lme_arm] DUPLICATE: another process holds the flock on ${LOCK}" >&2
    exit 3
fi

trap 'rm -f "$LOCK"' EXIT

echo "[run_lme_arm] starting at $(date -u +%FT%TZ): out=${OUT}" >&2
exec python -m evals.longmemeval_adapter "$@"
