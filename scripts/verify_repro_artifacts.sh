#!/usr/bin/env bash
# Reproducibility verification: every bench/results/*.json mentioned in
# the paper body (§4, §A.4, REPRODUCIBILITY.md) must exist on disk, and
# every reproduce-script reference under scripts/ or evals/ must resolve.
#
# Exit 0 = all good. Exit 1 = one or more missing.
set -u
cd "$(dirname "$0")/.."

fail=0

# 1. Artifact paths
mapfile -t arts < <(grep -hoE 'bench/results/[a-zA-Z_0-9]+\.json' \
  paper/A1_appendix_ablations.md paper/40_results.md paper/REPRODUCIBILITY.md \
  | sort -u)
for f in "${arts[@]}"; do
  if [ ! -f "$f" ]; then
    echo "MISSING $f"; fail=1
  fi
done
echo "checked ${#arts[@]} artifact paths"

# 2. Repro-script references
mapfile -t scripts < <(grep -hoE '(scripts|evals|tests/evals|tests/unit)/[a-zA-Z_0-9/]+\.(py|sh)' \
  paper/A1_appendix_ablations.md paper/40_results.md paper/REPRODUCIBILITY.md \
  | sort -u)
for s in "${scripts[@]}"; do
  if [ ! -e "$s" ]; then
    echo "MISSING $s"; fail=1
  fi
done
echo "checked ${#scripts[@]} script references"

if [ "$fail" -eq 0 ]; then
  echo "OK — all paper-cited artifacts and reproduce scripts present"
fi
exit "$fail"
