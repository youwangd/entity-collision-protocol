#!/usr/bin/env bash
# §D15 live re-run analysis: paired bootstrap on gated_pref + gated_kuPref vs baseline.
# Run this once both arm JSONs exist under bench/results/.
set -euo pipefail
cd "$(dirname "$0")/.."
source .venv/bin/activate

BASE=bench/results/lme_d15_baseline_n240.json
PREF=bench/results/lme_d15_gated_pref_n240.json
KUPF=bench/results/lme_d15_gated_kuPref_n240.json
OUT=bench/results/lme_d15_arms_delta.json

missing=0
for f in "$BASE" "$PREF" "$KUPF"; do
  if [[ ! -f "$f" ]]; then
    echo "[d15.finalize] MISSING: $f" >&2
    missing=1
  fi
done
if [[ "$missing" == "1" ]]; then
  echo "[d15.finalize] aborting — wait for D15 live arms to land." >&2
  exit 2
fi

echo "[d15.finalize] paired bootstrap → $OUT"
python -m evals.lme_compare_arms \
  --baseline "$BASE" \
  --arms "$PREF" "$KUPF" \
  --out "$OUT" \
  --resamples 5000 --seed 42

echo
echo "[d15.finalize] headline:"
python - <<'PY'
import json, pathlib
data = json.loads(pathlib.Path("bench/results/lme_d15_arms_delta.json").read_text())
print(f"baseline n={data['n_baseline']}")
for name, blk in data["arms"].items():
    o = blk["overall"]
    d1, dk = o["delta_hit_at_1"], o["delta_hit_at_k"]
    print(f"  {name}: n={o['n']}  Δh@1={d1['mean']:+.4f} CI=[{d1['ci_lo']:+.4f},{d1['ci_hi']:+.4f}]"
          f"  Δh@10={dk['mean']:+.4f} CI=[{dk['ci_lo']:+.4f},{dk['ci_hi']:+.4f}]")
PY
