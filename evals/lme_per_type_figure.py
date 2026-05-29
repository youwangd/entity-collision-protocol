"""Generate paper/figures/lme_per_type.png from two LongMemEval result JSONs.

Compares per-question-type session_hit@1 between a baseline run and a
contrast arm (typically PRF×SP). Wired into scripts/regen_figures.sh.

Reads the schema produced by `evals/longmemeval_adapter.py`:

    {
      "session_hit_at_1": 0.xxx,
      "per_type_session_hit_at_1": {<type>: float, ...},
      "per_type_n": {<type>: int, ...},
      "arm": "baseline" | "prfsp",
      ...
    }

Restrained, monospaced, low-chrome. Grouped bars per type, paired-baseline
delta annotated above each pair. Type order is fixed left→right by descending
n in the baseline so the most-evidenced bars sit nearest the y-axis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _short(t: str) -> str:
    """Fit type names on x-axis without crowding."""
    return {
        "single-session-user": "ss-user",
        "single-session-assistant": "ss-asst",
        "single-session-preference": "ss-pref",
        "multi-session": "multi",
        "temporal-reasoning": "temporal",
        "knowledge-update": "knowl-upd",
    }.get(t, t)


def _plot(baseline: dict, arm: dict, out: Path) -> None:
    base_per = baseline["per_type_session_hit_at_1"]
    arm_per = arm["per_type_session_hit_at_1"]
    per_n = baseline.get("per_type_n", {})
    types = sorted(base_per.keys(), key=lambda t: -per_n.get(t, 0))

    base_vals = [base_per[t] for t in types]
    arm_vals = [arm_per.get(t, 0.0) for t in types]

    overall_b = baseline["session_hit_at_1"]
    overall_a = arm["session_hit_at_1"]

    base_label = baseline.get("arm", "baseline")
    arm_label = arm.get("arm", "arm")

    x = np.arange(len(types))
    width = 0.38

    fig, ax = plt.subplots(figsize=(8.0, 3.4), dpi=150)
    _ = ax.bar(x - width / 2, base_vals, width,
                label=f"{base_label} (overall {overall_b:.3f})",
                color="#888")
    _ = ax.bar(x + width / 2, arm_vals, width,
                label=f"{arm_label} (overall {overall_a:.3f})",
                color="#222")

    # delta annotations
    for xi, bv, av in zip(x, base_vals, arm_vals):
        d = av - bv
        sign = "+" if d >= 0 else ""
        color = "#1a7f37" if d >= 0 else "#a00000"
        top = max(bv, av)
        ax.annotate(f"{sign}{d*100:.1f}pp",
                    (xi, top), textcoords="offset points",
                    xytext=(0, 4), ha="center", fontsize=7, color=color)

    # x labels with n
    xt = [f"{_short(t)}\nn={per_n.get(t,0)}" for t in types]
    ax.set_xticks(x)
    ax.set_xticklabels(xt, fontsize=7.5)
    ax.set_ylabel("session_hit@1", fontsize=8)
    ax.set_ylim(0, max(max(base_vals), max(arm_vals)) * 1.18 + 0.02)
    ax.set_title("LongMemEval-S per-type session_hit@1  "
                 f"(n={baseline.get('n_instances','?')}, k={baseline.get('k','?')})",
                 fontsize=9, loc="left")
    ax.grid(True, axis="y", linestyle=":", alpha=0.4)
    ax.tick_params(axis="y", labelsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(fontsize=7.5, frameon=False, loc="upper right")

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", required=True, help="baseline LME results JSON")
    ap.add_argument("--arm", required=True, help="contrast-arm LME results JSON")
    ap.add_argument("--out", required=True, help="output PNG path")
    args = ap.parse_args()

    baseline = json.loads(Path(args.baseline).read_text())
    arm = json.loads(Path(args.arm).read_text())
    _plot(baseline, arm, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
