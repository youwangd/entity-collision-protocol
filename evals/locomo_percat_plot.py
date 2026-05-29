"""LoCoMo per-category Δhit@1 figure for paper §4.5.

Renders a 2-panel (ST | Hash) figure of paired Δhit@1 vs BM25-only baseline
across LoCoMo categories c1..c5 at vw ∈ {0.3, 0.5, 0.7}, with 95% paired
bootstrap CIs. The story this figure tells: the dense-fusion lift seen on
synthetic entity-collision corpora **does not replicate** on real LoCoMo —
ST is at best neutral on c1-c3 and CI-negative on c4-c5; Hash is uniformly
null or CI-negative.

Reads:
  bench/results/locomo10_ht_sweep_ci_percat.json
  bench/results/locomo10_st_sweep_ci_percat.json

Writes:
  bench/results/locomo_percat_paper_figure.png

Usage:
  python -m evals.locomo_percat_plot \
      --hash bench/results/locomo10_ht_sweep_ci_percat.json \
      --st   bench/results/locomo10_st_sweep_ci_percat.json \
      --out  bench/results/locomo_percat_paper_figure.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CATS = ["1", "2", "3", "4", "5"]
CAT_LABELS = {
    "1": "c1 single-hop",
    "2": "c2 multi-hop",
    "3": "c3 temporal",
    "4": "c4 open-domain",
    "5": "c5 adversarial",
}
VWS = [0.3, 0.5, 0.7]


def _extract(path: Path) -> dict[float, dict[str, dict]]:
    """Return {vw: {cat: {mean, lo, hi, n}}} for hit_at_1 deltas."""
    d = json.loads(path.read_text())
    out: dict[float, dict[str, dict]] = {}
    for r in d["rows"]:
        vw = r.get("vector_weight")
        if vw not in VWS:
            continue
        pc = r["ci"]["per_category_delta"]
        out[vw] = {}
        for c, m in pc.items():
            h = m["hit_at_1"]
            out[vw][c] = {
                "mean": h["mean"],
                "lo": h["ci_lo"],
                "hi": h["ci_hi"],
                "n": m["n"],
            }
    return out


def _plot_panel(ax, data: dict[float, dict[str, dict]], title: str) -> None:
    n_vw = len(VWS)
    width = 0.8 / n_vw
    x = np.arange(len(CATS))
    colors = {0.3: "#4c78a8", 0.5: "#f58518", 0.7: "#54a24b"}

    for i, vw in enumerate(VWS):
        means, los, his = [], [], []
        for c in CATS:
            cell = data.get(vw, {}).get(c)
            if cell is None:
                means.append(0.0); los.append(0.0); his.append(0.0)
            else:
                means.append(cell["mean"])
                los.append(cell["mean"] - cell["lo"])
                his.append(cell["hi"] - cell["mean"])
        offset = (i - (n_vw - 1) / 2) * width
        ax.bar(
            x + offset, means, width=width * 0.9,
            yerr=[los, his], capsize=3,
            color=colors[vw], label=f"vw={vw}",
            edgecolor="black", linewidth=0.4,
        )

    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([CAT_LABELS[c] for c in CATS], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Δ hit@1 vs BM25-only (paired)")
    ax.set_title(title, fontsize=10)
    ax.grid(True, axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(fontsize=8, loc="lower left")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hash", dest="hash_path", type=Path,
                    default=Path("bench/results/locomo10_ht_sweep_ci_percat.json"))
    ap.add_argument("--st", dest="st_path", type=Path,
                    default=Path("bench/results/locomo10_st_sweep_ci_percat.json"))
    ap.add_argument("--out", type=Path,
                    default=Path("bench/results/locomo_percat_paper_figure.png"))
    args = ap.parse_args()

    h = _extract(args.hash_path)
    s = _extract(args.st_path)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    _plot_panel(axes[0], s, "ST MiniLM-384 — LoCoMo per-category Δhit@1")
    _plot_panel(axes[1], h, "HashTrigram-256 — LoCoMo per-category Δhit@1")
    fig.suptitle(
        "Synthetic entity-collision lift does NOT replicate on real LoCoMo "
        "(95% paired bootstrap CIs, n=1978)",
        fontsize=10, y=1.00,
    )
    fig.tight_layout()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
