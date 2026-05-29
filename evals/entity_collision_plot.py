"""Entity-collision paper figure.

Three-panel layout (after BGE-large addition, 2026-05-24):
  left   = HashTrigram-256
  middle = ST MiniLM-384
  right  = BGE-large-1024

Plots Δhit@1 vs collision degree K with paired 95% bootstrap CI bands.
Reads `*_ci.json` files produced by `evals.entity_collision_ci`.

Usage:
    python -m evals.entity_collision_plot --out bench/results/ec_paper_figure.png

Adding new (tag, embedder) cells: extend SERIES_HASH / SERIES_ST / SERIES_BGE.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

# Tag → (color, marker). Shared across panels so the eye reads "this tag"
# regardless of embedder.
TAG_STYLE: dict[str, dict] = {
    "technical":  {"color": "#1b4f72", "marker": "o"},
    "service":    {"color": "#117a65", "marker": "s"},
    "preference": {"color": "#6c3483", "marker": "^"},
    "project":    {"color": "#b9770e", "marker": "D"},
    "tool":       {"color": "#a93226", "marker": "v"},
}

SERIES_HASH: list[dict] = [
    {"tag": "technical",  "files": ["bench/results/ec_sweep_hash_technical_n32_K16_ci.json"]},
    {"tag": "service",    "files": ["bench/results/ec_sweep_hash_service_n32_ci.json",
                                    "bench/results/ec_sweep_hash_service_n32_K16_ci.json"]},
    {"tag": "preference", "files": ["bench/results/ec_sweep_hash_preference_n32_K16_ci.json"]},
    {"tag": "project",    "files": ["bench/results/ec_sweep_hash_project_n32_K16_ci.json"]},
    {"tag": "tool",       "files": ["bench/results/ec_sweep_hash_tool_n32_K16_ci.json"]},
]
SERIES_ST: list[dict] = [
    {"tag": "technical",  "files": ["bench/results/ec_sweep_st_technical_n32_K16_ci.json"]},
    {"tag": "service",    "files": ["bench/results/ec_sweep_st_service_n32_ci.json",
                                    "bench/results/ec_sweep_st_service_n32_K16_ci.json"]},
    {"tag": "preference", "files": ["bench/results/ec_sweep_st_preference_n32_K16_ci.json"]},
    {"tag": "project",    "files": ["bench/results/ec_sweep_st_project_n32_K16_ci.json"]},
    {"tag": "tool",       "files": ["bench/results/ec_sweep_st_tool_n32_K16_ci.json"]},
]
SERIES_BGE: list[dict] = [
    {"tag": "technical",  "files": ["bench/results/ec_bge_large_technical_n32_K16_ci.json"]},
    {"tag": "service",    "files": ["bench/results/ec_bge_large_service_n32_K16_ci.json"]},
    {"tag": "preference", "files": ["bench/results/ec_bge_large_preference_n32_K16_ci.json"]},
    {"tag": "project",    "files": ["bench/results/ec_bge_large_project_n32_K16_ci.json"]},
    {"tag": "tool",       "files": ["bench/results/ec_bge_large_tool_n32_K16_ci.json"]},
]


def _load_series(files: list[str]) -> list[tuple[int, float, float, float]]:
    out: dict[int, tuple[float, float, float]] = {}
    for f in files:
        p = Path(f)
        if not p.exists():
            print(f"[plot] WARNING missing {f}, skipping")
            continue
        d = json.loads(p.read_text())
        for row in d.get("rows", []):
            K = row["collision_degree"]
            ci = row.get("delta_ci", {}).get("hit_at_1")
            if ci is None:
                continue
            out[K] = (ci["mean"], ci["ci_lo"], ci["ci_hi"])
    rows = sorted(out.items())
    return [(K, m, lo, hi) for K, (m, lo, hi) in rows]


def _plot_panel(ax, series: list[dict], title: str) -> bool:
    plotted = False
    for s in series:
        rows = _load_series(s["files"])
        if not rows:
            continue
        plotted = True
        style = TAG_STYLE[s["tag"]]
        Ks = [r[0] for r in rows]
        means = [r[1] for r in rows]
        lo = [r[2] for r in rows]
        hi = [r[3] for r in rows]
        ax.plot(Ks, means, marker=style["marker"], color=style["color"],
                linewidth=1.6, markersize=6, label=s["tag"])
        ax.fill_between(Ks, lo, hi, color=style["color"], alpha=0.12,
                        linewidth=0)
    ax.axhline(0.0, color="#555", linewidth=0.8, linestyle=":")
    ax.set_xscale("log", base=2)
    ax.set_xticks([1, 2, 4, 8, 16])
    ax.set_xticklabels(["1", "2", "4", "8", "16"])
    ax.set_xlabel("Collision degree K")
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", linestyle=":", linewidth=0.5,
            color="#bbb", alpha=0.7)
    return plotted


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str,
                    default="bench/results/ec_paper_figure.png")
    ap.add_argument("--suptitle", type=str,
                    default="Vector retrieval lift under entity collision")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=140,
                             sharey=True)
    any_hash = _plot_panel(axes[0], SERIES_HASH, "HashTrigram-256")
    any_st = _plot_panel(axes[1], SERIES_ST, "ST MiniLM-384")
    any_bge = _plot_panel(axes[2], SERIES_BGE, "BGE-large-1024")
    if not (any_hash or any_st or any_bge):
        raise SystemExit("[plot] no series had data; aborting")

    axes[0].set_ylabel(r"$\Delta$ hit@1   (vector fusion − BM25-only)")
    # Single shared legend, ordered by tag effect-size on ST
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, frameon=False,
               fontsize=9, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(args.suptitle, fontsize=11)
    fig.tight_layout(rect=(0, 0.04, 1, 0.97))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"[plot] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
