"""Plot vector_weight sweep results across (embed × difficulty) cells.

Reads the four canonical sweep JSONs from bench/results/ and emits a 2x2
small-multiples PNG: rows = {hash, st}, columns = {easy, hard}.
Each panel shows hit@1 and MRR vs vector_weight, with horizontal BM25-only
baselines. Flat low-chrome style — monospace, no chartjunk.

Usage:
    python -m evals.plot_sweep [--out bench/results/sweep_2x2.png]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
BENCH = REPO / "bench" / "results"

CELLS = [
    ("hash", "easy", BENCH / "sweep_vw_hash_easy.json"),
    ("hash", "hard", BENCH / "sweep_vw_hash_hard.json"),
    ("st", "easy", BENCH / "sweep_vw_st_easy.json"),
    ("st", "hard", BENCH / "sweep_vw_st_hard.json"),
]


def _load(p: Path) -> dict:
    with p.open() as fh:
        return json.load(fh)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(BENCH / "sweep_2x2.png"))
    ap.add_argument(
        "--st-hard-override",
        default=None,
        help="Path to alternate ST-hard sweep JSON (e.g. n=100 result).",
    )
    args = ap.parse_args()

    plt.rcParams.update(
        {
            "font.family": "monospace",
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
            "lines.linewidth": 1.4,
        }
    )

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0), sharex=True)
    axes_by_cell = {
        ("hash", "easy"): axes[0, 0],
        ("hash", "hard"): axes[0, 1],
        ("st", "easy"): axes[1, 0],
        ("st", "hard"): axes[1, 1],
    }

    for embed, diff, path in CELLS:
        if embed == "st" and diff == "hard" and args.st_hard_override:
            path = Path(args.st_hard_override)
        data = _load(path)
        ax = axes_by_cell[(embed, diff)]
        sweep = data["sweep"]
        xs = [row["vector_weight"] for row in sweep]
        hit1 = [row["baseline_hit_at_1"] for row in sweep]
        mrr = [row["baseline_mrr"] for row in sweep]
        bm25_hit1 = sweep[0]["bm25_only_hit_at_1"]
        bm25_mrr = sweep[0]["bm25_only_mrr"]

        ax.plot(xs, hit1, "o-", color="#1b4965", label="hit@1")
        ax.plot(xs, mrr, "s--", color="#bc4749", label="MRR")
        ax.axhline(bm25_hit1, color="#1b4965", alpha=0.35, linewidth=0.8)
        ax.axhline(bm25_mrr, color="#bc4749", alpha=0.35, linewidth=0.8)
        cfg = data["config"]
        n = cfg["n_sessions"]
        ax.set_title(f"{embed} / {diff}  (n={n})", loc="left")
        ax.set_ylim(0.0, 1.0)
        if diff == "easy":
            ax.set_ylabel("score")
        if embed == "st":
            ax.set_xlabel("vector_weight (0=BM25, 1=vector)")

    axes[0, 0].legend(loc="lower right", framealpha=0.9, fontsize=8)
    fig.suptitle(
        "Engram retrieval: BM25 ↔ vector fusion sweep (hit@1, MRR)",
        x=0.02,
        ha="left",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
