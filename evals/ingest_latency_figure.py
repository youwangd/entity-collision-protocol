"""Generate paper/figures/ingest_1m_latency.png from a 1M-ingest result JSON.

Wired into scripts/regen_figures.sh as the ingest_1m_latency.png generator.
Reads the head/tail-100k p99 plus the global latency percentile bundle and
emits a small two-panel figure:

    panel A: latency-percentile staircase (p50, p95, p99, p99.9, max) — log y
    panel B: head_100k_p99 vs tail_100k_p99 bars, with drift % annotation

Designed to read the schema produced by `tests/scale/test_ingest_1m.py`:

    {
      "n": 1_000_000,
      "throughput_per_sec": 1390.7,
      "latency_ms": {"p50":..., "p95":..., "p99":..., "p999":..., "max":...},
      "head_100k_p99_ms": 3.377,
      "tail_100k_p99_ms": 3.656,
      "meta": {"sha": "...", "timestamp": "..."}
    }

Restrained, monospaced, low-chrome. No gradients, no chartjunk.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _plot(data: dict, out: Path) -> None:
    lat = data["latency_ms"]
    pcts = [("p50", lat["p50"]), ("p95", lat["p95"]), ("p99", lat["p99"]),
            ("p99.9", lat.get("p999", lat.get("p99_9"))), ("max", lat["max"])]
    head_p99 = data["head_100k_p99_ms"]
    tail_p99 = data["tail_100k_p99_ms"]
    drift_pct = (tail_p99 - head_p99) / head_p99 * 100.0
    tput = data["throughput_per_sec"]
    n = data["n"]
    sha = data.get("meta", {}).get("sha", "?")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8.0, 3.2), dpi=150)

    # ----- Panel A: latency staircase -----
    xs = list(range(len(pcts)))
    ys = [v for _, v in pcts]
    ax1.plot(xs, ys, marker="o", color="#222", linewidth=1.2, markersize=4)
    ax1.set_yscale("log")
    ax1.set_xticks(xs)
    ax1.set_xticklabels([k for k, _ in pcts], fontsize=8)
    ax1.set_ylabel("write latency (ms, log)", fontsize=8)
    ax1.set_title(f"1M-ingest latency  (n={n:,}, {tput:.0f} w/s)",
                  fontsize=9, loc="left")
    ax1.grid(True, which="both", linestyle=":", alpha=0.4)
    ax1.tick_params(axis="both", labelsize=8)
    for x, (k, v) in zip(xs, pcts):
        ax1.annotate(f"{v:.2f}", (x, v), textcoords="offset points",
                     xytext=(0, 6), ha="center", fontsize=7, color="#444")

    # ----- Panel B: head vs tail p99 -----
    bars = ax2.bar(["head 100k", "tail 100k"], [head_p99, tail_p99],
                   color=["#888", "#444"], width=0.55)
    ax2.set_ylabel("p99 latency (ms)", fontsize=8)
    ax2.set_title(f"drift = {drift_pct:+.1f}%   (sha {sha})",
                  fontsize=9, loc="left")
    ax2.tick_params(axis="both", labelsize=8)
    ax2.grid(True, axis="y", linestyle=":", alpha=0.4)
    for bar, val in zip(bars, [head_p99, tail_p99]):
        ax2.annotate(f"{val:.2f}", (bar.get_x() + bar.get_width() / 2, val),
                     textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8)

    for ax in (ax1, ax2):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True,
                    help="Path to bench/results/ingest_1m_*.json")
    ap.add_argument("--out", required=True,
                    help="Output PNG path (e.g. paper/figures/ingest_1m_latency.png)")
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text())
    _plot(data, Path(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
