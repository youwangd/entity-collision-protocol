"""Matched 10k → 100k → 1M ingest p50/p95/p99 curves on the same harness.

Reads the per-N latency JSON artifacts under bench/results/ and emits a
single PNG + Markdown summary into bench/results/ingest_curves_*.{png,md}.

Usage:
    python -m bench.plot_ingest_curves
"""
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "bench" / "results"

# Newest representative per N for the matched harness
ARTIFACTS = {
    10_000: RESULTS / "ingest_10k_cd941b3_20260519T231648.json",
    100_000: RESULTS / "ingest_100k_cda2631_20260520T072231.json",
    1_000_000: RESULTS / "ingest_1m_c52560c_20260523T021631.json",
}


def main() -> int:
    ns: list[int] = []
    p50: list[float] = []
    p95: list[float] = []
    p99: list[float] = []
    tput: list[float] = []
    rows = []
    for n, path in sorted(ARTIFACTS.items()):
        if not path.exists():
            print(f"missing artifact: {path}", file=sys.stderr)
            return 1
        d = json.loads(path.read_text())
        lat = d["latency_ms"]
        ns.append(d["n"])
        p50.append(lat["p50"])
        p95.append(lat["p95"])
        p99.append(lat["p99"])
        tput.append(d["throughput_per_sec"])
        rows.append((d["n"], lat["p50"], lat["p95"], lat["p99"],
                     d["throughput_per_sec"], path.name))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    ax1.plot(ns, p50, "o-", label="p50", color="#1f77b4")
    ax1.plot(ns, p95, "s-", label="p95", color="#ff7f0e")
    ax1.plot(ns, p99, "^-", label="p99", color="#d62728")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("Corpus size (memories)")
    ax1.set_ylabel("Per-write latency (ms, log)")
    ax1.set_title("Engram ingest latency vs N (matched harness)")
    ax1.grid(True, which="both", alpha=0.3)
    ax1.legend(loc="best")

    ax2.plot(ns, tput, "o-", color="#2ca02c")
    ax2.set_xscale("log")
    ax2.set_xlabel("Corpus size (memories)")
    ax2.set_ylabel("Throughput (writes/s)")
    ax2.set_title("Engram ingest throughput vs N")
    ax2.grid(True, which="both", alpha=0.3)

    fig.tight_layout()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_png = RESULTS / f"ingest_curves_{ts}.png"
    out_md = RESULTS / f"ingest_curves_{ts}.md"
    fig.savefig(out_png, dpi=140)
    plt.close(fig)

    lines = ["# Matched ingest curves — 10k → 100k → 1M", "",
             f"Generated {ts}.", "",
             "| N | p50 ms | p95 ms | p99 ms | tput w/s | artifact |",
             "|--:|-------:|-------:|-------:|---------:|----------|"]
    for n, a, b, c, t, name in rows:
        lines.append(f"| {n:>9,} | {a:.3f} | {b:.3f} | {c:.3f} | {t:.1f} | `{name}` |")
    lines.append("")
    out_md.write_text("\n".join(lines))
    print(f"wrote {out_png.name}")
    print(f"wrote {out_md.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
