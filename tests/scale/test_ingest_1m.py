"""§D5 — BEAM-1M scale extension (Mem0 v3 reports 10M-token regime).

This test extends the 100k harness to **1,000,000 memories** to answer
the reviewer question "does Engram scale to Mem0's regime?". Marked
``mega_scale`` so it runs only on demand. Budget: ~45-90 min wall on
tmpfs, ~2-4 GB disk depending on backend.

Decision rule:
    p99 write latency must stay < 500ms across the run.
    Throughput must stay > 50 writes/sec averaged.

If either bound trips, that's a publishable scaling-cliff finding;
if both hold, the SCALE_REPORT can claim "tested at 1M, no cliff".
"""
from __future__ import annotations

import statistics
import time
from pathlib import Path

import pytest

from engram import Engram, Config

from .test_ingest_scale import (
    _gen_content,
    _percentile,
    _record,
)


@pytest.mark.mega_scale
@pytest.mark.slow
def test_scale_ingest_1m(tmp_path: Path):
    """1M ingest. Opt-in only (-m mega_scale). Heavy."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        contents = _gen_content(1_000_000)
        latencies_ms: list[float] = []
        wall_start = time.monotonic()
        for i, c in enumerate(contents):
            t0 = time.monotonic()
            eng.remember(c)
            latencies_ms.append((time.monotonic() - t0) * 1000)
            if (i + 1) % 100_000 == 0:
                p99 = _percentile(latencies_ms[-100_000:], 99)
                tput = (i + 1) / (time.monotonic() - wall_start)
                print(
                    f"[scale-1M] {i+1:>8}/{len(contents)} writes, "
                    f"last-100k p99={p99:.2f}ms, tput={tput:.1f}/s"
                )
        wall_total = time.monotonic() - wall_start

        result = {
            "n": 1_000_000,
            "wall_seconds": round(wall_total, 3),
            "throughput_per_sec": round(1_000_000 / wall_total, 1),
            "latency_ms": {
                "p50": round(_percentile(latencies_ms, 50), 3),
                "p95": round(_percentile(latencies_ms, 95), 3),
                "p99": round(_percentile(latencies_ms, 99), 3),
                "p999": round(_percentile(latencies_ms, 99.9), 3),
                "max": round(max(latencies_ms), 3),
                "mean": round(statistics.mean(latencies_ms), 3),
            },
            # Detect a scaling cliff: tail of the run vs head.
            "head_100k_p99_ms": round(_percentile(latencies_ms[:100_000], 99), 3),
            "tail_100k_p99_ms": round(_percentile(latencies_ms[-100_000:], 99), 3),
            "status": eng.status(),
        }
        _record("ingest_1m", result)

        # Soft regression bounds — generous but sufficient to catch a
        # serious cliff. Hard fail surfaces in CI; the JSON has the
        # actual curve regardless.
        assert result["latency_ms"]["p99"] < 500, (
            f"p99 write latency too high at 1M: "
            f"{result['latency_ms']['p99']}ms (head→tail "
            f"{result['head_100k_p99_ms']}→{result['tail_100k_p99_ms']})"
        )
        assert result["throughput_per_sec"] > 50, (
            f"throughput cliff at 1M: {result['throughput_per_sec']}/s "
            f"(wall {result['wall_seconds']}s)"
        )
    finally:
        eng.close()
