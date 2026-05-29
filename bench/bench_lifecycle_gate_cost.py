"""Microbench: per-retrieval cost of `respect_schema_lifecycle=True`.

Question (NEXT.md item #1, paraphrased): the gate's hot path is
`snapshot_from_buffer(buffer)` followed by a set-membership filter over
SCHEMA candidates. The 1M-ingest D5 reps had **0 fading/faded** so the
gate's heavy path never fired. This bench instead loads the buffer with
a controllable number of CONSOLIDATION_SCHEMA_LIFECYCLE events and
measures retrieval latency as a function of that lifecycle-event count.

Two arms:
  - `respect_schema_lifecycle=False` (gate off, baseline)
  - `respect_schema_lifecycle=True`  (gate on, replay snapshot every call)

Sweeps the lifecycle-event count over {0, 100, 1k, 10k, 50k}. Fixed
1k SCHEMA memories in the store so the gate's filter actually has a
candidate set to filter; half of the lifecycle events deprecate
schemas (so deprecated_ids is non-empty and the filter loop runs).

Output: bench/results/lifecycle_gate_cost_<sha>_<ts>.json with
per-event-count median + p95 retrieve() latency for each arm,
plus the absolute and relative gate-on overhead.

Run: python bench/bench_lifecycle_gate_cost.py
"""
from __future__ import annotations

import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from engram import Config, Engram
from engram.consolidation.lifecycle_projection import make_lifecycle_event
from engram.consolidation.schema_lifecycle import EventKind
from engram.core import Memory, MemoryState, MemoryType


REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _percentile(values, p):
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def _seed_corpus(eng: Engram, n_schemas: int) -> list[str]:
    """Insert n_schemas SCHEMA-typed memories. Returns their IDs."""
    schema_ids: list[str] = []
    now = datetime.now(timezone.utc)
    for i in range(n_schemas):
        m = Memory(
            id=f"sch_{i:06d}",
            type=MemoryType.SCHEMA,
            state=MemoryState.ACTIVE,
            content=f"schema pattern {i}: when X then Y_{i % 17}",
            summary=f"schema {i}",
            salience=0.5,
            confidence=0.7,
            decay_rate=0.1,
            created_at=now,
            last_accessed=now,
        )
        # _store is internal but stable; this is a microbench script.
        eng._store.upsert(m)
        schema_ids.append(m.id)
    return schema_ids


def _seed_lifecycle_events(eng: Engram, schema_ids: list[str], n_events: int) -> int:
    """Append n_events CONSOLIDATION_SCHEMA_LIFECYCLE events. Returns
    the number of distinct schemas that end up in DEPRECATED state.

    Layout: first ``min(n_schemas, n_events)`` events are CREATEs (one
    per schema, in order) so the reducer has something to act on.
    Remaining events alternate DEPRECATE / RECOVER on already-created
    schemas. Even-index post-CREATE events DEPRECATE; odd-index RECOVER
    with a fresh window_id (recover requires distinct window_id per
    invariant #5). This produces a buffer where the gate's snapshot
    has a well-defined non-empty deprecated set, and where all events
    are valid (so reducer walks them all without dropping).
    """
    if n_events == 0:
        return 0
    n_schemas = len(schema_ids)
    deprecated: set[str] = set()
    last_op: dict[str, str] = {}  # sid → "dep" | "rec"
    for i in range(n_events):
        if i < n_schemas:
            sid = schema_ids[i]
            ev = make_lifecycle_event(
                schema_id=sid, kind=EventKind.CREATE, content=f"e{i}"
            )
        else:
            sid = schema_ids[i % n_schemas]
            prev = last_op.get(sid)
            if prev != "dep":
                ev = make_lifecycle_event(
                    schema_id=sid, kind=EventKind.DEPRECATE,
                    window_id=f"w{i}", content=f"e{i}",
                )
                last_op[sid] = "dep"
                deprecated.add(sid)
            else:
                ev = make_lifecycle_event(
                    schema_id=sid, kind=EventKind.RECOVER,
                    window_id=f"w{i}", content=f"e{i}",
                )
                last_op[sid] = "rec"
                deprecated.discard(sid)
        eng._buffer.append(ev)
    return len(deprecated)


def _measure_arm(tmp_root: Path, n_events: int, gate_on: bool, n_queries: int = 200) -> dict:
    cfg = Config(path=str(tmp_root / f"engram_{n_events}_{int(gate_on)}"))
    cfg.security.max_events_per_minute = 0
    cfg.retrieval.respect_schema_lifecycle = gate_on
    eng = Engram(config=cfg)
    try:
        schema_ids = _seed_corpus(eng, n_schemas=1000)
        n_dep = _seed_lifecycle_events(eng, schema_ids, n_events)

        # Warm-up
        for _ in range(5):
            eng.recall("schema pattern X then Y_3", limit=10)

        latencies_ms: list[float] = []
        for i in range(n_queries):
            q = f"schema pattern when then Y_{i % 17}"
            t0 = time.monotonic()
            eng.recall(q, limit=10)
            latencies_ms.append((time.monotonic() - t0) * 1000)

        return {
            "n_events": n_events,
            "gate_on": gate_on,
            "n_queries": n_queries,
            "n_deprecated_schemas": n_dep,
            "p50_ms": round(_percentile(latencies_ms, 50), 3),
            "p95_ms": round(_percentile(latencies_ms, 95), 3),
            "p99_ms": round(_percentile(latencies_ms, 99), 3),
            "mean_ms": round(statistics.mean(latencies_ms), 3),
            "max_ms": round(max(latencies_ms), 3),
        }
    finally:
        eng.close()


def main(out_path: Path | None = None) -> Path:
    import tempfile

    sha = _git_sha()
    sweep = [0, 100, 1_000, 10_000, 50_000]

    rows = []
    with tempfile.TemporaryDirectory() as td:
        tmp_root = Path(td)
        for n in sweep:
            for gate_on in (False, True):
                row = _measure_arm(tmp_root, n_events=n, gate_on=gate_on)
                rows.append(row)
                print(
                    f"[bench] n_events={n:>6} gate_on={gate_on!s:>5} "
                    f"p50={row['p50_ms']:.3f}ms p95={row['p95_ms']:.3f}ms "
                    f"mean={row['mean_ms']:.3f}ms"
                )

    # Pair off / off across event counts
    by_n = {n: {} for n in sweep}
    for r in rows:
        by_n[r["n_events"]][r["gate_on"]] = r
    overhead = []
    for n in sweep:
        off = by_n[n][False]
        on = by_n[n][True]
        overhead.append({
            "n_events": n,
            "p50_off_ms": off["p50_ms"],
            "p50_on_ms": on["p50_ms"],
            "p50_abs_overhead_ms": round(on["p50_ms"] - off["p50_ms"], 3),
            "p50_rel_overhead_pct": (
                round(100 * (on["p50_ms"] - off["p50_ms"]) / off["p50_ms"], 2)
                if off["p50_ms"] > 0 else None
            ),
            "p95_abs_overhead_ms": round(on["p95_ms"] - off["p95_ms"], 3),
        })

    payload = {
        "meta": {
            "sha": sha,
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S"),
            "name": "lifecycle_gate_cost",
            "n_schemas_in_store": 1000,
            "queries_per_arm": 200,
        },
        "arms": rows,
        "overhead": overhead,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_path is None:
        out_path = RESULTS_DIR / f"lifecycle_gate_cost_{sha}_{payload['meta']['timestamp']}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[bench] wrote {out_path}")
    return out_path


if __name__ == "__main__":
    main()
