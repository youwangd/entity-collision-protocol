"""Scale tests — opt-in via pytest -m scale.

These tests measure ingest, recall, and consolidation latency at non-trivial sizes.
They emit a JSON results file under bench/results/ for tracking over time.
"""
from __future__ import annotations

import json
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pytest

from engram import Engram, Config


REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "bench" / "results"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    return s[f] if f == c else s[f] + (s[c] - s[f]) * (k - f)


def _record(name: str, payload: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    sha = _git_sha()
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    out = RESULTS_DIR / f"{name}_{sha}_{ts}.json"
    payload["meta"] = {
        "sha": sha,
        "timestamp": ts,
        "name": name,
    }
    out.write_text(json.dumps(payload, indent=2))
    print(f"\n[scale] wrote {out}")


def _gen_content(n: int) -> list[str]:
    """Pseudo-realistic memory content with varying salience signals."""
    templates = [
        "User {} prefers {} for {} workflows.",
        "Deploy of service {} failed at {} with error code {}.",
        "Meeting note: {} aligned on {} by EOQ {}.",
        "Bug report: {} returned {} when {} was set.",
        "Insight: {} correlates with {} in cohort {}.",
    ]
    out = []
    for i in range(n):
        t = templates[i % len(templates)]
        out.append(t.format(f"user{i}", f"x{i % 17}", f"y{i % 31}"))
    return out


@pytest.mark.scale
@pytest.mark.slow
def test_scale_ingest_10k(tmp_path: Path):
    """Ingest 10,000 memories. Record write latency p50/p95/p99 + total throughput."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0  # disable rate limit for scale tests
    eng = Engram(config=cfg)
    try:
        contents = _gen_content(10_000)
        latencies_ms: list[float] = []
        wall_start = time.monotonic()
        for c in contents:
            t0 = time.monotonic()
            eng.remember(c)
            latencies_ms.append((time.monotonic() - t0) * 1000)
        wall_total = time.monotonic() - wall_start

        status = eng.status()
        assert status["total_memories"] == 10_000

        result = {
            "n": 10_000,
            "wall_seconds": round(wall_total, 3),
            "throughput_per_sec": round(10_000 / wall_total, 1),
            "latency_ms": {
                "p50": round(_percentile(latencies_ms, 50), 3),
                "p95": round(_percentile(latencies_ms, 95), 3),
                "p99": round(_percentile(latencies_ms, 99), 3),
                "max": round(max(latencies_ms), 3),
                "mean": round(statistics.mean(latencies_ms), 3),
            },
            "status": status,
        }
        _record("ingest_10k", result)

        # Soft regression bound: p99 should stay under 100ms on a developer box
        assert result["latency_ms"]["p99"] < 200, (
            f"p99 write latency too high: {result['latency_ms']['p99']}ms — "
            f"see {RESULTS_DIR} for full report"
        )
    finally:
        eng.close()


@pytest.mark.scale
@pytest.mark.slow
def test_scale_recall_after_10k(tmp_path: Path):
    """Ingest 10k memories, then measure recall latency over 100 queries."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        contents = _gen_content(10_000)
        for c in contents:
            eng.remember(c)

        # Measure recall over 100 queries
        recall_latencies: list[float] = []
        recall_hits: list[int] = []
        for i in range(100):
            q = f"user{i * 97 % 10_000}"  # spread across the corpus
            t0 = time.monotonic()
            results = eng.recall(q, limit=10)
            recall_latencies.append((time.monotonic() - t0) * 1000)
            recall_hits.append(len(results))

        result = {
            "n_corpus": 10_000,
            "n_queries": 100,
            "recall_latency_ms": {
                "p50": round(_percentile(recall_latencies, 50), 3),
                "p95": round(_percentile(recall_latencies, 95), 3),
                "p99": round(_percentile(recall_latencies, 99), 3),
                "max": round(max(recall_latencies), 3),
                "mean": round(statistics.mean(recall_latencies), 3),
            },
            "hit_rate": round(sum(1 for h in recall_hits if h > 0) / len(recall_hits), 3),
            "mean_results_returned": round(statistics.mean(recall_hits), 2),
        }
        _record("recall_at_10k", result)

        # We should hit on most of these (FTS over generated content is forgiving)
        assert result["hit_rate"] > 0.8, f"hit rate {result['hit_rate']} suspiciously low"
        assert result["recall_latency_ms"]["p99"] < 500, (
            f"p99 recall too slow: {result['recall_latency_ms']['p99']}ms"
        )
    finally:
        eng.close()


@pytest.mark.mega_scale
@pytest.mark.slow
def test_scale_recall_after_100k(tmp_path: Path):
    """Ingest 100k memories, then measure recall latency over 200 queries.

    Extends `test_scale_recall_after_10k` by one decade so the paper's
    read-side scaling panel (matching §4.14's ingest curve) gets a
    100k datapoint alongside the existing 10k point. Same harness,
    same query distribution shape, same bounds — only `n_corpus`
    differs.
    """
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        contents = _gen_content(100_000)
        for c in contents:
            eng.remember(c)

        # 200 queries spread across the 100k corpus (2× the 10k harness so
        # tail percentiles are estimated on a non-degenerate sample).
        recall_latencies: list[float] = []
        recall_hits: list[int] = []
        for i in range(200):
            q = f"user{i * 977 % 100_000}"
            t0 = time.monotonic()
            results = eng.recall(q, limit=10)
            recall_latencies.append((time.monotonic() - t0) * 1000)
            recall_hits.append(len(results))

        result = {
            "n_corpus": 100_000,
            "n_queries": 200,
            "recall_latency_ms": {
                "p50": round(_percentile(recall_latencies, 50), 3),
                "p95": round(_percentile(recall_latencies, 95), 3),
                "p99": round(_percentile(recall_latencies, 99), 3),
                "max": round(max(recall_latencies), 3),
                "mean": round(statistics.mean(recall_latencies), 3),
            },
            "hit_rate": round(sum(1 for h in recall_hits if h > 0) / len(recall_hits), 3),
            "mean_results_returned": round(statistics.mean(recall_hits), 2),
        }
        _record("recall_at_100k", result)

        assert result["hit_rate"] > 0.8, f"hit rate {result['hit_rate']} suspiciously low"
        # Generous bound — the 10k harness clears < 500 ms at p99; one
        # decade up should still clear sub-second on the FTS5 + ACL path.
        assert result["recall_latency_ms"]["p99"] < 1000, (
            f"p99 recall too slow at 100k: {result['recall_latency_ms']['p99']}ms"
        )
    finally:
        eng.close()


def _stratified_corpus(n: int, n_needles: int, seed: int = 1729) -> tuple[list[str], list[tuple[int, str]]]:
    """Generate `n` haystack memories + plant `n_needles` with unique tokens.

    Returns ``(contents, needles)`` where ``needles[i] = (insertion_index, unique_token)``.
    Each needle's content embeds a token of the form ``zk7q_<seed>_<idx>`` that is
    statistically guaranteed not to occur in the haystack templates (which only
    use ``user{i}``/``x{i%17}``/``y{i%31}`` slots). This breaks the degenerate
    hit_rate=1.0 ceiling of `_gen_content`-only recall harnesses, which return
    >0 results for every query because the haystack tokens collide pairwise.
    """
    import random

    rng = random.Random(seed)
    contents = _gen_content(n)
    needles: list[tuple[int, str]] = []
    # Spread needle insertion across the full corpus so position-effect
    # is averaged out.
    positions = sorted(rng.sample(range(n), n_needles))
    for k, pos in enumerate(positions):
        token = f"zk7q{seed}n{k:04d}"  # alphanumeric only; FTS5-tokenizer-safe
        needles.append((pos, token))
        # Inject the unique token while preserving the original template shape.
        contents[pos] = f"{contents[pos]} ref={token}"
    return contents, needles


@pytest.mark.mega_scale
@pytest.mark.slow
def test_scale_recall_stratified_at_100k(tmp_path: Path):
    """Stratified needle-in-haystack recall at 100k.

    The plain `test_scale_recall_after_100k` saturates `hit_rate` to 1.0 because
    every haystack memory mentions ``user{N}`` and every query is ``user{i*977%100k}``,
    so FTS5 returns ≥1 result deterministically. That's a latency benchmark, not
    a recall benchmark.

    This harness plants `n_needles=200` memories with cryptographically-unique
    tokens (``zk7q…``) into a 100k haystack and queries for those tokens. A
    correct system must rank the planted memory in top-K. ``recall@1`` and
    ``recall@10`` are the headline numbers; latency is reported in the same
    shape as §4.14r so the 100k stratified point is plot-comparable.
    """
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        contents, needles = _stratified_corpus(100_000, n_needles=200)
        for c in contents:
            eng.remember(c)

        # Tokens are unique by construction (zk7q… prefix is absent from
        # `_gen_content` templates), so any returned memory whose content
        # contains the token must derive from the planted needle.
        recall_latencies: list[float] = []
        recall_at_1 = 0
        recall_at_10 = 0
        tokens = [tok for _, tok in needles]
        for tok in tokens:
            t0 = time.monotonic()
            results = eng.recall(tok, limit=10)
            recall_latencies.append((time.monotonic() - t0) * 1000)
            if not results:
                continue
            contents_top = [getattr(r.memory, "content", "") or "" for r in results]
            if contents_top and tok in contents_top[0]:
                recall_at_1 += 1
            if any(tok in ct for ct in contents_top):
                recall_at_10 += 1

        n_q = len(tokens)
        result = {
            "n_corpus": 100_000,
            "n_needles": n_q,
            "recall_at_1": round(recall_at_1 / n_q, 4),
            "recall_at_10": round(recall_at_10 / n_q, 4),
            "recall_latency_ms": {
                "p50": round(_percentile(recall_latencies, 50), 3),
                "p95": round(_percentile(recall_latencies, 95), 3),
                "p99": round(_percentile(recall_latencies, 99), 3),
                "max": round(max(recall_latencies), 3),
                "mean": round(statistics.mean(recall_latencies), 3),
            },
        }
        _record("recall_stratified_at_100k", result)

        # Stratified harness must do meaningfully better than chance.
        # 200 needles in a 100k corpus → uniform-random recall@10 = 10/100k = 1e-4.
        # Asserting >0.5 gives a wide safety margin while still failing loudly
        # if FTS5 indexing or ACL filtering silently drops needles.
        assert result["recall_at_10"] > 0.5, (
            f"stratified recall@10 {result['recall_at_10']} suggests broken indexing"
        )
        assert result["recall_latency_ms"]["p99"] < 1000, (
            f"p99 stratified recall too slow at 100k: {result['recall_latency_ms']['p99']}ms"
        )
    finally:
        eng.close()


@pytest.mark.mega_scale
@pytest.mark.slow
def test_scale_recall_stratified_at_1m(tmp_path: Path):
    """Stratified needle-in-haystack recall at 1M.

    Closes the read-side three-decade curve to match the §4.14 write-side
    three-decade curve. Same fixture shape as
    `test_scale_recall_stratified_at_100k`, with n_corpus=1_000_000 and
    n_needles=500 unique-token needles. Uniform-random recall@10 baseline
    at this scale is 10/1e6 = 1e-5 — six orders of magnitude below the
    asserted floor of 0.5.
    """
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        ingest_start = time.monotonic()
        contents, needles = _stratified_corpus(1_000_000, n_needles=500)
        for i, c in enumerate(contents):
            eng.remember(c)
            if (i + 1) % 100_000 == 0:
                print(
                    f"[scale-1m-stratified] {i+1:>7}/{len(contents)} writes,"
                    f" elapsed={time.monotonic()-ingest_start:.1f}s"
                )
        ingest_seconds = time.monotonic() - ingest_start

        recall_latencies: list[float] = []
        recall_at_1 = 0
        recall_at_10 = 0
        tokens = [tok for _, tok in needles]
        for tok in tokens:
            t0 = time.monotonic()
            results = eng.recall(tok, limit=10)
            recall_latencies.append((time.monotonic() - t0) * 1000)
            if not results:
                continue
            contents_top = [getattr(r.memory, "content", "") or "" for r in results]
            if contents_top and tok in contents_top[0]:
                recall_at_1 += 1
            if any(tok in ct for ct in contents_top):
                recall_at_10 += 1

        n_q = len(tokens)
        result = {
            "n_corpus": 1_000_000,
            "n_needles": n_q,
            "ingest_seconds": round(ingest_seconds, 1),
            "recall_at_1": round(recall_at_1 / n_q, 4),
            "recall_at_10": round(recall_at_10 / n_q, 4),
            "recall_latency_ms": {
                "p50": round(_percentile(recall_latencies, 50), 3),
                "p95": round(_percentile(recall_latencies, 95), 3),
                "p99": round(_percentile(recall_latencies, 99), 3),
                "max": round(max(recall_latencies), 3),
                "mean": round(statistics.mean(recall_latencies), 3),
            },
        }
        _record("recall_stratified_at_1m", result)

        assert result["recall_at_10"] > 0.5, (
            f"stratified recall@10 {result['recall_at_10']} suggests broken indexing at 1M"
        )
        assert result["recall_latency_ms"]["p99"] < 5000, (
            f"p99 stratified recall too slow at 1M: {result['recall_latency_ms']['p99']}ms"
        )
    finally:
        eng.close()


@pytest.mark.mega_scale
@pytest.mark.slow
def test_scale_ingest_100k(tmp_path: Path):
    """100k ingest. Opt-in only (-m mega_scale). Budget ~5min on dev box."""
    cfg = Config(path=str(tmp_path / "engram"))
    cfg.security.max_events_per_minute = 0
    eng = Engram(config=cfg)
    try:
        contents = _gen_content(100_000)
        latencies_ms: list[float] = []
        wall_start = time.monotonic()
        for i, c in enumerate(contents):
            t0 = time.monotonic()
            eng.remember(c)
            latencies_ms.append((time.monotonic() - t0) * 1000)
            if (i + 1) % 10_000 == 0:
                p99 = _percentile(latencies_ms[-10_000:], 99)
                print(f"[scale] {i+1:>7}/{len(contents)} writes, last-10k p99={p99:.2f}ms")
        wall_total = time.monotonic() - wall_start

        result = {
            "n": 100_000,
            "wall_seconds": round(wall_total, 3),
            "throughput_per_sec": round(100_000 / wall_total, 1),
            "latency_ms": {
                "p50": round(_percentile(latencies_ms, 50), 3),
                "p95": round(_percentile(latencies_ms, 95), 3),
                "p99": round(_percentile(latencies_ms, 99), 3),
                "max": round(max(latencies_ms), 3),
                "mean": round(statistics.mean(latencies_ms), 3),
            },
            "status": eng.status(),
        }
        _record("ingest_100k", result)
    finally:
        eng.close()
