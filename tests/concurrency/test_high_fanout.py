"""High-fanout concurrency torture — 50+ writers × 50+ readers.

Mission item 2c explicitly calls for this scale.

Opt-in via `pytest -m concurrency`. Reasonable wall-time budget (<60s on dev box).
"""
from __future__ import annotations

import random
import string
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

from engram import Config, Engram

pytestmark = pytest.mark.concurrency


def _mk_config(tmp_path: Path) -> Config:
    cfg = Config.minimal()
    cfg.path = str(tmp_path)
    cfg.security.max_events_per_minute = 0
    return cfg


def _rand_text(n: int = 30) -> str:
    return "".join(random.choices(string.ascii_letters + " ", k=n)).strip() or "x"


def test_50_writers_50_readers_fanout(tmp_path: Path):
    """50 writer threads × 50 reader threads, no crashes, ≥95% writes land."""
    cfg = _mk_config(tmp_path)
    e = Engram(config=cfg)
    try:
        n_writers = 50
        n_readers = 50
        per_writer = 40  # 2,000 total writes — keeps wall < 30s

        write_errors: list[BaseException] = []
        read_errors: list[BaseException] = []
        stop = threading.Event()
        ready = threading.Barrier(n_writers + n_readers)

        def writer(wid: int):
            try:
                ready.wait()
                for j in range(per_writer):
                    e.remember(
                        f"writer-{wid} entry-{j} {_rand_text(25)}",
                        salience=0.3,
                    )
            except BaseException as ex:  # noqa: BLE001
                write_errors.append(ex)

        def reader(rid: int):
            try:
                ready.wait()
                while not stop.is_set():
                    res = e.recall(f"writer-{rid % n_writers}", limit=5)
                    assert isinstance(res, list)
                    time.sleep(0.001)
            except BaseException as ex:  # noqa: BLE001
                read_errors.append(ex)

        with ThreadPoolExecutor(max_workers=n_writers + n_readers) as pool:
            r_futs = [pool.submit(reader, i) for i in range(n_readers)]
            w_futs = [pool.submit(writer, i) for i in range(n_writers)]
            for f in as_completed(w_futs):
                f.result()
            stop.set()
            for f in as_completed(r_futs):
                f.result()

        assert not write_errors, f"writer errors ({len(write_errors)}): {write_errors[:3]}"
        assert not read_errors, f"reader errors ({len(read_errors)}): {read_errors[:3]}"

        mems = e._store.all_active()
        expected = per_writer * n_writers
        assert len(mems) >= int(expected * 0.95), (
            f"expected ~{expected}, got {len(mems)} (loss > 5%)"
        )
        # Schema integrity under fanout.
        for m in mems[:200]:
            assert m.id and m.content and m.created_at
    finally:
        e.close()


def test_64_readers_during_burst_writes(tmp_path: Path):
    """Read-heavy mix: 64 readers vs 8 burst writers. Readers must never see torn rows."""
    cfg = _mk_config(tmp_path)
    e = Engram(config=cfg)
    try:
        # Pre-seed so readers have something to read from t=0.
        for i in range(200):
            e.remember(f"seed-{i} {_rand_text(20)}", salience=0.2)

        n_readers = 64
        n_writers = 8
        per_writer = 100

        write_errors: list[BaseException] = []
        read_errors: list[BaseException] = []
        stop = threading.Event()
        ready = threading.Barrier(n_writers + n_readers)

        def writer(wid: int):
            try:
                ready.wait()
                for j in range(per_writer):
                    e.remember(f"burst-{wid}-{j} {_rand_text(20)}", salience=0.4)
            except BaseException as ex:  # noqa: BLE001
                write_errors.append(ex)

        def reader(rid: int):
            try:
                ready.wait()
                while not stop.is_set():
                    res = e.recall("seed", limit=10)
                    assert isinstance(res, list)
                    for r in res:
                        # Torn-row check: every row must have id+content+created_at.
                        m = r.memory if hasattr(r, "memory") else r
                        assert m.id and m.content and m.created_at
            except BaseException as ex:  # noqa: BLE001
                read_errors.append(ex)

        with ThreadPoolExecutor(max_workers=n_writers + n_readers) as pool:
            r_futs = [pool.submit(reader, i) for i in range(n_readers)]
            w_futs = [pool.submit(writer, i) for i in range(n_writers)]
            for f in as_completed(w_futs):
                f.result()
            stop.set()
            for f in as_completed(r_futs):
                f.result()

        assert not write_errors, f"writer errors: {write_errors[:3]}"
        assert not read_errors, f"reader errors: {read_errors[:3]}"
    finally:
        e.close()
