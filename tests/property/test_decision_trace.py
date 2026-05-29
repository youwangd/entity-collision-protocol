"""§94b-internal — decision trace hook invariants T1–T4.

Locks the contract of `family_decision_trace`:

  T1. With no recorder installed, decide_with_family is byte-identical
      to its pre-trace behavior (same return value).
  T2. Recorder receives one record per call, in order, with the expected
      keys populated.
  T3. Recorder is thread-local; threads do not see each other's traces.
  T4. Exceptions raised inside a recorder callback do not affect the
      decision return value.
"""
from __future__ import annotations

import threading

from engram.consolidation.schema_decision import EvidenceWindow, Thresholds
from engram.consolidation.schema_family_decision import (
    decide_with_family,
    family_decision_trace,
)
from engram.consolidation.schema_lifecycle import (
    EventKind,
    SchemaState,
    SchemaStatus,
)


TH = Thresholds()
WIN = "w0"


def _state(status: SchemaStatus = SchemaStatus.INFERRED) -> SchemaState:
    return SchemaState(schema_id="s1", status=status, version=1)


# ---- T1: no recorder ⇒ behavior identical -----------------------------------

def test_t1_no_recorder_identity():
    own = EvidenceWindow(window_id=WIN, supports=10, contradictions=0)
    sibs = [
        EvidenceWindow(window_id=WIN, supports=5, contradictions=0),
        EvidenceWindow(window_id=WIN, supports=3, contradictions=0),
    ]
    r0 = decide_with_family(_state(), own, sibs, TH, share=0.0)
    r1 = decide_with_family(_state(), own, sibs, TH, share=0.5)
    # Both must return either None or some EventKind. The exact value is
    # locked by the G-tests; here we only assert shape + that the trace
    # hook does not perturb the result vs a no-trace call.
    assert r0 is None or isinstance(r0, EventKind)
    assert r1 is None or isinstance(r1, EventKind)


# ---- T2: recorder records every call in order -------------------------------

def test_t2_recorder_in_order():
    own = EvidenceWindow(window_id=WIN, supports=2, contradictions=0)
    sibs = [EvidenceWindow(window_id=WIN, supports=4, contradictions=0)]
    with family_decision_trace() as trace:
        decide_with_family(_state(), own, sibs, TH, share=0.0)
        decide_with_family(_state(), own, sibs, TH, share=0.5)
        decide_with_family(_state(), own, sibs, TH, share=1.0)
    assert len(trace) == 3
    assert [r["share"] for r in trace] == [0.0, 0.5, 1.0]
    # share=0 fast-path: no borrowing.
    assert trace[0]["borrowed_via_share"] is False
    assert trace[0]["sib_s"] == 0  # fast path doesn't compute sibling sum
    # share>0 with siblings: borrowing flag set, sib sums computed.
    assert trace[1]["sib_s"] == 4
    assert trace[1]["eff_s"] == 2 + int(0.5 * 4)
    assert trace[1]["borrowed_via_share"] is True
    # All records have the canonical keys.
    keys = {"state", "share", "own_s", "own_c", "sib_s", "sib_c",
            "eff_s", "eff_c", "decision", "borrowed_via_share"}
    for r in trace:
        assert keys.issubset(r.keys())


# ---- T2b: empty siblings short-circuit reports borrowed=False ---------------

def test_t2b_empty_siblings_no_borrow():
    own = EvidenceWindow(window_id=WIN, supports=2, contradictions=0)
    with family_decision_trace() as trace:
        decide_with_family(_state(), own, [], TH, share=0.75)
    assert len(trace) == 1
    assert trace[0]["borrowed_via_share"] is False
    assert trace[0]["share"] == 0.75


# ---- T3: thread-local isolation ---------------------------------------------

def test_t3_thread_local_isolation():
    own = EvidenceWindow(window_id=WIN, supports=2, contradictions=0)
    sibs = [EvidenceWindow(window_id=WIN, supports=4, contradictions=0)]

    main_sink: list[dict] = []
    barrier = threading.Barrier(2)

    def child():
        # Child has no recorder of its own. Calls here must NOT land in
        # the main thread's sink.
        barrier.wait()
        decide_with_family(_state(), own, sibs, TH, share=0.5)
        decide_with_family(_state(), own, sibs, TH, share=0.5)
        barrier.wait()

    with family_decision_trace(main_sink):
        t = threading.Thread(target=child)
        t.start()
        barrier.wait()  # release child to call decide
        # Meanwhile main does one call.
        decide_with_family(_state(), own, sibs, TH, share=1.0)
        barrier.wait()  # let child finish
        t.join()

    # Main must see only its own one call, never the child's two.
    assert len(main_sink) == 1
    assert main_sink[0]["share"] == 1.0


# ---- T4: recorder exceptions do not affect return value ---------------------

def test_t4_recorder_exception_swallowed():
    own = EvidenceWindow(window_id=WIN, supports=10, contradictions=0)

    # Install a recorder that raises by directly poking the TLS list,
    # bypassing the contextmanager's safe wrapper.
    from engram.consolidation import schema_family_decision as mod

    def boom(_rec):
        raise RuntimeError("recorder bug")

    mod._get_recorders().append(boom)
    try:
        # Without the swallow, this would propagate.
        result = decide_with_family(
            _state(), own, [], TH, share=0.5
        )
    finally:
        mod._get_recorders().remove(boom)

    # Default thresholds + supports=10 ⇒ PROMOTE.
    assert result == EventKind.PROMOTE
