"""Parity check: LoCoMo `_build_arm_config` should mirror the LME adapter's
`_build_config` for every arm, so cross-benchmark comparisons use identical
knobs (PRF dominance gate, share-prior reranker, rerank pool size, etc.).

Bench-side guarantee for the v0.2 paper: when we report Δhit@1 on LME and
LoCoMo, the underlying treatment is the same code path with the same
parameter wiring."""
from __future__ import annotations

import pytest

from evals.locomo_adapter import _build_arm_config as locomo_build
from evals.longmemeval_adapter import _build_config as lme_build


@pytest.mark.parametrize("arm", ["prf", "share_prior", "both"])
def test_locomo_lme_arm_config_parity(arm):
    """Same args → same RetrievalConfig knobs on both adapters."""
    qe_dom = 0.3
    sp_alpha = 0.10
    sp_pool = 20

    locomo_cfg = locomo_build(arm, qe_dom, sp_alpha, sp_pool)
    lme_cfg = lme_build(arm, qe_dom, sp_alpha, sp_pool)

    assert locomo_cfg is not None
    assert lme_cfg is not None

    a, b = locomo_cfg.retrieval, lme_cfg.retrieval
    assert a.query_expansion_min_dominance == b.query_expansion_min_dominance
    assert a.reranker == b.reranker
    assert a.share_prior_alpha == b.share_prior_alpha
    assert a.rerank_pool_size == b.rerank_pool_size
    assert a.entity_ner == b.entity_ner


def test_locomo_baseline_returns_none():
    """Baseline arm preserves prior behavior: caller falls back to a fresh
    Config(path=tmp) per sample."""
    assert locomo_build("baseline", 0.3, 0.10, 20) is None


def test_locomo_typed_prf_purity_gate_threads_through():
    cfg = locomo_build("prf", 0.3, 0.10, 20, qe_type_purity_min=0.6,
                       qe_backend="spacy_sm")
    assert cfg is not None
    assert cfg.retrieval.query_expansion_type_purity_min == 0.6
    assert cfg.retrieval.entity_ner == "spacy_sm"
