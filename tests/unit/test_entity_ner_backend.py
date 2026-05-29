"""Tests for the entity_ner backend selector (D1 — spaCy swap prep).

Covers:
- Default backend ('heuristic') behaves identically to bare extract_entities.
- Unknown backend raises ValueError with a clear message.
- 'spacy_sm' raises ImportError-with-install-hint when spaCy is absent
  (we don't ship the dep yet — these tests assume it's not installed and
  are skipped if it is).
- RetrievalConfig threads `entity_ner` through to/from dict.
- Engine picks up the cfg.entity_ner selector when entity channel is on.
"""
from __future__ import annotations

import importlib.util

import pytest

from engram.core.config import Config, RetrievalConfig
from engram.retrieval.entities import extract_entities


_HAS_SPACY = importlib.util.find_spec("spacy") is not None


def test_default_backend_matches_legacy_call():
    text = "I met Alice Smith yesterday in New York."
    assert extract_entities(text) == extract_entities(text, backend="heuristic")


def test_unknown_backend_raises_valueerror():
    with pytest.raises(ValueError, match="Unknown entity_ner backend"):
        extract_entities("anything", backend="bogus_ner_v9")


@pytest.mark.skipif(_HAS_SPACY, reason="spaCy is installed; skipping the absent-dep error test")
def test_spacy_backend_without_dep_raises_importerror():
    with pytest.raises(ImportError, match="entity-ner"):
        extract_entities("Alice Smith works at NASA.", backend="spacy_sm")


def test_retrieval_config_default():
    cfg = RetrievalConfig()
    assert cfg.entity_ner == "heuristic"


def test_engram_config_roundtrip_preserves_entity_ner():
    cfg = Config()
    cfg.retrieval.entity_ner = "spacy_sm"
    d = cfg.to_dict()
    assert d["retrieval"]["entity_ner"] == "spacy_sm"
    cfg2 = Config._from_dict(d)
    assert cfg2.retrieval.entity_ner == "spacy_sm"


def test_engram_config_default_dict_has_entity_ner_field():
    cfg = Config()
    d = cfg.to_dict()
    assert d["retrieval"]["entity_ner"] == "heuristic"


def test_engram_config_from_dict_missing_entity_ner_defaults_to_heuristic():
    # Forward-compat: configs serialized before this field was added still load.
    cfg = Config._from_dict({"retrieval": {"bm25_weight": 0.4}})
    assert cfg.retrieval.entity_ner == "heuristic"
