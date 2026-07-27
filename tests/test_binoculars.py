# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Binoculars tests: refusal logic without any model, plus one real-math CPU smoke on a
tiny shared-tokenizer pair (gpt2 + distilgpt2, both already cached by earlier runs)."""

from __future__ import annotations

import pytest

from gaige.detectors import binoculars
from gaige.detectors.base import Detector


class _StubTok:
    def __init__(self, ids):
        self._ids = ids

    def encode(self, _text):
        return list(self._ids)


def test_tokenizer_mismatch_refuses_by_name():
    with pytest.raises(RuntimeError, match="share a tokenizer"):
        binoculars._check_tokenizer_match(_StubTok([1, 2, 3]), _StubTok([1, 2, 4]))
    binoculars._check_tokenizer_match(_StubTok([1, 2, 3]), _StubTok([1, 2, 3]))  # ok


def test_quant_rules_mirror_the_reference_detector():
    det = binoculars.Binoculars(quant="4bit")
    with pytest.raises(RuntimeError, match="CUDA-only"):
        det._effective_quant("cpu")
    with pytest.raises(RuntimeError, match="fp16 on CPU"):
        binoculars.Binoculars(quant="fp16")._effective_quant("cpu")
    assert binoculars.Binoculars(quant="fp32")._effective_quant("cpu") == "fp32"


def test_conforms_to_detector_protocol():
    assert isinstance(binoculars.Binoculars(), Detector)


def test_real_math_smoke_on_tiny_shared_tokenizer_pair():
    """gpt2 (observer) + distilgpt2 (performer) share the GPT-2 tokenizer: the actual
    ppl/x_ppl arithmetic runs on CPU in seconds. Asserts the score is finite, negative
    (negated ratio of two positive cross-entropies), deterministic, and that the
    fingerprint proves BOTH models."""
    pytest.importorskip("torch", reason="real-math smoke scores through the [gpu] extra")
    pytest.importorskip("transformers", reason="real-math smoke scores through the [gpu] extra")
    det = binoculars.Binoculars(
        observer_id="gpt2",
        performer_id="distilgpt2",
        quant="fp32",
        device="cpu",
        max_tokens=64,
        min_free_gb=2.0,
    )
    text = "The quick brown fox jumps over the lazy dog and keeps on running through the field."
    s1 = det.score(text)
    s2 = det.score(text)
    assert s1 == s2  # deterministic forward passes
    assert s1 < 0.0  # negated positive ratio
    assert abs(s1) < 10.0  # sane magnitude for a CE ratio
    meta = det.metadata()
    assert meta["observer_id"] == "gpt2" and meta["performer_id"] == "distilgpt2"
    assert meta["quant_verified"]["per_model"]["observer"]["resident_gb"] > 0
    assert "NEGATED" in meta["score_semantics"]
    assert det.score("") == 0.0  # degenerate input stays defined
