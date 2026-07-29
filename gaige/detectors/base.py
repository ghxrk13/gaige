# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Detector protocol: anything that maps text -> scalar score (higher = more AI-like)
and can fully describe itself for the receipt. Calibration is gaige's job, not the
detector's — detectors here emit raw criteria, never vendor-calibrated probabilities.
"""

from __future__ import annotations

import logging
import warnings
from contextlib import contextmanager
from typing import Protocol, runtime_checkable


@runtime_checkable
class Detector(Protocol):
    name: str

    def score(self, text: str) -> float:  # higher = more AI-like
        ...

    def metadata(self) -> dict:  # everything a receipt needs to reproduce this instrument
        ...


class _DropTorchDtypeMessage(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "torch_dtype" not in record.getMessage()


@contextmanager
def mute_torch_dtype_deprecation():
    """Silence the upstream `torch_dtype is deprecated, use dtype` notice during model load.

    Newer transformers renamed the from_pretrained dtype kwarg; the old spelling stays
    honored across the supported range (>=4.45,<5), but the rename notice surfaces to a
    stranger running the quickstart (found by the 0.0.1 post-publish pass). Depending on
    the transformers version the notice arrives via the warnings module or the
    transformers logger, so both channels are filtered, scoped to that one message and to
    this context only. The load path itself is untouched on every version, which is the
    point: a cosmetic fix must not change the instrument. Switch the kwarg to dtype= and
    delete this when the transformers floor moves past 5.
    """
    f = _DropTorchDtypeMessage()
    loggers = [logging.getLogger("transformers"), logging.getLogger("transformers.modeling_utils")]
    handlers = list(logging.getLogger("transformers").handlers)
    for target in loggers + handlers:
        target.addFilter(f)
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=r".*torch_dtype.*")
            yield
    finally:
        for target in loggers + handlers:
            target.removeFilter(f)
