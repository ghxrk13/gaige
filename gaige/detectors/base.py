# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Detector protocol: anything that maps text -> scalar score (higher = more AI-like)
and can fully describe itself for the receipt. Calibration is gaige's job, not the
detector's — detectors here emit raw criteria, never vendor-calibrated probabilities.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class Detector(Protocol):
    name: str

    def score(self, text: str) -> float:  # higher = more AI-like
        ...

    def metadata(self) -> dict:  # everything a receipt needs to reproduce this instrument
        ...
