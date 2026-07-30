# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Memory floors, single-sourced (0.0.2 acceptance finding, backlog 36).

The floor that refuses a load and the feasibility table `gaige plan` prints must be the
same numbers or they will eventually disagree; both consume this module. Floors are
MEASURED where a receipt exists and conservative-by-default where none does — a 1.7 GB
gpt2-large load under a flat 8 GB floor was the finding, and the deliberate escape hatch
(`--min-free-gb`) is the honest fix for unmeasured configurations, not a guessed
prediction (the plan table's own rule: configurations without a receipt say "unmeasured"
instead of guessing).

Precedence: an explicit user floor always wins, then the measured floor for exactly this
configuration, then the detector's conservative default. Note the fp16 direction: the flat
8.0 default UNDER-protected falcon-7b fp16, whose measured need is 13.7 GB — per-instrument
floors move both ways.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Floor:
    gb: float
    kind: str  # "vram" | "ram" — which resource the measurement is about
    receipt: str


# Keyed by (detector, instrument, quant); instrument is the model id, or
# "observer+performer" for two-model detectors. Numbers come from the measured anchors
# that `gaige plan` cites (bench 2026-07-22, pinned env).
MEASURED: dict[tuple[str, str, str], Floor] = {
    ("fast-detect-gpt", "tiiuae/falcon-7b", "4bit"): Floor(
        8.0, "vram", "bench 2026-07-22, report 163959"
    ),
    ("fast-detect-gpt", "tiiuae/falcon-7b", "fp16"): Floor(
        13.7, "vram", "bench 2026-07-22, report 221356"
    ),
    ("fast-detect-gpt", "gpt2-large", "fp32"): Floor(8.0, "ram", "bench 2026-07-22, report 174021"),
    ("binoculars", "tiiuae/falcon-7b+tiiuae/falcon-7b-instruct", "4bit"): Floor(
        9.0, "vram", "bench 2026-07-22, report 213952"
    ),
}

# Conservative fallbacks for configurations with no receipt. Deliberately high: the floor
# exists to protect co-resident work, and an unmeasured load does not get to guess lower.
DEFAULT_GB = {"fast-detect-gpt": 8.0, "binoculars": 9.0}


def effective_floor(
    explicit: float | None, detector: str, instrument: str, quant: str, kind: str
) -> tuple[float, str]:
    """Resolve the floor and say where it came from — the provenance string lands in the
    refusal message, so a refused user knows whether they hit a measurement or a default."""
    if explicit is not None:
        return explicit, "set with --min-free-gb"
    f = MEASURED.get((detector, instrument, quant))
    if f is not None and f.kind == kind:
        return f.gb, f"measured floor for this configuration ({f.receipt})"
    return (
        DEFAULT_GB[detector],
        "conservative default; this configuration has no measured floor receipt",
    )


def refusal(resource: str, free: float, floor: float, why: str, note: str = "") -> str:
    """Both refusal branches (VRAM and RAM) speak this one sentence shape, remedy included."""
    return (
        f"refusing to load: {free:.1f} GB {resource} is below the {floor:.1f} GB floor "
        f"({why}; the floor protects co-resident work{note}). "
        "Pass --min-free-gb to lower it deliberately, or use a smaller --model."
    )
