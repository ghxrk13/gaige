# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Memory-floor honesty rules (backlog 36, the 0.0.2 acceptance finding).

Pure logic only, so the whole file runs on the numpy+requests core: the floor's precedence
(explicit > measured > conservative default), both refusal messages naming the remedy, and
the single-source guarantee that `gaige plan` and the loaders cannot print different
numbers for the same configuration."""

from __future__ import annotations

from argparse import Namespace

from gaige import memfloor, plan
from gaige.cli import _build_detector

FALCON = "tiiuae/falcon-7b"
PAIR = "tiiuae/falcon-7b+tiiuae/falcon-7b-instruct"


def test_explicit_floor_always_wins():
    gb, why = memfloor.effective_floor(2.5, "fast-detect-gpt", FALCON, "4bit", "vram")
    assert gb == 2.5
    assert "--min-free-gb" in why


def test_measured_floors_apply_per_configuration():
    """The flat 8.0 both under- and over-protected: fp16's measured need is 13.7, and the
    two-model pair carries its own 9.0."""
    assert memfloor.effective_floor(None, "fast-detect-gpt", FALCON, "4bit", "vram")[0] == 8.0
    assert memfloor.effective_floor(None, "fast-detect-gpt", FALCON, "fp16", "vram")[0] == 13.7
    assert memfloor.effective_floor(None, "binoculars", PAIR, "4bit", "vram")[0] == 9.0


def test_unmeasured_configurations_get_the_conservative_default_and_say_so():
    """A 1.7 GB gpt2-large fp16 load still meets the 8.0 default — by design: unmeasured
    configurations do not get to guess lower, and the refusal names the deliberate
    escape hatch instead (the finding's honest fix)."""
    gb, why = memfloor.effective_floor(None, "fast-detect-gpt", "gpt2-large", "fp16", "vram")
    assert gb == 8.0
    assert "no measured floor receipt" in why


def test_both_refusal_shapes_name_the_remedy():
    for resource in ("free VRAM", "available RAM"):
        msg = memfloor.refusal(resource, 1.2, 8.0, "conservative default")
        assert resource in msg
        assert "--min-free-gb" in msg
        assert "smaller --model" in msg
        assert "co-resident" in msg


def test_plan_and_loader_floors_are_the_same_numbers():
    """Single source: every detector row in the plan table takes its needs from the same
    table the loaders enforce. The 0.0.2 finding was exactly this disagreement."""
    floors_in_plan = [row["needs"] for row in plan.CONFIGS if row["config"].startswith("detect ·")]
    assert floors_in_plan == [
        {"vram_free_gb": 8.0},
        {"vram_free_gb": 13.7},
        {"ram_gb": 8.0},
        {"vram_free_gb": 9.0},
    ]
    assert {k: (f.gb, f.kind) for k, f in memfloor.MEASURED.items()} == {
        ("fast-detect-gpt", FALCON, "4bit"): (8.0, "vram"),
        ("fast-detect-gpt", FALCON, "fp16"): (13.7, "vram"),
        ("fast-detect-gpt", "gpt2-large", "fp32"): (8.0, "ram"),
        ("binoculars", PAIR, "4bit"): (9.0, "vram"),
    }


def test_cli_threads_the_escape_hatch_into_both_detectors():
    fdg = _build_detector(
        Namespace(
            detector="fast-detect-gpt",
            model="gpt2-large",
            quant="fp16",
            max_tokens=64,
            device="cuda",
            min_free_gb=2.5,
        )
    )
    assert fdg.min_free_gb == 2.5
    bino = _build_detector(
        Namespace(
            detector="binoculars",
            observer="gpt2",
            performer="distilgpt2",
            quant="fp32",
            max_tokens=64,
            device="cpu",
            min_free_gb=3.0,
        )
    )
    assert bino.min_free_gb == 3.0
