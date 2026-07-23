# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Regression lock on the DETECTION workflow.

gaige is evolving toward longitudinal drift measurement for a research apparatus. The risk that
creates is ordinary and well known: the original capability rots because nothing exercises it,
and nobody notices until someone depends on it.

So the reference detection instrument is pinned here. The fixture holds the scores from the
2026-07-21 reference run — HC3-mini n=100 seed=17, Fast-DetectGPT on falcon-7b 4-bit CUDA,
corpus sha256 7d2819d3e83bd10d... — and these tests assert that gaige's analysis path still turns
them into exactly the same numbers.

No GPU required: scoring produced the fixture once; analysis reproduces from it forever. If these
numbers move, the detection path changed. That is either a bug or a deliberate change that has to
be argued for and re-pinned — never a surprise, and never something a refactor does quietly.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaige import analyze

FIXTURES = Path(__file__).parent / "fixtures"
SCORES = FIXTURES / "reference-hc3-falcon7b-4bit-cuda.scores.csv"
EXPECTED = json.loads((FIXTURES / "reference-expected.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def results():
    rows = analyze.read_scores_csv(SCORES)
    return analyze.compute_results(rows, n_boot=EXPECTED["n_boot"], seed=17)


def test_reference_corpus_shape_unchanged():
    rows = analyze.read_scores_csv(SCORES)
    assert len(rows) == 200
    assert sum(1 for r in rows if r["label"] == "human") == 100
    assert sum(1 for r in rows if r["label"] == "ai") == 100


def test_reference_auroc_is_exact(results):
    """AUROC 0.9720 on the reference instrument. Exact, not approximate."""
    assert results["auroc"] == EXPECTED["auroc"]


def test_reference_auroc_ci_is_exact(results):
    """Bootstrap CIs are seeded, so they are reproducible to the bit. Treat them that way.

    Compared as lists: compute_results returns a tuple, JSON round-trips it to a list, and a
    tuple never equals a list. Normalise rather than let a type mismatch masquerade as drift.
    """
    assert list(results["auroc_ci"]) == list(EXPECTED["auroc_ci"])


def test_reference_thresholds_are_exact(results):
    """The operating thresholds are the numbers a user would actually act on."""
    got = results["thresholds"]
    exp = EXPECTED["thresholds"]
    assert len(got) == len(exp)
    for g, e in zip(got, exp):
        for key in ("target_fpr", "threshold", "achieved_fpr", "achieved_tpr"):
            assert g[key] == e[key], f"{key} drifted: {g[key]} != {e[key]}"
        assert list(g["tpr_ci"]) == list(e["tpr_ci"]), (
            f"tpr_ci drifted: {g['tpr_ci']} != {e['tpr_ci']}"
        )


def test_achieved_fpr_never_exceeds_target(results):
    """The core promise of a target-FPR threshold. If this fails the tool is lying."""
    for row in results["thresholds"]:
        assert row["achieved_fpr"] <= row["target_fpr"] + 1e-12


def test_reference_eer_is_exact(results):
    """EER 0.07 at threshold 1.6398 on the reference instrument (added 2026-07-23)."""
    assert results["eer"] == EXPECTED["eer"]
    assert results["eer_threshold"] == EXPECTED["eer_threshold"]
