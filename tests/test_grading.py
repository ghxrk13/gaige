# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""The grading rule is part of the instrument, so its behavior is pinned by tests: any
change here should force a deliberate GRADING_VERSION bump, not slip through."""

import pytest

from gaige import grading


def test_normalize_pipeline_edges():
    assert grading.normalize("The Answer") == "answer"
    assert grading.normalize("the the answer") == "the answer"  # exactly ONE leading article
    assert grading.normalize("the") == "the"  # a lone article is the whole answer; kept
    assert grading.normalize("U.S.") == "us"  # punctuation removed, no space injected
    assert grading.normalize("well-known") == "wellknown"
    assert grading.normalize("  A\t Straße  ") == "strasse"  # casefold: ß -> ss
    assert grading.normalize("ﬁve") == "five"  # NFKC decomposes the ligature
    assert grading.normalize("don't know") == "dont know"


def test_free_text_matches_key_and_aliases():
    r = grading.grade_free_text("The U.S.", "US", aliases=["United States"])
    assert r["correct"] is True and r["matched"] == "US"
    r = grading.grade_free_text("united states!", "US", aliases=["United States"])
    assert r["correct"] is True and r["matched"] == "United States"
    r = grading.grade_free_text("Canada", "US", aliases=["United States"])
    assert r["correct"] is False and r["matched"] is None
    assert r["normalized_answer"] == "canada"


def test_choice_argmax_and_conservative_tie():
    assert grading.grade_choice({"A": -1.0, "B": -5.0}, "A")["correct"] is True
    r = grading.grade_choice({"A": -5.0, "B": -1.0}, "A")
    assert r["correct"] is False and r["chosen"] == "B" and r["tie"] is False
    r = grading.grade_choice({"A": -1.0, "B": -1.0}, "A")
    assert r == {"correct": False, "chosen": None, "tie": True}  # a tie is not an answer
    with pytest.raises(ValueError, match="not among options"):
        grading.grade_choice({"A": -1.0}, "C")


def test_version_constant_exists_and_is_pinned():
    assert grading.GRADING_VERSION == "nem-1"
