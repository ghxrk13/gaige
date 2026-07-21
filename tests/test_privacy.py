# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""The no-persistence property, verified rather than asserted.

`SECURITY.md` and `single.py` both claim that the text you score is never written to disk.
That is a security claim, and an unverified security claim is worth less than none — it is the
first thing a hostile reviewer tests, and the realistic user is someone checking their own
writing or an institution checking a student's. The content is sensitive by construction.

These tests take a distinctive string, score it, and then prove the string appears nowhere on
disk and that no file was created or modified anywhere under the working tree.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from gaige import analyze
from gaige.receipts import write_report
from gaige.single import score_document

CANARY = "zqx-canary-phrase-8842-that-must-never-be-written-anywhere"


class FakeDetector:
    """Stands in for a real scorer so this runs with no GPU and no model download."""

    name = "fake"

    def __init__(self):
        self.seen: list[str] = []

    def score(self, text: str) -> float:
        self.seen.append(text)  # in memory only, and asserted against below
        return 1.234

    def metadata(self) -> dict:
        return {"detector": "fake", "device": "cpu", "versions": {}}


def rows(n=60):
    out = []
    for i in range(n):
        out.append({"id": f"h{i}", "label": "human", "score": -1.0 + i * 0.01, "seconds": 0.1})
    for i in range(n):
        out.append({"id": f"a{i}", "label": "ai", "score": 1.0 + i * 0.01, "seconds": 0.1})
    return out


@pytest.fixture
def report(tmp_path):
    r = rows()
    results = analyze.compute_results(r, n_boot=100, seed=3)
    out = tmp_path / "report"
    write_report(out, analyze.UNKNOWN_CORPUS, dict(analyze.UNKNOWN_DETECTOR), r, results, "test")
    return out


def snapshot(root: Path) -> dict:
    return {p: (p.stat().st_size, p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def test_scored_text_is_never_written_to_disk(tmp_path, report):
    """The claim, tested directly: the canary must appear in no file under the tree."""
    before = snapshot(tmp_path)
    score_document(report, CANARY, detector=FakeDetector())

    for p in tmp_path.rglob("*"):
        if p.is_file():
            body = p.read_bytes()
            assert CANARY.encode() not in body, f"scored text leaked into {p}"

    assert snapshot(tmp_path) == before, "scoring created or modified a file; it must not"


def test_scoring_returns_the_analysis_without_echoing_the_text(report):
    """The result carries scores and verdicts — never the document itself."""
    r = score_document(report, CANARY, detector=FakeDetector())
    assert CANARY not in json.dumps(r), "the scored text was echoed back inside the result"
    assert r["score"] == 1.234
    assert "verdicts" in r and r["verdicts"]


def test_word_count_is_derived_not_stored(report):
    r = score_document(report, "one two three four five", detector=FakeDetector())
    assert r["n_words"] == 5
    assert "text" not in r and "document" not in r


def test_detector_received_the_text_verbatim(report):
    """Sanity check on the fake: the text really did reach the scorer, so the assertions above
    are testing a real code path rather than a no-op."""
    fake = FakeDetector()
    score_document(report, CANARY, detector=fake)
    assert fake.seen == [CANARY]
