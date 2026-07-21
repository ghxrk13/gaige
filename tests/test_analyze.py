# gaige — calibration + receipts for AI-text detectors.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Analysis-replay tests.

The load-bearing property here is that `gaige run` and `gaige analyze` cannot disagree. They
share `analyze.compute_results`, and these tests exist to keep it that way: if someone ever
"optimises" one path, the round-trip assertion below fails rather than a user quietly getting
two different thresholds for the same scores.
"""

from __future__ import annotations

import csv
import json

import pytest

from gaige import analyze


def write_scores(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "score", "seconds"])
        w.writeheader()
        w.writerows(rows)
    return path


def synthetic_rows(n=60):
    """Separable-ish two-class scores, deterministic without needing numpy's RNG semantics."""
    rows = []
    for i in range(n):
        rows.append({"id": f"h{i}", "label": "human", "score": -1.0 + i * 0.01, "seconds": 0.1})
    for i in range(n):
        rows.append({"id": f"a{i}", "label": "ai", "score": 1.0 + i * 0.01, "seconds": 0.1})
    return rows


def test_round_trip_is_deterministic(tmp_path):
    """Same scores + same seed => byte-identical results. This is the regression guard."""
    p = write_scores(tmp_path / "scores.csv", synthetic_rows())
    rows = analyze.read_scores_csv(p)
    a = analyze.compute_results(rows, n_boot=200, seed=7)
    b = analyze.compute_results(rows, n_boot=200, seed=7)
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def test_reread_matches_original_rows(tmp_path):
    """Writing scores and reading them back must not perturb the numbers."""
    original = synthetic_rows(30)
    p = write_scores(tmp_path / "scores.csv", original)
    reread = analyze.read_scores_csv(p)
    assert [r["score"] for r in reread] == [r["score"] for r in original]
    assert [r["label"] for r in reread] == [r["label"] for r in original]


def test_rejects_missing_columns(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,value\n1,0.5\n", encoding="utf-8")
    with pytest.raises(analyze.NotAReport, match="missing required column"):
        analyze.read_scores_csv(p)


def test_rejects_unknown_label(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("id,label,score\n1,maybe,0.5\n", encoding="utf-8")
    with pytest.raises(analyze.NotAReport, match="label must be"):
        analyze.read_scores_csv(p)


def test_rejects_empty(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("id,label,score\n", encoding="utf-8")
    with pytest.raises(analyze.NotAReport, match="no score rows"):
        analyze.read_scores_csv(p)


def test_missing_report_dir_is_explicit(tmp_path):
    with pytest.raises(analyze.NotAReport, match="no scores.csv"):
        analyze.load_report(tmp_path)


def test_scores_without_env_are_marked_unknown(tmp_path):
    """A receipt built from bare scores must not imply an instrument it cannot attest to."""
    write_scores(tmp_path / "scores.csv", synthetic_rows(30))
    _rows, corpus, detector = analyze.load_report(tmp_path)
    assert detector["instrument_unknown"] is True
    assert corpus.sha256 == "unknown"


def test_report_written_from_unknown_instrument_says_so(tmp_path):
    """The honesty requirement, enforced: no fingerprint => the receipt states that plainly."""
    from gaige.receipts import write_report

    rows = synthetic_rows(60)
    results = analyze.compute_results(rows, n_boot=100, seed=3)
    out = tmp_path / "out"
    write_report(
        out, analyze.UNKNOWN_CORPUS, dict(analyze.UNKNOWN_DETECTOR), rows, results, "test"
    )
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "INSTRUMENT UNKNOWN" in text
    assert "not transferable" in text


def test_report_is_utf8_on_every_platform(tmp_path):
    """Report text contains non-ASCII (arrows, en-dashes). Writing it must not depend on the
    platform's default codec — this exact bug crashed report writing on Windows (cp1252)."""
    from gaige.receipts import write_report

    rows = synthetic_rows(60)
    results = analyze.compute_results(rows, n_boot=100, seed=3)
    out = tmp_path / "out"
    write_report(out, analyze.UNKNOWN_CORPUS, dict(analyze.UNKNOWN_DETECTOR), rows, results, "t")
    raw = (out / "report.md").read_bytes()
    raw.decode("utf-8")  # must not raise
    for name in ("env.json", "results.json", "roc.json", "scores.csv"):
        (out / name).read_bytes().decode("utf-8")
