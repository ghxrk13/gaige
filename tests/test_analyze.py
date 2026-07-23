# gaige — calibration and drift receipts for AI measurement.
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
    write_report(out, analyze.UNKNOWN_CORPUS, dict(analyze.UNKNOWN_DETECTOR), rows, results, "test")
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "INSTRUMENT UNKNOWN" in text
    assert "not transferable" in text


def write_scores_v2(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "score", "seconds", "n_words", "meta"])
        w.writeheader()
        w.writerows(rows)
    return path


def synthetic_rows_with_words(n=120):
    rows = []
    for i in range(n):
        rows.append(
            {
                "id": f"h{i}",
                "label": "human",
                "score": -1.0 + i * 0.01,
                "seconds": 0.1,
                "n_words": 60 if i % 2 else 400,
                "meta": "",
            }
        )
    for i in range(n):
        rows.append(
            {
                "id": f"a{i}",
                "label": "ai",
                "score": 1.0 + i * 0.01,
                "seconds": 0.1,
                "n_words": 60 if i % 2 else 400,
                "meta": "",
            }
        )
    return rows


def test_results_carry_conformal_subgroups_base_rate(tmp_path):
    p = write_scores_v2(tmp_path / "scores.csv", synthetic_rows_with_words())
    rows = analyze.read_scores_csv(p)
    assert all(isinstance(r["n_words"], int) for r in rows)
    res = analyze.compute_results(rows, n_boot=100, seed=7)
    by_alpha = {r["alpha"]: r for r in res["conformal"]}
    assert "threshold" in by_alpha[0.05]  # 120 humans support alpha=0.05 and 0.01
    assert "threshold" in by_alpha[0.01]
    strata = res["subgroups"]["by_threshold"][0]["strata"]["length_bucket"]
    assert set(strata) == {"0-100w", "250-500w"}
    assert res["base_rate"]["at"][0]["expected_false_positives"] == pytest.approx(0.01 * 75000)


def test_rows_without_n_words_get_honest_unavailable(tmp_path):
    """Old score sets lack the n_words column; the subgroup block must say so, not guess."""
    p = write_scores(tmp_path / "scores.csv", synthetic_rows())
    rows = analyze.read_scores_csv(p)
    res = analyze.compute_results(rows, n_boot=100, seed=7)
    assert "unavailable" in res["subgroups"]


def test_results_json_ships_every_computed_key(tmp_path):
    """results.json is a wholesale write contract: every key compute_results emits lands
    in the file, except roc, which ships as its own artifact. A hand-kept key list is
    the same class of gap as a guard test that covers only one hop of a chain: a newly
    added statistic passes its unit tests and still never reaches the receipt."""
    import json as _json

    from gaige.receipts import write_report

    rows = synthetic_rows(60)
    results = analyze.compute_results(rows, n_boot=100, seed=3)
    out = tmp_path / "out"
    write_report(out, analyze.UNKNOWN_CORPUS, dict(analyze.UNKNOWN_DETECTOR), rows, results, "t")
    shipped = _json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert set(shipped.keys()) == set(results.keys()) - {"roc"}


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
