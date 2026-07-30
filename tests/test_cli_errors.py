# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.

"""The CLI boundary prints refusals as plain errors, never tracebacks.

Found by the 0.0.1 outside-user pass (2026-07-26): a stranger with a core-only
install hit a raw ModuleNotFoundError, and a malformed corpus surfaced a traceback
around a perfectly good message. Refusals are expected outputs; tracebacks are for
genuinely unexpected bugs only.
"""

import pytest

from gaige import cli


def test_bad_corpus_is_a_clean_error(tmp_path, capsys):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"text": "no label here"}\n')
    rc = cli.main(["run", "--corpus", str(bad), "--detector", "fast-detect-gpt"])
    err = capsys.readouterr().err
    assert rc == 2
    assert "rows need text + label" in err
    assert "Traceback" not in err


def test_missing_gpu_extra_names_the_fix(monkeypatch, capsys):
    def boom(_args):
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(cli, "cmd_corpora", boom)
    rc = cli.main(["corpora"])
    err = capsys.readouterr().err
    assert rc == 2
    assert 'pip install "gaige[gpu]"' in err
    assert "Analysis commands" in err


def test_admit_requires_exactly_one_source(capsys):
    # argparse enforces the mutually-exclusive required group before any code runs
    with pytest.raises(SystemExit):
        cli.main(["admit", "--baseline", "x"])
    with pytest.raises(SystemExit):
        cli.main(
            ["admit", "--baseline", "x", "--candidate", "a.jsonl", "--candidate-scores", "b.csv"]
        )


def test_admit_missing_baseline_is_a_clean_error(tmp_path, capsys):
    scores = tmp_path / "s.csv"
    scores.write_text("id,score\nc0,1.0\n", encoding="utf-8")
    rc = cli.main(
        ["admit", "--baseline", str(tmp_path / "nope"), "--candidate-scores", str(scores)]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "no scores.csv" in err
    assert "Traceback" not in err


def test_admit_garbage_alpha_is_a_clean_error(tmp_path, capsys):
    scores = tmp_path / "s.csv"
    scores.write_text("id,score\nc0,1.0\n", encoding="utf-8")
    rc = cli.main(
        [
            "admit",
            "--baseline",
            str(tmp_path / "nope"),
            "--candidate-scores",
            str(scores),
            "--alphas",
            "banana",
        ]
    )
    err = capsys.readouterr().err
    assert rc == 2
    assert "Traceback" not in err
