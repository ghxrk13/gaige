# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.

"""The CLI boundary prints refusals as plain errors, never tracebacks.

Found by the 0.0.1 outside-user pass (2026-07-26): a stranger with a core-only
install hit a raw ModuleNotFoundError, and a malformed corpus surfaced a traceback
around a perfectly good message. Refusals are expected outputs; tracebacks are for
genuinely unexpected bugs only.
"""

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
