# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Resumable-run tests.

Most of these guard one property: a resume must never stitch together scores from two different
instruments. Skipping already-done work is the easy half; refusing to continue into a changed
environment is the half that keeps the resulting receipt honest.
"""

from __future__ import annotations

import pytest

from gaige import runstate


class Corpus:
    def __init__(self, sha="abc123", name="test-corpus"):
        self.sha256 = sha
        self.name = name
        self.counts = {"human": 2, "ai": 2}


META = {
    "detector": "fast-detect-gpt",
    "model_id": "gpt2",
    "quant_requested": "fp32",
    "max_tokens": 512,
    "device": "cpu",
    "versions": {"torch": "2.13.0", "transformers": "4.49.0"},
}


def begun(tmp_path, corpus=None, meta=None):
    runstate.write_runstate(tmp_path, corpus or Corpus(), meta or dict(META), "cmd")
    return runstate.read_runstate(tmp_path)


def test_roundtrip_runstate(tmp_path):
    state = begun(tmp_path)
    assert state["corpus"]["sha256"] == "abc123"
    assert state["detector"]["model_id"] == "gpt2"
    assert state["complete"] is False


def test_missing_runstate_is_explicit(tmp_path):
    with pytest.raises(runstate.ResumeRefused, match="not a resumable"):
        runstate.read_runstate(tmp_path)


def test_scores_survive_and_reload(tmp_path):
    fh, w = runstate.open_partial(tmp_path)
    runstate.append_row(fh, w, {"id": "a", "label": "human", "score": 0.5, "seconds": 0.1})
    runstate.append_row(fh, w, {"id": "b", "label": "ai", "score": 1.5, "seconds": 0.1})
    fh.close()
    rows = runstate.load_partial(tmp_path)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[1]["score"] == 1.5


def test_appending_does_not_duplicate_the_header(tmp_path):
    fh, w = runstate.open_partial(tmp_path)
    runstate.append_row(fh, w, {"id": "a", "label": "human", "score": 0.5, "seconds": 0.1})
    fh.close()
    fh, w = runstate.open_partial(tmp_path)  # reopened, as a resume would
    runstate.append_row(fh, w, {"id": "b", "label": "ai", "score": 1.5, "seconds": 0.1})
    fh.close()
    assert len(runstate.load_partial(tmp_path)) == 2


def test_truncated_final_line_is_dropped_not_fatal(tmp_path):
    """A process killed mid-write leaves a partial line. That sample is simply re-scored."""
    fh, w = runstate.open_partial(tmp_path)
    runstate.append_row(fh, w, {"id": "a", "label": "human", "score": 0.5, "seconds": 0.1})
    fh.close()
    with open(tmp_path / runstate.PARTIAL, "a", encoding="utf-8") as f:
        f.write("b,ai,NOT_A_NUMBER")
    rows = runstate.load_partial(tmp_path)
    assert [r["id"] for r in rows] == ["a"]


# --- the refusals: the reason this module exists -------------------------------------------


def test_matching_run_resumes(tmp_path):
    state = begun(tmp_path)
    runstate.check_args_match(state, Corpus(), "gpt2", "fp32", 512)
    runstate.check_instrument_match(state, dict(META))  # no exception


def test_different_corpus_refused(tmp_path):
    state = begun(tmp_path)
    with pytest.raises(runstate.ResumeRefused, match="corpus"):
        runstate.check_args_match(state, Corpus(sha="different"), "gpt2", "fp32", 512)


def test_different_model_refused_before_load(tmp_path):
    """Caught cheaply, before spending minutes downloading and loading a model."""
    state = begun(tmp_path)
    with pytest.raises(runstate.ResumeRefused, match="model_id"):
        runstate.check_args_match(state, Corpus(), "tiiuae/falcon-7b", "fp32", 512)


def test_different_quant_refused(tmp_path):
    state = begun(tmp_path)
    with pytest.raises(runstate.ResumeRefused, match="quant_requested"):
        runstate.check_args_match(state, Corpus(), "gpt2", "4bit", 512)


def test_different_max_tokens_refused(tmp_path):
    state = begun(tmp_path)
    with pytest.raises(runstate.ResumeRefused, match="max_tokens"):
        runstate.check_args_match(state, Corpus(), "gpt2", "fp32", 1024)


def test_device_change_refused_after_load(tmp_path):
    """A CUDA box that fell back to CPU on the retry. Different numerics, different instrument."""
    state = begun(tmp_path)
    meta = dict(META, device="cuda")
    with pytest.raises(runstate.ResumeRefused, match="device"):
        runstate.check_instrument_match(state, meta)


def test_library_upgrade_refused_after_load(tmp_path):
    """The subtle one: an upgraded transformers between the two halves of a run."""
    state = begun(tmp_path)
    meta = dict(META, versions={"torch": "2.13.0", "transformers": "5.14.1"})
    with pytest.raises(runstate.ResumeRefused, match="versions"):
        runstate.check_instrument_match(state, meta)


def test_refusal_explains_the_consequence(tmp_path):
    state = begun(tmp_path)
    with pytest.raises(runstate.ResumeRefused) as e:
        runstate.check_instrument_match(state, dict(META, device="cuda"))
    msg = str(e.value)
    assert "two different instruments" in msg
    assert "Start a fresh run" in msg  # tells the operator what to do instead


def test_completion_clears_the_partial(tmp_path):
    begun(tmp_path)
    fh, w = runstate.open_partial(tmp_path)
    runstate.append_row(fh, w, {"id": "a", "label": "human", "score": 0.5, "seconds": 0.1})
    fh.close()
    assert (tmp_path / runstate.PARTIAL).exists()
    runstate.mark_complete(tmp_path)
    assert not (tmp_path / runstate.PARTIAL).exists()
    assert runstate.read_runstate(tmp_path)["complete"] is True
