# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Probe-runner tests via an injectable fake provider (the proven test_privacy pattern):
no model, no network, and every refusal path exercised for real."""

from __future__ import annotations

import pytest

from gaige import proberun, runstate
from gaige.probes import load_probes
from gaige.providers.base import (
    CAP_COMPLETE,
    Decoding,
    MissingCapability,
    require,
    require_local_or_optin,
)
from tests.test_probes import probe, write_probeset


class FakeProvider:
    """Deterministic canned answers; identity stable so resumes match."""

    name = "fake"

    def __init__(self, answers: dict[str, str], is_local: bool = True, fail_after: int = -1):
        self.answers = answers
        self.is_local = is_local
        self.fail_after = fail_after
        self.calls = 0

    def capabilities(self):
        return frozenset({CAP_COMPLETE})

    def complete(self, prompt, decoding):
        self.calls += 1
        if self.fail_after >= 0 and self.calls > self.fail_after:
            raise RuntimeError("simulated provider outage")
        return self.answers.get(prompt, "no idea")

    def option_logprobs(self, prompt, options):  # pragma: no cover - never declared
        raise NotImplementedError

    def metadata(self):
        return {
            "provider": self.name,
            "attestation": "self-reported",
            "attestation_basis": "test fixture",
            "is_local": self.is_local,
            "model_id": "fake-1",
        }


def make_set(tmp_path, n_t0=4, n_t1=4):
    rows = [probe(i, vintage="t0") for i in range(n_t0)]
    rows += [probe(100 + i, vintage="t1") for i in range(n_t1)]
    return load_probes(write_probeset(tmp_path / "set.jsonl", rows))


def right_answers(ps, wrong_ids=()):
    return {p["prompt"]: (p["answer"] if p["id"] not in wrong_ids else "wrong") for p in ps.probes}


def test_full_run_grades_and_reports(tmp_path):
    ps = make_set(tmp_path)
    prov = FakeProvider(right_answers(ps, wrong_ids=("p0",)))  # one wrong in t0
    out = tmp_path / "run"
    results = proberun.run_probes(
        ps, prov, Decoding(), cutoff="2024-06-01", outdir=out, n_boot=100, progress=lambda *_: None
    )
    assert results["by_vintage"]["t0"]["accuracy"] == 0.75
    assert results["by_vintage"]["t1"]["accuracy"] == 1.0
    lo, hi = results["by_vintage"]["t0"]["accuracy_ci"]
    assert lo <= 0.75 <= hi
    assert results["post_cutoff_share"]["t0"]["share"] == 1.0
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "attestation: self-reported" in text
    assert "post-cutoff" in text
    assert "grading `nem-1`" in text or "nem-1" in text
    assert (out / "answers.csv").exists()
    assert not (out / runstate.PARTIAL).exists()  # completed runs clean up their partial


def test_interrupted_run_resumes_and_matches(tmp_path):
    ps = make_set(tmp_path)
    out = tmp_path / "run"
    flaky = FakeProvider(right_answers(ps), fail_after=3)
    with pytest.raises(RuntimeError, match="outage"):
        proberun.run_probes(
            ps,
            flaky,
            Decoding(),
            cutoff="2024-06-01",
            outdir=out,
            n_boot=100,
            progress=lambda *_: None,
        )
    healthy = FakeProvider(right_answers(ps))
    results = proberun.run_probes(
        ps,
        healthy,
        Decoding(),
        cutoff="2024-06-01",
        outdir=out,
        n_boot=100,
        resume=True,
        progress=lambda *_: None,
    )
    assert results["by_vintage"]["t0"]["n"] == 4 and results["by_vintage"]["t1"]["n"] == 4
    assert healthy.calls == 8 - 3  # resumed, not re-asked


def test_resume_refuses_changed_decoding(tmp_path):
    ps = make_set(tmp_path)
    out = tmp_path / "run"
    flaky = FakeProvider(right_answers(ps), fail_after=2)
    with pytest.raises(RuntimeError):
        proberun.run_probes(
            ps,
            flaky,
            Decoding(),
            cutoff="2024-06-01",
            outdir=out,
            n_boot=100,
            progress=lambda *_: None,
        )
    with pytest.raises(runstate.ResumeRefused, match="decoding"):
        proberun.run_probes(
            ps,
            FakeProvider(right_answers(ps)),
            Decoding(temperature=0.7, seed=1),
            cutoff="2024-06-01",
            outdir=out,
            n_boot=100,
            resume=True,
            progress=lambda *_: None,
        )


def test_resume_refuses_changed_probeset(tmp_path):
    ps = make_set(tmp_path)
    out = tmp_path / "run"
    flaky = FakeProvider(right_answers(ps), fail_after=2)
    with pytest.raises(RuntimeError):
        proberun.run_probes(
            ps,
            flaky,
            Decoding(),
            cutoff="2024-06-01",
            outdir=out,
            n_boot=100,
            progress=lambda *_: None,
        )
    other = load_probes(write_probeset(tmp_path / "other.jsonl", [probe(9)]))
    with pytest.raises(runstate.ResumeRefused, match="probe set changed"):
        proberun.run_probes(
            other,
            FakeProvider({}),
            Decoding(),
            cutoff="2024-06-01",
            outdir=out,
            n_boot=100,
            resume=True,
            progress=lambda *_: None,
        )


def test_remote_provider_needs_explicit_optin():
    remote = FakeProvider({}, is_local=False)
    with pytest.raises(RuntimeError, match="allow-remote-text"):
        require_local_or_optin(remote, allow_remote_text=False)
    require_local_or_optin(remote, allow_remote_text=True)  # opt-in passes
    require_local_or_optin(FakeProvider({}), allow_remote_text=False)  # local never blocked


def test_missing_capability_named():
    class NoComplete(FakeProvider):
        def capabilities(self):
            return frozenset()

    with pytest.raises(MissingCapability, match="complete"):
        require(NoComplete({}), CAP_COMPLETE)
