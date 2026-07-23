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
    """Deterministic canned answers; identity stable so resumes match.

    With `conf=(c_correct, c_wrong)` it also answers P(True) queries as a perfectly
    controllable self-assessor: it parses the proposed answer out of the P(True) template
    and returns logprobs that make ptrue_score come out exactly c_correct or c_wrong.
    """

    name = "fake"

    def __init__(
        self,
        answers: dict[str, str],
        is_local: bool = True,
        fail_after: int = -1,
        conf: tuple[float, float] | None = None,
        truth: dict[str, str] | None = None,
    ):
        self.answers = answers  # what it SAYS (may include deliberate wrongs)
        self.truth = truth  # what is TRUE (defaults to answers: a fake that is never wrong)
        self.is_local = is_local
        self.fail_after = fail_after
        self.conf = conf
        self.calls = 0

    def capabilities(self):
        caps = {CAP_COMPLETE}
        if self.conf is not None:
            from gaige.providers.base import CAP_OPTION_LOGPROBS

            caps.add(CAP_OPTION_LOGPROBS)
        return frozenset(caps)

    def complete(self, prompt, decoding):
        self.calls += 1
        if self.fail_after >= 0 and self.calls > self.fail_after:
            raise RuntimeError("simulated provider outage")
        return self.answers.get(prompt, "no idea")

    def option_logprobs(self, prompt, options):
        import math

        if self.conf is None:  # pragma: no cover - never declared without conf
            raise NotImplementedError
        q = prompt.split("Question: ", 1)[1].split("\nProposed answer: ")[0]
        proposed = prompt.split("\nProposed answer: ", 1)[1].split("\nIs the proposed", 1)[0]
        c = self.conf[0] if proposed == (self.truth or self.answers).get(q) else self.conf[1]
        return {"True": math.log(c), "False": math.log(1.0 - c)}

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


def test_ptrue_measures_m3_with_controlled_calibration(tmp_path):
    """A fake self-assessor at 0.8-when-right / 0.9-when-wrong: every M3 number is
    hand-computable. t0 (2 wrong of 4): mean conf .85, acc .5, gap +.35, ECE .55 —
    the same arithmetic as the probcal hand case, arriving through the full runner."""
    ps = make_set(tmp_path)
    prov = FakeProvider(
        right_answers(ps, wrong_ids=("p0", "p1")), conf=(0.8, 0.9), truth=right_answers(ps)
    )
    out = tmp_path / "run"
    results = proberun.run_probes(
        ps,
        prov,
        Decoding(),
        cutoff="2024-06-01",
        outdir=out,
        n_boot=100,
        with_ptrue=True,
        progress=lambda *_: None,
    )
    m3_t0 = results["by_vintage"]["t0"]["m3"]
    assert m3_t0["mean_confidence"] == pytest.approx(0.85)
    assert m3_t0["gap"] == pytest.approx(0.35)
    assert m3_t0["ece"] == pytest.approx(0.55)
    m3_t1 = results["by_vintage"]["t1"]["m3"]
    assert m3_t1["gap"] == pytest.approx(0.8 - 1.0)  # all right, conf .8: UNDERconfident
    text = (out / "report.md").read_text(encoding="utf-8")
    assert "M3 — calibration drift" in text and "ptrue-1" in text


def test_toggling_ptrue_refuses_resume(tmp_path):
    ps = make_set(tmp_path)
    out = tmp_path / "run"
    flaky = FakeProvider(right_answers(ps), fail_after=2, conf=(0.8, 0.9))
    with pytest.raises(RuntimeError):
        proberun.run_probes(
            ps,
            flaky,
            Decoding(),
            cutoff="2024-06-01",
            outdir=out,
            n_boot=100,
            with_ptrue=True,
            progress=lambda *_: None,
        )
    with pytest.raises(runstate.ResumeRefused, match="ptrue"):
        proberun.run_probes(
            ps,
            FakeProvider(right_answers(ps)),
            Decoding(),
            cutoff="2024-06-01",
            outdir=out,
            n_boot=100,
            resume=True,
            with_ptrue=False,
            progress=lambda *_: None,
        )


def test_ptrue_requires_option_logprobs(tmp_path):
    ps = make_set(tmp_path)
    with pytest.raises(MissingCapability, match="option_logprobs"):
        proberun.run_probes(
            ps,
            FakeProvider(right_answers(ps)),
            Decoding(),
            cutoff="2024-06-01",
            outdir=tmp_path / "run",
            n_boot=100,
            with_ptrue=True,
            progress=lambda *_: None,
        )
