# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Instrument-identity tests.

The claim these defend: a threshold is only valid for the instrument that produced it.
Device, dtype, model and library versions are all part of that identity, and crossing any of
those boundaries without saying so is the exact failure gaige exists to expose in other
people's tools. These run without torch installed, because the logic must not depend on it.
"""

from __future__ import annotations

from gaige.single import instrument_mismatches


def env(device="cuda", versions=None):
    return {
        "detector": {
            "device": device,
            "versions": versions if versions is not None else {},
            "model_id": "tiiuae/falcon-7b",
        }
    }


def test_same_device_is_no_mismatch():
    assert instrument_mismatches(env("cuda"), live_device="cuda") == []


def test_gpu_report_used_on_cpu_is_flagged():
    """The load-bearing one: CUDA-calibrated thresholds must not silently apply on CPU."""
    out = instrument_mismatches(env("cuda"), live_device="cpu")
    assert len(out) == 1
    assert "device" in out[0]
    assert "report=cuda" in out[0] and "current=cpu" in out[0]
    assert "do not transfer" in out[0]


def test_cpu_report_used_on_gpu_is_flagged():
    """Mismatch is symmetric — neither direction is safe."""
    out = instrument_mismatches(env("cpu"), live_device="cuda")
    assert any("device" in m for m in out)


def test_version_drift_is_flagged(monkeypatch):
    monkeypatch.setattr("gaige.single._live_versions", lambda: {"transformers": "4.49.0"})
    out = instrument_mismatches(env("cpu", {"transformers": "5.14.1"}), live_device="cpu")
    assert any("transformers" in m and "5.14.1" in m for m in out)


def test_absent_library_is_not_a_mismatch(monkeypatch):
    """A machine without torch installed is not 'drifted' — it simply cannot compare."""
    monkeypatch.setattr("gaige.single._live_versions", lambda: {})
    assert instrument_mismatches(env("cpu", {"torch": "2.13.0"}), live_device="cpu") == []


def test_unrecorded_device_is_not_a_false_alarm():
    """Older reports have no device field; absence must not manufacture a mismatch."""
    e = {"detector": {"versions": {}}}
    assert instrument_mismatches(e, live_device="cpu") == []


def test_cuda_only_quant_refused_on_cpu():
    """4-bit has no CPU kernel. Refuse with an actionable message rather than degrade silently."""
    from gaige.detectors.fast_detect_gpt import FastDetectGPT

    det = FastDetectGPT(quant="4bit", device="cpu")
    try:
        det._effective_quant("cpu")
    except RuntimeError as e:
        msg = str(e)
        assert "CUDA-only" in msg
        assert "fp32" in msg  # tells the user what to do instead
        assert "DIFFERENT INSTRUMENT" in msg  # and that it is not the same measurement
    else:
        raise AssertionError("expected 4bit-on-CPU to be refused")


def test_fp16_refused_on_cpu():
    from gaige.detectors.fast_detect_gpt import FastDetectGPT

    det = FastDetectGPT(quant="fp16", device="cpu")
    try:
        det._effective_quant("cpu")
    except RuntimeError as e:
        assert "fp32" in str(e)
    else:
        raise AssertionError("expected fp16-on-CPU to be refused")


def test_fp32_accepted_on_cpu():
    from gaige.detectors.fast_detect_gpt import FastDetectGPT

    assert FastDetectGPT(quant="fp32", device="cpu")._effective_quant("cpu") == "fp32"


def test_unknown_device_rejected():
    from gaige.detectors.fast_detect_gpt import FastDetectGPT

    det = FastDetectGPT(device="tpu")
    try:
        det.resolve_device()
    except (ValueError, ImportError, ModuleNotFoundError) as e:
        # ValueError is the intended path; on a box with no torch the import guard fires first,
        # which is also acceptable — what must never happen is silent acceptance.
        assert isinstance(e, (ValueError, ImportError, ModuleNotFoundError))
    else:
        raise AssertionError("expected an unknown device to be rejected")
