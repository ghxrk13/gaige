# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""M5 monitor tests: known injected shifts must be caught with sane latency, flat series
must stay quiet, the conformal alarm must refuse young references, and the scorecard
arithmetic (latency, false alarms) is exact on constructed sequences."""

from __future__ import annotations

import numpy as np
import pytest

from gaige import monitors
from gaige.conformal import InsufficientCalibration


def drifting(n_flat=10, n_drift=6, level=0.8, drop=0.3, noise=0.01, seed=5):
    rng = np.random.default_rng(seed)
    flat = level + rng.normal(0, noise, n_flat)
    fallen = level - drop + rng.normal(0, noise, n_drift)
    return np.concatenate([flat, fallen]), n_flat  # onset index


def test_page_hinkley_catches_drop_and_stays_quiet_on_flat():
    values, onset = drifting()
    res = monitors.page_hinkley(values, delta=0.005, lam=0.05, direction="down")
    score = monitors.evaluate(res, onset)
    assert score["detected"] and score["false_alarms"] == 0
    assert score["detection_latency"] <= 2  # a 0.3 drop against 0.01 noise: near-immediate
    flat = 0.8 + np.random.default_rng(6).normal(0, 0.01, 20)
    assert monitors.page_hinkley(flat, delta=0.005, lam=0.05, direction="down")["alarms"] == []


def test_cusum_catches_drop_and_stays_quiet_on_flat():
    values, onset = drifting()
    res = monitors.cusum(values, reference_mean=0.8, k=0.01, h=0.05, direction="down")
    score = monitors.evaluate(res, onset)
    assert score["detected"] and score["false_alarms"] == 0
    assert score["detection_latency"] <= 2
    flat = 0.8 + np.random.default_rng(7).normal(0, 0.01, 20)
    assert (
        monitors.cusum(flat, reference_mean=0.8, k=0.01, h=0.05, direction="down")["alarms"] == []
    )


def test_subthreshold_shift_is_not_hallucinated():
    """Specificity: a shift smaller than the slack must NOT alarm — a monitor that flags
    everything has perfect latency and no honesty."""
    rng = np.random.default_rng(8)
    values = np.concatenate([0.8 + rng.normal(0, 0.002, 10), 0.795 + rng.normal(0, 0.002, 10)])
    assert (
        monitors.cusum(values, reference_mean=0.8, k=0.02, h=0.08, direction="down")["alarms"] == []
    )
    assert monitors.page_hinkley(values, delta=0.02, lam=0.08, direction="down")["alarms"] == []


def test_conformal_alarm_bound_and_refusal():
    rng = np.random.default_rng(9)
    ref = 0.8 + rng.normal(0, 0.01, 19)  # exactly min_samples_for(0.05)
    values, onset = drifting()
    res = monitors.conformal_alarm(ref, values, alpha=0.05, direction="down")
    score = monitors.evaluate(res, onset)
    assert score["detected"] and score["false_alarms"] == 0
    assert "marginal" in res["guarantee"]
    with pytest.raises(InsufficientCalibration):
        monitors.conformal_alarm(ref[:3], values, alpha=0.05)  # young reference refuses


def test_conformal_alarm_direction_up():
    """Gap alarms watch a RISING quantity; orientation must not invert the guarantee."""
    rng = np.random.default_rng(10)
    ref = 0.05 + rng.normal(0, 0.01, 19)
    values = np.concatenate([0.05 + rng.normal(0, 0.01, 5), [0.4, 0.45]])
    res = monitors.conformal_alarm(ref, values, alpha=0.05, direction="up")
    assert res["alarms"] == [5, 6]


def test_evaluate_arithmetic_exact():
    fake = {"monitor": "x", "alarms": [2, 7, 9]}
    s = monitors.evaluate(fake, onset=6)
    assert s["false_alarms"] == 1 and s["false_alarm_indices"] == [2]
    assert s["detection_latency"] == 1 and s["detected"] is True
    assert monitors.evaluate({"monitor": "x", "alarms": [1]}, onset=5)["detection_latency"] is None


def test_watch_panel_returns_refusal_rows_not_exceptions():
    values, _ = drifting()
    panel = monitors.watch([0.8, 0.81, 0.79], values, alpha=0.05)  # 3 refs < 19 needed
    names = {r["monitor"]: r for r in panel}
    assert "refused" in names["conformal-interval-down"]
    assert names["page-hinkley-down"]["alarms"]  # cumulative detectors still ran
    assert names["cusum-down"]["alarms"]
