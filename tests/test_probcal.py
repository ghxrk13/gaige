# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Probability-calibration tests: exact hand-computable cases (an off-by-one in binning
changes them — inherent teeth), plus recovery of an INJECTED miscalibration."""

from __future__ import annotations

import numpy as np
import pytest

from gaige import probcal


def test_ece_exact_two_bin_hand_case():
    """Half the mass at conf .8 / acc 1.0, half at conf .9 / acc 0.0.

    ECE = 0.5*|1.0-0.8| + 0.5*|0.0-0.9| = 0.1 + 0.45 = 0.55, exactly. A binning
    off-by-one merges or shifts the groups and this number moves — proven red once by
    breaking the bin index during development.
    """
    conf = np.array([0.8, 0.8, 0.9, 0.9])
    corr = np.array([1.0, 1.0, 0.0, 0.0])
    e = probcal.ece(conf, corr)
    assert e["ece"] == pytest.approx(0.55)
    assert e["bins"][8]["n"] == 2 and e["bins"][9]["n"] == 2  # .8->bin 8, .9->bin 9
    assert probcal.confidence_accuracy_gap(conf, corr) == pytest.approx(0.85 - 0.5)


def test_conf_one_lands_in_top_bin():
    e = probcal.ece(np.array([1.0, 1.0]), np.array([1.0, 1.0]))
    assert e["bins"][probcal.DEFAULT_BINS - 1]["n"] == 2
    assert e["ece"] == pytest.approx(0.0)


def test_calibrated_synthetic_scores_near_zero():
    """correct ~ Bernoulli(conf) means ECE should be small; injecting a 0.25 confidence
    inflation must be recovered as ~0.25 — the property, not mere execution."""
    rng = np.random.default_rng(11)
    conf = rng.uniform(0.05, 0.95, 4000)
    corr = (rng.random(4000) < conf).astype(float)
    assert probcal.ece(conf, corr)["ece"] < 0.03

    inflated = np.clip(conf + 0.25, 0.0, 1.0)
    e = probcal.ece(inflated, corr)["ece"]
    assert e == pytest.approx(0.25, abs=0.04)
    assert probcal.confidence_accuracy_gap(inflated, corr) == pytest.approx(0.25, abs=0.03)


def test_ece_ci_deterministic_and_brackets_point():
    rng = np.random.default_rng(3)
    conf = rng.uniform(0, 1, 300)
    corr = (rng.random(300) < conf).astype(float)
    a = probcal.ece_ci(conf, corr, n_boot=300, seed=5)
    assert a == probcal.ece_ci(conf, corr, n_boot=300, seed=5)
    point = probcal.ece(conf, corr)["ece"]
    lo, hi = a
    assert lo <= point <= hi or point < 0.05  # tiny-ECE points can sit below the bootstrap band


def test_validation():
    with pytest.raises(ValueError, match="equal-length"):
        probcal.ece(np.array([0.5]), np.array([]))
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        probcal.ece(np.array([1.5]), np.array([1.0]))


def test_brier_perfect_predictions_score_zero():
    r = probcal.brier(np.array([1.0, 0.0, 1.0]), np.array([1.0, 0.0, 1.0]))
    assert r["brier"] == 0.0
    assert r["n"] == 3


def test_brier_constant_half_scores_quarter():
    r = probcal.brier(np.array([0.5, 0.5, 0.5, 0.5]), np.array([1.0, 0.0, 1.0, 0.0]))
    assert r["brier"] == pytest.approx(0.25)


def test_brier_hand_computed_case():
    # (0.8-1)^2 = 0.04 and (0.4-0)^2 = 0.16, mean 0.10.
    r = probcal.brier(np.array([0.8, 0.4]), np.array([1.0, 0.0]))
    assert r["brier"] == pytest.approx(0.10)


def test_brier_empty_is_honest_nan():
    # An empty vintage must refuse a number, not invent one: NaN with n=0.
    r = probcal.brier(np.array([]), np.array([]))
    assert r["n"] == 0
    assert np.isnan(r["brier"])
