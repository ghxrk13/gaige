# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Simulation tests asserting the statistical PROPERTY, not mere execution.

The claim under test is marginal: P(human flagged) <= alpha averaged over calibration
draws. The harness draws many independent (calibration, test) pairs and checks the realized
false-positive rate against the bound — and, sharper, against the exact conditional law
Beta(n+1-k, k) whose mean and sd conformal_threshold reports. An off-by-one in the order
statistic shifts that law by 1/(n+1), which these tolerances resolve decisively;
test_off_by_one_would_break_the_bound demonstrates the detection margin, and the suite was
additionally run once against a deliberately broken k to confirm it goes red (recorded in
the private research review of 2026-07-22).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from gaige import conformal


def mc_fprs(alpha, n_cal, trials, n_test=2000, seed=11):
    """Realized per-trial FPRs of conformal thresholds over independent draws."""
    rng = np.random.default_rng(seed)
    fprs = []
    for _ in range(trials):
        cal = rng.normal(size=n_cal)
        thr = conformal.conformal_threshold(cal, alpha)["threshold"]
        test = rng.normal(size=n_test)
        fprs.append(float((test >= thr).mean()))
    return np.array(fprs)


def test_marginal_bound_holds_across_alphas():
    for alpha, n_cal in ((0.05, 100), (0.01, 120), (0.005, 250)):
        fprs = mc_fprs(alpha, n_cal, trials=300)
        k = min(math.ceil((n_cal + 1) * (1.0 - alpha)), n_cal)
        exact_mean = (n_cal + 1 - k) / (n_cal + 1.0)
        assert exact_mean <= alpha + 1e-12  # the theory itself
        assert fprs.mean() == pytest.approx(exact_mean, abs=0.004)  # the code matches it
        assert fprs.mean() <= alpha + 0.004  # and therefore the bound, up to MC noise


def test_off_by_one_would_break_the_bound():
    """Proof the tolerance above has teeth: one order statistic lower shifts the marginal
    mean a full 1/(n+1) upward, and the same Monte Carlo separates that decisively."""
    alpha, n_cal, trials, n_test = 0.05, 100, 300, 2000
    rng = np.random.default_rng(11)
    fprs = []
    for _ in range(trials):
        cal = np.sort(rng.normal(size=n_cal))
        k = math.ceil((n_cal + 1) * (1.0 - alpha))
        thr = float(np.nextafter(cal[k - 2], np.inf))  # broken: s[k-1] would be correct
        test = rng.normal(size=n_test)
        fprs.append(float((test >= thr).mean()))
    assert np.mean(fprs) > alpha + 0.004


def test_conditional_dispersion_matches_beta_law():
    alpha, n_cal = 0.05, 100
    row = conformal.conformal_threshold(np.random.default_rng(0).normal(size=n_cal), alpha)
    n, k = row["n_calibration"], row["order_statistic"]
    a, b = n + 1 - k, k
    assert row["conditional_fpr_mean"] == pytest.approx(a / (n + 1.0))
    assert row["conditional_fpr_sd"] == pytest.approx(
        math.sqrt(a * b / ((n + 1.0) ** 2 * (n + 2.0)))
    )
    # The reported sd must describe reality: recover it from simulated draws after
    # subtracting the finite-test-set binomial noise.
    fprs = mc_fprs(alpha, n_cal, trials=400, n_test=4000, seed=5)
    binom_var = fprs.mean() * (1.0 - fprs.mean()) / 4000.0
    est_sd = math.sqrt(max(float(fprs.var()) - binom_var, 0.0))
    assert est_sd == pytest.approx(row["conditional_fpr_sd"], rel=0.25)


def test_min_samples_exact_and_refusal_fires():
    for alpha, need in ((0.05, 19), (0.01, 99), (0.005, 199), (0.03, 33)):
        assert conformal.min_samples_for(alpha) == need
        with pytest.raises(conformal.InsufficientCalibration):
            conformal.conformal_threshold(np.zeros(need - 1), alpha)
        conformal.conformal_threshold(np.linspace(0.0, 1.0, need), alpha)  # must not raise


def test_alpha_validation():
    for bad in (0.0, 1.0, -0.1, 1.5):
        with pytest.raises(ValueError):
            conformal.min_samples_for(bad)
        with pytest.raises(ValueError):
            conformal.conformal_threshold(np.zeros(500), bad)


def test_flag_rule_is_strictly_greater_than_order_stat():
    cal = np.arange(1.0, 101.0)
    row = conformal.conformal_threshold(cal, 0.05)
    s_k = float(np.sort(cal)[row["order_statistic"] - 1])
    assert row["threshold"] == np.nextafter(s_k, np.inf)
    assert not (s_k >= row["threshold"])  # a score EQUAL to the order stat does not flag
    assert np.nextafter(s_k, np.inf) >= row["threshold"]  # the next float up does


def test_no_pseudo_achieved_rate_in_output():
    """The in-sample flag rate is (n-k)/n by construction — a function of n and alpha, not
    a measurement. It must not appear dressed as an achieved rate."""
    row = conformal.conformal_threshold(np.random.default_rng(1).normal(size=100), 0.05)
    assert "empirical_fpr" not in row
    assert "marginal" in row["guarantee"]


def test_table_reports_tprs_and_refusals():
    rng = np.random.default_rng(3)
    rows = conformal.conformal_table(rng.normal(0, 1, 100), rng.normal(3, 1, 100))
    by_alpha = {r["alpha"]: r for r in rows}
    assert "unavailable" in by_alpha[0.005]  # 100 humans < the 199 needed
    assert 0.0 <= by_alpha[0.05]["tpr"] <= 1.0
    assert by_alpha[0.01]["order_statistic"] == 100  # k = ceil(101*0.99) = 100 = n


def test_deterministic():
    cal = np.random.default_rng(4).normal(size=150)
    assert conformal.conformal_threshold(cal, 0.01) == conformal.conformal_threshold(cal, 0.01)
