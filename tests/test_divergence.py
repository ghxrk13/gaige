# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Two-sample divergence math, held to hand-computed cases.

The two-sided band is exact order statistics, so tiny constructed references make every
number checkable by hand. Bootstrap values are asserted for same-seed reproducibility,
never for exact floats across environments (numpy stream posture, 2026-07-22).
"""

from __future__ import annotations

import numpy as np
import pytest

from gaige import divergence
from gaige.conformal import InsufficientCalibration, min_samples_for

# ------------------------------------------------------------------ KS distance


def test_ks_identical_samples_is_zero():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert divergence.ks_statistic(a, a) == 0.0


def test_ks_disjoint_samples_is_one():
    a = np.array([0.0, 0.1, 0.2])
    b = np.array([5.0, 5.1, 5.2])
    assert divergence.ks_statistic(a, b) == 1.0


def test_ks_hand_computed():
    # ECDFs at the pooled points {0, 1}: |3/4 - 0| = 0.75, |1 - 1| = 0.
    a = np.array([0.0, 0.0, 0.0, 1.0])
    b = np.array([1.0, 1.0, 1.0])
    assert divergence.ks_statistic(a, b) == pytest.approx(0.75)


def test_ks_symmetric_and_tie_safe():
    a = np.array([1.0, 1.0, 2.0, 2.0])
    b = np.array([1.0, 2.0, 2.0])
    assert divergence.ks_statistic(a, b) == pytest.approx(divergence.ks_statistic(b, a))


def test_ks_empty_refuses():
    with pytest.raises(ValueError):
        divergence.ks_statistic(np.array([]), np.array([1.0]))


def test_ks_with_ci_same_seed_reproducible():
    rng = np.random.default_rng(1)
    a, b = rng.normal(0, 1, 60), rng.normal(0.5, 1, 60)
    r1 = divergence.ks_with_ci(a, b, n_boot=50, seed=17)
    r2 = divergence.ks_with_ci(a, b, n_boot=50, seed=17)
    assert r1 == r2
    lo, hi = r1["ci"]
    assert 0.0 <= lo <= hi <= 1.0


# ------------------------------------------------------------------ two-sided band


def test_two_sided_floor_formula():
    # ceil(2/alpha) - 1, via the per-side feasibility at alpha/2.
    assert divergence.two_sided_min_samples(0.05) == 39
    assert divergence.two_sided_min_samples(0.01) == 199
    assert divergence.two_sided_min_samples(0.005) == 399
    for a in (0.05, 0.01, 0.005):
        assert divergence.two_sided_min_samples(a) == min_samples_for(a / 2.0)


def test_band_refuses_below_floor_with_two_sided_wording():
    ref = np.arange(38, dtype=np.float64)
    with pytest.raises(InsufficientCalibration) as e:
        divergence.conformal_band(ref, 0.05)
    msg = str(e.value)
    assert "two-sided alpha=0.05" in msg
    assert ">= 39" in msg


def test_band_exact_order_statistics_at_the_floor():
    # ref = 1..39 at alpha 0.05: per side k = ceil(40 * 0.975) = 39, so the band is
    # (just below 1, just above 39) and each side's conditional exceedance mean is 1/40.
    ref = np.arange(1, 40, dtype=np.float64)
    band = divergence.conformal_band(ref, 0.05)
    assert band["hi"] > 39.0 and band["hi"] == pytest.approx(39.0)
    assert band["lo"] < 1.0 and band["lo"] == pytest.approx(1.0)
    assert band["upper"]["order_statistic"] == 39
    assert band["lower"]["order_statistic"] == 39
    assert band["upper"]["conditional_exceedance_mean"] == pytest.approx(1 / 40)
    assert band["lower"]["conditional_exceedance_mean"] == pytest.approx(1 / 40)
    # Boundary values sit INSIDE the band (strict one-sidedness preserved per side).
    assert not (39.0 >= band["hi"]) and not (39.0 <= band["lo"])
    assert not (1.0 >= band["hi"]) and not (1.0 <= band["lo"])
    # Clear exceedances sit outside.
    assert 40.0 >= band["hi"]
    assert 0.5 <= band["lo"]


def test_band_constant_reference_is_degenerate_but_honest():
    ref = np.full(39, 5.0)
    band = divergence.conformal_band(ref, 0.05)
    assert not (5.0 >= band["hi"]) and not (5.0 <= band["lo"])  # the value itself: inside
    assert 6.0 >= band["hi"]  # anything above: outside
    assert 4.0 <= band["lo"]  # anything below: outside


# ------------------------------------------------------------------ novelty rows


def test_novelty_refusal_row_at_unsupported_alpha():
    ref = np.arange(1, 101, dtype=np.float64)  # n=100: supports .05, refuses .01 and .005
    cand = np.array([50.0, 200.0])
    rows = divergence.novelty_rows(ref, cand, alphas=(0.05, 0.01), n_boot=50, seed=17)
    assert "unavailable" not in rows[0]
    assert rows[1]["alpha"] == 0.01
    assert "unavailable" in rows[1]


def test_novelty_rate_hand_computed():
    # ref = 1..199 supports alpha 0.01 exactly (needs 199); band = (just under 1, just
    # over 199). Candidates 0 and 300 are outside, 100 and 150 inside: rate 0.5.
    ref = np.arange(1, 200, dtype=np.float64)
    cand = np.array([0.0, 100.0, 300.0, 150.0])
    rows = divergence.novelty_rows(ref, cand, alphas=(0.01,), n_boot=50, seed=17)
    r = rows[0]
    assert "unavailable" not in r
    assert r["n_outside"] == 2
    assert r["outside_rate"] == pytest.approx(0.5)
    assert r["expected_outside_if_exchangeable"] == 0.01
    lo, hi = r["outside_ci"]
    assert 0.0 <= lo <= r["outside_rate"] <= hi <= 1.0


def test_novelty_rows_deterministic_same_seed():
    rng = np.random.default_rng(2)
    ref = rng.normal(0, 1, 250)
    cand = rng.normal(1, 1, 60)
    r1 = divergence.novelty_rows(ref, cand, n_boot=50, seed=17)
    r2 = divergence.novelty_rows(ref, cand, n_boot=50, seed=17)
    assert r1 == r2


def test_novelty_empty_candidate_refuses():
    with pytest.raises(ValueError):
        divergence.novelty_rows(np.arange(100.0), np.array([]), n_boot=50)


# ------------------------------------------------------------------ quantile shift


def test_quantile_shift_zero_for_identical_samples():
    a = np.arange(1, 101, dtype=np.float64)
    for row in divergence.quantile_shift(a, a, n_boot=50, seed=17):
        assert row["delta"] == pytest.approx(0.0)


def test_quantile_shift_detects_a_location_shift():
    rng = np.random.default_rng(4)
    ref = rng.normal(0, 1, 300)
    cand = ref + 1.0
    for row in divergence.quantile_shift(ref, cand, n_boot=50, seed=17):
        assert row["delta"] == pytest.approx(1.0, abs=1e-9)
        assert row["candidate_q"] == pytest.approx(row["reference_q"] + 1.0, abs=1e-9)


# ------------------------------------------------------------------ per-document p


def test_conformal_p_extremes_and_center():
    ref = np.arange(1, 100, dtype=np.float64)  # n=99
    far_out = divergence.conformal_p_two_sided(ref, 1000.0)
    assert far_out == pytest.approx(2.0 / 100.0)
    center = divergence.conformal_p_two_sided(ref, 50.0)
    assert center == 1.0


def test_conformal_p_monotone_toward_the_tail():
    ref = np.arange(1, 100, dtype=np.float64)
    ps = [divergence.conformal_p_two_sided(ref, v) for v in (50.0, 80.0, 95.0, 1000.0)]
    assert ps == sorted(ps, reverse=True)


def test_conformal_p_empty_reference_is_nan():
    assert np.isnan(divergence.conformal_p_two_sided(np.array([]), 1.0))
