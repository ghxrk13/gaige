# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

import numpy as np
import pytest

from gaige import calibrate


def make(scores_h, scores_a):
    s = np.array(list(scores_h) + list(scores_a), dtype=np.float64)
    lab = np.array(["human"] * len(scores_h) + ["ai"] * len(scores_a))
    return s, lab


def test_auroc_separable():
    s, lab = make(np.linspace(-3, -1, 60), np.linspace(1, 3, 60))
    assert calibrate.auroc(s, lab) == 1.0


def test_auroc_random_is_half():
    rng = np.random.default_rng(0)
    s, lab = make(rng.normal(size=500), rng.normal(size=500))
    assert abs(calibrate.auroc(s, lab) - 0.5) < 0.05


def test_auroc_inverted():
    s, lab = make(np.linspace(1, 3, 60), np.linspace(-3, -1, 60))
    assert calibrate.auroc(s, lab) == 0.0


def test_threshold_at_fpr_respects_target():
    rng = np.random.default_rng(1)
    s, lab = make(rng.normal(0, 1, 300), rng.normal(2, 1, 300))
    for target in (0.01, 0.05):
        row = calibrate.threshold_at_fpr(s, lab, target)
        assert row["achieved_fpr"] <= target
        human = s[lab == "human"]
        assert (human >= row["threshold"]).mean() == pytest.approx(row["achieved_fpr"])


def test_small_corpus_refused():
    s, lab = make(np.zeros(10), np.ones(10))
    with pytest.raises(calibrate.CorpusTooSmall):
        calibrate.auroc(s, lab)


def test_bootstrap_ci_brackets_point():
    rng = np.random.default_rng(2)
    s, lab = make(rng.normal(0, 1, 200), rng.normal(1.5, 1, 200))
    a = calibrate.auroc(s, lab)
    lo, hi = calibrate.bootstrap_ci(s, lab, calibrate.auroc, n_boot=200, seed=3)
    assert lo <= a <= hi
    assert hi - lo < 0.15


def _auroc_reference_scalar(scores, labels):
    """The pre-vectorization midrank implementation, kept as the equivalence oracle."""
    human = scores[labels == "human"]
    ai = scores[labels == "ai"]
    both = np.concatenate([human, ai])
    ranks = both.argsort().argsort().astype(np.float64) + 1.0
    order = both.argsort()
    sorted_vals = both[order]
    i = 0
    while i < len(sorted_vals):
        j = i
        while j + 1 < len(sorted_vals) and sorted_vals[j + 1] == sorted_vals[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = ranks[order[i : j + 1]].mean()
        i = j + 1
    r_ai = ranks[len(human) :].sum()
    u = r_ai - len(ai) * (len(ai) + 1) / 2.0
    return float(u / (len(human) * len(ai)))


def test_vectorized_auroc_identical_to_scalar_reference():
    """Midranks are exact math; the vectorization must be value-IDENTICAL, ties included."""
    rng = np.random.default_rng(7)
    cases = [
        make(rng.normal(0, 1, 150), rng.normal(1, 1, 150)),  # continuous
        make(
            rng.integers(0, 5, 200).astype(float), rng.integers(2, 7, 200).astype(float)
        ),  # heavy ties
        make(np.zeros(60), np.zeros(60)),  # total tie: AUROC exactly 0.5
    ]
    for s, lab in cases:
        assert calibrate.auroc(s, lab) == _auroc_reference_scalar(s, lab)


def test_proportion_ci_deterministic_and_brackets():
    rng = np.random.default_rng(4)
    v = (rng.random(300) < 0.3).astype(float)
    a = calibrate.proportion_ci(v, n_boot=500, seed=9)
    b = calibrate.proportion_ci(v, n_boot=500, seed=9)
    assert a == b  # seeded, reproducible to the bit
    lo, hi = a
    # The guaranteed property: the interval brackets the SAMPLE mean. (It misses the
    # population truth ~5% of the time by construction — asserting truth on one seeded
    # draw would be the kind of overclaim this codebase exists to refuse.)
    assert lo <= v.mean() <= hi
    se = float(np.sqrt(v.mean() * (1 - v.mean()) / len(v)))
    assert 2 * se < (hi - lo) < 6 * se  # width is binomial-plausible, not degenerate
    with pytest.raises(ValueError, match="at least one value"):
        calibrate.proportion_ci(np.array([]))


def test_eer_perfect_separation_is_zero():
    s, y = make([1.0] * 50, [5.0] * 50)
    r = calibrate.eer(s, y)
    assert r["eer"] == 0.0


def test_eer_lands_on_measured_crossing():
    # human = 50x1 + 50x3, ai = 50x2 + 50x4: at threshold 3 both error rates
    # are exactly 0.5.
    s, y = make([1.0] * 50 + [3.0] * 50, [2.0] * 50 + [4.0] * 50)
    r = calibrate.eer(s, y)
    assert r["eer"] == pytest.approx(0.5)
    assert r["threshold"] == pytest.approx(3.0)


def test_eer_interpolates_between_sweep_points():
    # human = 75x0 + 25x10, ai = 100x5: FPR-FNR jumps -0.75 -> +0.25 between
    # thresholds 10 and 5, so the crossing sits 3/4 of the way along, where the
    # interpolated FPR (constant 0.25 on that segment) equals interpolated FNR.
    s, y = make([0.0] * 75 + [10.0] * 25, [5.0] * 100)
    r = calibrate.eer(s, y)
    assert r["eer"] == pytest.approx(0.25)
    assert r["threshold"] == pytest.approx(6.25)


def test_eer_matches_definition_oracle_on_random_corpora():
    """Differential property: eer() against an independent, definition-based oracle —
    grid-search the linearly interpolated FPR and FNR curves for their closest approach.
    The oracle shares no code path with the vectorized crossing logic, so an argmax or
    interpolation-weight transcription bug fails here even if the exact hand cases pass.
    Half the draws come from tiny value grids to force heavy score ties (where ROC sweeps
    actually break)."""
    rng = np.random.default_rng(7)
    for trial in range(30):
        # Respect the library's own honesty floor (CorpusTooSmall below 50/class).
        n_h = int(rng.integers(50, 120))
        n_a = int(rng.integers(50, 120))
        if trial % 2 == 0:
            grid = np.array([0.0, 0.5, 1.0, 1.5, 2.0])
            s_h = rng.choice(grid, size=n_h)
            s_a = rng.choice(grid, size=n_a) + rng.choice([0.0, 0.5], size=n_a)
        else:
            s_h = rng.normal(0.0, 1.0, size=n_h)
            s_a = rng.normal(1.0, 1.2, size=n_a)
        s, y = make(s_h, s_a)
        r = calibrate.eer(s, y)
        assert 0.0 <= r["eer"] <= 1.0
        pts = calibrate.roc_points(s, y)
        thr = np.asarray(pts["thresholds"], dtype=np.float64)
        fpr = np.asarray(pts["fpr"], dtype=np.float64)
        fnr = 1.0 - np.asarray(pts["tpr"], dtype=np.float64)
        # np.interp needs ascending x; the sweep is descending, so flip everything.
        t_asc = thr[::-1]
        dense = np.linspace(t_asc[0], t_asc[-1], 200_001)
        f_dense = np.interp(dense, t_asc, fpr[::-1])
        m_dense = np.interp(dense, t_asc, fnr[::-1])
        k = int(np.argmin(np.abs(f_dense - m_dense)))
        oracle_eer = (f_dense[k] + m_dense[k]) / 2.0
        assert r["eer"] == pytest.approx(oracle_eer, abs=1e-3), (
            f"trial {trial}: eer {r['eer']} vs oracle {oracle_eer}"
        )
