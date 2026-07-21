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
