# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Property tests: a KNOWN injected disparity must surface with a truthful interval, and
the refusal floor must actually refuse (count shown, rate withheld) — not annotate."""

from __future__ import annotations

import math

import numpy as np
import pytest

from gaige import subgroups

# Injected truth at threshold 2.0: short humans score N(1.5,1) -> FPR 1-Phi(0.5); long
# humans score N(0,1) -> FPR 1-Phi(2). The disparity is the construction, not an accident.
TRUE_FPR_SHORT = 0.30854
TRUE_FPR_LONG = 0.02275


def synth_rows(rng, n_short=200, n_long=200, n_ai=120):
    rows = []
    rows += [
        {"label": "human", "score": rng.normal(1.5, 1), "n_words": 60, "meta": None}
        for _ in range(n_short)
    ]
    rows += [
        {"label": "human", "score": rng.normal(0.0, 1), "n_words": 400, "meta": None}
        for _ in range(n_long)
    ]
    rows += [
        {"label": "ai", "score": rng.normal(3.0, 1), "n_words": 60 if i % 2 else 400, "meta": None}
        for i in range(n_ai)
    ]
    return rows


def test_injected_length_disparity_surfaces_with_truthful_intervals():
    rows = synth_rows(np.random.default_rng(17))
    strata = subgroups.stratified_rates(rows, threshold=2.0, n_boot=400, seed=7)
    lb = strata["length_bucket"]
    for bucket, truth in (("0-100w", TRUE_FPR_SHORT), ("250-500w", TRUE_FPR_LONG)):
        grp = lb[bucket]
        assert grp["rate_withheld"] is False
        assert grp["fpr"] == pytest.approx(truth, abs=0.07)
        lo, hi = grp["fpr_ci"]
        assert lo <= truth <= hi  # the interval brackets the injected truth
        assert hi - lo < 0.20
    disp = subgroups.max_disparity(strata)["length_bucket"]
    assert disp["worst_group"] == "0-100w"
    assert disp["gap"] == pytest.approx(TRUE_FPR_SHORT - TRUE_FPR_LONG, abs=0.09)


def test_below_floor_rate_actually_withheld():
    """Five humans at score 9.9 would be a 100% FPR subgroup — loud, and meaningless at
    n=5. The floor must withhold the rate entirely and keep the group out of disparity."""
    rows = synth_rows(np.random.default_rng(2))
    rows += [{"label": "human", "score": 9.9, "n_words": 150, "meta": None} for _ in range(5)]
    strata = subgroups.stratified_rates(rows, threshold=2.0, n_boot=100, seed=3)
    tiny = strata["length_bucket"]["100-250w"]
    assert tiny["n_human"] == 5
    assert tiny["fpr"] is None and tiny["fpr_ci"] is None
    assert tiny["rate_withheld"] is True
    disp = subgroups.max_disparity(strata)["length_bucket"]
    assert disp["worst_group"] != "100-250w"


def test_tpr_floor_is_independent_of_fpr_floor():
    rows = synth_rows(np.random.default_rng(4), n_ai=10)  # 5 ai per bucket: below floor
    strata = subgroups.stratified_rates(rows, threshold=2.0, n_boot=100, seed=3)
    grp = strata["length_bucket"]["0-100w"]
    assert grp["fpr"] is not None  # 200 humans: reported
    assert grp["tpr"] is None  # 5 ai: withheld
    assert grp["rate_withheld"] is True


def test_auto_keys_requires_presence_on_every_row():
    rows = [
        {"label": "human", "score": 0.1, "n_words": 10, "meta": {"domain": "qa"}},
        {"label": "ai", "score": 2.2, "n_words": 500, "meta": {"domain": "news"}},
    ]
    assert subgroups.auto_keys(rows) == ["length_bucket", "domain"]
    rows[1] = {**rows[1], "meta": {}}
    assert subgroups.auto_keys(rows) == ["length_bucket"]


def test_length_bucket_edges():
    assert subgroups.length_bucket(0) == "0-100w"
    assert subgroups.length_bucket(99) == "0-100w"
    assert subgroups.length_bucket(100) == "100-250w"
    assert subgroups.length_bucket(250) == "250-500w"
    assert subgroups.length_bucket(500) == "500+w"
    assert subgroups.length_bucket(10**7) == "500+w"


def test_base_rate_and_ppv_arithmetic():
    h = subgroups.base_rate_harm(0.01, 75000)
    assert h["expected_false_positives"] == pytest.approx(750.0)
    expected = 0.86 * 0.01 / (0.86 * 0.01 + 0.01 * 0.99)
    assert subgroups.ppv(0.01, 0.86, 0.01) == pytest.approx(expected)
    assert math.isnan(subgroups.ppv(0.0, 0.0, 0.5))
