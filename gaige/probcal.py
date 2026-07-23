# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Probability calibration: does a stated 80% confidence come true 80% of the time?

DO NOT confuse this with `calibrate.py`. That module is DECISION-THRESHOLD calibration
(what score cuts at 1% FPR); this one is PROBABILITY calibration (Expected Calibration
Error, confidence-accuracy gap). Different mathematics, different failure modes — the name
overlap is a documented trap, which is why this module is not called calibrate-anything.

The drift application (M3): a frozen model's confidence stays high while its accuracy on
dated probes falls — "fluent and authoritative whilst quietly wrong" made numeric. The gap
and ECE per vintage, tracked across a registry series, are that warning light.

Binning policy: equal-width bins on [0,1], bin COUNT fixed per series (default 10) — a
changed bin count is an instrument change, exactly like a changed threshold rule.
"""

from __future__ import annotations

import numpy as np

from .calibrate import proportion_ci  # noqa: F401  (re-exported convenience for callers)

DEFAULT_BINS = 10


def ece(confidences: np.ndarray, corrects: np.ndarray, n_bins: int = DEFAULT_BINS) -> dict:
    """Expected Calibration Error with the per-bin table that explains it.

    ECE = sum_b (n_b / N) * |accuracy_b - mean_confidence_b| over equal-width bins.
    Confidence exactly 1.0 lands in the top bin (right edge inclusive there only).
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=np.float64)
    if len(conf) != len(corr) or len(conf) == 0:
        raise ValueError("confidences and corrects must be equal-length and non-empty")
    if conf.min() < 0.0 or conf.max() > 1.0:
        raise ValueError("confidences must lie in [0, 1]")
    # digitize with right-open bins; clip so conf == 1.0 joins the last bin.
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    total = 0.0
    bins = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        if n == 0:
            bins.append({"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": 0})
            continue
        mc, acc = float(conf[mask].mean()), float(corr[mask].mean())
        total += (n / len(conf)) * abs(acc - mc)
        bins.append(
            {"lo": b / n_bins, "hi": (b + 1) / n_bins, "n": n, "mean_conf": mc, "accuracy": acc}
        )
    return {"ece": float(total), "n_bins": n_bins, "n": int(len(conf)), "bins": bins}


def ece_ci(
    confidences: np.ndarray,
    corrects: np.ndarray,
    n_bins: int = DEFAULT_BINS,
    n_boot: int = 1000,
    seed: int = 17,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI for ECE, resampling (confidence, correct) PAIRS together."""
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(conf), size=(n_boot, len(conf)))
    vals = [ece(conf[row], corr[row], n_bins=n_bins)["ece"] for row in idx]
    lo = (1.0 - ci) / 2.0
    return float(np.quantile(vals, lo)), float(np.quantile(vals, 1.0 - lo))


def brier(confidences: np.ndarray, corrects: np.ndarray) -> dict:
    """Brier score: mean squared error of P(True) against the 0/1 outcome.

    Complements ECE: ECE bins and averages the calibration gap, so it moves
    with the binning; the Brier score is the un-binned proper score
    (calibration and refinement together). 0 is perfect; 0.25 is what
    always answering 0.5 earns.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=np.float64)
    if len(conf) == 0:
        return {"brier": float("nan"), "n": 0}
    return {"brier": float(np.mean((conf - corr) ** 2)), "n": int(len(conf))}


def confidence_accuracy_gap(confidences: np.ndarray, corrects: np.ndarray) -> float:
    """Mean stated confidence minus realized accuracy. Positive = overconfident.

    The M3 headline: on dated vintages this gap WIDENS if the model stays confident while
    the world moves; on the static control it should stay flat within run variance.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    corr = np.asarray(corrects, dtype=np.float64)
    return float(conf.mean() - corr.mean())
