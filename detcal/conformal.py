# detcal — calibration + receipts for AI-text detectors.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Conformal (split-conformal) threshold calibration with a distribution-free FPR bound.

Standard calibration picks the threshold whose *empirical* FPR on the calibration sample hits
the target. That estimate is itself noisy: on 100 human samples, an observed 1% FPR is
consistent with a true rate several times higher, and the error lands on real people.

Split conformal prediction fixes the threshold at an order statistic of the calibration
scores such that, for exchangeable data, P(false positive) <= alpha holds in finite samples —
no distributional assumptions. Following Wang et al. (arXiv:2505.05084), which applies
conformal prediction to machine-generated-text detection specifically and reports empirical
FPRs staying within the theoretical bound across seven detectors at alpha from 0.2 to 0.005.

Cost: the guarantee needs samples. At alpha = 0.01 you need at least 99 human calibration
samples for the bound to be attainable at all; detcal refuses rather than pretending.
"""

from __future__ import annotations

import math

import numpy as np


class InsufficientCalibration(ValueError):
    """Calibration set too small for the requested guarantee."""


def min_samples_for(alpha: float) -> int:
    """Smallest calibration-set size at which a conformal threshold for alpha exists."""
    return int(math.ceil(1.0 / alpha)) - 1


def conformal_threshold(human_scores: np.ndarray, alpha: float) -> dict:
    """Threshold with a finite-sample guarantee that P(human flagged) <= alpha.

    Uses the ceil((n+1)(1-alpha))-th order statistic of the human calibration scores, the
    standard split-conformal quantile with the finite-sample correction.
    """
    n = len(human_scores)
    need = min_samples_for(alpha)
    if n < need:
        raise InsufficientCalibration(
            f"alpha={alpha} needs >= {need} human calibration samples, got {n}. "
            "A tighter guarantee than your data supports is not a guarantee."
        )
    s = np.sort(np.asarray(human_scores, dtype=np.float64))
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    k = min(k, n)  # k == n means "the largest observed score"
    thr = float(s[k - 1])
    # Strictly greater-than at the order statistic keeps the guarantee one-sided.
    thr = float(np.nextafter(thr, np.inf))
    return {
        "alpha": alpha,
        "threshold": thr,
        "n_calibration": n,
        "order_statistic": k,
        "empirical_fpr": float((s >= thr).mean()),
        "guarantee": f"P(human flagged) <= {alpha} under exchangeability (split conformal)",
    }


def conformal_table(
    human_scores: np.ndarray,
    ai_scores: np.ndarray,
    alphas: tuple[float, ...] = (0.05, 0.01, 0.005),
) -> list[dict]:
    """Conformal thresholds at several alphas, with the TPR each one actually buys."""
    rows = []
    for a in alphas:
        try:
            row = conformal_threshold(human_scores, a)
        except InsufficientCalibration as e:
            rows.append({"alpha": a, "unavailable": str(e)})
            continue
        row["tpr"] = float((np.asarray(ai_scores) >= row["threshold"]).mean())
        rows.append(row)
    return rows
