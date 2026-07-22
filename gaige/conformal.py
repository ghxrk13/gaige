# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Conformal (split-conformal) threshold calibration with a distribution-free FPR bound.

Standard calibration picks the threshold whose *empirical* FPR on the calibration sample hits
the target. That estimate is itself noisy: on 100 human samples, an observed 1% FPR is
consistent with a true rate several times higher, and the error lands on real people.

Split conformal prediction fixes the threshold at an order statistic of the calibration
scores such that, for exchangeable data, P(false positive) <= alpha holds in finite samples —
no distributional assumptions. Following Zhu et al. (arXiv:2505.05084), which applies
conformal prediction to machine-generated-text detection specifically and reports empirical
FPRs staying within the theoretical bound across seven detectors at alpha from 0.2 to 0.005.
(The paper squashes detector output through a monotone sigmoid first; quantiles are
equivariant under monotone maps, so operating on raw scores is mathematically identical.)

Two honesty notes that the reported numbers must carry, verified against the paper:

- The guarantee is MARGINAL, averaged over draws of the calibration set. Conditionally on
  the particular calibration set in hand, the true FPR of the emitted threshold is a random
  variable with law Beta(n+1-k, k) for continuous scores (conservative under ties), so each
  threshold ships that law's exact mean and sd instead of a pseudo-"achieved" rate.
- The bound assumes calibration and deployment human text are exchangeable. Domain shift
  voids it; the report says so.

Cost: the guarantee needs samples. At alpha = 0.01 you need at least 99 human calibration
samples for the bound to be attainable at all; gaige refuses rather than pretending.
"""

from __future__ import annotations

import math

import numpy as np


class InsufficientCalibration(ValueError):
    """Calibration set too small for the requested guarantee."""


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha!r}")


def min_samples_for(alpha: float) -> int:
    """Smallest calibration-set size at which a conformal threshold for alpha exists.

    Feasibility needs ceil((n+1)(1-alpha)) <= n, i.e. n >= 1/alpha - 1; ceil(1/alpha) - 1
    equals ceil(1/alpha - 1) for every alpha in (0, 1).
    """
    _check_alpha(alpha)
    return int(math.ceil(1.0 / alpha)) - 1


def conformal_threshold(human_scores: np.ndarray, alpha: float) -> dict:
    """Threshold with a finite-sample marginal guarantee that P(human flagged) <= alpha.

    Uses the ceil((n+1)(1-alpha))-th order statistic of the human calibration scores, the
    standard split-conformal quantile with the finite-sample correction (Zhu et al.
    arXiv:2505.05084 Eq. 1; flag rule Eq. 4). With tied scores the strict-inequality rule
    can only flag less, so the bound stays valid.

    No "achieved FPR" is returned, deliberately: the in-sample flag rate on the calibration
    scores is (n - k)/n by construction — a function of n and alpha, not a measurement.
    What IS returned is the exact conditional law of the true FPR given this calibration
    set: Beta(n+1-k, k) for continuous scores, reported as mean and sd.
    """
    _check_alpha(alpha)
    n = len(human_scores)
    need = min_samples_for(alpha)
    if n < need:
        raise InsufficientCalibration(
            f"alpha={alpha} needs >= {need} human calibration samples, got {n}. "
            "A tighter guarantee than your data supports is not a guarantee."
        )
    s = np.sort(np.asarray(human_scores, dtype=np.float64))
    k = int(math.ceil((n + 1) * (1.0 - alpha)))
    # Mathematically k <= n whenever n >= min_samples_for(alpha); the clamp only defends
    # against float rounding in (n+1)*(1-alpha), and it errs upward (fewer flags): safe.
    k = min(k, n)
    thr = float(s[k - 1])
    # Strictly greater-than at the order statistic keeps the guarantee one-sided.
    thr = float(np.nextafter(thr, np.inf))
    a, b = n + 1 - k, k  # conditional FPR | calibration ~ Beta(a, b), continuous scores
    return {
        "alpha": alpha,
        "threshold": thr,
        "n_calibration": n,
        "order_statistic": k,
        "conditional_fpr_mean": a / (n + 1.0),
        "conditional_fpr_sd": math.sqrt(a * b / ((n + 1.0) ** 2 * (n + 2.0))),
        "guarantee": (
            f"P(human flagged) <= {alpha}, marginal over calibration draws, "
            "under exchangeability (split conformal)"
        ),
    }


def conformal_table(
    human_scores: np.ndarray,
    ai_scores: np.ndarray,
    alphas: tuple[float, ...] = (0.05, 0.01, 0.005),
) -> list[dict]:
    """Conformal thresholds at several alphas, with the TPR each buys on THESE ai_scores.

    The tpr field is descriptive of the supplied corpus; the guarantee applies only to the
    human-flag rate. Alphas the calibration set cannot support come back as refusal rows.
    """
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
