# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Two-sample divergence math for corpus admission. Pure numpy, no model dependencies.

Everything here compares an unlabeled candidate sample against a reference sample drawn
from an accepted baseline. The guarantee-bearing piece is the two-sided split-conformal
band: two one-sided conformal thresholds at alpha/2 each (the upper one is
`conformal.conformal_threshold` as-is; the lower one is the same machinery run on negated
scores), so that under exchangeability the expected share of candidate documents outside
the band is at most alpha — finite samples, no distributional assumptions, union bound
over the two sides. The supporting statistics (Kolmogorov-Smirnov distance, quantile
shifts) carry bootstrap intervals and deliberately no p-values: the interval carries the
uncertainty, and a p-value invites a verdict.
"""

from __future__ import annotations

import numpy as np

from . import conformal
from .calibrate import proportion_ci

SAMPLE_NOUN = "baseline reference scores"


def two_sided_min_samples(alpha: float) -> int:
    """Smallest reference size at which a two-sided band for alpha exists.

    Each side runs at alpha/2, so the floor is ceil(2/alpha) - 1: 39 at alpha 0.05,
    199 at 0.01, 399 at 0.005. The floor is the guarantee's price, not a limitation.
    """
    return conformal.min_samples_for(alpha / 2.0)


def conformal_band(reference: np.ndarray, alpha: float) -> dict:
    """Two-sided acceptance band from the reference scores, alpha/2 per side.

    Raises InsufficientCalibration below the two-sided floor, with two-sided wording —
    the per-side call would name alpha/2 and confuse the reader about what was asked.
    Each side ships its exact conditional exceedance law Beta(n+1-k, k), the same
    bookkeeping the one-sided table already reports.
    """
    ref = np.asarray(reference, dtype=np.float64)
    n = len(ref)
    need = two_sided_min_samples(alpha)
    if n < need:
        raise conformal.InsufficientCalibration(
            f"two-sided alpha={alpha:g} needs >= {need} {SAMPLE_NOUN} "
            f"(ceil(2/alpha)-1; each side runs at alpha/2), got {n}. "
            "A tighter guarantee than your data supports is not a guarantee."
        )
    upper = conformal.conformal_threshold(ref, alpha / 2.0, sample_noun=SAMPLE_NOUN)
    lower_neg = conformal.conformal_threshold(-ref, alpha / 2.0, sample_noun=SAMPLE_NOUN)
    # On the negated axis the flag rule is (-score) >= t, i.e. score <= -t in real space.
    lo = -lower_neg["threshold"]
    hi = upper["threshold"]
    return {
        "alpha": alpha,
        "per_side_alpha": alpha / 2.0,
        "n_reference": n,
        "lo": lo,
        "hi": hi,
        "upper": {
            "order_statistic": upper["order_statistic"],
            "conditional_exceedance_mean": upper["conditional_fpr_mean"],
            "conditional_exceedance_sd": upper["conditional_fpr_sd"],
        },
        "lower": {
            "order_statistic": lower_neg["order_statistic"],
            "conditional_exceedance_mean": lower_neg["conditional_fpr_mean"],
            "conditional_exceedance_sd": lower_neg["conditional_fpr_sd"],
        },
        "guarantee": (
            f"P(outside band) <= {alpha:g}, marginal over calibration draws, under "
            "exchangeability (two one-sided split-conformal bounds at alpha/2, union bound)"
        ),
    }


def novelty_rows(
    reference: np.ndarray,
    candidate: np.ndarray,
    alphas: tuple[float, ...] = (0.05, 0.01, 0.005),
    n_boot: int = 1000,
    seed: int = 17,
) -> list[dict]:
    """Share of candidate documents outside the reference band, per alpha, with a CI.

    The yardstick each row carries: were the candidate exchangeable with the baseline,
    the expected outside share is at most alpha. Alphas the reference cannot support come
    back as refusal rows, the conformal_table idiom — the table prints WHY beside the
    alphas that did run.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if len(cand) == 0:
        raise ValueError("novelty_rows needs at least one candidate score")
    rows: list[dict] = []
    for a in alphas:
        try:
            band = conformal_band(ref, a)
        except conformal.InsufficientCalibration as e:
            rows.append({"alpha": a, "unavailable": str(e)})
            continue
        outside = ((cand >= band["hi"]) | (cand <= band["lo"])).astype(np.float64)
        band["outside_rate"] = float(outside.mean())
        band["outside_ci"] = proportion_ci(outside, n_boot=n_boot, seed=seed)
        band["n_outside"] = int(outside.sum())
        band["n_candidate"] = len(cand)
        band["expected_outside_if_exchangeable"] = a
        rows.append(band)
    return rows


def ks_statistic(a: np.ndarray, b: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov distance: sup |ECDF_a - ECDF_b|. Tie-safe."""
    a = np.sort(np.asarray(a, dtype=np.float64))
    b = np.sort(np.asarray(b, dtype=np.float64))
    if len(a) == 0 or len(b) == 0:
        raise ValueError("ks_statistic needs non-empty samples on both sides")
    grid = np.concatenate([a, b])
    ca = np.searchsorted(a, grid, side="right") / len(a)
    cb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.abs(ca - cb).max())


def ks_with_ci(
    reference: np.ndarray,
    candidate: np.ndarray,
    n_boot: int = 1000,
    seed: int = 17,
    ci: float = 0.95,
) -> dict:
    """KS distance with a percentile-bootstrap interval, resampling BOTH samples.

    No p-value, deliberately: under resampling the statistic never tests a null, it
    measures a distance, and the interval says how well that distance is pinned down.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    stat = ks_statistic(ref, cand)
    rng = np.random.default_rng(seed)
    idx_r = rng.integers(0, len(ref), size=(n_boot, len(ref)))
    idx_c = rng.integers(0, len(cand), size=(n_boot, len(cand)))
    vals = [ks_statistic(ref[idx_r[i]], cand[idx_c[i]]) for i in range(n_boot)]
    lo = (1.0 - ci) / 2.0
    return {
        "stat": stat,
        "ci": (float(np.quantile(vals, lo)), float(np.quantile(vals, 1.0 - lo))),
        "n_reference": len(ref),
        "n_candidate": len(cand),
    }


def quantile_shift(
    reference: np.ndarray,
    candidate: np.ndarray,
    qs: tuple[float, ...] = (0.10, 0.50, 0.90),
    n_boot: int = 1000,
    seed: int = 17,
    ci: float = 0.95,
) -> list[dict]:
    """Candidate-minus-reference quantile deltas with bootstrap intervals.

    Says WHERE the distributions differ: a shifted p90 with a flat p10 reads very
    differently from a wholesale location shift, and the novelty rate alone cannot
    tell those apart.
    """
    ref = np.asarray(reference, dtype=np.float64)
    cand = np.asarray(candidate, dtype=np.float64)
    if len(ref) == 0 or len(cand) == 0:
        raise ValueError("quantile_shift needs non-empty samples on both sides")
    rng = np.random.default_rng(seed)
    idx_r = rng.integers(0, len(ref), size=(n_boot, len(ref)))
    idx_c = rng.integers(0, len(cand), size=(n_boot, len(cand)))
    lo_q = (1.0 - ci) / 2.0
    rows: list[dict] = []
    for q in qs:
        rq = float(np.quantile(ref, q))
        cq = float(np.quantile(cand, q))
        deltas = np.quantile(cand[idx_c], q, axis=1) - np.quantile(ref[idx_r], q, axis=1)
        rows.append(
            {
                "q": q,
                "reference_q": rq,
                "candidate_q": cq,
                "delta": cq - rq,
                "delta_ci": (
                    float(np.quantile(deltas, lo_q)),
                    float(np.quantile(deltas, 1.0 - lo_q)),
                ),
            }
        )
    return rows


def conformal_p_two_sided(sorted_reference, value: float) -> float:
    """Rank-based two-sided conformal p-value for one candidate score.

    p = min(1, 2 * min((1 + #{ref >= s}) / (n+1), (1 + #{ref <= s}) / (n+1))) —
    finite-sample valid under exchangeability. Small means the score sits in a tail of
    the reference distribution; it never means the document is anything.
    """
    ref = np.asarray(sorted_reference, dtype=np.float64)
    n = len(ref)
    if n == 0:
        return float("nan")
    le = int(np.searchsorted(ref, value, side="right"))
    ge = n - int(np.searchsorted(ref, value, side="left"))
    p_hi = (1 + ge) / (n + 1.0)
    p_lo = (1 + le) / (n + 1.0)
    return float(min(1.0, 2.0 * min(p_hi, p_lo)))
