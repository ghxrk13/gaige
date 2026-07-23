# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""ROC / threshold / uncertainty math. Pure numpy, no model dependencies.

Convention: higher score = more AI-like. Labels are the strings "human" and "ai".
"""

from __future__ import annotations

import numpy as np

MIN_PER_CLASS = 50


class CorpusTooSmall(ValueError):
    """Refuse to emit thresholds from corpora too small to mean anything."""


def _split(scores: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    human = scores[labels == "human"]
    ai = scores[labels == "ai"]
    if len(human) < MIN_PER_CLASS or len(ai) < MIN_PER_CLASS:
        raise CorpusTooSmall(
            f"need >= {MIN_PER_CLASS} samples per class, got human={len(human)} ai={len(ai)}; "
            "a threshold fitted on less is a guess wearing a lab coat"
        )
    return human, ai


def _midranks(values: np.ndarray) -> np.ndarray:
    """1-based midranks, fully vectorized. Tied values share the mean of their positions.

    Midranks are determined by the values alone, so this is value-identical to the scalar
    tie-averaging loop it replaced (asserted against a reference implementation in tests).
    """
    order = np.argsort(values, kind="stable")
    s = values[order]
    # Tie-group boundaries: positions where a new value starts, plus the end sentinel.
    starts = np.flatnonzero(np.r_[True, s[1:] != s[:-1]])
    ends = np.r_[starts[1:], len(s)]
    avg = (starts + 1 + ends) / 2.0  # mean of 1-based positions [start+1 .. end]
    ranks = np.empty(len(s), dtype=np.float64)
    ranks[order] = np.repeat(avg, ends - starts)
    return ranks


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank / Mann-Whitney identity (ties get midranks)."""
    human, ai = _split(scores, labels)
    both = np.concatenate([human, ai])
    ranks = _midranks(both)
    r_ai = ranks[len(human) :].sum()
    n_h, n_a = len(human), len(ai)
    u = r_ai - n_a * (n_a + 1) / 2.0
    return float(u / (n_h * n_a))


def roc_points(scores: np.ndarray, labels: np.ndarray) -> dict:
    """Full ROC sweep over unique thresholds. Predicted-AI when score >= threshold."""
    human, ai = _split(scores, labels)
    thresholds = np.unique(scores)[::-1]
    fpr = [(human >= t).mean() for t in thresholds]
    tpr = [(ai >= t).mean() for t in thresholds]
    return {
        "thresholds": [float(t) for t in thresholds],
        "fpr": [float(x) for x in fpr],
        "tpr": [float(x) for x in tpr],
    }


def threshold_at_fpr(scores: np.ndarray, labels: np.ndarray, target_fpr: float) -> dict:
    """Smallest threshold whose measured FPR on the human class is <= target.

    Returns achieved (not target) rates — the honest number is the measured one.
    """
    human, ai = _split(scores, labels)
    best = None
    for t in sorted(
        np.unique(scores)
    ):  # ascending: first t meeting target = most sensitive legal threshold
        f = float((human >= t).mean())
        if f <= target_fpr:
            best = {
                "target_fpr": target_fpr,
                "threshold": float(t),
                "achieved_fpr": f,
                "achieved_tpr": float((ai >= t).mean()),
            }
            break
    if best is None:  # no threshold reaches the target; report the strictest available
        t = float(np.max(scores)) + 1e-9
        best = {
            "target_fpr": target_fpr,
            "threshold": t,
            "achieved_fpr": 0.0,
            "achieved_tpr": 0.0,
        }
    return best


def bootstrap_ci(
    scores: np.ndarray,
    labels: np.ndarray,
    stat_fn,
    n_boot: int = 1000,
    seed: int = 17,
    ci: float = 0.95,
) -> tuple[float, float]:
    """Percentile bootstrap CI, resampling within each class. Vectorized draws (2026-07-22).

    All resample index matrices are drawn in one rng call per class instead of one per
    iteration. NOTE, honestly: that is a DIFFERENT random stream than the pre-vectorization
    loop, so CI values differ slightly from receipts generated before this change — an
    analysis-layer instrument change, changelogged, with the reference pin re-measured.
    AUROC and thresholds are deterministic and unaffected.

    A single-class input (e.g. one subgroup's human rows) resamples the class that exists.
    stat_fn is still evaluated per resample row (arbitrary statistics stay supported); for
    plain proportion/mean statistics use `proportion_ci`, which vectorizes the statistic too.
    """
    rng = np.random.default_rng(seed)
    h_idx = np.flatnonzero(labels == "human")
    a_idx = np.flatnonzero(labels == "ai")
    parts = [
        rng.choice(cls, size=(n_boot, len(cls)), replace=True) for cls in (h_idx, a_idx) if len(cls)
    ]
    idx_matrix = np.concatenate(parts, axis=1)
    vals = []
    for row in idx_matrix:
        try:
            vals.append(stat_fn(scores[row], labels[row]))
        except CorpusTooSmall:
            continue
    lo = (1.0 - ci) / 2.0
    return float(np.quantile(vals, lo)), float(np.quantile(vals, 1.0 - lo))


def proportion_ci(
    values: np.ndarray, n_boot: int = 1000, seed: int = 17, ci: float = 0.95
) -> tuple[float, float]:
    """Fully vectorized percentile bootstrap CI for a mean/proportion.

    One rng draw, one axis-mean: the hot path for TPR-at-threshold, per-subgroup rates, and
    probe accuracy (all indicator means). Measured ~40x faster than the generic path at
    n_boot=1000; at RAID scale the difference is the feature.
    """
    v = np.asarray(values, dtype=np.float64)
    if len(v) == 0:
        raise ValueError("proportion_ci needs at least one value")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(v), size=(n_boot, len(v)))
    means = v[idx].mean(axis=1)
    lo = (1.0 - ci) / 2.0
    return float(np.quantile(means, lo)), float(np.quantile(means, 1.0 - lo))
