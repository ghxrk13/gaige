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


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC via the rank / Mann-Whitney identity (ties get midranks)."""
    human, ai = _split(scores, labels)
    both = np.concatenate([human, ai])
    ranks = both.argsort().argsort().astype(np.float64) + 1.0
    # midrank correction for ties
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
    for t in sorted(np.unique(scores)):  # ascending: first t meeting target = most sensitive legal threshold
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
    """Percentile bootstrap CI, resampling within each class."""
    rng = np.random.default_rng(seed)
    h_idx = np.flatnonzero(labels == "human")
    a_idx = np.flatnonzero(labels == "ai")
    vals = []
    for _ in range(n_boot):
        hs = rng.choice(h_idx, size=len(h_idx), replace=True)
        as_ = rng.choice(a_idx, size=len(a_idx), replace=True)
        idx = np.concatenate([hs, as_])
        try:
            vals.append(stat_fn(scores[idx], labels[idx]))
        except CorpusTooSmall:
            continue
    lo = (1.0 - ci) / 2.0
    return float(np.quantile(vals, lo)), float(np.quantile(vals, 1.0 - lo))
