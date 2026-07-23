# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Subgroup-stratified error reporting.

Aggregate error rates hide who pays for them. The measured literature is blunt about this:
static thresholds produce substantially higher false-positive rates on short texts (Jung
et al., arXiv:2502.04528), and seven commercial detectors misclassified over half of
human-written TOEFL essays by non-native speakers as AI — an average 61.2% FPR versus ~5.2%
on native-speaker essays (Liang et al., arXiv:2304.02819).

gaige therefore reports error rates per subgroup by default. A corpus can carry arbitrary
metadata keys on each row (e.g. "length_bucket", "l1", "domain"); any key present on every
row (both classes) becomes a stratification axis.

Length bucketing is built in because it needs no annotation and is the disparity every
corpus can measure.
"""

from __future__ import annotations

import numpy as np

from . import calibrate

# Buckets chosen so the shortest reflects the "noise-dominated" regime detectors fail on.
LENGTH_BUCKETS = ((0, 100), (100, 250), (250, 500), (500, 10**9))
MIN_SUBGROUP = 20  # below this, report the count and refuse to report a rate


def length_bucket(n_words: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n_words < hi:
            return f"{lo}-{hi}w" if hi < 10**9 else f"{lo}+w"
    return "unknown"


def auto_keys(rows: list[dict]) -> list[str]:
    """Stratification axes = length (always) + any metadata key present on EVERY row.

    Every row, both classes: a subgroup TPR needs the ai rows bucketed on the same axis as
    the human rows, so an axis missing from either side is no axis at all.
    """
    keys = ["length_bucket"]
    meta_keys = set(rows[0].get("meta", {}) or {}) if rows else set()
    for k in sorted(meta_keys):
        if all(k in (r.get("meta") or {}) for r in rows):
            keys.append(k)
    return keys


def _value(row: dict, key: str) -> str:
    if key == "length_bucket":
        return length_bucket(row["n_words"])
    return str((row.get("meta") or {}).get(key, "unknown"))


def _rate_with_ci(
    grp: list[dict], threshold: float, n_boot: int, seed: int
) -> tuple[float | None, tuple[float, float] | None]:
    """Flag rate for one subgroup's rows of one class, with a bootstrap CI — or a refusal.

    Below MIN_SUBGROUP the rate is withheld (None), not shown-but-flagged: a rate on a
    handful of samples is noise wearing a percent sign, and downstream renderers print
    whatever they are given. The count still gets reported by the caller.
    """
    if len(grp) < MIN_SUBGROUP:
        return None, None
    flags = np.array([g["score"] >= threshold for g in grp], dtype=np.float64)
    rate = float(flags.mean())
    ci = calibrate.proportion_ci(flags, n_boot=n_boot, seed=seed)
    return rate, ci


def stratified_rates(
    rows: list[dict],
    threshold: float,
    keys: list[str] | None = None,
    n_boot: int = 1000,
    seed: int = 17,
) -> dict:
    """Per-subgroup FPR (on human rows) and TPR (on ai rows) at a fixed threshold.

    rows: [{"label", "score", "n_words", "meta": {...}}]
    Every reported rate carries a bootstrap CI (reusing calibrate.bootstrap_ci in
    single-class mode). Below MIN_SUBGROUP samples in a class, that class's rate and CI are
    None and only the count speaks. `rate_withheld` is True when either class was refused.
    Returns {key: {value: {n_human, fpr, fpr_ci, n_ai, tpr, tpr_ci, rate_withheld}}}
    """
    keys = keys or auto_keys(rows)
    out: dict = {}
    for key in keys:
        groups: dict = {}
        for r in rows:
            groups.setdefault(_value(r, key), []).append(r)
        per_value = {}
        for value, grp in sorted(groups.items()):
            h = [g for g in grp if g["label"] == "human"]
            a = [g for g in grp if g["label"] == "ai"]
            fpr, fpr_ci = _rate_with_ci(h, threshold, n_boot, seed)
            tpr, tpr_ci = _rate_with_ci(a, threshold, n_boot, seed)
            per_value[value] = {
                "n_human": len(h),
                "n_ai": len(a),
                "fpr": fpr,
                "fpr_ci": fpr_ci,
                "tpr": tpr,
                "tpr_ci": tpr_ci,
                "rate_withheld": fpr is None or tpr is None,
            }
        out[key] = per_value
    return out


def max_disparity(strata: dict) -> dict:
    """Largest FPR gap between subgroups on each axis, over groups whose rate was reported.

    `gap` is exactly the FPR-disparity metric of FairOPT (arXiv:2502.04528 Eq. 8:
    max_g FPR_g - min_g FPR_g), so the number is directly comparable to that literature.
    """
    out = {}
    for key, values in strata.items():
        fprs = {v: d["fpr"] for v, d in values.items() if d["fpr"] is not None}
        if len(fprs) < 2:
            out[key] = None
            continue
        hi_v, hi = max(fprs.items(), key=lambda kv: kv[1])
        lo_v, lo = min(fprs.items(), key=lambda kv: kv[1])
        out[key] = {
            "gap": hi - lo,
            "worst_group": hi_v,
            "worst_fpr": hi,
            "best_group": lo_v,
            "best_fpr": lo,
            "ratio": (hi / lo) if lo > 0 else None,
        }
    return out


def base_rate_harm(fpr: float, volume: int) -> dict:
    """Translate an FPR into people wrongly flagged at a given volume.

    This is the calculation Vanderbilt published when it disabled Turnitin's AI detector:
    a claimed 1% false-positive rate against 75,000 submissions/year is ~750 papers wrongly
    flagged. Every calibration report should force the reader to see this number.
    Prevalence-dependent PPV is `ppv()` below; the two are deliberately separate calls.
    """
    return {
        "fpr": fpr,
        "volume": volume,
        "expected_false_positives": fpr * volume,
    }


def ppv(fpr: float, tpr: float, prevalence: float) -> float:
    """P(actually AI | flagged) — Bayes, the number nobody puts on a detector dashboard."""
    tp = tpr * prevalence
    fp = fpr * (1.0 - prevalence)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
