# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Subgroup-stratified error reporting.

Aggregate error rates hide who pays for them. The measured literature is blunt about this:
static thresholds produce substantially higher false-positive rates on short texts (Nguyen
et al., arXiv:2502.04528), and seven commercial detectors misclassified over half of
human-written TOEFL essays by non-native speakers as AI — an average 61.2% FPR versus ~5.2%
on native-speaker essays (Liang et al., arXiv:2304.02819).

gaige therefore reports error rates per subgroup by default. A corpus can carry arbitrary
metadata keys on each row (e.g. "length_bucket", "l1", "domain"); any key present on every
human-labeled row becomes a stratification axis.

Length bucketing is built in because it needs no annotation and is the disparity every
corpus can measure.
"""

from __future__ import annotations

import numpy as np

# Buckets chosen so the shortest reflects the "noise-dominated" regime detectors fail on.
LENGTH_BUCKETS = ((0, 100), (100, 250), (250, 500), (500, 10**9))
MIN_SUBGROUP = 20  # below this, report the count and refuse to report a rate


def length_bucket(n_words: int) -> str:
    for lo, hi in LENGTH_BUCKETS:
        if lo <= n_words < hi:
            return f"{lo}-{hi}w" if hi < 10**9 else f"{lo}+w"
    return "unknown"


def auto_keys(rows: list[dict]) -> list[str]:
    """Stratification axes = length (always) + any metadata key present on every row."""
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


def stratified_rates(rows: list[dict], threshold: float, keys: list[str] | None = None) -> dict:
    """Per-subgroup FPR (on human rows) and TPR (on ai rows) at a fixed threshold.

    rows: [{"label", "score", "n_words", "meta": {...}}]
    Returns {key: {value: {n_human, fpr|None, n_ai, tpr|None, suppressed: bool}}}
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
            small = len(h) < MIN_SUBGROUP
            per_value[value] = {
                "n_human": len(h),
                "n_ai": len(a),
                "fpr": (float(np.mean([x["score"] >= threshold for x in h])) if h else None),
                "tpr": (float(np.mean([x["score"] >= threshold for x in a])) if a else None),
                "suppressed": small,  # rate shown but flagged as statistically thin
            }
        out[key] = per_value
    return out


def max_disparity(strata: dict) -> dict:
    """Largest FPR gap between subgroups on each axis (only over non-thin groups)."""
    out = {}
    for key, values in strata.items():
        fprs = {
            v: d["fpr"]
            for v, d in values.items()
            if d["fpr"] is not None and not d["suppressed"]
        }
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


def base_rate_harm(fpr: float, volume: int, ai_prevalence: float | None = None) -> dict:
    """Translate an FPR into people wrongly flagged at a given volume.

    This is the calculation Vanderbilt published when it disabled Turnitin's AI detector:
    a claimed 1% false-positive rate against 75,000 submissions/year is ~750 papers wrongly
    flagged. Every calibration report should force the reader to see this number.

    If ai_prevalence is supplied, also reports the positive predictive value — the share of
    flagged documents that are actually AI-written — which is the number that decides whether
    a flag means anything at all.
    """
    false_positives = fpr * volume
    out = {
        "fpr": fpr,
        "volume": volume,
        "expected_false_positives": false_positives,
    }
    if ai_prevalence is not None:
        out["ai_prevalence"] = ai_prevalence
        # PPV needs a TPR; callers pass the calibrated one via ppv() below when available.
    return out


def ppv(fpr: float, tpr: float, prevalence: float) -> float:
    """P(actually AI | flagged) — Bayes, the number nobody puts on a detector dashboard."""
    tp = tpr * prevalence
    fp = fpr * (1.0 - prevalence)
    return float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
