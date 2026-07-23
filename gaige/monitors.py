# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""M5: sequential drift monitors over registered series, with honest alarm thresholds.

The comparison the spec needs: run candidate monitors over the SAME longitudinal series
and report, per monitor, DETECTION LATENCY (intervals from a known onset to the first
alarm) and FALSE ALARMS (alarms before onset / on zero-drift data). Monitors never touch a
model — they replay what the registry recorded, which is why building them after real
series exist costs nothing.

Where the thresholds come from is the point (longitudinal spec section 5, scoped honestly):

- PER-INTERVAL monitors (an alarm rule applied to each interval's value independently) get
  a **conformal threshold** calibrated on zero-drift reference values (Day-0 replicates,
  control-vintage intervals). That carries a finite-sample guarantee: per-interval
  false-alarm probability <= alpha, MARGINAL over the calibration draw, under
  exchangeability of zero-drift intervals. Expected false alarms over a series = alpha x
  number of looks — stated, not hidden.
- CUMULATIVE detectors (Page-Hinkley, CUSUM — the Gama 2014 / Webb 2016 lineage) carry NO
  such guarantee here: their statistics accumulate, so interval exchangeability does not
  apply to the statistic. They run with their tuning parameters recorded on the receipt,
  which is exactly the ad-hoc practice the literature uses — reported as such. Conformal
  test martingales are the principled sequential extension; future work, cited, unclaimed.
"""

from __future__ import annotations

import numpy as np

from . import conformal

DIRECTIONS = ("down", "up")


def _oriented(values, direction: str) -> np.ndarray:
    """Map to 'bigger = more alarming'. down-alarms (accuracy drops) negate the series."""
    if direction not in DIRECTIONS:
        raise ValueError(f"direction must be one of {DIRECTIONS}, got {direction!r}")
    v = np.asarray(values, dtype=np.float64)
    return -v if direction == "down" else v


def conformal_alarm(reference, values, alpha: float, direction: str = "down") -> dict:
    """Per-interval alarm with a conformal threshold from zero-drift reference values.

    Raises conformal.InsufficientCalibration when the reference cannot support alpha —
    at alpha=0.05 that means >= 19 zero-drift intervals, and a young series simply does not
    have them yet. The refusal IS the honest answer; the report says what is needed.
    """
    ref = _oriented(reference, direction)
    obs = _oriented(values, direction)
    row = conformal.conformal_threshold(ref, alpha, sample_noun="zero-drift reference intervals")
    thr = row["threshold"]
    alarms = [int(i) for i in np.flatnonzero(obs >= thr)]
    return {
        "monitor": f"conformal-interval-{direction}",
        "alarms": alarms,
        "threshold": (-thr if direction == "down" else thr),
        "alpha": alpha,
        "n_reference": row["n_calibration"],
        "guarantee": (
            f"per-interval false-alarm probability <= {alpha}, marginal over the "
            "calibration draw, under exchangeability of zero-drift intervals; expected "
            f"false alarms over k looks = {alpha} x k"
        ),
    }


def page_hinkley(values, delta: float = 0.005, lam: float = 0.05, direction: str = "down") -> dict:
    """Page-Hinkley mean-shift detector (Gama 2014 formulation), parameters recorded.

    Increase-form on the oriented series: m_t = sum_i (x_i - xbar_i - delta) with running
    mean xbar_i; PH_t = m_t - min_{i<=t} m_i; alarm whenever PH_t > lam. NO false-alarm
    guarantee is claimed for this detector; delta and lam are tuning constants, and the
    receipt says so.
    """
    x = _oriented(values, direction)
    alarms: list[int] = []
    m = 0.0
    m_min = 0.0
    mean = 0.0
    ph_trace: list[float] = []
    for t, xt in enumerate(x, start=1):
        mean += (xt - mean) / t
        m += xt - mean - delta
        m_min = min(m_min, m)
        ph = m - m_min
        ph_trace.append(float(ph))
        if ph > lam:
            alarms.append(t - 1)
    return {
        "monitor": f"page-hinkley-{direction}",
        "alarms": alarms,
        "params": {"delta": delta, "lambda": lam},
        "trace": ph_trace,
        "guarantee": "none claimed (cumulative statistic; tuned constants per drift-literature practice)",
    }


def cusum(
    values, reference_mean: float, k: float = 0.01, h: float = 0.05, direction: str = "down"
) -> dict:
    """One-sided CUSUM against a reference mean, parameters recorded, no guarantee claimed.

    Oriented form: S_t = max(0, S_{t-1} + (x_t - mu0 - k)); alarm whenever S_t > h, where
    mu0 is the oriented reference mean (e.g. the Day-0 replicate mean).
    """
    x = _oriented(values, direction)
    mu0 = -reference_mean if direction == "down" else reference_mean
    alarms: list[int] = []
    s = 0.0
    trace: list[float] = []
    for t, xt in enumerate(x):
        s = max(0.0, s + (xt - mu0 - k))
        trace.append(float(s))
        if s > h:
            alarms.append(t)
    return {
        "monitor": f"cusum-{direction}",
        "alarms": alarms,
        "params": {"k": k, "h": h, "reference_mean": reference_mean},
        "trace": trace,
        "guarantee": "none claimed (cumulative statistic; tuned constants per drift-literature practice)",
    }


def evaluate(monitor_result: dict, onset: int) -> dict:
    """Score one monitor run against a KNOWN drift onset (index into the value sequence).

    detection_latency = first alarm at/after onset, minus onset (None = missed).
    false_alarms = alarms strictly before onset. This is M5's per-technique scorecard;
    onset is known by construction in evaluation settings (injected shifts, or dated
    vintages whose decay window is designed).
    """
    alarms = monitor_result["alarms"]
    fa = [a for a in alarms if a < onset]
    post = [a for a in alarms if a >= onset]
    return {
        "monitor": monitor_result["monitor"],
        "false_alarms": len(fa),
        "false_alarm_indices": fa,
        "detection_latency": (post[0] - onset) if post else None,
        "detected": bool(post),
    }


def watch(
    reference,
    values,
    alpha: float = 0.2,
    direction: str = "down",
    ph_delta: float = 0.005,
    ph_lambda: float = 0.05,
    cusum_k: float = 0.01,
    cusum_h: float = 0.05,
) -> list[dict]:
    """Run the standard monitor panel over one interval-value sequence.

    The conformal monitor may refuse (young reference); the refusal is returned as a
    result row rather than raised, so a report can print WHY alongside the detectors that
    did run.
    """
    results: list[dict] = []
    try:
        results.append(conformal_alarm(reference, values, alpha, direction))
    except conformal.InsufficientCalibration as e:
        results.append(
            {
                "monitor": f"conformal-interval-{direction}",
                "refused": str(e),
                "alarms": [],
            }
        )
    ref_mean = float(np.asarray(reference, dtype=np.float64).mean())
    results.append(page_hinkley(values, delta=ph_delta, lam=ph_lambda, direction=direction))
    results.append(
        cusum(values, reference_mean=ref_mean, k=cusum_k, h=cusum_h, direction=direction)
    )
    return results
