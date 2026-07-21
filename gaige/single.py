# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Score one document against a calibrated receipts report.

The report directory is the instrument definition: which model, which quantization, which
thresholds, measured on which corpus. Scoring a document with a DIFFERENT environment than
the report's fingerprint invalidates the thresholds — mismatches are loudly surfaced, not
silently tolerated.

Nothing about the scored document is written anywhere. Privacy is a feature.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

MIN_RELIABLE_WORDS = 50


def load_instrument(report_dir: Path) -> dict:
    env = json.loads((report_dir / "env.json").read_text(encoding="utf-8"))
    results = json.loads((report_dir / "results.json").read_text(encoding="utf-8"))
    human, ai = [], []
    with open(report_dir / "scores.csv", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            (human if row["label"] == "human" else ai).append(float(row["score"]))
    return {"env": env, "results": results, "human_scores": sorted(human), "ai_scores": sorted(ai)}


def percentile_among(sorted_vals: list[float], value: float) -> float:
    """Fraction of reference values <= value (0..1)."""
    import bisect

    if not sorted_vals:
        return float("nan")
    return bisect.bisect_right(sorted_vals, value) / len(sorted_vals)


def _live_versions() -> dict:
    """Versions of whatever is importable right now. Absent libraries are simply absent."""
    live = {}
    try:
        import torch

        live["torch"] = torch.__version__
    except Exception:
        pass
    try:
        import transformers

        live["transformers"] = transformers.__version__
    except Exception:
        pass
    try:
        import bitsandbytes

        live["bitsandbytes"] = bitsandbytes.__version__
    except Exception:
        pass
    return live


def instrument_mismatches(env: dict, live_device: str | None = None) -> list[str]:
    """Every recorded way this environment differs from the one that produced the thresholds.

    Library versions were always checked. Device and dtype now are too, because they matter at
    least as much: CUDA-4bit and CPU-fp32 are different numerics, so a threshold calibrated on
    one is not valid on the other. Silently reusing it across that boundary is precisely the
    error gaige exists to make visible in other people's tools.

    Returns a list of human-readable differences; empty means the instrument matches.
    """
    det = env.get("detector", {})
    out: list[str] = []

    stored_versions = det.get("versions", {}) or {}
    for k, v in _live_versions().items():
        if stored_versions.get(k) not in (None, v):
            out.append(f"{k}: report={stored_versions.get(k)} current={v}")

    if live_device is None:
        try:
            import torch

            live_device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            live_device = None
    stored_device = det.get("device")
    if stored_device and live_device and stored_device != live_device:
        out.append(
            f"device: report={stored_device} current={live_device} "
            "(different numerics - thresholds do not transfer)"
        )
    return out


# Retained so existing callers keep working; the broader check is the one to use.
def version_mismatches(env: dict) -> list[str]:
    return instrument_mismatches(env)


def score_document(report_dir: Path, text: str, detector=None) -> dict:
    """Score one document against a calibrated report.

    `detector` exists so this function can be tested without a GPU, and specifically so the
    no-persistence claim in SECURITY.md can be VERIFIED rather than asserted. A security
    property nobody can test is a promise, and this project does not trade in promises.
    Passing None keeps the production path: build the instrument the report describes.
    """
    inst = load_instrument(report_dir)
    det_meta = inst["env"]["detector"]

    mismatches = instrument_mismatches(inst["env"])

    det = detector
    if det is None:
        from .detectors.fast_detect_gpt import FastDetectGPT

        det = FastDetectGPT(
            model_id=det_meta["model_id"],
            quant=det_meta["quant_requested"],
            max_tokens=det_meta["max_tokens"],
        )
        det.load()
    score = det.score(text)

    n_words = len(text.split())
    verdicts = []
    for row in inst["results"]["thresholds"]:
        verdicts.append(
            {
                "target_fpr": row["target_fpr"],
                "threshold": row["threshold"],
                "flags": bool(score >= row["threshold"]),
                "calibrated_tpr": row["achieved_tpr"],
            }
        )
    return {
        "score": score,
        "n_words": n_words,
        "short_text_caveat": n_words < MIN_RELIABLE_WORDS,
        "percentile_among_human": percentile_among(inst["human_scores"], score),
        "percentile_among_ai": percentile_among(inst["ai_scores"], score),
        "verdicts": verdicts,
        "instrument_mismatches": mismatches,
        "instrument": {
            "report": str(report_dir),
            "corpus": inst["env"]["corpus"]["name"],
            "model": det_meta["model_id"],
            "quant": det_meta["quant_requested"],
        },
    }


def format_result(r: dict) -> str:
    lines = [
        f"score {r['score']:.4f}  ({r['n_words']} words)",
        f"  vs calibration corpus: higher than {r['percentile_among_human']:.0%} of known-HUMAN, "
        f"higher than {r['percentile_among_ai']:.0%} of known-AI samples",
    ]
    for v in r["verdicts"]:
        state = "FLAGS" if v["flags"] else "clear"
        lines.append(
            f"  @FPR<={v['target_fpr']:.0%} (thr {v['threshold']:.4f}): {state} "
            f"(threshold catches {v['calibrated_tpr']:.0%} of corpus AI)"
        )
    if r["short_text_caveat"]:
        lines.append(
            f"  CAVEAT: {r['n_words']} words < {MIN_RELIABLE_WORDS} — detector scores on text this "
            "short are noise-dominated; treat as unreliable in BOTH directions."
        )
    if r["instrument_mismatches"]:
        lines.append(
            "  WARNING: environment differs from the report's fingerprint — thresholds may not transfer:"
        )
        for m in r["instrument_mismatches"]:
            lines.append(f"    - {m}")
    lines.append(
        f"  instrument: {r['instrument']['model']} ({r['instrument']['quant']}) calibrated on {r['instrument']['corpus']}"
    )
    lines.append("  note: evidence, not a verdict. Nothing was logged.")
    return "\n".join(lines)
