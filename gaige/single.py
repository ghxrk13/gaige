# gaige — calibration + receipts for AI-text detectors.
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


def version_mismatches(env: dict) -> list[str]:
    import bitsandbytes
    import torch
    import transformers

    stored = env["detector"]["versions"]
    current = {
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "bitsandbytes": bitsandbytes.__version__,
    }
    return [
        f"{k}: report={stored.get(k)} current={v}"
        for k, v in current.items()
        if stored.get(k) not in (None, v)
    ]


def score_document(report_dir: Path, text: str) -> dict:
    inst = load_instrument(report_dir)
    det_meta = inst["env"]["detector"]

    mismatches = version_mismatches(inst["env"])

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
        "version_mismatches": mismatches,
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
    if r["version_mismatches"]:
        lines.append("  WARNING: environment differs from the report's fingerprint — thresholds may not transfer:")
        for m in r["version_mismatches"]:
            lines.append(f"    - {m}")
    lines.append(f"  instrument: {r['instrument']['model']} ({r['instrument']['quant']}) calibrated on {r['instrument']['corpus']}")
    lines.append("  note: evidence, not a verdict. Nothing was logged.")
    return "\n".join(lines)
