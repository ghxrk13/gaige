# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Replay the analysis layer over scores that already exist.

Scoring needs the model; *analysing* the scores needs nothing but arithmetic. Separating the
two is what lets calibration run where the GPU isn't — a laptop, a CPU-only enclave, the
machine you happen to be sitting at — and it is the building block a longitudinal series
needs, since a series is many runs' scores analysed together.

`compute_results` is deliberately the ONE path that turns scores into results. `gaige run`
and `gaige analyze` both call it, so a replay cannot silently disagree with the original run;
if it ever does, that is a real regression and the round-trip test will say so.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import __version__, calibrate

TARGET_FPRS = (0.01, 0.05)


class NotAReport(ValueError):
    """Directory does not contain the artifacts a gaige report is made of."""


@dataclass
class CorpusView:
    """Just enough of a Corpus for `receipts.write_report` to describe what was measured.

    Reconstructed from a report's env.json rather than re-derived, so a replay describes the
    corpus the scores actually came from — never a fresh guess at it.
    """

    name: str
    sha256: str
    counts: dict
    meta: dict = field(default_factory=dict)


# Used when scores arrive without a report to describe them. Honest about the hole rather
# than filling it: a receipt whose instrument is unknown must SAY the instrument is unknown.
UNKNOWN_CORPUS = CorpusView(
    name="(supplied scores — corpus not recorded)",
    sha256="unknown",
    counts={},
    meta={
        "source": "scores supplied directly to `gaige analyze`",
        "note": "Corpus provenance was not recorded with these scores; nothing here attests to "
        "what they were measured on.",
    },
)

UNKNOWN_DETECTOR = {
    "detector": "unknown",
    "model_id": "unknown",
    "quant_requested": "unknown",
    "quant_verified": {},
    "max_tokens": None,
    "versions": {},
    "device": "unknown",
    "score_semantics": "unknown — scores supplied directly; higher assumed more AI-like",
    "instrument_unknown": True,
}


def read_scores_csv(path: Path) -> list[dict]:
    """Read a gaige scores.csv back into rows. Requires at minimum `label` and `score`."""
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        missing = {"label", "score"} - set(reader.fieldnames or [])
        if missing:
            raise NotAReport(f"{path}: scores file missing required column(s): {sorted(missing)}")
        for i, row in enumerate(reader):
            label = (row.get("label") or "").strip()
            if label not in ("human", "ai"):
                raise NotAReport(
                    f"{path} row {i + 1}: label must be 'human' or 'ai', got {label!r}"
                )
            rows.append(
                {
                    "id": row.get("id") or f"row{i}",
                    "label": label,
                    "score": float(row["score"]),
                    "seconds": float(row["seconds"]) if row.get("seconds") else "",
                }
            )
    if not rows:
        raise NotAReport(f"{path}: no score rows found")
    return rows


def load_report(report_dir: Path) -> tuple[list[dict], CorpusView, dict]:
    """Load an existing report's scores plus the corpus/instrument it recorded."""
    scores_path = report_dir / "scores.csv"
    if not scores_path.exists():
        raise NotAReport(f"{report_dir}: no scores.csv (is this a gaige report directory?)")
    rows = read_scores_csv(scores_path)

    env_path = report_dir / "env.json"
    if not env_path.exists():
        return rows, UNKNOWN_CORPUS, dict(UNKNOWN_DETECTOR)
    env = json.loads(env_path.read_text(encoding="utf-8"))
    c = env.get("corpus") or {}
    corpus = CorpusView(
        name=c.get("name", UNKNOWN_CORPUS.name),
        sha256=c.get("sha256", "unknown"),
        counts=c.get("counts", {}),
        meta=c.get("meta", {}),
    )
    detector = env.get("detector") or dict(UNKNOWN_DETECTOR)
    return rows, corpus, detector


def compute_results(
    rows: list[dict],
    target_fprs: tuple[float, ...] = TARGET_FPRS,
    n_boot: int = 1000,
    seed: int = 17,
) -> dict:
    """Scores -> AUROC, thresholds at target FPRs, bootstrap CIs, ROC sweep.

    The single shared path for `run` and `analyze`. Keep it that way: two code paths that
    compute the same statistic will eventually disagree, and this is the number people act on.
    """
    scores = np.array([r["score"] for r in rows], dtype=np.float64)
    labels = np.array([r["label"] for r in rows])

    auroc = calibrate.auroc(scores, labels)
    auroc_ci = calibrate.bootstrap_ci(scores, labels, calibrate.auroc, n_boot=n_boot, seed=seed)

    thresholds = []
    for tf in target_fprs:
        row = calibrate.threshold_at_fpr(scores, labels, tf)
        thr = row["threshold"]
        row["tpr_ci"] = calibrate.bootstrap_ci(
            scores,
            labels,
            lambda s, lb, thr=thr: float((s[lb == "ai"] >= thr).mean()),
            n_boot=n_boot,
            seed=seed,
        )
        thresholds.append(row)

    return {
        "gaige_version": __version__,
        "auroc": auroc,
        "auroc_ci": auroc_ci,
        "thresholds": thresholds,
        "roc": calibrate.roc_points(scores, labels),
        "n_boot": n_boot,
    }
