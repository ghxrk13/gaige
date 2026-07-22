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

from . import __version__, calibrate, conformal, subgroups

TARGET_FPRS = (0.01, 0.05)
CONFORMAL_ALPHAS = (0.05, 0.01, 0.005)
HARM_VOLUME_DEFAULT = 75000  # Vanderbilt's published annual submission volume
PPV_PREVALENCES = (0.01, 0.10, 0.50)


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
            out = {
                "id": row.get("id") or f"row{i}",
                "label": label,
                "score": float(row["score"]),
                "seconds": float(row["seconds"]) if row.get("seconds") else "",
            }
            # Optional columns: derived word count + corpus metadata (for subgroup
            # receipts). Older score sets simply lack them; the report then says so.
            nw = row.get("n_words")
            if nw not in (None, ""):
                out["n_words"] = int(float(nw))
            m = row.get("meta")
            if m:
                try:
                    out["meta"] = json.loads(m)
                except ValueError:
                    pass
            rows.append(out)
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
    harm_volume: int = HARM_VOLUME_DEFAULT,
) -> dict:
    """Scores -> AUROC, thresholds, conformal table, subgroup rates, base-rate arithmetic.

    The single shared path for `run` and `analyze`. Keep it that way: two code paths that
    compute the same statistic will eventually disagree, and this is the number people act on.
    Everything here is deterministic given (rows, n_boot, seed), which is what keeps the
    replay round-trip byte-identical.
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

    conformal_rows = conformal.conformal_table(
        scores[labels == "human"], scores[labels == "ai"], CONFORMAL_ALPHAS
    )

    if all(isinstance(r.get("n_words"), int) for r in rows):
        sub_rows = [
            {
                "label": r["label"],
                "score": r["score"],
                "n_words": r["n_words"],
                "meta": r.get("meta"),
            }
            for r in rows
        ]
        by_threshold = []
        for row in thresholds:
            strata = subgroups.stratified_rates(
                sub_rows, row["threshold"], n_boot=n_boot, seed=seed
            )
            by_threshold.append(
                {
                    "target_fpr": row["target_fpr"],
                    "threshold": row["threshold"],
                    "strata": strata,
                    "max_fpr_disparity": subgroups.max_disparity(strata),
                }
            )
        subgroups_block: dict = {
            "min_subgroup": subgroups.MIN_SUBGROUP,
            "by_threshold": by_threshold,
        }
    else:
        subgroups_block = {
            "unavailable": "n_words was not recorded with these scores (older report or bare "
            "scores.csv); re-score, or supply an n_words column, to get subgroup receipts"
        }

    base_rate = {
        "volume": harm_volume,
        "volume_note": "default is Vanderbilt's published 75,000 submissions/year",
        "at": [
            {
                "target_fpr": row["target_fpr"],
                "expected_false_positives": subgroups.base_rate_harm(
                    row["target_fpr"], harm_volume
                )["expected_false_positives"],
                "ppv_at_prevalence": [
                    {
                        "prevalence": p,
                        "ppv": subgroups.ppv(row["target_fpr"], row["achieved_tpr"], p),
                    }
                    for p in PPV_PREVALENCES
                ],
            }
            for row in thresholds
        ],
    }

    return {
        "gaige_version": __version__,
        "auroc": auroc,
        "auroc_ci": auroc_ci,
        "thresholds": thresholds,
        "conformal": conformal_rows,
        "subgroups": subgroups_block,
        "base_rate": base_rate,
        "roc": calibrate.roc_points(scores, labels),
        "n_boot": n_boot,
    }
