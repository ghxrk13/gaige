# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Receipts writer: the point of the whole exercise.

A run emits a self-contained report directory:
  report.md    — human-readable receipt (env fingerprint, corpus fingerprint, results, caveats)
  scores.csv   — every sample's id, label, raw score
  roc.json     — full ROC sweep
  env.json     — detector + host fingerprint, corpus meta, exact reproduce command
"""

from __future__ import annotations

import csv
import json
import platform
from datetime import datetime, timezone
from pathlib import Path


def _fingerprint_lines(m: dict) -> list[str]:
    """Describe the instrument using only what was actually recorded.

    Different devices record different things — a CUDA 4-bit load can prove itself by counting
    Linear4bit modules and resident VRAM; a CPU fp32 load cannot, because there is nothing to
    quantize. Rendering must therefore report what exists and stay silent about what doesn't,
    rather than KeyError or, worse, imply a verification that never happened.
    """
    if m.get("instrument_unknown"):
        return [
            "- **INSTRUMENT UNKNOWN.** These scores arrived without a fingerprint, so nothing here "
            "identifies the model, quantization, device, or library versions that produced them. "
            "The thresholds below describe this score set and nothing else — they are not "
            "transferable, and they are not evidence about any instrument.",
        ]

    lines = [f"- model: `{m.get('model_id', 'unknown')}`"]

    quant = m.get("quant_requested", "unknown")
    qv = m.get("quant_verified") or {}
    if qv.get("linear4bit_modules"):
        lines[0] += (
            f" · quant requested **{quant}**, verified: {qv['linear4bit_modules']} Linear4bit "
            f"modules, {qv.get('resident_gb', '?')} GB resident"
        )
    elif qv.get("resident_gb") is not None:
        lines[0] += f" · dtype **{quant}**, {qv['resident_gb']} GB resident"
    else:
        lines[0] += f" · dtype/quant **{quant}**"

    device = m.get("device", "unknown")
    compute = m.get("compute", {}) or {}
    dev_line = f"- device: **{device}**"
    if compute.get("name"):
        dev_line += f" ({compute['name']})"
    if m.get("device_fallback"):
        dev_line += "  ← **fell back from CUDA; this is a DIFFERENT instrument than a GPU run**"
    lines.append(dev_line)

    if m.get("versions"):
        lines.append(f"- versions: {json.dumps(m['versions'])}")
    if m.get("max_tokens") is not None:
        lines.append(f"- max_tokens: {m['max_tokens']}")
    if m.get("score_semantics"):
        lines.append(f"- score semantics: {m['score_semantics']}")
    return lines


def write_report(
    outdir: Path,
    corpus,
    detector_meta: dict,
    scores_rows: list[dict],
    results: dict,
    reproduce_cmd: str,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(outdir / "scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "label", "score", "seconds"])
        w.writeheader()
        w.writerows(scores_rows)

    (outdir / "roc.json").write_text(json.dumps(results["roc"], indent=1), encoding="utf-8")
    (outdir / "results.json").write_text(
        json.dumps(
            {
                "gaige_version": results["gaige_version"],
                "auroc": results["auroc"],
                "auroc_ci": results["auroc_ci"],
                "thresholds": results["thresholds"],
                "n_boot": results["n_boot"],
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    env = {
        "generated_utc": ts,
        "gaige_version": results["gaige_version"],
        "host": {"platform": platform.platform(), "device": detector_meta.get("device", "unknown")},
        "detector": detector_meta,
        "corpus": {
            "name": corpus.name,
            "sha256": corpus.sha256,
            "counts": corpus.counts,
            "meta": corpus.meta,
        },
        "reproduce": reproduce_cmd,
    }
    (outdir / "env.json").write_text(json.dumps(env, indent=1), encoding="utf-8")

    a_lo, a_hi = results["auroc_ci"]
    lines = [
        f"# gaige receipt — {corpus.name} × {detector_meta['detector']}",
        "",
        f"generated: {ts} · gaige {results['gaige_version']}",
        "",
        "## Instrument fingerprint",
        *_fingerprint_lines(detector_meta),
        "",
        "## Corpus fingerprint",
        f"- {corpus.name} — sha256 `{corpus.sha256[:16]}…` · counts {corpus.counts}",
        f"- source: {corpus.meta.get('source', '?')} · filters: {corpus.meta.get('filters', '{}')}",
        "",
        "## Results",
        f"- **AUROC {results['auroc']:.4f}** (95% bootstrap CI {a_lo:.4f}–{a_hi:.4f}, n_boot={results['n_boot']})",
        "",
        "| target FPR | threshold | achieved FPR | TPR at threshold | TPR 95% CI |",
        "|---|---|---|---|---|",
    ]
    for row in results["thresholds"]:
        t_lo, t_hi = row["tpr_ci"]
        lines.append(
            f"| {row['target_fpr']:.0%} | {row['threshold']:.4f} | {row['achieved_fpr']:.3%} "
            f"| {row['achieved_tpr']:.1%} | {t_lo:.1%}–{t_hi:.1%} |"
        )
    lines += [
        "",
        "## Honest caveats (read before using these thresholds)",
        "- Thresholds are valid ONLY for this instrument fingerprint on text like this corpus. "
        "Different model, quantization, library versions, text domain, or model-family of the "
        "AI side ⇒ re-calibrate.",
        f"- {corpus.meta.get('note', 'Corpus generalization limits apply.')}",
        "- Single detector, single corpus: no ensemble, no cross-domain claim, no style-matched "
        "adversarial evaluation (documented failure mode of this detector class).",
        "- A score crossing a threshold is evidence, not a verdict. Error bars above are the point.",
        "",
        "## Reproduce",
        f"```\n{reproduce_cmd}\n```",
    ]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outdir / "report.md"
