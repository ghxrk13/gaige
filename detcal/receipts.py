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

    (outdir / "roc.json").write_text(json.dumps(results["roc"], indent=1))
    (outdir / "results.json").write_text(
        json.dumps(
            {
                "detcal_version": results["detcal_version"],
                "auroc": results["auroc"],
                "auroc_ci": results["auroc_ci"],
                "thresholds": results["thresholds"],
                "n_boot": results["n_boot"],
            },
            indent=1,
        )
    )
    env = {
        "generated_utc": ts,
        "detcal_version": results["detcal_version"],
        "host": {"platform": platform.platform(), "node_role": "reference GPU box"},
        "detector": detector_meta,
        "corpus": {
            "name": corpus.name,
            "sha256": corpus.sha256,
            "counts": corpus.counts,
            "meta": corpus.meta,
        },
        "reproduce": reproduce_cmd,
    }
    (outdir / "env.json").write_text(json.dumps(env, indent=1))

    a_lo, a_hi = results["auroc_ci"]
    lines = [
        f"# detcal receipt — {corpus.name} × {detector_meta['detector']}",
        "",
        f"generated: {ts} · detcal {results['detcal_version']}",
        "",
        "## Instrument fingerprint",
        f"- model: `{detector_meta['model_id']}` · quant requested **{detector_meta['quant_requested']}**, "
        f"verified: {detector_meta['quant_verified']['linear4bit_modules']} Linear4bit modules, "
        f"{detector_meta['quant_verified']['resident_gb']} GB resident",
        f"- versions: {json.dumps(detector_meta['versions'])}",
        f"- gpu: {detector_meta['gpu']} · max_tokens {detector_meta['max_tokens']}",
        f"- score semantics: {detector_meta['score_semantics']}",
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
    (outdir / "report.md").write_text("\n".join(lines) + "\n")
    return outdir / "report.md"
