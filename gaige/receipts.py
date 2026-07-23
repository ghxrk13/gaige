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


def _conformal_lines(results: dict) -> list[str]:
    rows = results.get("conformal", [])
    if not rows:
        return []
    lines = [
        "",
        "## Conformal thresholds (distribution-free FPR bound)",
        "Split conformal (Zhu et al., arXiv:2505.05084): the threshold is an order statistic "
        "of the human calibration scores, giving **P(human flagged) ≤ α marginally over "
        "calibration draws** — finite-sample, no distributional assumptions. Conditionally on "
        "THIS calibration set the true FPR is Beta(n+1−k, k); its exact mean ± sd is shown. "
        'No "achieved FPR" appears here by design: the in-sample flag rate is (n−k)/n by '
        "construction and measures nothing.",
        "",
        "| alpha | threshold | order stat k/n | conditional FPR mean ± sd | TPR on this corpus |",
        "|---|---|---|---|---|",
    ]
    refusals = []
    for r in rows:
        if "unavailable" in r:
            refusals.append(f"- α={r['alpha']:g}: refused — {r['unavailable']}")
            continue
        lines.append(
            f"| {r['alpha']:g} | {r['threshold']:.4f} | {r['order_statistic']}/{r['n_calibration']} "
            f"| {r['conditional_fpr_mean']:.2%} ± {r['conditional_fpr_sd']:.2%} | {r['tpr']:.1%} |"
        )
    return lines + refusals


def _fmt_rate(rate, ci, n: int, floor: int) -> str:
    if rate is None:
        return f"withheld (n={n} < {floor})"
    lo, hi = ci
    return f"{rate:.1%} [{lo:.1%}–{hi:.1%}]"


def _subgroup_lines(results: dict) -> list[str]:
    block = results.get("subgroups")
    if not block:
        return []
    lines = ["", "## Subgroup error rates (measured on this corpus only)"]
    if "unavailable" in block:
        return lines + [f"- {block['unavailable']}"]
    floor = block["min_subgroup"]
    lines += [
        f"Axes are what this corpus carries: length bucket always, metadata axes only when "
        f"present on every row. A class with fewer than {floor} samples gets its count shown "
        "and its rate withheld. Intervals are 95% bootstrap. These axes make **no demographic "
        "claim**; disparities documented in the literature (ESL: arXiv:2304.02819) are cited, "
        "not measured here, unless the corpus carries that axis.",
    ]
    for bt in block["by_threshold"]:
        lines += ["", f"At target FPR ≤ {bt['target_fpr']:.0%} (threshold {bt['threshold']:.4f}):"]
        for axis, groups in bt["strata"].items():
            lines += [
                "",
                f"| {axis} | n_human | FPR [95% CI] | n_ai | TPR [95% CI] |",
                "|---|---|---|---|---|",
            ]
            for value, d in groups.items():
                lines.append(
                    f"| {value} | {d['n_human']} "
                    f"| {_fmt_rate(d['fpr'], d['fpr_ci'], d['n_human'], floor)} "
                    f"| {d['n_ai']} | {_fmt_rate(d['tpr'], d['tpr_ci'], d['n_ai'], floor)} |"
                )
            disp = (bt.get("max_fpr_disparity") or {}).get(axis)
            if disp and disp["gap"] == 0:
                lines.append(
                    f"- max FPR disparity on {axis}: **0.0%** (all reported groups equal) "
                    "— FairOPT Δ_FPR, arXiv:2502.04528 Eq. 8"
                )
            elif disp:
                lines.append(
                    f"- max FPR disparity on {axis}: **{disp['gap']:.1%}** "
                    f"(worst {disp['worst_group']} {disp['worst_fpr']:.1%} vs best "
                    f"{disp['best_group']} {disp['best_fpr']:.1%}) — FairOPT Δ_FPR, "
                    "arXiv:2502.04528 Eq. 8"
                )
            else:
                lines.append(
                    f"- max FPR disparity on {axis}: not computable "
                    "(fewer than two groups with a reported rate)"
                )
    return lines


def _base_rate_lines(results: dict) -> list[str]:
    br = results.get("base_rate")
    if not br:
        return []
    lines = [
        "",
        "## Base-rate arithmetic (what an FPR means at volume)",
        f"Volume: {br['volume']:,} documents/year ({br['volume_note']}; change with "
        "`--harm-volume`). This is the calculation Vanderbilt published when it disabled "
        "Turnitin's AI detector. PPV = share of flags that would actually be AI, at an "
        "assumed prevalence, using this corpus's in-sample TPR.",
        "",
    ]
    for at in br["at"]:
        ppvs = " / ".join(
            f"{p['ppv']:.0%} @ {p['prevalence']:.0%} AI" for p in at["ppv_at_prevalence"]
        )
        lines.append(
            f"- operating at FPR ≤ {at['target_fpr']:.0%}: {at['target_fpr']:.0%} × "
            f"{br['volume']:,} = **{at['expected_false_positives']:,.0f} wrongly flagged "
            f"per year**. PPV: {ppvs}."
        )
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
        w = csv.DictWriter(
            f,
            fieldnames=["id", "label", "score", "seconds", "n_words", "meta"],
            extrasaction="ignore",
            restval="",
        )
        w.writeheader()
        for r in scores_rows:
            rr = dict(r)
            if isinstance(rr.get("meta"), dict):
                rr["meta"] = json.dumps(rr["meta"], separators=(",", ":"), sort_keys=True)
            w.writerow({k: ("" if rr.get(k) is None else rr.get(k, "")) for k in w.fieldnames})

    (outdir / "roc.json").write_text(json.dumps(results["roc"], indent=1), encoding="utf-8")
    (outdir / "results.json").write_text(
        json.dumps(
            # Wholesale write contract: everything compute_results emits ships,
            # except roc, which is written as its own artifact above. A hand-kept
            # key list here would silently drop a newly added statistic.
            {k: v for k, v in results.items() if k != "roc"},
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
        f"- **EER {results['eer']:.4f}** at threshold {results['eer_threshold']:.4f} (measured FPR = FNR crossing on this calibration sample; it does not transfer to other corpora)",
        "",
        "| target FPR | threshold | FPR on calibration sample | TPR at threshold | TPR 95% CI |",
        "|---|---|---|---|---|",
    ]
    for row in results["thresholds"]:
        t_lo, t_hi = row["tpr_ci"]
        lines.append(
            f"| {row['target_fpr']:.0%} | {row['threshold']:.4f} | {row['achieved_fpr']:.3%} "
            f"| {row['achieved_tpr']:.1%} | {t_lo:.1%}–{t_hi:.1%} |"
        )
    lines += _conformal_lines(results)
    lines += _subgroup_lines(results)
    lines += _base_rate_lines(results)
    lines += [
        "",
        "## Honest caveats (read before using these thresholds)",
        "- Thresholds are valid ONLY for this instrument fingerprint on text like this corpus. "
        "Different model, quantization, library versions, text domain, or model-family of the "
        "AI side ⇒ re-calibrate.",
        '- "FPR on calibration sample" is an in-sample observation, not a guarantee. The '
        "conformal table is the guarantee, and it is **marginal over calibration draws**: the "
        "conditional mean ± sd column shows how much the realized rate wobbles around α.",
        "- Every conformal guarantee assumes deployment human text is exchangeable with the "
        "calibration humans. Domain shift, a different population of writers, or a different "
        "AI family on the flagged side voids the bound. This assumption is stated, not waived.",
        "- Subgroup tables report only the axes this corpus carries; withheld cells are the "
        "floor working as intended, and interval width — not the point estimate — is the "
        "honest summary of a small group.",
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
