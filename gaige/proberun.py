# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""The probe runner: dated probes -> model answers -> deterministic grades -> a receipt.

This is the acquisition layer the longitudinal spec's M1 metric needs: accuracy per vintage,
with an interval, under an instrument fingerprint that includes the provider identity and
attestation, the full decoding block, the grading rule version, the probe-set hash, and the
per-vintage post-cutoff share (the contamination demonstration).

Crash-safe and resumable via the runstate pattern: every answer hits disk before the next
probe is asked, and a resume refuses if ANY pinned fingerprint field changed — an answer
set spanning two instruments is not a measurement.
"""

from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import calibrate, probcal, ptrue, runstate
from .grading import GRADING_VERSION, grade_free_text
from .probes import ProbeSet
from .providers.base import CAP_COMPLETE, CAP_OPTION_LOGPROBS, Decoding, Provider, require

PROBE_FIELDS = ["id", "vintage", "response", "normalized", "correct", "ptrue", "seconds"]
# Everything here changing mid-run means the second half was measured by a different
# instrument. The whole provider metadata dict is pinned deliberately — strict beats sorry.
PROBE_PINNED = (
    "provider",
    "decoding",
    "grading_version",
    "probes_sha256",
    "training_cutoff",
    "ptrue",
)


def probe_instrument(
    provider_meta: dict, decoding: Decoding, probeset, cutoff: str, with_ptrue: bool = False
) -> dict:
    return {
        "provider": provider_meta,
        "decoding": decoding.fingerprint(),
        "grading_version": GRADING_VERSION,
        "probes_sha256": probeset.sha256,
        "training_cutoff": cutoff,
        # The P(True) template is an instrument component when M3 runs; toggling it or
        # changing the template mid-series is a fork, so it is pinned like everything else.
        "ptrue": ptrue.template_fingerprint() if with_ptrue else None,
    }


def _load_partial(outdir: Path) -> dict[str, dict]:
    p = outdir / runstate.PARTIAL
    if not p.exists():
        return {}
    done: dict[str, dict] = {}
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                done[row["id"]] = {
                    "id": row["id"],
                    "vintage": row["vintage"],
                    "response": row["response"],
                    "normalized": row["normalized"],
                    "correct": row["correct"] == "True",
                    "ptrue": float(row["ptrue"]) if row.get("ptrue") not in (None, "") else "",
                    "seconds": float(row["seconds"]) if row.get("seconds") else "",
                }
            except (KeyError, TypeError, ValueError):
                continue  # a partial final line from an abrupt kill; re-asked on resume
    return done


def vintage_accuracy(rows: list[dict], n_boot: int = 1000, seed: int = 17) -> dict:
    """Per-vintage accuracy with a bootstrap CI (vectorized proportion path)."""
    out: dict = {}
    for v in sorted({r["vintage"] for r in rows}):
        correct = np.array([float(r["correct"]) for r in rows if r["vintage"] == v])
        entry: dict = {"n": int(len(correct)), "accuracy": float(correct.mean())}
        if len(correct) >= 2:
            entry["accuracy_ci"] = calibrate.proportion_ci(correct, n_boot=n_boot, seed=seed)
        out[v] = entry
    return out


def m3_by_vintage(
    rows: list[dict], n_bins: int = probcal.DEFAULT_BINS, n_boot: int = 1000, seed: int = 17
) -> dict:
    """Per-vintage M3: mean P(True), confidence-accuracy gap, ECE with CI."""
    out: dict = {}
    for v in sorted({r["vintage"] for r in rows}):
        conf = np.array([r["ptrue"] for r in rows if r["vintage"] == v], dtype=np.float64)
        corr = np.array([float(r["correct"]) for r in rows if r["vintage"] == v], dtype=np.float64)
        e = probcal.ece(conf, corr, n_bins=n_bins)
        out[v] = {
            "mean_confidence": float(conf.mean()),
            "gap": probcal.confidence_accuracy_gap(conf, corr),
            "ece": e["ece"],
            "ece_ci": probcal.ece_ci(conf, corr, n_bins=n_bins, n_boot=n_boot, seed=seed),
            "brier": probcal.brier(conf, corr)["brier"],
            "n_bins": n_bins,
        }
    return out


def run_probes(
    probeset: ProbeSet,
    provider: Provider,
    decoding: Decoding,
    cutoff: str,
    outdir: Path,
    n_boot: int = 1000,
    seed: int = 17,
    resume: bool = False,
    reproduce_cmd: str = "",
    with_ptrue: bool = False,
    progress=print,
) -> dict:
    require(provider, CAP_COMPLETE)
    if with_ptrue:
        require(provider, CAP_OPTION_LOGPROBS)
    provider_meta = provider.metadata()
    instrument = probe_instrument(provider_meta, decoding, probeset, cutoff, with_ptrue)

    done: dict[str, dict] = {}
    if resume:
        state = runstate.read_runstate(outdir)
        if state.get("corpus", {}).get("sha256") != probeset.sha256:
            raise runstate.ResumeRefused(
                f"probe set changed: run was {state.get('corpus', {}).get('sha256', '?')[:16]}..., "
                f"now {probeset.sha256[:16]}.... A resumed run must ask the same questions."
            )
        runstate.check_instrument_match(state, instrument, pinned=PROBE_PINNED)
        done = _load_partial(outdir)
        progress(f"[resume] {len(done)}/{len(probeset.probes)} already answered; continuing")
    else:
        runstate.write_runstate(outdir, probeset, instrument, reproduce_cmd, pinned=PROBE_PINNED)

    fh, writer = runstate.open_partial(outdir, fields_override=PROBE_FIELDS)
    rows: list[dict] = []
    t_all = time.time()
    try:
        for i, probe in enumerate(probeset.probes, 1):
            if probe["id"] in done:
                rows.append(done[probe["id"]])
                continue
            t0 = time.time()
            response = provider.complete(probe["prompt"], decoding)
            g = grade_free_text(response, probe["answer"], probe.get("aliases", ()))
            row = {
                "id": probe["id"],
                "vintage": probe["vintage"],
                "response": response,
                "normalized": g["normalized_answer"],
                "correct": g["correct"],
                "ptrue": (
                    round(ptrue.ptrue_score(provider, probe["prompt"], response), 6)
                    if with_ptrue
                    else ""
                ),
                "seconds": round(time.time() - t0, 3),
            }
            runstate.append_row(fh, writer, row)
            rows.append(row)
            if i % 10 == 0 or i == len(probeset.probes):
                progress(f"[probe] {i}/{len(probeset.probes)} ({time.time() - t_all:.0f}s)")
    except (KeyboardInterrupt, Exception):
        fh.close()
        progress(
            f"\n[interrupted] {len(rows)}/{len(probeset.probes)} answers are safe on disk; "
            f"resume with --resume {outdir}"
        )
        raise
    fh.close()

    by_vintage = vintage_accuracy(rows, n_boot=n_boot, seed=seed)
    if with_ptrue:
        for v, m3 in m3_by_vintage(rows, n_boot=n_boot, seed=seed).items():
            by_vintage[v]["m3"] = m3
    results = {
        "gaige_version": _version(),
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "by_vintage": by_vintage,
        "post_cutoff_share": probeset.post_cutoff_share(cutoff),
        "vintage_hashes": probeset.vintage_hashes,
        "n_boot": n_boot,
        "instrument": instrument,
    }
    write_probe_report(outdir, probeset, instrument, rows, results, reproduce_cmd)
    runstate.mark_complete(outdir)
    return results


def _version() -> str:
    from . import __version__

    return __version__


def write_probe_report(
    outdir: Path,
    probeset: ProbeSet,
    instrument: dict,
    rows: list[dict],
    results: dict,
    reproduce_cmd: str,
) -> Path:
    outdir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    with open(outdir / "answers.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=PROBE_FIELDS, extrasaction="ignore", restval="")
        w.writeheader()
        w.writerows(rows)
    (outdir / "probe-results.json").write_text(
        json.dumps(results, indent=1, default=str), encoding="utf-8"
    )

    prov = instrument["provider"]
    att = prov.get("attestation", "unknown")
    lines = [
        f"# gaige probe receipt — {probeset.name} × {prov.get('provider', '?')}",
        "",
        f"generated: {ts} · gaige {results['gaige_version']} · grading {instrument['grading_version']}",
        "",
        "## Instrument fingerprint",
        f"- provider: `{prov.get('provider', '?')}` · model `{prov.get('model_id', prov.get('model_requested', '?'))}` "
        f"· **attestation: {att}** ({prov.get('attestation_basis', 'in-process')})",
        f"- decoding: {json.dumps(instrument['decoding'])} — any change forks the series",
        f"- grading rule: normalized exact match, version `{instrument['grading_version']}`",
        "",
        "## Probe-set fingerprint",
        f"- {probeset.name} — sha256 `{probeset.sha256[:16]}…` · vintages {probeset.vintages}",
        f"- training cutoff {instrument['training_cutoff']} · post-cutoff share per vintage: "
        + " · ".join(
            f"{v}: {d['post_cutoff']}/{d['n']} ({d['share']:.0%})"
            for v, d in sorted(results["post_cutoff_share"].items())
        ),
        "",
        "## Accuracy by vintage",
        "",
        "| vintage | n | accuracy | 95% CI |",
        "|---|---|---|---|",
    ]
    for v, d in sorted(results["by_vintage"].items()):
        ci = d.get("accuracy_ci")
        ci_s = f"{ci[0]:.1%}–{ci[1]:.1%}" if ci else "—"
        lines.append(f"| {v} | {d['n']} | {d['accuracy']:.1%} | {ci_s} |")
    m3_rows = {v: d["m3"] for v, d in results["by_vintage"].items() if "m3" in d}
    if m3_rows:
        pt = instrument["ptrue"]
        lines += [
            "",
            "## M3 — calibration drift (P(True) vs accuracy)",
            f"Kadavath-style P(True), template `{pt['version']}` (sha256 "
            f"`{pt['template_sha256'][:16]}…`) — a forward-pass logit read, no sampling. "
            "Positive gap = overconfident. Bin count is fixed per series.",
            "",
            "| vintage | mean P(True) | accuracy | gap (conf−acc) | ECE | ECE 95% CI |",
            "|---|---|---|---|---|---|",
        ]
        for v, m in sorted(m3_rows.items()):
            acc = results["by_vintage"][v]["accuracy"]
            lo, hi = m["ece_ci"]
            lines.append(
                f"| {v} | {m['mean_confidence']:.1%} | {acc:.1%} | {m['gap']:+.1%} "
                f"| {m['ece']:.3f} | {lo:.3f}–{hi:.3f} |"
            )
    lines += [
        "",
        "## Honest caveats",
        "- Accuracy is defined by the versioned grading rule above (deterministic normalized "
        "exact match + authored aliases). A different rule is a different instrument.",
        "- A vintage's post-cutoff share below 100% means some probes may sit inside the "
        "model's training data: correct answers there can measure memorization, not currency.",
    ]
    if att != "verified":
        lines.append(
            f"- **Attestation is `{att}`**, not verified: the model identity rests on the "
            "provider's report"
            + (
                ". At `opaque`, these numbers are valid at most while the endpoint's "
                "behavior is unchanged — which nothing here can attest — and drift "
                "attribution is impossible."
                if att == "opaque"
                else "; an unchanged report across runs is the evidence of an unchanged instrument."
            )
        )
    lines += [
        "- Single run: this receipt is a point, not a series. Movement claims need the run "
        "registry, replicates, and the variance bound.",
        "",
        "## Reproduce",
        f"```\n{reproduce_cmd}\n```",
    ]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outdir / "report.md"
