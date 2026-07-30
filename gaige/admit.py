# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Corpus admission: measure how far unlabeled candidate material diverges from an
accepted baseline receipt, under the baseline's fingerprinted instrument.

The framing is trusted-vs-new, not human-vs-AI: the baseline is whatever an organization
has vetted and accepted (its provenance does not matter; the acceptance is the label), and
new material is measured for divergence from that accepted distribution. One verb, one
receipt: conformal novelty rate with an exact false-alarm law as the guarantee-bearing
number, KS distance and quantile shifts as support, per-document placements, stratified
where-it-differs, and refusal floors everywhere the evidence is too thin.

What this module will never do: emit admit / reject / pass / fail. It reports divergence
measurements with intervals; the decision is the organization's. It never claims material
is AI or human. Divergence is not badness, baselines age, and every number is relative to
this corpus, this scorer, this operating point.
"""

from __future__ import annotations

import csv
import json
import platform
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__, divergence, runstate, subgroups
from .analyze import NotAReport, read_scores_csv
from .calibrate import proportion_ci
from .corpus import _sha256
from .receipts import _fingerprint_lines
from .single import (
    MIN_RELIABLE_WORDS,
    instrument_mismatches,
    load_instrument,
    percentile_among,
)

# One floor, shared with the subgroup machinery: below this many candidate documents,
# slice-level rates are noise wearing a percent sign. Per-document placements still write.
CANDIDATE_FLOOR = subgroups.MIN_SUBGROUP


@dataclass
class CandidateSlice:
    """An unlabeled candidate slice. Deliberately not `corpus.Corpus`: that type's
    counts property keys on labels, and a candidate being admitted has none."""

    name: str
    path: Path | None
    items: list[dict]
    sha256: str
    labels_present: bool = False

    @property
    def n(self) -> int:
        return len(self.items)

    @property
    def counts(self) -> dict:
        return {"candidate": len(self.items)}


def load_candidate_jsonl(path: Path) -> CandidateSlice:
    """JSONL rows {id?, text, meta?}. Labels present on rows are noted, never refused —
    admission measures divergence, not class membership."""
    items: list[dict] = []
    labels_present = False
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not row.get("text"):
                raise ValueError(f"{path}:{i + 1}: candidate rows need non-empty text")
            if "label" in row:
                labels_present = True
            item = {"id": row.get("id") or f"{path.stem}-{i}", "text": row["text"]}
            if isinstance(row.get("meta"), dict):
                item["meta"] = row["meta"]
            items.append(item)
    if not items:
        raise ValueError(f"{path}: no candidate rows found")
    return CandidateSlice(
        name=path.stem,
        path=path,
        items=items,
        sha256=_sha256(path),
        labels_present=labels_present,
    )


def read_candidate_scores_csv(path: Path) -> tuple[list[dict], str, bool]:
    """The no-GPU lane: a CSV with a `score` column (id, seconds, n_words, meta optional).

    Returns (rows, file sha256, labels_present). Scores arriving this way carry no
    instrument attestation; the receipt says so loudly rather than refusing — candidate
    attestations have no mechanism yet, so honesty is a label here, not a gate.
    """
    rows: list[dict] = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames or []
        if "score" not in fields:
            raise NotAReport(f"{path}: candidate scores file needs a 'score' column")
        for i, row in enumerate(reader):
            out: dict = {"id": row.get("id") or f"row{i}", "score": float(row["score"])}
            sec = row.get("seconds")
            if sec not in (None, ""):
                out["seconds"] = float(sec)
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
        raise NotAReport(f"{path}: no candidate score rows found")
    return rows, _sha256(path), "label" in fields


def _mismatches(
    env: dict, live_versions: dict | None = None, live_device: str | None = None
) -> list[str]:
    """Instrument-mismatch check with injectable live state, so the refusal path is
    testable on machines that could never load a model. Production (both None for
    versions) defers to `single.instrument_mismatches` unchanged."""
    if live_versions is None:
        return instrument_mismatches(env, live_device=live_device)
    det = env.get("detector", {}) or {}
    stored = det.get("versions", {}) or {}
    out: list[str] = []
    for k, v in live_versions.items():
        if stored.get(k) not in (None, v):
            out.append(f"{k}: report={stored.get(k)} current={v}")
    stored_device = det.get("device")
    if stored_device and live_device and stored_device != live_device:
        out.append(
            f"device: report={stored_device} current={live_device} "
            "(different numerics - thresholds do not transfer)"
        )
    return out


def _axes(rows: list[dict]) -> list[str]:
    """Stratification axes for candidate rows: length bucket when every row recorded a
    word count, plus any metadata key present on every row (the subgroups rule)."""
    keys: list[str] = []
    if rows and all(isinstance(r.get("n_words"), int) for r in rows):
        keys.append("length_bucket")
    metas = [r.get("meta") or {} for r in rows]
    common = set(metas[0]) if metas else set()
    for m in metas[1:]:
        common &= set(m)
    keys += sorted(common)
    return keys


def _baseline_counts(b_rows: list[dict], key: str) -> dict | None:
    """Per-value counts of the baseline rows on one axis, or None when the baseline
    never recorded that axis."""
    if not b_rows:
        return None
    if key == "length_bucket":
        if not all(isinstance(r.get("n_words"), int) for r in b_rows):
            return None
    elif not all(key in (r.get("meta") or {}) for r in b_rows):
        return None
    counts: dict = {}
    for r in b_rows:
        v = subgroups._value(r, key)
        counts[v] = counts.get(v, 0) + 1
    return counts


def _strata(
    cand_rows: list[dict],
    novelty: list[dict],
    primary_alpha: float | None,
    b_rows: list[dict],
    n_boot: int,
    seed: int,
) -> dict:
    """Where the candidate differs: per-axis, per-value novelty against the GLOBAL band
    at the primary alpha. Below the subgroup floor the rate is withheld and the count
    speaks — the existing subgroups rule, unchanged."""
    if primary_alpha is None:
        return {
            "unavailable": "no requested alpha is supported by the reference size; "
            "there is no banded rate to stratify"
        }
    band = next(r for r in novelty if r.get("alpha") == primary_alpha and "unavailable" not in r)
    lo, hi = band["lo"], band["hi"]
    keys = _axes(cand_rows)
    if not keys:
        return {
            "unavailable": "candidate rows carry no word counts and no metadata key "
            "present on every row; there is no axis to stratify on"
        }
    axes: dict = {}
    for key in keys:
        groups: dict = {}
        for r in cand_rows:
            groups.setdefault(subgroups._value(r, key), []).append(r)
        base_counts = _baseline_counts(b_rows, key)
        per_value: dict = {}
        for value, grp in sorted(groups.items()):
            entry: dict = {"n": len(grp)}
            if base_counts is not None:
                entry["n_baseline"] = base_counts.get(value, 0)
            if len(grp) >= subgroups.MIN_SUBGROUP:
                flags = np.array(
                    [(g["score"] >= hi) or (g["score"] <= lo) for g in grp], dtype=np.float64
                )
                entry["outside_rate"] = float(flags.mean())
                entry["outside_ci"] = proportion_ci(flags, n_boot=n_boot, seed=seed)
                entry["median_percentile_ref"] = float(
                    np.median([g["percentile_ref"] for g in grp])
                )
                entry["rate_withheld"] = False
            else:
                entry["rate_withheld"] = True
            per_value[value] = entry
        axes[key] = per_value
    return {
        "primary_alpha": primary_alpha,
        "band": {"lo": lo, "hi": hi},
        "min_subgroup": subgroups.MIN_SUBGROUP,
        "axes": axes,
    }


def run_admit(
    baseline_dir: Path,
    candidate_path: Path | None = None,
    candidate_scores_path: Path | None = None,
    reference: str = "all",
    alphas: tuple[float, ...] = (0.05, 0.01, 0.005),
    n_boot: int = 1000,
    seed: int = 17,
    out: Path | None = None,
    root: Path = Path("."),
    detector=None,
    live_versions: dict | None = None,
    live_device: str | None = None,
) -> dict:
    """The admit pipeline. `detector`, `live_versions`, `live_device` exist so the live
    lane and its refusal are testable without a GPU (the score_document precedent)."""
    baseline_dir = Path(baseline_dir)
    if not (baseline_dir / "scores.csv").exists():
        raise NotAReport(f"{baseline_dir}: no scores.csv (is this a gaige report directory?)")
    if not (baseline_dir / "env.json").exists():
        raise ValueError(
            f"{baseline_dir}: no env.json — a baseline without an instrument fingerprint "
            "defines no standard to diverge from. Re-run the baseline with `gaige run` so "
            "its receipt records the instrument."
        )
    if (candidate_path is None) == (candidate_scores_path is None):
        raise ValueError("give exactly one of candidate_path or candidate_scores_path")
    if reference not in ("all", "human", "ai"):
        raise ValueError(f"--reference must be all, human, or ai, got {reference!r}")

    inst = load_instrument(baseline_dir)
    env = inst["env"]
    det_meta = env.get("detector", {}) or {}
    n_h, n_a = len(inst["human_scores"]), len(inst["ai_scores"])
    if reference == "all":
        ref_sorted = sorted(inst["human_scores"] + inst["ai_scores"])
    else:
        ref_sorted = inst["human_scores"] if reference == "human" else inst["ai_scores"]
        if not ref_sorted:
            raise ValueError(
                f"--reference {reference}: the baseline has no {reference}-labeled rows"
            )
    ref = np.asarray(ref_sorted, dtype=np.float64)

    outdir = Path(out) if out else Path(root) / "reports" / f"{datetime.now():%Y%m%d-%H%M%S}-admit"
    src_bit = (
        f"--candidate {candidate_path}"
        if candidate_path is not None
        else f"--candidate-scores {candidate_scores_path}"
    )
    reproduce = (
        f"gaige admit --baseline {baseline_dir} {src_bit} --reference {reference} "
        f"--alphas {','.join(f'{a:g}' for a in alphas)} --n-boot {n_boot} --seed {seed}"
    )

    if candidate_scores_path is not None:
        cand_rows, cand_sha, labels_ignored = read_candidate_scores_csv(Path(candidate_scores_path))
        cand_name = Path(candidate_scores_path).stem
        scoring = "supplied-unattested"
        mismatch_check = "not run (scores supplied; no live instrument to compare)"
        print(
            "[admit] WARNING: candidate scores arrived without an instrument attestation. "
            "If they were not produced by the baseline instrument, every number below "
            "compares two different instruments and measures nothing."
        )
    else:
        slice_ = load_candidate_jsonl(Path(candidate_path))
        mism = _mismatches(env, live_versions=live_versions, live_device=live_device)
        if mism:
            raise RuntimeError(
                "refusing the live lane — this environment is not the baseline's instrument:\n  "
                + "\n  ".join(mism)
                + "\nScore the candidate on the matching environment, or supply scores it "
                "produced via --candidate-scores."
            )
        mismatch_check = "clean (live environment matches the baseline fingerprint)"
        det = detector
        if det is None:
            if det_meta.get("detector") != "fast-detect-gpt":
                raise ValueError(
                    "live admit supports fast-detect-gpt baselines; this baseline's "
                    f"instrument is {det_meta.get('detector', 'unknown')!r}. Score the "
                    "candidate with that instrument yourself and pass --candidate-scores."
                )
            from .detectors.fast_detect_gpt import FastDetectGPT

            det = FastDetectGPT(
                model_id=det_meta["model_id"],
                quant=det_meta["quant_requested"],
                max_tokens=det_meta["max_tokens"],
            )
            det.load()
        runstate.write_runstate(outdir, slice_, det_meta, reproduce)
        fh, writer = runstate.open_partial(
            outdir, fields_override=["id", "score", "seconds", "n_words", "meta"]
        )
        cand_rows = []
        try:
            for it in slice_.items:
                t0 = time.time()
                s = float(det.score(it["text"]))
                row = {
                    "id": it["id"],
                    "score": s,
                    "seconds": round(time.time() - t0, 3),
                    "n_words": len(it["text"].split()),
                    "meta": it.get("meta"),
                }
                runstate.append_row(fh, writer, row)
                cand_rows.append(row)
        finally:
            fh.close()
        cand_sha = slice_.sha256
        cand_name = slice_.name
        scoring = "live"
        labels_ignored = slice_.labels_present
        if labels_ignored:
            print(
                "[admit] labels on candidate rows are ignored: admission measures "
                "divergence, not class membership"
            )

    n_cand = len(cand_rows)
    cand_scores = np.array([r["score"] for r in cand_rows], dtype=np.float64)
    n_words_known = all(isinstance(r.get("n_words"), int) for r in cand_rows)
    n_short = sum(
        1
        for r in cand_rows
        if isinstance(r.get("n_words"), int) and r["n_words"] < MIN_RELIABLE_WORDS
    )
    slice_stats = n_cand >= CANDIDATE_FLOOR

    novelty = ks = qshift = None
    primary_alpha = None
    if slice_stats:
        novelty = divergence.novelty_rows(ref, cand_scores, alphas, n_boot=n_boot, seed=seed)
        ks = divergence.ks_with_ci(ref, cand_scores, n_boot=n_boot, seed=seed)
        qshift = divergence.quantile_shift(ref, cand_scores, n_boot=n_boot, seed=seed)
        primary_alpha = next((r["alpha"] for r in novelty if "unavailable" not in r), None)

    for r in cand_rows:
        r["percentile_ref"] = percentile_among(ref_sorted, r["score"])
        r["conformal_p_two_sided"] = divergence.conformal_p_two_sided(ref, r["score"])
        nw = r.get("n_words")
        r["short_text"] = bool(isinstance(nw, int) and nw < MIN_RELIABLE_WORDS)

    if slice_stats:
        try:
            b_rows = read_scores_csv(baseline_dir / "scores.csv")
        except NotAReport:
            b_rows = []
        if reference != "all":
            b_rows = [r for r in b_rows if r["label"] == reference]
        strata_block = _strata(cand_rows, novelty, primary_alpha, b_rows, n_boot, seed)
    else:
        strata_block = {
            "unavailable": f"n={n_cand} candidate documents is below the "
            f"{CANDIDATE_FLOOR}-document floor for slice-level rates"
        }

    results = {
        "kind": "admit",
        "gaige_version": __version__,
        "baseline": {
            "report": baseline_dir.name,
            "corpus": {k: (env.get("corpus") or {}).get(k) for k in ("name", "sha256", "counts")},
        },
        "reference": {"mode": reference, "n": int(len(ref)), "n_human": n_h, "n_ai": n_a},
        "candidate": {
            "name": cand_name,
            "sha256": cand_sha,
            "n": n_cand,
            "n_short_text": n_short,
            "n_words_recorded": n_words_known,
            "scoring": scoring,
            "labels_ignored": labels_ignored,
        },
        "novelty": novelty,
        "ks": ks,
        "quantile_shift": qshift,
        "primary_alpha": primary_alpha,
        "strata": strata_block,
        "floors": {
            "candidate_min_for_slice_stats": CANDIDATE_FLOOR,
            "min_subgroup": subgroups.MIN_SUBGROUP,
            "min_reliable_words": MIN_RELIABLE_WORDS,
            "slice_stats_withheld": not slice_stats,
            "two_sided_reference_min": {
                f"{a:g}": divergence.two_sided_min_samples(a) for a in alphas
            },
        },
        "n_boot": n_boot,
        "seed": seed,
    }

    outdir.mkdir(parents=True, exist_ok=True)
    fields = [
        "id",
        "score",
        "seconds",
        "n_words",
        "meta",
        "percentile_ref",
        "conformal_p_two_sided",
        "short_text",
    ]
    with open(outdir / "candidate-scores.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore", restval="")
        w.writeheader()
        for r in cand_rows:
            rr = dict(r)
            if isinstance(rr.get("meta"), dict):
                rr["meta"] = json.dumps(rr["meta"], separators=(",", ":"), sort_keys=True)
            w.writerow({k: ("" if rr.get(k) is None else rr.get(k, "")) for k in fields})

    (outdir / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    env_out = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "gaige_version": __version__,
        "host": {"platform": platform.platform()},
        "detector": det_meta,
        "candidate_scoring": {"mode": scoring, "instrument_check": mismatch_check},
        "baseline": {
            "report": str(baseline_dir),
            "corpus": env.get("corpus", {}),
            "scores_sha256": _sha256(baseline_dir / "scores.csv"),
        },
        "candidate": {
            "name": cand_name,
            "path": str(candidate_path if candidate_path is not None else candidate_scores_path),
            "sha256": cand_sha,
            "n": n_cand,
            "scoring": scoring,
        },
        "reproduce": reproduce,
    }
    (outdir / "env.json").write_text(json.dumps(env_out, indent=1), encoding="utf-8")

    report_path = _write_report_md(outdir, results, det_meta)
    if scoring == "live":
        runstate.mark_complete(outdir)
    return {"outdir": outdir, "results": results, "report_path": report_path}


SHARP_EDGES = [
    "Divergence != badness. Novel-but-good material diverges. Receipts read as triage "
    "signal; a human decides. Evidence, not a verdict — unchanged.",
    "Baselines age. Re-vintaging via the existing registry vintage machinery; the "
    "baseline's own drift is a first-class measurement, not an embarrassment.",
    "Scorer-relative. Divergence exists only relative to corpus + scorer + operating "
    "point; the fingerprint carries all three (unchanged discipline).",
    "Exchangeability. The conformal guarantee assumes it; org corpora drift over time. "
    "State the bound with the monitors' honesty (marginal, per-interval, no i.i.d. "
    "pretense).",
    "Gameability. Known scorer -> material can be tuned into the band. True of every "
    "scorer; caveat carried, never solved-by-claim.",
]


def _fmt_ci(ci) -> str:
    lo, hi = ci
    return f"[{lo:.1%}–{hi:.1%}]"


def _write_report_md(outdir: Path, results: dict, det_meta: dict) -> Path:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    cand = results["candidate"]
    base = results["baseline"]
    refr = results["reference"]
    lines = [
        f"# gaige admit receipt — {cand['name']} vs {base['corpus'].get('name')} × "
        f"{det_meta.get('detector', 'unknown')}",
        "",
        f"generated: {ts} · gaige {results['gaige_version']}",
    ]
    if cand["scoring"] == "supplied-unattested":
        lines += [
            "",
            "**UNATTESTED CANDIDATE SCORES.** These candidate scores arrived without an "
            "instrument attestation. If they were not produced by the baseline instrument "
            "below, every number in this receipt compares two different instruments and "
            "measures nothing.",
        ]
    lines += [
        "",
        "## Instrument fingerprint (the baseline's — the standard being diverged from)",
        *_fingerprint_lines(det_meta),
        f"- candidate scoring: **{cand['scoring']}**",
        "",
        "## Baseline",
        f"- report: `{base['report']}` · corpus {base['corpus'].get('name')} — sha256 "
        f"`{str(base['corpus'].get('sha256'))[:16]}…` · counts {base['corpus'].get('counts')}",
        f"- reference = **{refr['mode']}** (n={refr['n']}: {refr['n_human']} human + "
        f"{refr['n_ai']} ai rows in the baseline). "
        + (
            "The whole trusted corpus defines 'accepted', regardless of label — the "
            "trusted-vs-new reading."
            if refr["mode"] == "all"
            else f"Only the baseline's {refr['mode']}-labeled rows define 'accepted'."
        ),
        "",
        "## Candidate",
        f"- {cand['name']} — sha256 `{str(cand['sha256'])[:16]}…` · {cand['n']} documents"
        + (
            f" · {cand['n_short_text']} under {MIN_RELIABLE_WORDS} words"
            if cand["n_short_text"]
            else ""
        ),
    ]
    if cand["labels_ignored"]:
        lines.append(
            "- labels were present on candidate rows and were ignored: admission measures "
            "divergence, not class membership"
        )

    lines += ["", "## Divergence"]
    if results["floors"]["slice_stats_withheld"]:
        lines.append(
            f"- slice-level statistics withheld: n={cand['n']} candidate documents is below "
            f"the {results['floors']['candidate_min_for_slice_stats']}-document floor. "
            "Per-document placements are in candidate-scores.csv."
        )
    else:
        lines += [
            "",
            "### Conformal novelty (the guarantee-bearing number)",
            "| alpha | band | outside | rate [95% CI] | expected if exchangeable |",
            "|---|---|---|---|---|",
        ]
        refusals = []
        guarantee = None
        for r in results["novelty"]:
            if "unavailable" in r:
                refusals.append(f"- α={r['alpha']:g}: refused — {r['unavailable']}")
                continue
            guarantee = guarantee or r["guarantee"]
            lines.append(
                f"| {r['alpha']:g} | {r['lo']:.4f} – {r['hi']:.4f} | "
                f"{r['n_outside']}/{r['n_candidate']} | {r['outside_rate']:.1%} "
                f"{_fmt_ci(r['outside_ci'])} | <= {r['expected_outside_if_exchangeable']:g} |"
            )
        lines += refusals
        if guarantee:
            lines.append(f"- guarantee: {guarantee}")
        ksr = results["ks"]
        lines += [
            "",
            f"### Shape: Kolmogorov–Smirnov distance **{ksr['stat']:.4f}** "
            f"(95% bootstrap {ksr['ci'][0]:.4f}–{ksr['ci'][1]:.4f}, both sides resampled). "
            "A distance, not a test: no p-value by design.",
            "",
            "### Where the distribution moved (quantile shift, candidate − reference)",
            "| quantile | reference | candidate | delta [95% CI] |",
            "|---|---|---|---|",
        ]
        for r in results["quantile_shift"]:
            lines.append(
                f"| p{int(r['q'] * 100)} | {r['reference_q']:.4f} | {r['candidate_q']:.4f} | "
                f"{r['delta']:+.4f} [{r['delta_ci'][0]:+.4f} – {r['delta_ci'][1]:+.4f}] |"
            )

    lines += ["", "## Where it differs (candidate strata vs the global band)"]
    strata = results["strata"]
    if "unavailable" in strata:
        lines.append(f"- {strata['unavailable']}")
    else:
        lines.append(
            f"At α={strata['primary_alpha']:g} (band {strata['band']['lo']:.4f} – "
            f"{strata['band']['hi']:.4f}); rates below n={strata['min_subgroup']} are "
            "withheld and the count speaks."
        )
        for axis, groups in strata["axes"].items():
            lines += [
                "",
                f"| {axis} | n | n_baseline | outside rate [95% CI] | median percentile among reference |",
                "|---|---|---|---|---|",
            ]
            for value, d in groups.items():
                nb = d.get("n_baseline")
                nb_s = "-" if nb is None else str(nb)
                if d["rate_withheld"]:
                    lines.append(f"| {value} | {d['n']} | {nb_s} | withheld (n < floor) | — |")
                else:
                    lines.append(
                        f"| {value} | {d['n']} | {nb_s} | {d['outside_rate']:.1%} "
                        f"{_fmt_ci(d['outside_ci'])} | {d['median_percentile_ref']:.0%} |"
                    )

    floors = results["floors"]
    mins = " · ".join(f"α={a} needs {n}" for a, n in floors["two_sided_reference_min"].items())
    lines += [
        "",
        "## Floors and refusals",
        f"- two-sided reference floors (ceil(2/alpha)-1): {mins} reference scores",
        f"- slice-level statistics need >= {floors['candidate_min_for_slice_stats']} candidate "
        "documents; strata rates need >= "
        f"{floors['min_subgroup']} per stratum; documents under "
        f"{floors['min_reliable_words']} words carry a short_text caveat",
        "",
        "## Honest caveats (read before acting on this receipt)",
        "- this receipt never says admit or reject. it measures divergence; the decision is yours.",
    ]
    for i, edge in enumerate(SHARP_EDGES, 1):
        lines.append(f"- {i}. {edge}")
    if cand["scoring"] == "supplied-unattested":
        lines.append(
            "- the candidate scores are unattested (see the block at the top); the receipt "
            "is only as meaningful as their provenance."
        )
    env_doc = json.loads((outdir / "env.json").read_text(encoding="utf-8"))
    lines += ["", "## Reproduce", f"```\n{env_doc['reproduce']}\n```"]
    (outdir / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return outdir / "report.md"
