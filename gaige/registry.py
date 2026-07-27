# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""The run registry: comparable series over time, with instrument constancy ENFORCED.

The longitudinal spec's hardest validity question is "is the change in the model, or in the measuring
pipeline?" The registry is the mechanical answer: runs land in a series keyed by the hash
of the instrument identity (provider identity + decoding + grading + cutoff + gaige
version), and a run whose identity differs is REFUSED with the mismatched fields named —
never silently appended. Slope and drift statistics are only ever computed within a series.

Two constancy rules, distinct on purpose:
- INSTRUMENT identity keys the series. Change the model, the decoding, the grading rule —
  that is a different instrument; start (or continue) that instrument's own series.
- MATERIAL constancy is enforced within a series: a vintage label, once measured, must hash
  to the same content forever. New vintage labels may be added at any interval (that is the
  longitudinal design); editing an existing one would change the measurand mid-study.

Replicates (same-day repeat runs) establish the run-variance bound: the measured dispersion
of accuracy under an identical instrument on identical material. With a greedy, in-process
pipeline that dispersion is typically zero — which is then a measured fact on the receipt,
not an assumption — and any later movement beyond the bound is signal, not noise.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

# Provider-metadata keys that describe HOW something was attested rather than WHAT the
# instrument is. Everything else in the provider dict counts toward identity — strict
# beats sorry.
_NON_IDENTITY_PROVIDER_KEYS = {"attestation_basis"}


class SeriesMismatch(RuntimeError):
    """The run's instrument or material does not belong to this series; fields are named."""


def series_identity(instrument: dict, gaige_version: str) -> dict:
    """The identity-defining subset of a probe-run instrument.

    Deliberately excludes probes_sha256: the probe FILE grows as vintages are authored, and
    material constancy is enforced per vintage instead (see vintage hashes). Includes the
    harness version per the longitudinal spec — a gaige release is an instrument change until shown
    otherwise.
    """
    provider = {
        k: v for k, v in instrument["provider"].items() if k not in _NON_IDENTITY_PROVIDER_KEYS
    }
    return {
        "provider": provider,
        "decoding": instrument["decoding"],
        "grading_version": instrument["grading_version"],
        "training_cutoff": instrument["training_cutoff"],
        "gaige_version": gaige_version,
    }


def series_id(identity: dict) -> str:
    canonical = json.dumps(identity, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()[:12]


def _diff_fields(stored: dict, new: dict, prefix: str = "") -> list[str]:
    out: list[str] = []
    for k in sorted(set(stored) | set(new)):
        a, b = stored.get(k), new.get(k)
        path = f"{prefix}{k}"
        if isinstance(a, dict) and isinstance(b, dict):
            out += _diff_fields(a, b, prefix=f"{path}.")
        elif a != b:
            out.append(f"{path}: series has {a!r}, this run has {b!r}")
    return out


def _series_path(registry_dir: Path, sid: str) -> Path:
    return Path(registry_dir) / sid


def _load_series(series_dir: Path) -> dict:
    return json.loads((series_dir / "series.json").read_text(encoding="utf-8"))


def _save_series(series_dir: Path, series: dict) -> None:
    series_dir.mkdir(parents=True, exist_ok=True)
    (series_dir / "series.json").write_text(json.dumps(series, indent=1), encoding="utf-8")


def record_run(registry_dir: Path, run_dir: Path, replicate: bool = False) -> dict:
    """Register a completed probe run into its instrument's series.

    Returns the updated series dict. Refuses (SeriesMismatch) if the run's identity collides
    with a stored series id but differs in content — impossible in practice unless files
    were edited — or if an existing vintage label arrives with different content.
    """
    run_dir = Path(run_dir)
    results = json.loads((run_dir / "probe-results.json").read_text(encoding="utf-8"))
    identity = series_identity(results["instrument"], results["gaige_version"])
    sid = series_id(identity)
    series_dir = _series_path(registry_dir, sid)

    if (series_dir / "series.json").exists():
        series = _load_series(series_dir)
        problems = _diff_fields(series["identity"], identity)
        if problems:
            raise SeriesMismatch(
                "series id collision with differing identity (were files edited?):\n  "
                + "\n  ".join(problems)
            )
        for v, h in results["vintage_hashes"].items():
            stored = series["vintage_hashes"].get(v)
            if stored is not None and stored != h:
                raise SeriesMismatch(
                    f"vintage {v!r} content changed: series has {stored[:16]}..., this run "
                    f"has {h[:16]}.... A measured vintage is frozen; editing it changes the "
                    "measurand mid-study. Author a NEW vintage label instead."
                )
    else:
        series = {
            "series_id": sid,
            "identity": identity,
            "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "vintage_hashes": {},
            "runs": [],
        }

    # M3 constancy: toggling P(True) on/off does not fork (the M1 instrument is unchanged —
    # option_logprobs are separate forward passes), but once a series has measured M3 under
    # one template, later M3 runs must use THAT template or gaps/ECEs are not comparable.
    run_ptrue = results["instrument"].get("ptrue")
    if run_ptrue is not None:
        stored_pt = series.get("ptrue")
        if stored_pt is not None and stored_pt != run_ptrue:
            raise SeriesMismatch(
                f"ptrue template changed: series measured M3 under {stored_pt}, this run "
                f"used {run_ptrue}. Confidence numbers across templates are not comparable; "
                "a changed template needs its own series."
            )
        series["ptrue"] = run_ptrue

    series["vintage_hashes"].update(results["vintage_hashes"])
    series["runs"].append(
        {
            "run_dir": str(run_dir),
            "generated_utc": results.get("generated_utc", ""),
            "replicate": replicate,
            "by_vintage": results["by_vintage"],
            "post_cutoff_share": results["post_cutoff_share"],
        }
    )
    _save_series(series_dir, series)
    write_series_report(series_dir, series)
    return series


def variance_bound(series: dict) -> dict | None:
    """Per-vintage run-variance bound from the replicate runs, or None if none exist.

    bound = 2 * population std of accuracy across replicate runs, per vintage — the
    pre-registered movement rule's denominator. Zero is a legitimate measured value for a
    deterministic pipeline, and the report says it was measured.
    """
    reps = [r for r in series["runs"] if r["replicate"]]
    if len(reps) < 2:
        return None
    out: dict = {}
    vintages = sorted({v for r in reps for v in r["by_vintage"]})
    for v in vintages:
        accs = [r["by_vintage"][v]["accuracy"] for r in reps if v in r["by_vintage"]]
        if len(accs) < 2:
            continue
        mean = sum(accs) / len(accs)
        var = sum((a - mean) ** 2 for a in accs) / len(accs)
        out[v] = {"n_replicates": len(accs), "mean": mean, "bound": 2.0 * var**0.5}
    return out or None


def write_series_report(series_dir: Path, series: dict) -> Path:
    ident = series["identity"]
    prov = ident["provider"]
    bound = variance_bound(series)
    vintages = sorted({v for r in series["runs"] for v in r["by_vintage"]})

    lines = [
        f"# gaige series — {series['series_id']}",
        "",
        f"created: {series['created_utc']} · runs: {len(series['runs'])}",
        "",
        "## Instrument (constant across every run below, enforced)",
        f"- provider `{prov.get('provider', '?')}` · model "
        f"`{prov.get('model_id', prov.get('model_requested', '?'))}` · attestation "
        f"**{prov.get('attestation', '?')}**",
        f"- decoding {json.dumps(ident['decoding'])} · grading `{ident['grading_version']}` · "
        f"cutoff {ident['training_cutoff']} · gaige {ident['gaige_version']}",
        "- Instrument constancy is asserted mechanically: a run whose fingerprint differs "
        "from the above is refused by the registry, not compared.",
        "",
        "## Accuracy by vintage, per run",
        "",
        "| run (UTC) | replicate | " + " | ".join(vintages) + " |",
        "|---|---|" + "---|" * len(vintages),
    ]
    for r in series["runs"]:
        cells = []
        for v in vintages:
            d = r["by_vintage"].get(v)
            if d:
                cell = f"{d['accuracy']:.1%} (n={d['n']})"
                if "m3" in d:
                    cell += f" · gap {d['m3']['gap']:+.1%}"
                cells.append(cell)
            else:
                cells.append("—")
        lines.append(
            f"| {r['generated_utc'] or '?'} | {'yes' if r['replicate'] else ''} | "
            + " | ".join(cells)
            + " |"
        )

    lines += ["", "## Run-variance bound"]
    if bound is None:
        lines.append(
            "- Not established: fewer than two replicate runs. Movement claims are premature "
            "until the Day-0 replicate protocol has run (`gaige probe run --replicates k`)."
        )
    else:
        for v, b in sorted(bound.items()):
            lines.append(
                f"- {v}: mean {b['mean']:.1%} over {b['n_replicates']} replicates, "
                f"bound ±{b['bound']:.1%} (2σ, measured — zero means the pipeline is "
                "deterministic, which is a result, not an assumption)"
            )
        first_non_rep = [r for r in series["runs"] if not r["replicate"]]
        if first_non_rep:
            lines += ["", "### Movement vs bound (non-replicate runs against replicate mean)"]
            for r in first_non_rep:
                for v in vintages:
                    d, b = r["by_vintage"].get(v), (bound or {}).get(v)
                    if not d or not b:
                        continue
                    delta = d["accuracy"] - b["mean"]
                    verdict = (
                        "BEYOND the bound — signal"
                        if abs(delta) > b["bound"]
                        else "within run variance"
                    )
                    lines.append(
                        f"- {r['generated_utc']} · {v}: Δ {delta:+.1%} vs replicate mean → {verdict}"
                    )

    lines += [
        "",
        "## Honest caveats",
        "- Each row is the SAME instrument on frozen-per-vintage material; that is the only "
        "reason rows are comparable. Cross-series comparison is not offered.",
        "- Accuracy is defined by the versioned grading rule in the identity block.",
        "- Slope claims need more intervals than most series here have; a two-run series "
        "shows deltas, not trends.",
    ]
    (series_dir / "series-report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return series_dir / "series-report.md"


def vintage_sequences(series: dict, vintage: str, quantity: str = "accuracy") -> dict:
    """Interval-value sequences for one vintage: zero-drift reference vs observed runs.

    reference = replicate runs' values (the Day-0 zero-drift sample); observed = the
    non-replicate runs in recorded order, with their timestamps as labels. Runs that lack
    the vintage or the quantity (e.g. gap on a run without M3) are skipped, labels kept
    aligned. quantity: "accuracy" or "gap" (M3 confidence-accuracy gap).
    """

    def q(run: dict):
        d = run["by_vintage"].get(vintage)
        if d is None:
            return None
        if quantity == "accuracy":
            return d["accuracy"]
        if quantity == "gap":
            return d.get("m3", {}).get("gap")
        raise ValueError(f"unknown quantity {quantity!r}: expected accuracy or gap")

    reference = [v for r in series["runs"] if r["replicate"] and (v := q(r)) is not None]
    observed, labels = [], []
    for r in series["runs"]:
        if r["replicate"]:
            continue
        v = q(r)
        if v is None:
            continue
        observed.append(v)
        labels.append(r.get("generated_utc", "?"))
    return {"reference": reference, "observed": observed, "labels": labels}


def list_series(registry_dir: Path) -> list[dict]:
    registry_dir = Path(registry_dir)
    out = []
    if not registry_dir.exists():
        return out
    for d in sorted(registry_dir.iterdir()):
        if (d / "series.json").exists():
            s = _load_series(d)
            prov = s["identity"]["provider"]
            out.append(
                {
                    "series_id": s["series_id"],
                    "model": prov.get("model_id", prov.get("model_requested", "?")),
                    "provider": prov.get("provider", "?"),
                    "runs": len(s["runs"]),
                    "vintages": sorted(s["vintage_hashes"]),
                    "dir": str(d),
                }
            )
    return out
