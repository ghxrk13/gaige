# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Probe authoring toolchain: template generator + schema lint.

Probe authoring is constrained by signed decisions (the longitudinal instruments spec, 2026-07-22). The
four that bind the probe-set artifact are enforced here mechanically, not editorially:

  a. grading is normalized exact match plus authored aliases — the set DECLARES the grading
     rule and version it was authored against, and answers must be short and checkable;
  b. the control is a frozen MMLU-style subset scored by per-option logprob argmax — the set
     LINKS to its control (name + sha256 + scoring shape), so a study set cannot exist
     without its flat reference;
  d. greedy (temperature-0) decoding is pre-registered — the set declares it, and the probe
     runner refuses a run whose decoding contradicts the declaration;
  e. every probe carries contamination provenance (source, source_date, authored), and every
     source_date must post-date the declared training cutoff, so vintages are demonstrably
     post-cutoff by construction.

(The fifth signed decision, the M2 rescope, governs a different instrument; no tooling here.)

Set-level declarations live in a sidecar manifest `<name>.manifest.json` beside the JSONL so
the probe file itself stays row-pure: editing a declaration never moves the probe-file sha256
or the frozen per-vintage content hashes. `gaige probe new` scaffolds both files with the
fixed decisions pre-filled and EDIT-ME placeholders wherever a human must author; the
placeholders fail lint by design, so an unedited template cannot pass as a study set.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path

from .grading import GRADING_VERSION, normalize
from .probes import BadProbeSet, load_probes

MANIFEST_VERSION = 1
PLACEHOLDER = "EDIT-ME"
GRADING_RULE = "normalized-exact-match"
CONTROL_SCORING = "option-logprob-argmax"
# Advisory bar for decision a's "short checkable answers": longer keys grade fine under nem
# but are fragile to phrasing, so the lint warns rather than refuses.
SHORT_ANSWER_WORDS = 5

_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class ManifestViolation(ValueError):
    """A probe run contradicts the set's signed authoring declarations."""


def manifest_path(probes_path: Path) -> Path:
    return probes_path.with_suffix(".manifest.json")


def load_manifest(probes_path: Path) -> dict | None:
    p = manifest_path(probes_path)
    if not p.exists():
        return None
    try:
        m = json.loads(p.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise ManifestViolation(f"{p}: not valid JSON ({e.msg})") from None
    if not isinstance(m, dict):
        raise ManifestViolation(f"{p}: manifest must be a JSON object")
    return m


def new_probe_set(
    out: Path, vintage: str = "t0", cutoff: str = "", n_examples: int = 2
) -> tuple[Path, Path]:
    """Scaffold a probe-set JSONL + manifest. Refuses to overwrite; placeholders fail lint."""
    try:
        cutoff_date = date.fromisoformat(cutoff)
    except (TypeError, ValueError):
        raise ValueError(
            f"cutoff {cutoff!r} is not an ISO date (YYYY-MM-DD); the manifest pins the "
            "training cutoff the set is authored against (decision e)."
        ) from None
    mpath = manifest_path(out)
    for p in (out, mpath):
        if p.exists():
            raise FileExistsError(f"{p} exists; refusing to overwrite a probe set")
    out.parent.mkdir(parents=True, exist_ok=True)

    first_valid = (cutoff_date + timedelta(days=1)).isoformat()
    rows = []
    for i in range(1, n_examples + 1):
        rows.append(
            {
                "id": f"{PLACEHOLDER}-{i:03d}",
                "prompt": (
                    "Answer with only the answer, nothing else.\n"
                    f"Question: {PLACEHOLDER} (one question with a short checkable answer)\n"
                    "Answer:"
                ),
                "answer": PLACEHOLDER,
                "aliases": [],
                "vintage": vintage,
                "source": f"{PLACEHOLDER}: named world-coupled source (document, feed, standard)",
                "source_date": first_valid,
                "authored": first_valid,
            }
        )
    with open(out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    manifest = {
        "manifest_version": MANIFEST_VERSION,
        "probe_file": out.name,
        "training_cutoff": cutoff,
        "grading": {"rule": GRADING_RULE, "version": GRADING_VERSION},
        "decoding": {"policy": "greedy", "temperature": 0.0, "top_p": 1.0, "top_k": 0},
        "control": {
            "benchmark": f"{PLACEHOLDER}: name the frozen MMLU-subset control (decision b)",
            "sha256": None,
            "path": None,
            "scoring": CONTROL_SCORING,
        },
    }
    mpath.write_text(json.dumps(manifest, indent=1) + "\n", encoding="utf-8")
    return out, mpath


@dataclass
class LintReport:
    path: Path
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    vintages: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not self.errors


def _placeholder_scan(obj, where: str, errors: list[str]) -> None:
    if isinstance(obj, str):
        if PLACEHOLDER in obj:
            errors.append(f"{where}: unedited template placeholder ({PLACEHOLDER!r})")
    elif isinstance(obj, dict):
        for k, v in obj.items():
            _placeholder_scan(v, f"{where}.{k}", errors)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _placeholder_scan(v, f"{where}[{i}]", errors)


def _lint_manifest(m: dict, probes_path: Path, errors: list[str]) -> date | None:
    """Manifest declarations vs the signed decisions. Returns the parsed cutoff if valid."""
    mp = manifest_path(probes_path)
    if m.get("manifest_version") != MANIFEST_VERSION:
        errors.append(
            f"{mp}: manifest_version {m.get('manifest_version')!r} unknown "
            f"(this gaige knows {MANIFEST_VERSION})"
        )
    if m.get("probe_file") != probes_path.name:
        errors.append(
            f"{mp}: probe_file {m.get('probe_file')!r} does not name {probes_path.name!r}; "
            "a manifest binds exactly one probe file"
        )

    g = m.get("grading") or {}
    if g.get("rule") != GRADING_RULE:
        errors.append(
            f"{mp}: grading.rule {g.get('rule')!r} — decision a fixes grading to "
            f"{GRADING_RULE!r}; a different rule is a different instrument"
        )
    if g.get("version") != GRADING_VERSION:
        errors.append(
            f"{mp}: grading.version {g.get('version')!r} != current {GRADING_VERSION!r}; "
            "re-review answers and aliases under the current normalization, then update "
            "the declaration"
        )

    d = m.get("decoding") or {}
    if d.get("policy") != "greedy" or d.get("temperature") != 0.0:
        errors.append(
            f"{mp}: decoding {json.dumps(d)} — decision d pre-registers greedy "
            "(policy 'greedy', temperature 0.0) for probe answers"
        )
    if d.get("top_p", 1.0) != 1.0 or d.get("top_k", 0) != 0:
        errors.append(f"{mp}: decoding declares sampling parameters; greedy is top_p=1, top_k=0")

    c = m.get("control") or {}
    if c.get("scoring") != CONTROL_SCORING:
        errors.append(
            f"{mp}: control.scoring {c.get('scoring')!r} — decision b fixes the control to "
            f"{CONTROL_SCORING!r} (pure forward pass, no decoding parameters)"
        )
    if not isinstance(c.get("benchmark"), str) or not c.get("benchmark").strip():
        errors.append(
            f"{mp}: control.benchmark missing — decision b links every study set to a "
            "frozen MMLU-subset control by name"
        )
    sha = c.get("sha256")
    if not (isinstance(sha, str) and _HEX64.match(sha)):
        errors.append(
            f"{mp}: control.sha256 {sha!r} is not a sha256 hex digest; freeze the control "
            "subset and record its hash (decision b: frozen and hashed)"
        )
    elif c.get("path"):
        cpath = (probes_path.parent / c["path"]).resolve()
        if cpath.exists():
            h = hashlib.sha256(cpath.read_bytes()).hexdigest()
            if h != sha:
                errors.append(
                    f"{mp}: control file {c['path']} hashes {h[:16]}…, manifest declares "
                    f"{sha[:16]}… — the control moved or the declaration is stale"
                )

    cutoff = m.get("training_cutoff")
    try:
        return date.fromisoformat(cutoff)
    except (TypeError, ValueError):
        errors.append(f"{mp}: training_cutoff {cutoff!r} is not an ISO date (YYYY-MM-DD)")
        return None


def lint(probes_path: Path) -> LintReport:
    """Lint one probe set + manifest against the signed authoring decisions.

    Errors violate a signed decision or make a probe ungradeable; warnings are authoring
    advice. A study set must lint with zero errors — the probe runner enforces that.
    """
    rep = LintReport(path=probes_path)
    cutoff: date | None = None

    try:
        m = load_manifest(probes_path)
    except ManifestViolation as e:
        rep.errors.append(str(e))
        m = None
    if m is None:
        rep.errors.append(
            f"{manifest_path(probes_path)}: missing — the manifest carries the set's signed "
            "declarations (grading, decoding, control linkage, cutoff); "
            "`gaige probe new` writes one"
        )
    else:
        cutoff = _lint_manifest(m, probes_path, rep.errors)
        _placeholder_scan(m, str(manifest_path(probes_path)), rep.errors)

    try:
        ps = load_probes(probes_path)
    except (BadProbeSet, OSError) as e:
        rep.errors.append(str(e))
        return rep

    seen_prompts: dict[str, str] = {}
    for p in ps.probes:
        pid = p["id"]
        row = {k: v for k, v in p.items() if not k.startswith("_")}
        _placeholder_scan(row, f"probe {pid!r}", rep.errors)

        if not p.get("authored"):
            rep.errors.append(
                f"probe {pid!r}: missing 'authored' — decision e requires per-probe "
                "provenance (source, source_date, authored)"
            )
        else:
            authored = date.fromisoformat(p["authored"])  # format guaranteed by the loader
            if authored < p["_source_date"]:
                rep.errors.append(
                    f"probe {pid!r}: authored {p['authored']} predates source_date "
                    f"{p['source_date']} — a probe cannot be authored before its source"
                )
        if cutoff is not None and p["_source_date"] <= cutoff:
            rep.errors.append(
                f"probe {pid!r}: source_date {p['source_date']} does not post-date the "
                f"training cutoff {cutoff.isoformat()} — decision e makes vintages "
                "post-cutoff by construction; a pre-cutoff probe can measure memorization"
            )

        keys = [p["answer"], *p.get("aliases", [])]
        seen_norm: set[str] = set()
        for k in keys:
            nk = normalize(k)
            if not nk:
                rep.errors.append(
                    f"probe {pid!r}: {k!r} normalizes to nothing under {GRADING_VERSION}; "
                    "ungradeable"
                )
            elif nk in seen_norm:
                rep.warnings.append(
                    f"probe {pid!r}: alias {k!r} duplicates another key under normalization"
                )
            seen_norm.add(nk)
        if len(normalize(p["answer"]).split()) > SHORT_ANSWER_WORDS:
            rep.warnings.append(
                f"probe {pid!r}: answer is {len(normalize(p['answer']).split())} words — "
                "decision a wants short checkable answers; exact match on long keys is "
                "fragile to phrasing"
            )

        if p["prompt"] in seen_prompts:
            rep.warnings.append(
                f"probe {pid!r}: prompt duplicates probe {seen_prompts[p['prompt']]!r}"
            )
        else:
            seen_prompts[p["prompt"]] = pid

    rep.vintages = {v: {"n": n} for v, n in ps.vintages.items()}
    if cutoff is not None:
        for v, d in ps.post_cutoff_share(cutoff.isoformat()).items():
            rep.vintages[v]["post_cutoff"] = d["post_cutoff"]
    return rep


def check_run_against_manifest(probes_path: Path, decoding) -> str:
    """Gate a probe run on the set's own declarations (mechanical, not editorial).

    No manifest → legacy/demo set, nothing declared, nothing enforced (says so).
    Manifest present → the set must lint clean, and the run's decoding must honor the
    pre-registered greedy block (decision d). Raises ManifestViolation otherwise.
    """
    if load_manifest(probes_path) is None:
        return (
            "no manifest beside the probe set; authoring declarations unenforced (gaige probe lint)"
        )
    rep = lint(probes_path)
    if not rep.ok:
        shown = "\n  ".join(rep.errors[:8])
        more = f"\n  … and {len(rep.errors) - 8} more" if len(rep.errors) > 8 else ""
        raise ManifestViolation(
            f"probe set fails its own manifest lint; refusing the run:\n  {shown}{more}\n"
            f"(gaige probe lint --probes {probes_path})"
        )
    if decoding.temperature != 0.0:
        raise ManifestViolation(
            f"manifest pre-registers greedy decoding (temperature 0); this run asked "
            f"temperature={decoding.temperature:g}. Decision d makes the decoding block "
            "identity-defining — if a non-greedy arm is intended, change the manifest "
            "declaration first so the fork is visible in history."
        )
    return "manifest honored: nem grading declared, greedy decoding, control linked"
