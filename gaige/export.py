# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Export a receipt as public site data.

A report directory is private working state; an export is a public artifact. The exporter
joins results.json (the statistics) with env.json (the instrument) into one self-contained
JSON document that carries, for every number, the fingerprint that produced it and a
reproduce command a stranger can run. Nothing is recomputed here: every statistic is copied
verbatim from results.json, so the analyze replay gate covers exports transitively.

Schema versioning ("gaige-receipt-export/1"): consumers must ignore unknown keys and
tolerate absent optional sections (conformal, subgroups, base_rate, and the eer pair are
present only when the source receipt carries them). Additions are allowed within /1; any
rename, removal, or change of meaning bumps the version string. The index
("gaige-export-index/1") is rebuilt by rescanning the export directory on every run, so it
is deterministic and self-healing.

Redaction is structural and fail-closed. The document is built by allowlist projection
(only the sections named below are copied), then every string value is scanned for shapes
that do not belong in public data: absolute paths, home-directory references, IP addresses,
and URLs whose host is not on the public allowlist. A hit refuses the whole export and
names the field. No personal term list ships in this code; named-term sweeps are an
ops-side release gate.

Exports are deterministic: same inputs, same bytes, on every platform (LF newlines, no
export-time clock). A report without env.json is INSTRUMENT UNKNOWN and is not exportable,
because a public number without its instrument is exactly what gaige exists to prevent.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import __version__
from .analyze import NotAReport

SCHEMA = "gaige-receipt-export/1"
INDEX_SCHEMA = "gaige-export-index/1"

# The full possible top-level key set, in emission order. Optional sections are emitted
# only when the source receipt carries them; everything else is required.
TOP_KEYS = (
    "schema",
    "exported_by",
    "receipt",
    "instrument",
    "corpus",
    "metrics",
    "thresholds",
    "conformal",
    "subgroups",
    "base_rate",
    "reproduce",
)
OPTIONAL_KEYS = frozenset({"conformal", "subgroups", "base_rate"})

URL_ALLOWED_HOSTS = frozenset({"huggingface.co", "arxiv.org"})

_URL = re.compile(r"https?://([^/\s\"'`]+)")
_IP = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
# Absolute filesystem shapes: POSIX roots that mean a real machine, Windows drives and
# UNC paths, and home-relative references. Bare relative paths (corpora/x.jsonl, model
# ids like org/name) are legitimate receipt content and pass.
_ABS_PATH = re.compile(
    r"/(?:home|root|users|tmp|var|opt|mnt|srv|etc)/"
    r"|(?<![A-Za-z0-9])[A-Za-z]:[\\/]"
    r"|\\\\"
    r"|~[/\\]",
    re.IGNORECASE,
)

_ARCH_TOKENS = ("x86_64", "amd64", "aarch64", "arm64")


def _host_public(host: dict) -> dict:
    """Project the host block down to what identifies the instrument class.

    The raw platform string (kernel build, libc version) fingerprints a specific machine
    while adding nothing to instrument identity; os family, architecture, and device are
    what a reader needs to interpret the numbers.
    """
    platform_str = str(host.get("platform", ""))
    os_family = platform_str.split("-", 1)[0] if platform_str else "unknown"
    arch = next((t for t in _ARCH_TOKENS if t.lower() in platform_str.lower()), "unknown")
    return {"os": os_family or "unknown", "arch": arch, "device": host.get("device", "unknown")}


def _scan_strings(node, path: str) -> None:
    """Refuse the export if any string value carries a shape private to a machine."""
    if isinstance(node, dict):
        for k, v in node.items():
            _scan_strings(v, f"{path}.{k}" if path else str(k))
        return
    if isinstance(node, list):
        for i, v in enumerate(node):
            _scan_strings(v, f"{path}[{i}]")
        return
    if not isinstance(node, str):
        return
    remainder = node
    for m in _URL.finditer(node):
        host = m.group(1).split("@")[-1].split(":")[0].lower()
        if host not in URL_ALLOWED_HOSTS:
            raise NotAReport(
                f"refusing to export: field {path} carries a URL to a non-public host "
                f"({host!r}); exports may reference only {sorted(URL_ALLOWED_HOSTS)}"
            )
        remainder = remainder.replace(m.group(0), " ")
    for name, rx in (("an absolute path", _ABS_PATH), ("an IP address", _IP)):
        m = rx.search(remainder)
        if m:
            raise NotAReport(
                f"refusing to export: field {path} carries {name} ({m.group(0)!r}); "
                "public receipts must not reference a specific machine"
            )


def build_export(report_dir: Path) -> dict:
    """Assemble the export document for one report directory. Raises, never redacts silently."""
    report_dir = Path(report_dir).resolve()
    results_path = report_dir / "results.json"
    env_path = report_dir / "env.json"
    if not results_path.exists():
        raise NotAReport(f"{report_dir}: no results.json (is this a gaige report directory?)")
    if not env_path.exists():
        raise NotAReport(
            f"{report_dir}: no env.json, so these numbers have no instrument fingerprint. "
            "INSTRUMENT UNKNOWN receipts are not exportable: a public number without its "
            "instrument is what receipts exist to prevent."
        )
    results = json.loads(results_path.read_text(encoding="utf-8"))
    env = json.loads(env_path.read_text(encoding="utf-8"))
    if results.get("kind") == "admit":
        raise NotAReport(
            f"{report_dir}: admission receipts are not exportable yet; the "
            "gaige-admit-export/1 document type is planned for a future release"
        )
    if "auroc" not in results:
        raise NotAReport(
            f"{report_dir}: results.json carries no calibration statistics; only "
            "calibration receipts are exportable in this schema"
        )

    metrics = {"auroc": results["auroc"], "auroc_ci": results["auroc_ci"]}
    if "eer" in results:
        metrics["eer"] = results["eer"]
        metrics["eer_threshold"] = results["eer_threshold"]
    metrics["n_boot"] = results.get("n_boot")

    corpus = env.get("corpus", {})
    reproduce: dict = {"run": env.get("reproduce", "")}
    meta = corpus.get("meta", {}) or {}
    source = str(meta.get("source", ""))
    if "url" not in meta and source.endswith(".jsonl"):
        note = (
            "This receipt was produced from a locally prepared corpus file. The corpus "
            "sha256 above pins the exact bytes; the file itself is not redistributed."
        )
        if str(corpus.get("name", "")).startswith("raid-"):
            note += (
                " RAID slices (Dugan et al., ACL 2024) are prepared from the public dataset "
                "with `gaige corpus prepare-raid` and are never redistributed."
            )
        reproduce["corpus_note"] = note

    doc = {
        "schema": SCHEMA,
        "exported_by": f"gaige {__version__}",
        "receipt": {
            "id": report_dir.name,
            "generated_utc": env.get("generated_utc", "unknown"),
            "gaige_version": results.get("gaige_version", env.get("gaige_version", "unknown")),
        },
        "instrument": {
            "host": _host_public(env.get("host", {})),
            "detector": env.get("detector", {}),
        },
        "corpus": corpus,
        "metrics": metrics,
        "thresholds": results.get("thresholds", []),
    }
    for key in ("conformal", "subgroups", "base_rate"):
        if key in results:
            doc[key] = results[key]
    doc["reproduce"] = reproduce

    _scan_strings(doc, "")
    return doc


def _dumps(doc: dict) -> str:
    return json.dumps(doc, indent=1)


def write_export(report_dir: Path, out_root: Path, force: bool = False) -> tuple[Path, bool]:
    """Write <out>/receipts/<id>.json. Returns (path, wrote).

    Re-running over an unchanged receipt is a no-op. Different bytes without --force is an
    error, because a changed export means the receipt or the exporter changed and that
    should be a deliberate act, not a side effect.
    """
    doc = build_export(report_dir)
    out_dir = Path(out_root) / "receipts"
    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{doc['receipt']['id']}.json"
    data = _dumps(doc)
    if target.exists():
        if target.read_text(encoding="utf-8") == data:
            return target, False
        if not force:
            raise NotAReport(
                f"{target}: an export for this receipt exists with different bytes, so the "
                "receipt or the exporter changed. Re-run with --force to overwrite "
                "deliberately."
            )
    target.write_text(data, encoding="utf-8", newline="\n")
    return target, True


def rebuild_index(out_root: Path) -> Path:
    """Rebuild <out>/index.json from whatever conforming exports are on disk."""
    out_root = Path(out_root)
    rows = []
    for p in sorted((out_root / "receipts").glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except ValueError:
            continue
        if doc.get("schema") != SCHEMA:
            continue
        rows.append(
            {
                "id": doc["receipt"]["id"],
                "path": f"receipts/{p.name}",
                "detector": doc["instrument"]["detector"].get("detector", "unknown"),
                "corpus": doc["corpus"].get("name", "unknown"),
                "corpus_sha256": doc["corpus"].get("sha256", "unknown"),
                "auroc": doc["metrics"].get("auroc"),
                "gaige_version": doc["receipt"].get("gaige_version", "unknown"),
                "generated_utc": doc["receipt"].get("generated_utc", "unknown"),
            }
        )
    rows.sort(key=lambda r: r["id"])
    index = {"schema": INDEX_SCHEMA, "exported_by": f"gaige {__version__}", "receipts": rows}
    target = out_root / "index.json"
    target.write_text(_dumps(index), encoding="utf-8", newline="\n")
    return target


def export_report(report_dir: Path, out_root: Path, force: bool = False) -> dict:
    """The CLI's whole job: write one export, rebuild the index, describe what happened."""
    receipt_path, wrote = write_export(report_dir, out_root, force=force)
    index_path = rebuild_index(out_root)
    doc = json.loads(receipt_path.read_text(encoding="utf-8"))
    return {
        "id": doc["receipt"]["id"],
        "detector": doc["instrument"]["detector"].get("detector", "unknown"),
        "corpus": doc["corpus"].get("name", "unknown"),
        "auroc": doc["metrics"].get("auroc"),
        "receipt_path": receipt_path,
        "index_path": index_path,
        "wrote": wrote,
    }
