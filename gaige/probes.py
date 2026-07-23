# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Probe sets: dated, provenance-carrying question/answer collections.

Format: JSONL, one probe per row:
  {"id", "prompt", "answer", "vintage", "source", "source_date",
   "aliases": [...]?, "authored"?}

Every probe carries its source and source_date because contamination is a standing validity
threat: if a probe's answer was in the model's training data, a correct answer measures
memorization, not currency. Recording authorship dates against the model's training cutoff
lets every receipt DEMONSTRATE, per vintage, how much of the probe set post-dates the model
— instead of asserting it.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

REQUIRED = ("id", "prompt", "answer", "vintage", "source", "source_date")


class BadProbeSet(ValueError):
    """The probe file does not meet the schema; the message names row and remedy."""


def _parse_date(value: str, path: Path, row_n: int, fieldname: str) -> date:
    try:
        return date.fromisoformat(value)
    except (TypeError, ValueError):
        raise BadProbeSet(
            f"{path}:{row_n}: {fieldname}={value!r} is not an ISO date (YYYY-MM-DD). "
            "Dates gate the contamination check; fix the probe, don't drop the field."
        ) from None


@dataclass
class ProbeSet:
    name: str
    path: Path
    probes: list[dict]
    sha256: str
    meta: dict = field(default_factory=dict)

    @property
    def vintages(self) -> dict:
        c: dict = {}
        for p in self.probes:
            c[p["vintage"]] = c.get(p["vintage"], 0) + 1
        return c

    # Shape-compatibility with runstate.write_runstate (which describes a "corpus" as
    # name/sha256/counts): a probe set's counts are its per-vintage counts.
    @property
    def counts(self) -> dict:
        return self.vintages

    @property
    def vintage_hashes(self) -> dict:
        """Content hash per vintage, order-independent.

        The longitudinal contract: once a vintage label has been measured, its questions are
        frozen — re-running "t0" with edited probes would silently change the measurand. The
        registry enforces that by comparing these hashes across runs; new vintage labels may
        be added, existing ones must hash identically forever.
        """
        buckets: dict[str, list[str]] = {}
        for p in self.probes:
            clean = {k: v for k, v in p.items() if not k.startswith("_")}
            buckets.setdefault(p["vintage"], []).append(json.dumps(clean, sort_keys=True))
        return {
            v: hashlib.sha256("\n".join(sorted(rows)).encode("utf-8")).hexdigest()
            for v, rows in buckets.items()
        }

    def post_cutoff_share(self, training_cutoff: str) -> dict:
        """Per-vintage share of probes whose source_date post-dates the model's cutoff.

        The number every receipt prints so "the vintages post-date the model" is shown,
        not claimed. 1.0 everywhere is what a well-authored study set looks like.
        """
        cutoff = date.fromisoformat(training_cutoff)
        out: dict = {}
        for p in self.probes:
            v = p["vintage"]
            d = out.setdefault(v, {"n": 0, "post_cutoff": 0})
            d["n"] += 1
            if p["_source_date"] > cutoff:
                d["post_cutoff"] += 1
        for d in out.values():
            d["share"] = d["post_cutoff"] / d["n"]
        return out


def load_probes(path: Path, name: str | None = None) -> ProbeSet:
    probes: list[dict] = []
    seen_ids: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise BadProbeSet(f"{path}:{n}: not valid JSON ({e.msg})") from None
            missing = [k for k in REQUIRED if not row.get(k)]
            if missing:
                raise BadProbeSet(
                    f"{path}:{n}: missing required field(s) {missing}. Every probe needs "
                    f"{list(REQUIRED)}; aliases/authored are optional."
                )
            if row["id"] in seen_ids:
                raise BadProbeSet(
                    f"{path}:{n}: duplicate id {row['id']!r}. Ids key resume and joins; "
                    "make them unique."
                )
            seen_ids.add(row["id"])
            aliases = row.get("aliases", [])
            if not isinstance(aliases, list) or not all(isinstance(a, str) for a in aliases):
                raise BadProbeSet(
                    f"{path}:{n}: aliases must be a list of strings, got {aliases!r}."
                )
            # Parsed date rides along under a private key; the raw string stays authoritative.
            row["_source_date"] = _parse_date(row["source_date"], path, n, "source_date")
            if row.get("authored"):
                _parse_date(row["authored"], path, n, "authored")
            probes.append(row)
    if not probes:
        raise BadProbeSet(f"{path}: no probes found")

    h = hashlib.sha256()
    with open(path, "rb") as fb:
        for chunk in iter(lambda: fb.read(1 << 20), b""):
            h.update(chunk)
    return ProbeSet(
        name=name or Path(path).stem,
        path=Path(path),
        probes=probes,
        sha256=h.hexdigest(),
        meta={"source": str(path)},
    )
