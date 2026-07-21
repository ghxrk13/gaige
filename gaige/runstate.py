# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Crash-safe, resumable scoring runs.

Scoring is the slow part. A 7B model on CPU runs ~20-36 s per sample, so a 200-sample corpus is
about two hours, and losing it at sample 190 to a dropped SSH session is a real way to lose an
evening. So scores are appended to disk as they are produced rather than held in memory until
the end.

**The correctness hazard this module exists to prevent.** Resuming is not just "skip what's
done." If the environment changed between the two halves of a run — a different model, a
different device, an upgraded transformers — then the resulting scores.csv would contain
measurements from TWO DIFFERENT INSTRUMENTS, silently interleaved, and every threshold derived
from it would be meaningless. gaige exists to make exactly that class of error visible in other
people's tools; it would be indefensible here.

So a resume is verified in two stages: the cheap arguments check before the model loads (fail
fast, do not download 14 GB to discover a mismatch), then the full instrument fingerprint after
the load, which is the only point at which library versions and the resolved device are known.
Any mismatch refuses the resume rather than producing a plausible-looking corrupt receipt.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

PARTIAL = "scores.partial.csv"
RUNSTATE = "run.json"
FIELDS = ["id", "label", "score", "seconds"]

# Fields of the instrument fingerprint that must not change mid-run. Anything here changing
# means the second half of the corpus was measured with a different instrument.
PINNED = ("detector", "model_id", "quant_requested", "max_tokens", "device", "versions")


class ResumeRefused(RuntimeError):
    """The run on disk was not produced by the instrument now in hand."""


def write_runstate(outdir: Path, corpus, detector_meta: dict, reproduce_cmd: str) -> None:
    """Record what this run IS, before any scoring, so a resume can be checked against it."""
    outdir.mkdir(parents=True, exist_ok=True)
    state = {
        "corpus": {"name": corpus.name, "sha256": corpus.sha256, "counts": corpus.counts},
        "detector": {k: detector_meta.get(k) for k in PINNED},
        "reproduce": reproduce_cmd,
        "complete": False,
    }
    (outdir / RUNSTATE).write_text(json.dumps(state, indent=1), encoding="utf-8")


def read_runstate(outdir: Path) -> dict:
    p = outdir / RUNSTATE
    if not p.exists():
        raise ResumeRefused(
            f"{outdir}: no {RUNSTATE} — this is not a resumable gaige run directory."
        )
    return json.loads(p.read_text(encoding="utf-8"))


def check_args_match(state: dict, corpus, model_id: str, quant: str, max_tokens: int) -> None:
    """Cheap pre-load check: corpus identity and the arguments we know before loading a model.

    Runs first specifically so a mismatch is caught before spending minutes on a model load.
    """
    problems = []
    stored_corpus = state.get("corpus", {})
    if stored_corpus.get("sha256") != corpus.sha256:
        problems.append(
            f"corpus: run was {stored_corpus.get('name')} sha256={str(stored_corpus.get('sha256'))[:16]}..., "
            f"now {corpus.name} sha256={corpus.sha256[:16]}..."
        )
    stored_det = state.get("detector", {})
    for key, now in (
        ("model_id", model_id),
        ("quant_requested", quant),
        ("max_tokens", max_tokens),
    ):
        was = stored_det.get(key)
        if was is not None and was != now:
            problems.append(f"{key}: run was {was!r}, now {now!r}")
    if problems:
        raise ResumeRefused(_refusal("before loading the model", problems))


def check_instrument_match(state: dict, detector_meta: dict) -> None:
    """Post-load check: the full fingerprint, including library versions and resolved device.

    This is the one that catches an environment that drifted between the two halves of a run —
    an upgraded transformers, or a CUDA box that fell back to CPU on the second attempt.
    """
    stored = state.get("detector", {})
    problems = []
    for key in PINNED:
        was, now = stored.get(key), detector_meta.get(key)
        if was is not None and was != now:
            problems.append(f"{key}: run was {was!r}, now {now!r}")
    if problems:
        raise ResumeRefused(_refusal("after loading the model", problems))


def _refusal(when: str, problems: list[str]) -> str:
    return (
        f"refusing to resume ({when}) — the instrument changed:\n  "
        + "\n  ".join(problems)
        + "\nResuming would interleave scores from two different instruments into one report, "
        "and every threshold derived from it would be meaningless. Start a fresh run instead."
    )


def load_partial(outdir: Path) -> list[dict]:
    """Rows already scored. Tolerates a truncated final line from an abrupt kill."""
    p = outdir / PARTIAL
    if not p.exists():
        return []
    rows = []
    with open(p, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            try:
                rows.append(
                    {
                        "id": row["id"],
                        "label": row["label"],
                        "score": float(row["score"]),
                        "seconds": float(row["seconds"]) if row.get("seconds") else "",
                    }
                )
            except (KeyError, TypeError, ValueError):
                # A process killed mid-write can leave a partial final line. Dropping it is
                # correct: that sample simply gets scored again.
                continue
    return rows


def open_partial(outdir: Path):
    """Append-mode handle, writing the header only for a new file."""
    outdir.mkdir(parents=True, exist_ok=True)
    p = outdir / PARTIAL
    new = not p.exists() or p.stat().st_size == 0
    fh = open(p, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=FIELDS)
    if new:
        writer.writeheader()
        fh.flush()
    return fh, writer


def append_row(fh, writer, row: dict) -> None:
    """Write one score and flush. The flush is the point — an unflushed buffer is not a receipt."""
    writer.writerow(row)
    fh.flush()


def mark_complete(outdir: Path) -> None:
    p = outdir / RUNSTATE
    if p.exists():
        state = json.loads(p.read_text(encoding="utf-8"))
        state["complete"] = True
        p.write_text(json.dumps(state, indent=1), encoding="utf-8")
    partial = outdir / PARTIAL
    if partial.exists():
        partial.unlink()  # the finished scores.csv supersedes it
