# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""The claims policy, enforced rather than remembered.

gaige's credibility rests on refusing to overclaim — including about itself, and including
in prose. The research fold of record keeps a blocked list of claims that were adversarially
REFUTED (0-3 or 2-1 panels) or that survive only through a secondary witness; none of them
may appear in any shipped document. And the positioning section holds every quantitative
claim to the same bar as a report: a number appears beside its source or its instrument, or
it does not appear.

An editorial rule that nobody runs is a rule that drifts. These tests run on every push.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

# Every prose document that ships with the repo. The golden export fixture is included
# because exported receipt JSON is shipped prose the moment it reaches the site.
SHIPPED_DOCS = [
    "README.md",
    "CHANGELOG.md",
    "NOTICE.md",
    "RUNBOOK.md",
    "CONTRIBUTING.md",
    "SECURITY.md",
    "SUPPORT.md",
    "TRADEMARK.md",
    "COMMERCIAL.md",
    "docs/what-the-aggregate-hides.md",
    "tests/fixtures/export-golden/receipts/export-clean.json",
]

AGGREGATE_NOTE = "docs/what-the-aggregate-hides.md"

# Refuted or unusably-sourced claims (fold of record: detection-research-2026-07-21.md,
# corrected 2026-07-22). Each entry: (pattern, why it is blocked).
BLOCKED = [
    (
        r"reduces\s+disparit\w+\s+(?:by\s+)?27\.4\s*%",
        (
            "FairOPT composite claim as originally worded — refuted 0-3; the corrected form "
            "is a 27.4 percentage-POINT drop vs a static baseline, with a measured F1 cost"
        ),
    ),
    (
        r"<\s*0\.1\s*%\s+accuracy",
        "FairOPT's '<0.1% accuracy cost' — contradicted by the paper's own Table 2",
    ),
    (
        r"cuts?\s+FPR\s+disparity\s+~?\s*51\s*%",
        (
            "group-threshold ~51% disparity claim as originally worded — panel ended 2-1, "
            "stays out of public text"
        ),
    ),
    (
        r"99\.67",
        (
            "the 'Binoculars is ESL-safe' figure — killed 0-3 in one form, 2-1 in the other; "
            "counter-evidence exists (DivEye measures proficiency-correlated degradation)"
        ),
    ),
    (
        r"98\.1\s*%?\s*(?:AI)?.{0,40}5\.3\s*%?",
        (
            "the Hix Bypass 98.1->5.3 pair survives only via a secondary witness quoting a "
            "paywalled interactive; never public without the primary"
        ),
    ),
    (
        r"no\s+false[- ]positive\s+bias",
        (
            "no detector may be described as bias-free; refuted for the one case tested, and "
            "the claims policy forbids the class"
        ),
    ),
    (
        r"bias[- ]free",
        "same rule, stated form",
    ),
]

# A quantitative claim is a percentage or a bare 2-4 decimal statistic (AUROC, KS, F1...).
NUMERIC = re.compile(r"\d+(?:\.\d+)?\s*%|\b0\.\d{2,4}\b|\b\d\.\d{4}\b")

# What counts as naming a source or an instrument, per the every-number-has-a-receipt rule.
SOURCED = re.compile(
    r"arXiv[:\s]*\d{4}\.\d{4,5}"  # a citable paper
    r"|ACL 2024"
    r"|hc3-mini"  # our reference corpus, named with n and seed where it appears
    r"|raid g2"  # the RAID slice, named with seed and instrument where it appears
    r"|measured in RAID",
    re.IGNORECASE,
)


def _read(name: str) -> str:
    return (ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize("doc", SHIPPED_DOCS)
def test_no_blocked_claims_in_shipped_docs(doc):
    text = _read(doc)
    hits = []
    for pattern, why in BLOCKED:
        m = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if m:
            line = text.count("\n", 0, m.start()) + 1
            hits.append(f"{doc}:{line} matches blocked claim {pattern!r} — {why}")
    assert not hits, "refuted claims in shipped prose:\n" + "\n".join(hits)


def _positioning_section() -> str:
    text = _read("README.md")
    m = re.search(r"^## Where gaige sits\n(.*?)(?=^## )", text, re.MULTILINE | re.DOTALL)
    assert m, "README.md must carry the 'Where gaige sits' section"
    return m.group(1)


def test_positioning_section_paragraphs_carry_sources():
    """Every paragraph in the positioning section that states a number names a source.

    Third-party numbers name the paper; first-party numbers name the corpus (with n and
    seed) or the slice — the same bar a report holds itself to.
    """
    body = re.sub(r"<!--.*?-->", "", _positioning_section(), flags=re.DOTALL)
    unsourced = []
    for para in re.split(r"\n\s*\n", body):
        flat = " ".join(para.split())
        if not flat:
            continue
        if NUMERIC.search(flat) and not SOURCED.search(flat):
            unsourced.append(flat[:100])
    assert not unsourced, (
        "numbers without a source/instrument in 'Where gaige sits':\n" + "\n".join(unsourced)
    )


def test_positioning_first_party_numbers_name_their_instrument():
    """The receipts we cite as our own must sit beside their instrument fingerprint."""
    body = _positioning_section()
    for number, needs in [
        ("0.9285", ["falcon-7b", "4-bit", "seed 17"]),  # first RAID slice receipt
        ("2.4446", ["falcon-7b", "4-bit", "n=100", "seed=17"]),  # conformal receipt
    ]:
        para = next(
            (p for p in re.split(r"\n\s*\n", body) if number in p),
            None,
        )
        assert para is not None, f"first-party receipt number {number} missing from section"
        flat = " ".join(para.split())
        missing = [tok for tok in needs if tok not in flat]
        assert not missing, f"{number} appears without instrument tokens {missing}"


def test_no_verdict_language_in_positioning():
    """gaige measures, gaige does not judge — the section may not parse gaige as a detector."""
    body = re.sub(r"<!--.*?-->", "", _positioning_section(), flags=re.DOTALL)
    for pattern in [r"gaige\s+detects", r"gaige\s+catches", r"gaige\s+flags\b"]:
        assert not re.search(pattern, body, re.IGNORECASE), f"detector framing: {pattern!r}"


# --------------------------------------------------- the aggregates-hide note (docs/)
# The note promises, in its own closing paragraph, to be held to report bar. These tests
# are that bar.


def _note_body() -> str:
    return re.sub(r"<!--.*?-->", "", _read(AGGREGATE_NOTE), flags=re.DOTALL)


def test_aggregate_note_paragraphs_carry_sources():
    """Every paragraph in the note that states a number names a source or an instrument."""
    body = re.sub(r"```.*?```", "", _note_body(), flags=re.DOTALL)
    unsourced = []
    for para in re.split(r"\n\s*\n", body):
        flat = " ".join(para.split())
        if not flat:
            continue
        if NUMERIC.search(flat) and not SOURCED.search(flat):
            unsourced.append(flat[:100])
    assert not unsourced, f"numbers without a source/instrument in {AGGREGATE_NOTE}:\n" + "\n".join(
        unsourced
    )


def test_aggregate_note_numbers_sit_beside_their_instrument():
    """The note's flagship numbers must appear beside their instrument fingerprint tokens."""
    body = _note_body()
    for number, needs in [
        ("0.9285", ["falcon-7b", "4-bit", "seed 17"]),  # slice aggregate
        ("61.5", ["falcon-7b", "4-bit", "seed 17"]),  # slice operating point
        ("86.0", ["hc3-mini", "n=100", "seed=17"]),  # reference-corpus comparison
        ("87.6", ["raid g2", "39.7", "same instrument"]),  # the decoding split
    ]:
        para = next((p for p in re.split(r"\n\s*\n", body) if number in p), None)
        assert para is not None, f"flagship number {number} missing from {AGGREGATE_NOTE}"
        flat = " ".join(para.split())
        missing = [tok for tok in needs if tok not in flat]
        assert not missing, f"{number} appears without instrument tokens {missing}"


def test_aggregate_note_agrees_with_readme():
    """One receipt, two surfaces: the note and the README fold may never drift apart."""
    note, readme = _note_body(), _read("README.md")
    for number in ["0.9285", "61.5", "87.6", "39.7"]:
        assert number in note, f"{number} missing from {AGGREGATE_NOTE}"
        assert number in readme, (
            f"{number} is in {AGGREGATE_NOTE} but not README.md — the surfaces have drifted"
        )


def test_aggregate_note_no_verdict_language():
    """The note may not parse gaige as a detector either."""
    body = _note_body()
    for pattern in [r"gaige\s+detects", r"gaige\s+catches", r"gaige\s+flags\b"]:
        assert not re.search(pattern, body, re.IGNORECASE), f"detector framing: {pattern!r}"
