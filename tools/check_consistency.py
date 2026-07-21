#!/usr/bin/env python3
# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Identity drift check: does gaige still describe itself consistently?

gaige exists to catch instruments that quietly stopped measuring the same thing. The same
failure mode applies to the project itself: a definition changes, and the README, the package
metadata, the module headers, and the published package all drift apart at different rates.
That drift is how a project ends up making a claim on one surface it has abandoned on another.

So this is a drift detector pointed at gaige. It runs in CI and fails loudly, because catching
this by memory does not scale and forgetting it is exactly the kind of avoidable embarrassment
a public artifact cannot afford.

Run:  python tools/check_consistency.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import tomllib

ROOT = Path(__file__).resolve().parent.parent
HEADER = "# gaige — calibration and drift receipts for AI measurement."

problems: list[str] = []


def fail(msg: str) -> None:
    problems.append(msg)


def check_version_matches() -> None:
    """The version a user installs must equal the version a receipt records."""
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    declared = pyproject["project"]["version"]
    init = (ROOT / "gaige" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'__version__\s*=\s*"([^"]+)"', init)
    if not m:
        fail("gaige/__init__.py: no __version__ found")
        return
    if m.group(1) != declared:
        fail(
            f"version drift: pyproject={declared!r} but gaige/__init__.py={m.group(1)!r}. "
            "Every receipt records __version__; a mismatch makes receipts untraceable."
        )


def check_headers() -> None:
    """Every source file carries the current one-line identity."""
    stale = []
    for p in sorted(list((ROOT / "gaige").rglob("*.py")) + list((ROOT / "tests").rglob("*.py"))):
        first = p.read_text(encoding="utf-8").splitlines()[:1]
        if not first or first[0].strip() != HEADER:
            stale.append(str(p.relative_to(ROOT)))
    if stale:
        fail(
            "stale or missing identity header in: "
            + ", ".join(stale)
            + f"\n    expected first line: {HEADER}"
        )


def check_description_alignment() -> None:
    """pyproject summary, README headline, and the package docstring must agree.

    Not string-identical (each surface has its own voice) but they must share the defining
    phrase. If the definition moves, it has to move everywhere in the same commit.
    """
    anchor = "calibration and drift receipts for ai measurement"

    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    if anchor not in pyproject["project"]["description"].lower():
        fail(f"pyproject description does not contain the defining phrase: {anchor!r}")

    readme = (ROOT / "README.md").read_text(encoding="utf-8").lower()
    if anchor not in readme[:800]:
        fail(f"README headline does not contain the defining phrase: {anchor!r}")

    init = (ROOT / "gaige" / "__init__.py").read_text(encoding="utf-8").lower()
    if anchor not in init:
        fail(f"gaige/__init__.py docstring does not contain the defining phrase: {anchor!r}")


def check_required_docs() -> None:
    """Documents a public repo is judged for having."""
    for name in (
        "LICENSE",
        "README.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "COMMERCIAL.md",
        "TRADEMARK.md",
        "NOTICE.md",
        "PROGRESS.md",
    ):
        if not (ROOT / name).exists():
            fail(f"missing expected document: {name}")


def check_no_heavy_imports_at_module_scope() -> None:
    """torch/transformers/bitsandbytes must stay lazily imported.

    If one escapes to module scope, gaige silently stops working on every machine without a
    GPU stack — which is most machines, including the enclave and the author's own laptop.
    """
    heavy = ("torch", "transformers", "bitsandbytes", "accelerate")
    for p in sorted((ROOT / "gaige").rglob("*.py")):
        if "detectors" in p.parts:
            continue  # detector modules are themselves imported lazily
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if line.startswith(("import ", "from ")) and any(
                re.match(rf"(import|from)\s+{h}\b", line) for h in heavy
            ):
                fail(
                    f"{p.relative_to(ROOT)}:{i}: heavy import at module scope -> {line.strip()!r}. "
                    "Move it inside the function that needs it."
                )


def main() -> int:
    check_version_matches()
    check_headers()
    check_description_alignment()
    check_required_docs()
    check_no_heavy_imports_at_module_scope()

    if problems:
        print("gaige consistency check FAILED:\n")
        for p in problems:
            print(f"  - {p}")
        print(
            "\nThese are identity-drift failures. Fix them in the same commit that moved the "
            "definition, not later."
        )
        return 1
    print("gaige consistency check passed: version, headers, description, docs, import hygiene.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
