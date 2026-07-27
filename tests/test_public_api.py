# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""The supported import surface, enforced.

gaige/__init__.py re-exports the calibrate/conformal/analyze spine so a library user works
from `import gaige` without spelunking submodules. These tests pin that surface: every
promised name resolves, the promise list cannot drift silently, and importing gaige never
drags torch into the process — the analysis side must run on machines that could never
load a model.
"""

import subprocess
import sys

import gaige

EXPECTED = [
    "Corpus",
    "CorpusTooSmall",
    "CorpusView",
    "InsufficientCalibration",
    "NotAReport",
    "__version__",
    "auroc",
    "base_rate_harm",
    "bootstrap_ci",
    "compute_results",
    "conformal_table",
    "conformal_threshold",
    "eer",
    "fetch_hc3_mini",
    "load_jsonl",
    "load_report",
    "max_disparity",
    "min_samples_for",
    "ppv",
    "proportion_ci",
    "read_scores_csv",
    "roc_points",
    "stratified_rates",
    "threshold_at_fpr",
    "write_report",
]


def test_all_is_exactly_the_promised_list():
    assert gaige.__all__ == EXPECTED
    assert EXPECTED == sorted(EXPECTED)  # kept sorted, so diffs read as adds/removes


def test_every_promised_name_resolves():
    missing = [name for name in gaige.__all__ if not hasattr(gaige, name)]
    assert not missing, f"__all__ promises names gaige does not carry: {missing}"


def test_import_gaige_never_imports_torch():
    # A fresh interpreter, because this test process may carry torch from other tests.
    code = (
        "import sys, gaige; "
        "leaked = sorted({m.split('.')[0] for m in sys.modules} "
        "& {'torch', 'transformers', 'bitsandbytes', 'accelerate'}); "
        "assert not leaked, f'gpu deps leaked into module scope: {leaked}'"
    )
    proc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
