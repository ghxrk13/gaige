# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""gaige — calibration and drift receipts for AI measurement.

Thesis: a score is meaningless without the operating threshold, and a threshold is meaningless
without the corpus, model, quantization, device, and versions it was measured on. Change any of
those and you have a different instrument, whether or not anyone noticed.

gaige turns (corpus or probe set, pluggable scorer) into a receipts report: ROC, AUROC,
thresholds at target false-positive rates, bootstrap error bars, and a complete instrument
fingerprint — so the number you act on is one you can defend, and so you can tell an instrument
that drifted from a system that did.

Two applications of one machine:
  - Detector calibration - what is this detector's real false-positive rate on YOUR material,
    rather than on a vendor's marketing page.
  - Instrument drift - has the thing being measured changed, or has the measuring pipeline?
    Answering that is the difference between a finding and an artifact.

The `gaige` CLI is the primary interface. The names re-exported here are the supported library
surface — the calibrate/conformal/analyze spine, which never imports torch, so it runs on
machines that could never load a model:

    import gaige
    rows, corpus, meta = gaige.load_report(report_dir)  # replay an existing report, no GPU
    results = gaige.compute_results(rows)
    cal = gaige.conformal_threshold(human_scores, alpha=0.01)

Everything not listed in __all__ is internal and may move without notice.
"""

# Assigned before the re-exports below: analyze.py reads `from . import __version__` at
# module scope, so the attribute must exist while this package is partially initialized.
__version__ = "0.0.1"

from gaige.analyze import CorpusView, NotAReport, compute_results, load_report, read_scores_csv
from gaige.calibrate import (
    CorpusTooSmall,
    auroc,
    bootstrap_ci,
    eer,
    proportion_ci,
    roc_points,
    threshold_at_fpr,
)
from gaige.conformal import (
    InsufficientCalibration,
    conformal_table,
    conformal_threshold,
    min_samples_for,
)
from gaige.corpus import Corpus, fetch_hc3_mini, load_jsonl
from gaige.receipts import write_report
from gaige.subgroups import base_rate_harm, max_disparity, ppv, stratified_rates

__all__ = [
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
