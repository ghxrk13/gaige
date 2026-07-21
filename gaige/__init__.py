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
"""

__version__ = "0.0.1"
