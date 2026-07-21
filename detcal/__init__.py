"""detcal — calibration + receipts harness for AI-text detectors.

Thesis: a detector score is meaningless without the operating threshold, and an operating
threshold is meaningless without the corpus, model, quantization, and versions it was
measured on. detcal turns (labeled corpus, detector) into a receipts report: ROC, AUROC,
thresholds at target false-positive rates, bootstrap error bars, and a full environment
fingerprint — so the number you act on is one you can defend.
"""

__version__ = "0.0.1"
