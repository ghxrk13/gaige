# Third-party notices and attribution

gaige implements published detection criteria. The algorithms are described in the papers
below; the implementations in this repository were written from those descriptions rather
than copied, but attribution is owed either way, and a tool about provenance should model
good provenance.

## Fast-DetectGPT (analytic sampling discrepancy)

Bao, Zhao, Teng, Yang, Zhang. *Fast-DetectGPT: Efficient Zero-Shot Detection of
Machine-Generated Text via Conditional Probability Curvature.* ICLR 2024.
arXiv:2310.05130 · reference implementation: https://github.com/baoguangsheng/fast-detect-gpt
(MIT License). The MIT license permits reuse with attribution; this notice provides it.

## Calibration methodology

- Split-conformal thresholding follows Zhu et al., *Reliably Bounding False Positives: A
  Zero-Shot Machine-Generated Text Detection Framework via Multiscaled Conformal Prediction*,
  arXiv:2505.05084. (Corrected 2026-07-22: an earlier revision of this file misattributed the
  paper and carried a different paper's title.)
- Fixed-FPR evaluation practice follows Dugan et al., *RAID: A Shared Benchmark for Robust
  Evaluation of Machine-Generated Text Detectors*, ACL 2024 (arXiv:2405.07940).
- Subgroup-stratified error reporting is motivated by Liang et al., arXiv:2304.02819
  (non-native-speaker false positives) and Jung et al., arXiv:2502.04528 (length/style
  threshold disparity; corrected from "Nguyen et al." 2026-07-22).

## Trademarks (additional term under AGPL-3.0 section 7(e))

As permitted by section 7(e) of the GNU AGPL-3.0, the following non-permissive additional
term applies to this work: **this license does not grant permission to use the "gaige" name,
logo, or any associated trademarks or service marks**, except as required for reasonable and
customary use in describing the origin of the work. See `TRADEMARK.md` for the usage policy.
(AGPL-3.0's operative grant is scoped to copyright and patents; unlike Apache-2.0 §6 it
contains no express trademark clause, so this term states the reservation explicitly rather
than relying on license silence.)

## Corpora

`hc3-mini` subsamples HC3 (Hello-SimpleAI), distributed on the Hugging Face Hub under its own
terms. gaige downloads it at run time and redistributes none of it. Every report records the
corpus sha256 so a reader can verify which subsample produced a number.

`gaige corpus prepare-raid` slices RAID (Dugan et al., *RAID: A Shared Benchmark for Robust
Evaluation of Machine-Generated Text Detectors*, ACL 2024, arXiv:2405.07940), distributed by
its authors on the Hugging Face Hub under their terms. gaige fetches rows at preparation time
and redistributes none of it; every slice records the dataset revision sha, the full selection
parameters, and the slice sha256 so a reader can verify exactly which cut produced a number.
