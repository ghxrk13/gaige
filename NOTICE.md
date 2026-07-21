# Third-party notices and attribution

gaige implements published detection criteria. The algorithms are described in the papers
below; the implementations in this repository were written from those descriptions rather
than copied, but attribution is owed either way — and a tool about provenance should model
good provenance.

## Fast-DetectGPT (analytic sampling discrepancy)

Bao, Zhao, Teng, Yang, Zhang. *Fast-DetectGPT: Efficient Zero-Shot Detection of
Machine-Generated Text via Conditional Probability Curvature.* ICLR 2024.
arXiv:2310.05130 · reference implementation: https://github.com/baoguangsheng/fast-detect-gpt
(MIT License). The MIT license permits reuse with attribution; this notice provides it.

## Calibration methodology

- Split-conformal thresholding follows Wang et al., *Are AI Detectors Good Enough? A Survey on
  Quality of Datasets With Machine-Generated Texts* / conformal MGT detection, arXiv:2505.05084.
- Fixed-FPR evaluation practice follows Dugan et al., *RAID: A Shared Benchmark for Robust
  Evaluation of Machine-Generated Text Detectors*, ACL 2024 (arXiv:2405.07940).
- Subgroup-stratified error reporting is motivated by Liang et al., arXiv:2304.02819
  (non-native-speaker false positives) and Nguyen et al., arXiv:2502.04528 (length/style
  threshold disparity).

## Corpora

`hc3-mini` subsamples HC3 (Hello-SimpleAI), distributed on the Hugging Face Hub under its own
terms. gaige downloads it at run time and redistributes none of it. Every report records the
corpus sha256 so a reader can verify which subsample produced a number.
