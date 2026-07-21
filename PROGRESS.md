# PROGRESS — detcal (yoto pattern)

## Burst 1 — 2026-07-21 (bench)
- Skeleton: corpus loader + HC3 subsample fetcher, Fast-DetectGPT analytic detector w/ quantization VERIFICATION (refuses silent-fp16 loads), ROC/AUROC/threshold-at-FPR + bootstrap CIs, receipts writer, CLI, unit tests.
- License Apache-2.0 (adoption over copyleft; open-core still possible later).
- NEXT (Burst 2): run receipts on 2-3 detectors incl. bf16-vs-4bit A/B; flagship report; then Burst 3 = three customer conversations before any more code.
- Rules: clean-room corpora only; local repo (ghxrk13); GitHub creation goes through the approval queue.
