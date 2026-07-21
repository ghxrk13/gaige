# PROGRESS — detcal (yoto pattern)

## Burst 1 incident receipt — 2026-07-21 (the thesis, live, on day one)
First e2e run: transformers **5.14.1** silently misrouted `BitsAndBytesConfig` (log showed an
eetq quantizer path, "no linear modules found"), attempted a **12.89 GiB fp16** load of a
model requested at 4-bit (~5 GiB), and OOM'd. Exactly the instrument-integrity failure class
detcal exists to catch: same command, same model, different library version ⇒ different
instrument. Response: (a) detector now warns on transformers>=5 + 4bit and converts load-OOM
into an instrument-integrity error with the honest explanation; (b) repo-local pinned env
(.venv: transformers 4.49 / torch cu130 / bnb) is the reference runtime; (c) this incident is
flagship-report material for Burst 2 (measured, reproducible, version-pinned).

## Burst 1 — 2026-07-21 (bench)
- Skeleton: corpus loader + HC3 subsample fetcher, Fast-DetectGPT analytic detector w/ quantization VERIFICATION (refuses silent-fp16 loads), ROC/AUROC/threshold-at-FPR + bootstrap CIs, receipts writer, CLI, unit tests.
- License Apache-2.0 (adoption over copyleft; open-core still possible later).
- NEXT (Burst 2): run receipts on 2-3 detectors incl. bf16-vs-4bit A/B; flagship report; then Burst 3 = three customer conversations before any more code.
- Rules: clean-room corpora only; local repo (ghxrk13); GitHub creation goes through the approval queue.
