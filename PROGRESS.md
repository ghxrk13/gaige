# PROGRESS — gaige (yoto pattern)

## Burst 1 incident receipt — 2026-07-21 (the thesis, live, on day one)
First e2e run: transformers **5.14.1** silently misrouted `BitsAndBytesConfig` (log showed an
eetq quantizer path, "no linear modules found"), attempted a **12.89 GiB fp16** load of a
model requested at 4-bit (~5 GiB), and OOM'd. Exactly the instrument-integrity failure class
gaige exists to catch: same command, same model, different library version ⇒ different
instrument. Response: (a) detector now warns on transformers>=5 + 4bit and converts load-OOM
into an instrument-integrity error with the honest explanation; (b) repo-local pinned env
(.venv: transformers 4.49 / torch cu130 / bnb) is the reference runtime; (c) this incident is
flagship-report material for Burst 2 (measured, reproducible, version-pinned).

## Burst 1 — 2026-07-21
- Skeleton: corpus loader + HC3 subsample fetcher, Fast-DetectGPT analytic detector w/ quantization VERIFICATION (refuses silent-fp16 loads), ROC/AUROC/threshold-at-FPR + bootstrap CIs, receipts writer, CLI, unit tests.
- License Apache-2.0 (adoption over copyleft; open-core still possible later).
- NEXT (Burst 2): run receipts on 2-3 detectors incl. bf16-vs-4bit A/B; flagship report; then Burst 3 = three customer conversations before any more code.
- Rules: clean-room corpora only; local repo (ghxrk13); GitHub creation goes through the approval queue.

## Research base — 2026-07-21
Literature sweep (26 sources, 23 adversarially-verified claims) landed. Key external
validation: RAID (ACL 2024) measured open-source detectors at 47-100% FPR on naive default
thresholds vs commercial <=1.7%, and recommends verbatim what this tool does ("calibrate
detectors on in-domain data before using them"); an ICLR-2026 peer-review study runs this
exact workflow (calibration set -> FPR-targeted thresholds -> re-measured on held-out ->
bootstrap CIs). Research-ranked build queue supersedes the earlier Burst-2 sketch:
1. Conformal / FPR-bounded calibration mode (human-only calibration set; measured TPR gains
   at strict operating points: Fast-DetectGPT 51.22->69.32 @0.5%FPR, Binoculars 70.16->84.34).
2. Subgroup-stratified receipts (length first, then style/formality) + optional group-adaptive
   thresholds; static thresholds measurably penalize short and non-native-speaker text.
3. Binoculars as detector #2; then quantization/version A/B receipts.
4. Adversarial-degradation receipts using published attack panels (synonym swap, homoglyph,
   LLM-rewrite-of-human -- the last is the hardest real case).
5. RAID / TH-Bench corpus adapters; "provably-human, provably-unseen" corpus recipe.
6. Watermark-verifier adapter (SynthID ships a detector and delegates threshold choice to
   deployers -- a funded audience for exactly this tool).
7. Base-rate calculator in every report (FPR x volume = wrongly-accused count).
Claims policy: cite measured third-party numbers only; never claim detection is "accurate",
never claim a threshold generalizes, never claim any detector is bias-free.
