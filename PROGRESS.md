# PROGRESS — gaige (yoto pattern)

## 2026-07-22 — the statistical core: reviewed, fixed, wired, and property-tested

**The adversarial review happened before the wiring**, in writing
(`private-notes/research/conformal-subgroups-review-2026-07-22.md`), against the actual papers.
`conformal.py`'s construction verified exact against Wang arXiv:2505.05084 (quantile, strict
flag rule, minimum-n bound — no off-by-one). Two things it *said* were wrong and now aren't:
the "empirical_fpr" it returned was (n−k)/n by construction — a function of n and α dressed
as a measurement — and is gone; and the guarantee now states what it is, **marginal over
calibration draws**, with the exact conditional dispersion (Beta(n+1−k, k) mean ± sd) printed
beside it. `subgroups.py`'s floor claimed to refuse and didn't; it now withholds rates below
n=20 (counts speak), and every reported rate carries a bootstrap interval.

**Wired into the ONE shared path** (`compute_results`), so `run` and `analyze` both emit the
conformal table, per-subgroup tables, and base-rate arithmetic in every report — replay
verified **bit-identical** on the reference receipt. scores.csv grew `n_words`/`meta`
(derived data only, never text; the privacy tests still pass), old score sets degrade
honestly ("subgroup receipts unavailable"), old partials still resume.

**Tests 47 → 64, and the new ones assert the property, not execution**: simulated marginal
coverage across alphas, conditional dispersion recovered from simulation matching the Beta
law, refusal boundaries exact. Teeth proven: breaking the order statistic by one rank turned
the suite red (2 failed) before reverting — recorded in the review doc.

**Reference receipt re-measured, unchanged**: AUROC 0.9720 (CI 0.9458–0.9938), thr@1% 2.1229,
corpus sha256 7d2819d3…, 128 Linear4bit / 4.04 GB verified. New alongside it: conformal
α=0.05 → 1.8468 (TPR 90%), α=0.01 → **2.4446 (TPR 76%)** vs the in-sample 2.1229 (86%) — the
measured price of an actual guarantee — and α=0.005 refused at n=100, correctly. First real
subgroup finding on our own corpus: short-text FPR 4.8% vs 2.6% at the 5% point, the
literature's direction, now with intervals.

**Deferred deliberately**: group-adaptive thresholds. The honest version is per-bucket
(Mondrian) conformal — which is also what the paper's "multiscaled" headline is — and it
needs ≥99 humans per bucket at α=0.01. Banked, not half-built.

**Same day, second pass — the four Fable-tagged judgment items, closed.** (1) Attestation
levels sharpened and signed: `verified` now includes an artifact-hash path (GGUF sha256 /
ollama digest matched against server identity), so the longitudinal spec's llama.cpp apparatus
will not rest on a server's word; `self-reported` is defined by whether an unchanged report
is evidence of an unchanged instrument; `opaque` has an explicit permitted-claims list and a
measured-during window. (2) Cross-instrument presentation rules adopted (ops
engineering-standards): rankings only as fixed-corpus selection decisions; `gaige plan` will
show feasibility and cost, never a quality column. (3) Device policy signed: `auto` keeps its
loud CPU fallback, and the real hole is fixed — **reproduce commands now record the resolved
device, never "auto"** (a receipt can no longer silently swap instruments when re-run
elsewhere). (4) The CPU default is **measured**: six candidates, one protocol, receipts
retained — gpt2-large alone separated the reference corpus perfectly (AUROC 1.0000, TPR 100%
at 1%-FPR and conformal α=0.01) at 0.64 s/sample, dominating gpt2-xl and gpt-neo-1.3B; and
separation was non-monotone in size, which is exactly why defaults get measured. Bonus
instrument datum from the verification run: gpt2/fp32 scored 0.9944 on GPU vs 0.9942 on CPU —
same model, different device, different number, as the fingerprint has claimed all along.

## 2026-07-21 evening — runs without a GPU; identity is now enforced, not asserted

**Redefined.** "Calibration + receipts for AI-text detectors" was accurate while text detection
was the only application. gaige is now **calibration and drift receipts for AI measurement**, with
two applications of one machine: detector calibration, and telling whether an observed change is
in the system measured or in the measuring pipeline. Summary, README, package docstring and all
source headers moved together; `tools/check_consistency.py` (in CI) fails the build if they ever
drift apart again.

**`gaige analyze`** re-derives thresholds and reports from scores that already exist — no model, no
GPU. `run` and `analyze` now share one `compute_results`, and a round-trip on the HC3 reference
reproduces AUROC 0.9720 and both threshold rows **exactly**, bootstrap CIs included.

**Runs on CPU.** Six CUDA assumptions removed from the detector. First CPU receipt, measured:
gpt2/fp32/cpu on hc3-mini(n=50) — **AUROC 0.9908**, thr@1%FPR 4.2276, 100 samples scored in 11 s.
Set against the reference GPU instrument (falcon-7b/4bit/cuda, AUROC 0.9720, thr@1%FPR 2.1229) this
is the thesis measured rather than argued: **same corpus, same detector, different device — a
threshold that is off by a factor of two.** 4-bit is refused on CPU rather than silently degraded.

**Instrument identity is now checked, not described.** `version_mismatches` became
`instrument_mismatches`: device joins library versions as part of identity, so CUDA-calibrated
thresholds cannot be quietly applied to CPU scores. Verified in both directions, with no false
alarm on reports that predate the field.

**Engineering baseline.** CI added where there was none (pytest + import hygiene + CLI smoke on
Linux **and Windows** across three Pythons, plus ruff). Tests 6 -> 25. `SECURITY.md` states trust
boundaries, the no-persistence property of `score`, and what gaige explicitly does not defend
against. Report writing crashed on Windows (`write_text` used cp1252, the caveats contain an
arrow); all report IO is now explicit UTF-8 with a test asserting it — the class of defect the
Windows CI leg exists to catch.

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
- License Apache-2.0 at first commit; **relicensed to AGPL-3.0 + commercial dual-license the same day** after the research showed permissive licensing leaves the code open to vendor wrapping with no return. Clean relicense (sole copyright, zero distribution). Renamed detcal -> gaige same day; trademark policy written.
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
