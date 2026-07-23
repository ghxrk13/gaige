# PROGRESS — gaige (yoto pattern)

## 2026-07-23 — EER + Brier land in the receipt (additive analysis-layer change)
- calibrate.eer: the FPR = FNR crossing on the measured ROC sweep, linearly interpolated
  between adjacent measured points (interpolated FPR equals interpolated FNR by
  construction). Reported in every analyze receipt (json + md) beside AUROC, with the
  carried caveat that a crossing measured on one calibration sample does not transfer.
- probcal.brier: the un-binned proper score beside ECE in the M3 per-vintage table
  (0 perfect; 0.25 = always answering 0.5).
- Reference instrument pin EXTENDED, not moved: AUROC/CIs/thresholds byte-identical;
  the fixture now also pins EER 0.07 @ threshold 1.6398.
- Tests 123 -> 130 (hand-computable EER/Brier cases survive the 50-per-class floor,
  plus the new reference pin).
- Cross-project lesson applied same day (knowledge only, no code crosses either way):
  a reviewer on an unrelated codebase flagged the hand-kept-field-list class of bug in a
  persistence step. gaige had the identical shape in receipts' results.json packing.
  results.json is now a wholesale write contract ({k: v for k != "roc"}) with a guard
  test that every computed key ships. Standalone-interface stance recorded: gaige stays
  a standalone CLI that other pipelines call; integration = adapters (lm-eval-harness,
  RAID queued), never entanglement. Tests 130 -> 131.

## 2026-07-22 — the statistical core: reviewed, fixed, wired, and property-tested

**The adversarial review happened before the wiring**, in writing
(`private-notes/research/conformal-subgroups-review-2026-07-22.md`), against the actual papers.
`conformal.py`'s construction verified exact against Zhu arXiv:2505.05084 (quantile, strict
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

**Same day, ninth pass — the served-model door and the feasibility mirror.** The ollama
provider earns `verified` the hard way: ollama's store is a content-addressed chain
(tags digest → manifest → model-layer digest → weights blob) and gaige re-hashes BOTH
links with its own hands — first live chain verification on qwen2.5:7b-instruct (weights
sha256 2bada8a745…, 4.68 GB), probe series `10c246457f8d` registered same hour. Any served
model is now a probe target with honest attestation; the bench store needed a reversible
read-ACL (documented in the RUNBOOK with its undo). And `gaige plan` closes the loop the
presentation rules demanded: feasibility + measured cost anchors that each NAME their
receipt, attestation level, and deliberately NO quality column — a test enforces the
legend. Live on bench it correctly said falcon-fp16 doesn't fit beside the daemon. Tests
109 → 123. Alongside the code: THE CHECKPOINT strategy doc
(`private-notes/strategy/continuity-longevity-plan-2026-07-22.md`) — HAVE/NEED/PATH/LONGEVITY
against both goals, P1–P4 standing order, the continuity contract stated plainly.

**Same day, eighth pass — Burst 2b: the public line ships its two receipts.** Binoculars
landed the way a measurement tool should land a detector: construction verified against
the released implementation BEFORE coding, tokenizer-identity refusal, both models
quant-proven on one fingerprint (256 Linear4bit, 8.07 GB — the VRAM ceiling beside the
daemon, measured 12.9/16.4 at peak). First receipt on the reference corpus: AUROC 0.9992,
TPR 97% @1%FPR, conformal α=.01 → 95%. Then the two flagship findings, both in-house now:
(1) **the paper's own global thresholds run at 16% and 3% FPR on our corpus** — RAID's
"calibrate in-domain" reproduced on the strongest zero-shot detector by our own tooling;
(2) **the quant A/B**: fp32-cpu vs fp16-cuda agree on thr@1% to four decimals, while 4-bit
moves it ~10% — quantization is an instrument parameter, and there is finally a receipt
that says so (falcon fp16 arm deferred to a daemon-free window). Tests 105 → 109. The
build queue's public line (items 1-4 + 8) is now DONE.

**Same day, seventh pass — M5: the battery is COMPLETE.** the operator pulled the change detectors
forward, and they were the right shape to pull: monitors replay registered series, so the
evening's real series became their first input. Per-interval conformal alarms carry the
marginal false-alarm bound (the methodological contribution, scoped exactly as the spec's
§5 wrote it); Page-Hinkley and CUSUM run as the literature runs them — tuned constants on
the receipt, no guarantee claimed, martingales cited as future work. The live watch on the
real series did precisely the honest thing: REFUSED conformal alarms on a 3-replicate
reference ("a tighter guarantee than your data supports is not a guarantee" — and told the
operator that Day-0 with `--replicates 4` fixes it), while PH/CUSUM sat correctly quiet on
a within-variance interval. Specificity is asserted in tests, not assumed: sub-threshold
shifts must NOT alarm. Tests 98 → 105. Every instrument in the M1/M1c/M3/M5 battery now
exists; M2r waits on an external sign-off by design.

**Same day, sixth pass — Phase D: the apparatus is REAL, end to end.** llama.cpp b10091 +
Qwen2.5-1.5B GGUF (the staged non-gated stand-in for the gated Llama-3.2-1B), served on
CPU, and every design decision met reality without flinching: `test-connection` earned
**verified** attestation on the first try (GGUF sha256 matched the server's reported
artifact); the Day-0 protocol measured the run-variance bound at **±0.0% on a SERVED
model** — greedy llama.cpp is deterministic, now a measured fact; the follow-up run landed
"within run variance"; and a temperature-0.3 run forked its own series live instead of
mixing. t0 75% / t1 50% under the signed grading rule, 100% post-cutoff. **The apparatus
burst — staged this evening as four phases — is complete in one session**: probe sets with
provenance → graded answers → attested providers → registered series → measured variance
bounds → M3's warning light. What remains before the pilot is the operator's side (real probe
authoring, HF Llama access, external sign-offs) and the post-pilot M5 detectors, which replay
registered series and lose nothing by waiting. Receipt of record:
`private-notes/research/first-longitudinal-receipt-2026-07-22.md`.

**Same day, fifth pass — apparatus Phase C: the warning light works.** M3 is real:
P(True) read from logits under a versioned, fingerprinted, series-frozen template; ECE and
the confidence-accuracy gap per vintage in a new `probcal` module (deliberately not
`calibrate.py` — that trap stays documented). Property-tested with a controllable fake
self-assessor (every number hand-computable through the full runner; injected
miscalibration recovered; ECE teeth proven by breaking the binning once). Semantics done
carefully: M3 riding along does not fork a series, but the template freezes per series the
first time M3 runs. And the first live measurement wrote the thesis's sentence by itself:
gpt2 at **0% accuracy, 79% mean P(True), gap +79%** — fluent and authoritative whilst
quietly wrong, measured. Tests 89 → 98. Remaining: Phase D (change detectors + the
real-model e2e).

**Same day, fourth pass — apparatus Phase B: the validity backbone exists.** The run
registry keys every probe run to its instrument-identity hash: a changed instrument forks a
new series (never mixes), a measured vintage is frozen forever (an edited "t0" is refused by
name; new vintages are welcome — that IS the longitudinal design), and the Day-0 replicate
protocol measures the run-variance bound instead of assuming one. Live e2e on the toy set:
three replicates → bound ±0.0% ("zero means the pipeline is deterministic, which is a
result, not an assumption"), follow-up run flagged "within run variance" — structurally,
**the first longitudinal receipt**, the artifact shape the longitudinal spec describes. Tests 81 → 87.
Remaining: P(True)/ECE (Phase C), change detectors + the real-model e2e (Phase D).

**Same day, third pass — apparatus Phase A: the acquisition layer exists.** The provider
seam (graded attestation: local-hf `verified` in-process; llamacpp earns
verified/self-reported/opaque, with the GGUF-sha256 path so the longitudinal apparatus won't rest on
a server's word), probe sets with per-probe provenance and the post-cutoff demonstration,
versioned deterministic grading (`nem-1`, plus MC argmax with conservative ties), and
`gaige probe run` — crash-safe, resumable, refusing any mid-run fingerprint change. Tests
64 → 81, all refusal paths exercised via an injectable fake provider; e2e smoked on gpt2
(which answered "The capital of France is" with "the capital of the French Republic, and" —
graded False, correctly: strict grading working as signed, and the argument for an instruct
model made empirically). The MC control path verified: " blue" beats " purple" after "The
color of the sky on a clear day is". Prompts cannot leave the machine for a non-local
endpoint without an explicit flag. Phases B-D staged next: registry/series, P(True)/ECE,
the first longitudinal receipt.

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
