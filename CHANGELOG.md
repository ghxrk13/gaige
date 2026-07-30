# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Measured numbers are quoted with the instrument that produced them, because a number without its
instrument is not a result. Where a change altered what gaige *measures*, it says so explicitly.

## [Unreleased]

The provenance/trust release, in preparation.

### Added

- **`gaige verify`: provenance evidence sweep** (`gaige/provenance.py`): deterministic
  checks with honest negatives: C2PA Content Credentials (including the standardized
  generative-AI source-type declaration), the publicly checkable dwtDct image watermark,
  and keyed text schemes reported as NEEDS_KEYS rather than pretended at. Emits evidence
  statuses, never an AI-likeness score. The honesty rules are structural: a watermark
  negative may read ABSENT only when a probe payload embedded into that same image
  survives a round trip proving the carrier could have held the mark, otherwise the
  result is INCONCLUSIVE (on an unfavourable carrier the decoder returns silence either
  way); C2PA absence is a typed signal measured against c2pa-python 0.37.2 (only
  ManifestNotFound reads ABSENT; unsupported file types read INCONCLUSIVE; validation
  failures stay ERROR, never "no manifest"); image-only scheme rows appear only on image
  sweeps; carriers below the watermark scheme's 256x256-pixel contract read INCONCLUSIVE,
  never a crash. Every result states what a negative from it means, and every sweep
  carries the verifier fingerprint including which optional libraries were absent. Live
  arms verified on real fixtures: a locally signed C2PA asset (validation state Valid,
  generative source declared), a watermark written by the real ecosystem encoder and
  recovered from disk, honest ABSENT with a passing probe, INCONCLUSIVE below the size
  contract, and a corrupted manifest reading ERROR.
- **Vendored dwtDct codec** (`gaige/_dwtdct.py`): invisible-watermark 0.2.0 imports torch
  at package-import time for an unrelated scheme, so its torch-free dwtDct path is
  unreachable without installing torch; the roughly one hundred lines gaige needs are
  vendored (MIT attribution in NOTICE.md) and the codec becomes part of the fingerprinted
  instrument. Bit-format compatibility is cross-validated against the real library in
  both directions on three carriers, including bit-identical encoder output arrays; a
  real-encoder golden fixture (`tests/data/`) pins ecosystem compatibility and
  numeric-stack drift in one byte comparison, with proven teeth (a self-consistent codec
  mutation greens every roundtrip test and reds only the golden). The install remedy is
  now actually curative: PyWavelets + opencv-python-headless, or the new `gaige[verify]`
  extra covering both sweep arms in one step.
- **Measured per-instrument memory floors with a deliberate escape hatch**
  (`gaige/memfloor.py`; the 0.0.2 acceptance finding): the floor the loaders enforce and
  the needs `gaige plan` prints are now the same single-sourced numbers and cannot
  disagree. Configurations with a measured receipt get their measured floor (which raised
  falcon-7b fp16 to its measured 13.7 GB; the flat 8.0 under-protected it), unmeasured
  configurations keep a conservative default rather than a guess, and `--min-free-gb` on
  `gaige run` is the always-winning deliberate override. Both refusal messages now name
  the floor's provenance and the remedy.
- **"What the aggregate hides"** (`docs/what-the-aggregate-hides.md`): the first RAID
  slice receipt written up as a short note. One instrument (Fast-DetectGPT, falcon-7b,
  4-bit) reads TPR@1%FPR 86.0% on its reference corpus (hc3-mini n=100 seed=17), 61.5%
  aggregate on the harder raid g2x d2x a2 slice, and inside that aggregate greedy 87.6%
  vs sampled 39.7%: a 47.9-point spread no aggregate row can show. Conformal a=.005
  refused at slice n by construction. The claims-policy tests hold the note to report
  bar: every number beside its instrument, blocked claims swept.

### Changed

- CI pins the ruff lint rule selection explicitly: ruff 0.16.0 widened its default rule
  set, which would have turned the unpinned gate red repo-wide on its next run. Lint
  behavior no longer moves when the linter's defaults do.

## [0.0.2] - 2026-07-29

### Added

- **`gaige export`: receipts as public site data.** Exports a report as one self-contained
  JSON document (schema `gaige-receipt-export/1`) joining every statistic with the
  instrument fingerprint that produced it and a stranger-runnable reproduce command, plus
  a rebuilt `index.json` (schema `gaige-export-index/1`). Nothing is recomputed: values
  are copied verbatim from results.json, so the analyze replay gate covers exports
  transitively. Redaction is structural and fail-closed: an absolute path, an IP address,
  or a URL off the public host allowlist refuses the whole export and names the offending
  field, and the host block is projected down to os family, architecture, and device.
  Reports without env.json refuse: INSTRUMENT UNKNOWN receipts are not exportable,
  mechanically. Exports are deterministic (LF bytes, no export-time clock), the format is
  golden-pinned, and the golden is held to the same blocked-claims scan as every shipped
  document. Tests: 182 → 200.

### Fixed

- csv field-size probe no longer overflows the Windows C long (found by the first public
  CI run; `corpus_raid` on Windows).
- `tools/check_consistency.py` reads pyproject on py3.10 via the tomli fallback (same CI
  run).

### Changed

- The upstream `torch_dtype` rename notice from newer transformers stays out of a
  stranger's terminal during model load (found by the 0.0.1 post-publish pass). The mute
  is message-scoped and context-scoped on both channels the notice can arrive on; the
  load path itself is untouched on every supported version, verified by replaying a 0.0.1
  reference receipt bit-identically under 0.0.2.
- README quickstart now says that CPU saturates hc3-mini (auto-selected gpt2-large,
  AUROC 1.0000) and points at the calibrated GPU receipt (AUROC 0.9720, Fast-DetectGPT on
  falcon-7b, 4bit, hc3-mini(n=100,seed=17)) as the honest example.

## [0.0.1] - 2026-07-26 (release-ready cut; PyPI publish rides the public flip)

The first real release, cut and verified 2026-07-26. PyPI still holds the `0.0.0`
name-reservation stub until the public flip: publishing is a separate, gated act.

**Release verification (transcripts in the ops repo's release receipt):** 167 tests green ·
ruff format clean · wheel + sdist built and twine-check PASSED (wheel sha256 `49161069…`) ·
clean-environment installs on Python 3.10 / 3.12 / 3.13 · core install proven torch-free
with the analyze lane replaying a pinned report bit-identically from the wheel · the
documented quickstart command ran verbatim end-to-end on a pristine py3.10 CPU-only
container (200 texts, 104 s, full receipt including the α=.005 conformal refusal).

### Added (2026-07-27, launch prep: the import surface and the pointers)

- **A public Python API.** `import gaige` now carries the supported library surface: the
  calibrate/conformal/analyze spine re-exported with a 25-name `__all__`, including the
  refusal types (`CorpusTooSmall`, `InsufficientCalibration`, `NotAReport`) so a caller can
  catch them where the functions live. Enforced, not remembered: `tests/test_public_api.py`
  pins the exact list, proves every promised name resolves, and asserts in a fresh
  interpreter that `import gaige` never imports torch. Everything outside `__all__` stays
  internal. (Fixed en route: the re-exports exposed a circular import because `analyze.py` reads
  `from . import __version__` at module scope, so the version is now assigned before the
  re-exports.)
- **PyPI metadata for strangers**: `[project.urls]`, which lists Homepage (gaige.dev), Documentation,
  Repository, Issues, Changelog, and Funding (ko-fi.com/gaigedev, rendered natively by
  PyPI), plus trove classifiers and keywords. The project page stops being bare.
- **README rewritten for someone who has never seen the repo**: `pip install gaige`
  quickstart (the old `pip install -e .` line was a from-source habit), the CPU-torch note
  (on CPU-only machines the `[gpu]` extra pulls the multi-gigabyte CUDA torch; install the
  CPU build first and the extra leaves it alone), a Python API example (sized to clear the
  50-per-class floor, because the first draft raised `CorpusTooSmall`, which running the
  example caught), and the gaige.dev / PyPI / issues / fuel pointers.
- **SUPPORT.md** (best-effort, receipts-first, no SLAs; registered in the claims-policy
  shipped-docs list so its prose is held to the same bar) and **.github/FUNDING.yml**
  (`ko_fi: gaigedev`, which lights the repo Sponsor button). Tests: 176 → 182.

### Added (2026-07-26, release night: the CLI refusal surface)

- **Refusals print as plain errors, never tracebacks.** Found by the outside-user pass:
  a core-only install running a scoring command now gets
  `error: scoring needs the GPU extra (torch is not installed)` with the exact
  `pip install "gaige[gpu]"` remedy and a note that analysis commands run without it,
  instead of a raw `ModuleNotFoundError`. Malformed corpora, missing files, and every
  honest refusal (`ValueError`/`RuntimeError` family) surface as one-line errors with
  exit code 2; tracebacks remain for genuinely unexpected bugs. Tests: 165 → 167.

### Added

- **"Where gaige sits" positioning section** in the README (accepted at the 2026-07-27
  launch review): the 3-0-verified research fold (default-threshold
  miscalibration (RAID), cross-dataset collapse, global-threshold flagships, conformal
  calibration with its printed TPR price, and subgroup/decoding disparities) beside our own
  receipts (reference corpus, Binoculars global-threshold demo, first RAID slice). A
  claims-policy test now enforces the fold's blocked list across every shipped document and
  requires each number in the section to name its paper or its instrument: the
  no-overclaiming rule as a test, not an editorial habit.
- **Probe-authoring toolchain** (`gaige probe new` + `gaige probe lint`): scaffolds a
  probe-set template with a sidecar manifest declaring the signed authoring decisions
  (2026-07-22: nem grading version, pre-registered greedy temperature-0 decoding, and a
  hashed control linkage scored by option-logprob argmax) and lints probe sets against
  them mechanically: per-probe `source`/`source_date`/`authored` provenance required, every
  `source_date` must post-date the declared training cutoff, answers must survive `nem-1`
  normalization, placeholders and stale control hashes refuse. `gaige probe run` now
  enforces a present manifest: a set failing its own lint, or a run whose decoding
  contradicts the declared greedy block, is refused with the remedy named. Manifest-less
  sets run unchanged (declarations reported as unenforced). The manifest is a sidecar so
  declaration edits never move the probe-file sha256 or the frozen vintage hashes.
- **RAID corpus adapter** (`gaige corpus prepare-raid`): seeded slices of the RAID benchmark
  (Dugan et al., ACL 2024) with generator/domain/attack/decoding carried as subgroup axes on
  every row. Two sources: datasets-server paging (no 11.8 GB download; dataset revision sha
  recorded) or a locally downloaded RAID csv (streamed, stdlib `csv`, per-cell reservoir
  sampling). No new dependencies; no third-party text enters the repo: slices land in the
  gitignored `corpora/`, and the adapter's tests fabricate synthetic rows in RAID's verified
  column shape. Batched scoring stays banked per the map until RAID-scale runs demand it.
- `gaige analyze`: re-derive AUROC, thresholds, CIs and a full report from scores that already
  exist. No model, no GPU. `run` and `analyze` share one `compute_results`, so a replay cannot
  disagree with the original run.
- **CPU support.** `--device auto|cuda|cpu` with `--quant fp32`; `auto` prefers CUDA and records
  the fallback. First CPU receipt: gpt2/fp32/cpu on hc3-mini(n=50), **AUROC 0.9908**, 100 samples
  in ~11 s.
- **Resumable scoring.** Scores are appended and flushed per sample; `--resume <run-dir>`
  continues an interrupted run. A resume is refused if the corpus, model, quant, `max_tokens`,
  device or library versions changed, because continuing would interleave two instruments into one
  report. Verified end-to-end: SIGKILL at 185/300 then resumed, result bit-identical to the
  uninterrupted run.
- Device-aware default model when `--model` is omitted (falcon-7b/4bit on CUDA, gpt2-large/fp32
  on CPU), recorded on the receipt as `model_auto_selected`. Ergonomic default only: which model
  detects best is an empirical question gaige should answer with a published comparison.
- `instrument_mismatches()`: device now counts toward instrument identity alongside library
  versions, so CUDA-calibrated thresholds cannot be silently applied to CPU scores.
- CI: pytest, import hygiene, CLI smoke and `ruff` on **Linux and Windows** across Python
  3.10/3.12/3.13.
- `tools/check_consistency.py`: identity-drift check (version, headers, description alignment,
  required docs, no heavy imports at module scope), run in CI.
- `SECURITY.md`: trust boundaries, the no-persistence property, and an explicit list of what
  gaige does *not* defend against.
- `RUNBOOK.md`: operate gaige with no chat session; exact commands, reference numbers, and the
  traps that have actually bitten.
- Tests: 6 → 47. Including the pinned reference detection instrument (AUROC 0.9720 and both
  thresholds, exact, no GPU needed) so detection cannot rot while other capabilities are built.
- **Conformal thresholds, wired into every report** (2026-07-22, after adversarial review
  against arXiv:2505.05084, review of record:
  the private research notes, conformal/subgroups review 2026-07-22). Split-conformal order
  statistic verified against the paper's construction (quantile, strict-inequality flag rule,
  minimum-n bound, all exact). The report states the guarantee as it is: **marginal over
  calibration draws, under exchangeability**, and prints the exact conditional dispersion
  (Beta(n+1−k, k) mean ± sd) instead of any in-sample "achieved" rate, which was removed as
  vacuous ((n−k)/n by construction). `gaige score` gains guarantee-backed conformal verdicts.
- **Subgroup-stratified receipts, wired into every report** (same review). Length buckets
  always; metadata axes when the corpus carries them on every row. Every reported rate now
  carries a 95% bootstrap interval (reusing `calibrate.bootstrap_ci`, generalized to
  single-class resampling with a bit-identical RNG path for existing two-class calls). The
  n<20 floor now actually WITHHOLDS rates (count shown) instead of printing them with a flag.
  Max-disparity line = FairOPT's Δ_FPR (arXiv:2502.04528 Eq. 8), so the number is comparable
  to the literature.
- **Base-rate arithmetic in every report**: FPR × volume = expected wrongly flagged per year
  (`--harm-volume`, default Vanderbilt's published 75,000), plus PPV at illustrative
  prevalences. The previously dead `base_rate_harm`/`ppv` are now wired; `base_rate_harm`
  lost a dangling `ai_prevalence` parameter that did nothing.
- **scores.csv gains `n_words` and `meta` columns** (derived count + corpus metadata, never
  text), so `gaige analyze` can replay subgroup receipts. Old score sets and partials are
  read tolerantly; their reports say subgroup receipts are unavailable rather than guessing.
- Tests: 47 → 64. The conformal tests assert the STATISTICAL PROPERTY by simulation
  (marginal bound across alphas, conditional-dispersion match to the Beta law, refusal
  boundaries) and their teeth are proven: a deliberately broken order statistic turns the
  suite red (recorded in the review doc), and an in-suite test demonstrates the detection
  margin permanently. Subgroup tests inject a KNOWN length disparity and require the
  interval to bracket the injected truth, and the refusal floor to actually refuse.

### Added (2026-07-22, late: ollama provider and `gaige plan`)

- **Ollama provider** (`gaige/providers/ollama.py`): probe runs against any locally served
  model, attestation EARNED the artifact way where possible. Ollama's store is a
  content-addressed CHAIN: the `/api/tags` digest names the MANIFEST, the manifest's
  model-layer digest names the weights blob, so gaige re-hashes BOTH with its own hands:
  manifest hash must equal the server digest AND weights-blob hash must equal the
  manifest's declared layer digest → `verified`. Digest reported but store unreadable
  (remote endpoint, permissions) → `self-reported`; no digest → `opaque`; any mismatch
  anywhere is reported loudly and never upgraded. COMPLETE only (no stable full-vocab
  logprob API, so MC control and P(True) stay on local-hf, same honesty as llamacpp).
  First live receipt: qwen2.5:7b-instruct on the bench, chain `verified` (weights sha256
  `2bada8a745…`, 4.68 GB), 20 probes scored, series `10c246457f8d`.
- **`gaige plan`**: what can THIS machine run, at what measured cost, with what
  attestation. Inspects CPUs/RAM/CUDA-free-VRAM/ollama/llama-server, prints fits-now
  verdicts against measured floors and runtime anchors that each NAME their receipt.
  Deliberately no quality column, because separation is a one-instrument-on-one-corpus property
  that lives in receipts (the legend says so on every table, and a test enforces it).
- Tests: 109 → 123 (9 ollama: full chain / manifest mismatch / weights mismatch / missing
  blob / unreadable store / opaque / decoding-options mapping / serializable fingerprint;
  5 plan: injected environments incl. cuda-tight and cpu-only, legend enforced).

### Added (2026-07-22, Burst 2b: detector #2 and the quantization A/B; the public line)

- **Binoculars** (`gaige/detectors/binoculars.py`), construction verified against the
  released implementation before writing a line (ppl under the performer / x-ppl of
  softmax(observer) against the performer's log-probs; gaige emits the NEGATED ratio to
  keep higher-is-AI, documented in score_semantics). Tokenizer-identity refusal; BOTH
  models quant-verified on the receipt; two-model VRAM floor. The paper's global
  thresholds are deliberately unused. First receipt (falcon pair, 4bit/cuda, reference
  corpus): **AUROC 0.9992, TPR 97% @1%FPR, conformal α=.01 → 95%**, and the paper's
  global thresholds measured at **16%** (accuracy-mode) and **3%** (low-FPR-mode) FPR on
  the same corpus: the receipts gap, demonstrated in-house.
- **Quantization A/B receipts** (gpt2-large, one corpus, three instruments): fp32-cpu and
  fp16-cuda agree on the 1%-FPR threshold to FOUR decimals (2.0461 vs 2.0462); **4-bit
  moves it ~10% (→2.2580)** and costs a point of TPR. Quantization is an instrument
  parameter, measured. **The falcon-7b fp16 arm was completed same night** in an
  operator-called daemon window (clean stop → run → restore; canary pre=post 0.0946, |Δ|=0):
  **4-bit shifts the 7B flagship's 1%-FPR threshold from 1.9540 to 2.1229 (~8.6%)**, AUROC
  CIs overlapping (no separation claim). The finding holds at both scales.
- Tests: 105 → 109 (tokenizer-mismatch refusal via stubs, quant rules, protocol
  conformance, and a real-math CPU smoke on the gpt2+distilgpt2 shared-tokenizer pair,
  deterministic, negated, both models fingerprinted).

### Added (2026-07-22, M5: drift monitors; the last unbuilt battery instrument)

- **`gaige/monitors.py`**: the detector-comparison arm. Per-interval **conformal alarms**
  reuse `conformal_threshold` on zero-drift reference values (Day-0 replicates) and carry
  the marginal finite-sample false-alarm bound, stated with the α×looks expected-false-alarm
  arithmetic; **Page-Hinkley** and **CUSUM** run with tuning constants recorded and NO
  guarantee claimed (cumulative statistics, per the honesty scoping of the longitudinal-measurement spec §5;
  conformal test martingales cited as the principled extension, unbuilt, unclaimed).
  `monitors.evaluate` scores any monitor against a known onset: detection latency + false
  alarms, M5's per-technique scorecard.
- **`gaige series watch`** replays a registered series through the panel and writes
  `monitors-report.md` beside the series. Direction-aware (accuracy alarms down, gap alarms
  up). Practical note surfaced by the live run: α=0.2 conformal alarms need FOUR zero-drift
  reference intervals: run Day-0 with `--replicates 4` to enable them from the start.
- The `InsufficientCalibration` message noun is now context-parametric ("zero-drift
  reference intervals" in monitor context vs "human calibration samples" in detector
  calibration): same refusal, right words.
- Tests: 98 → 105 (injected shifts caught with latency ≤ 2, flat and sub-threshold series
  stay quiet; specificity asserted, not assumed; refusal paths; exact scorecard
  arithmetic; direction orientation).

### Added (2026-07-22, apparatus Phase D: the real-model e2e; the apparatus burst is COMPLETE)

- `probes/demo.jsonl`: a committed, clean-room, 20-probe general-knowledge demo set in the
  signed schema (two vintages, aliases, provenance dates), the documented material for
  smoke tests and the RUNBOOK walk-through.
- First REAL longitudinal series, run live (receipt of record in the private notes): Qwen2.5-1.5B
  q4_k_m served by llama.cpp b10091 on CPU, attestation **verified** (GGUF sha256 matched
  the server's reported artifact: the sharpened attestation design working on real
  infrastructure, first try); Day-0 replicates measured the variance bound at ±0.0% (served
  greedy decoding is deterministic: measured, not assumed); follow-up run within variance;
  a temperature change forked its own series live. Qwen is the staged non-gated fallback
  for the gated Llama-3.2-1B the longitudinal spec names.
- `registry/` added to .gitignore (measurement state is machine-local; receipts of record
  are copied into the private notes deliberately).

### Added (2026-07-22, apparatus Phase C: M3, calibration drift)

- **P(True)** (`gaige/ptrue.py`): Kadavath-style self-assessment read from logits via the
  provider option_logprobs capability, a deterministic forward pass, no sampling. The prompt
  template is versioned (`ptrue-1`) and sha256-hashed into the instrument fingerprint;
  resuming a run with `--ptrue` toggled refuses, and the registry FREEZES the template per
  series once M3 has been measured (a changed template refuses at registration, because gaps
  across templates are not comparable). Toggling M3 on/off does not fork a series: the M1
  instrument is untouched.
- **Probability calibration** (`gaige/probcal.py`, deliberately NOT `calibrate.py`; that
  module is decision thresholds, this one is probability calibration, and the name overlap
  is a documented trap): ECE with the per-bin table, bootstrap CI over (confidence,
  correctness) pairs, and the confidence-accuracy gap. Bin count fixed per series.
- `gaige probe run --ptrue` wires M3 into probe receipts (per-vintage table) and series
  reports (gap beside each accuracy cell).
- Tests: 89 → 98. The ECE hand-case is exact (teeth proven: a one-bin off-by-one turned
  the suite red before being reverted); a controllable fake self-assessor makes every M3
  number hand-computable through the full runner; an injected 0.25 confidence inflation is
  recovered as ECE ≈ 0.25 by property test.
- First live M3 measurement (gpt2, toy probes): accuracy 0%, mean P(True) 79%, gap +79%:
  the failure mode the longitudinal apparatus instruments, produced by the smoke test.

### Changed (2026-07-22, analysis-layer performance: an instrument change, stated)

- **Bootstrap vectorized in pure numpy** (no new dependency; the deliberate alternative to a
  numba suggestion, which was rejected: an LLVM dependency against the numpy-only core, and
  parallel reductions reorder floats). Resample index matrices are drawn in one rng call per
  class; a new `calibrate.proportion_ci` fully vectorizes the indicator-mean hot path now
  used for TPR-at-threshold, per-subgroup rates, and probe accuracy. **Measured** (n_boot
  =1000): auroc-CI 0.17s→0.04s at n=200 and 7.93s→0.86s at n=10k (9.2×); proportion path
  0.02s→0.001s (20×). The auroc rows still sort per resample: the next lever at RAID scale.
- **Honesty accounting:** the vectorized draw is a different random stream, so bootstrap CI
  values shift slightly, whilst AUROC and every threshold are deterministic and verified UNCHANGED
  (the re-pin script hard-asserts them before touching the fixture). Reference pin
  re-measured: auroc_ci [0.9458, 0.9938] → **[0.9448, 0.9929]**; both tpr_ci values landed
  identical. The auroc midrank tie-loop was also vectorized: that one is exact math,
  asserted value-identical against the old implementation (ties included) in tests.
- Tests: 87 → 89.

### Added (2026-07-22, apparatus Phase B)

- **The run registry** (`gaige/registry.py`), the validity backbone: runs land in a series
  keyed by the hash of the instrument identity (provider identity minus attestation prose +
  decoding block + grading version + cutoff + gaige version; identity is order-independent
  and deliberately excludes the probe-file hash). A changed instrument **forks a new
  series**, never mixes; within a series, **vintages are frozen**: an edited vintage label
  is refused by name, new labels are welcome (the longitudinal design). `--register` on
  `gaige probe run`, plus `gaige series list/show`.
- **The Day-0 replicate protocol**: `--replicates k` runs the set k times and the series
  report derives the per-vintage **run-variance bound** (2σ across replicates, measured:
  ±0.0% on the greedy in-process pipeline, printed as a result rather than an assumption).
  Every later run is flagged within-variance or BEYOND the bound: the pre-registered
  movement rule from the longitudinal spec, mechanical.
- The series report carries the fingerprint-constancy statement the spec pre-wrote for the
  chapter: constancy is asserted mechanically, refused rather than compared.
- Tests: 81 → 87 (identity order-independence + decoding sensitivity, fork-not-mix,
  frozen-vintage refusal, new-vintage growth, bound measurement + movement flags).

### Added (2026-07-22, apparatus Phase A)

- **The probe runner**, the acquisition layer the longitudinal spec's M1 metric needs:
  `gaige probe run` takes a dated probe set (JSONL with per-probe provenance:
  source/source_date/vintage) through a model provider to a graded receipt: accuracy per
  vintage with bootstrap CIs (single-class reuse of `calibrate.bootstrap_ci`) and the
  per-vintage **post-cutoff share** against `--cutoff`, so "the probes post-date the model"
  is demonstrated, not asserted. Crash-safe + resumable (runstate pattern, parametrized);
  a resume refuses if ANY pinned fingerprint field changed (provider identity, decoding
  block, grading version, probe-set hash, cutoff).
- **Providers with graded attestation** (`gaige/providers/`): `local-hf` (in-process,
  attestation `verified`, greedy/seeded completion + per-option continuation logprobs for
  the MC control path) and `llamacpp` (OpenAI-compat `/v1`; attestation EARNED:
  `verified` with a `--gguf` sha256 matching the server's reported artifact,
  `self-reported` from `/props`, `opaque` otherwise; declares COMPLETE only until its
  logprob path is verified against the in-process reference). Capability declarations with
  refusal-naming-what's-missing; **prompts never leave the machine for a non-local endpoint
  without `--allow-remote-text`**. Plus `gaige providers` and `gaige test-connection`.
- **Deterministic grading** (`gaige/grading.py`): versioned normalized-exact-match pipeline
  (`nem-1`: NFKC → casefold → strip punctuation → collapse whitespace → drop one leading
  article) + authored aliases; MC argmax over option logprobs with conservative tie
  handling (a tie is not an answer). The grading version is part of the fingerprint.
- **Probe sets** (`gaige/probes.py`): schema-validated loader (errors name the row and the
  remedy), sha256 fingerprint, per-vintage counts, post-cutoff arithmetic.
- Tests: 64 → 81 (grading edges incl. unicode/ligature/casefold; probe schema refusals;
  runner resume, both refusal paths, remote opt-in, capability naming, all via an
  injectable fake provider, no model or network in CI).

### Changed

- **The CPU default is now measured, not asserted** (2026-07-22): six candidates
  (distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl, gpt-neo-1.3B) on
  hc3-mini(n=100,seed=17), fp32/cpu, fixed protocol. gpt2-large was the only candidate with
  perfect separation (AUROC 1.0000; TPR 100% at 1%-FPR and at conformal α=.01) and did it at
  0.64 s/sample, dominating both larger candidates. `DEFAULT_MODEL` unchanged; its comment
  now cites the measurement. Selection on one corpus under one protocol, not a detector
  ranking; separation was measurably non-monotone in model size (gpt2-large > gpt2-xl here).

### Fixed (2026-07-22, second pass)

- **The reproduce command could silently swap instruments.** A `--device auto` run recorded
  the literal string "auto" in its reproduce command, so re-running the receipt on a
  different machine would resolve to a different device (a different instrument) with no
  warning. Reproduce lines now record the RESOLVED device (verified live: an auto run's
  receipt says `--device cuda`). Policy alongside it in the RUNBOOK: `auto` is for
  exploratory runs; pre-registered runs pin `--device` explicitly.

### Changed (original 0.0.1 notes)

- **Redefined.** "Calibration + receipts for AI-text detectors" → **"calibration and drift
  receipts for AI measurement."** Text detection was the definition while it was the only
  application; it is now one of two, alongside telling whether an observed change is in the system
  measured or in the measuring pipeline. Summary, README, package docstring and every source
  header moved in the same commit, and the consistency check now enforces that they stay together.
- Receipt rendering describes only what was actually recorded, because a CPU load cannot count
  `Linear4bit` modules, so it says so rather than raising or implying a verification that never
  happened.
- Detector warnings moved from `print` to `logging`, so a library consumer can control them.

### Fixed

- **Report writing crashed on Windows.** `write_text` used the platform default codec (cp1252),
  which cannot encode the arrow in the caveats section, so gaige could not produce a report on
  Windows at all. All report IO is now explicit UTF-8, console output is ASCII, and a test asserts
  every artifact decodes as UTF-8.
- The repo did not pass its own lint job; `ruff` is now configured to the codebase's ~100-column
  style and all gates pass.

### Security

- The no-persistence claim for `gaige score` is now **verified rather than asserted**: the
  detector is injectable, and four tests prove a canary phrase reaches no file on disk, that no
  file is created or modified, and that the result never echoes the scored text. Confirmed the
  tests fail when a leak is deliberately injected.

### Removed

- Dead `version_mismatches` shim (nothing called it), a test whose assertion restated its own
  `except` clause, and the `detcal.egg-info` left over from the rename.

### Known limitations

- Group-adaptive thresholds ship as REPORTING (per-subgroup rates with intervals), not as
  per-group thresholds. The guarantee-backed version (Mondrian conformal per bucket) needs
  ≥99 human samples per bucket at α=0.01 and is deliberately deferred rather than half-built.
- (A prior note here said longitudinal drift measurement was not built; the 2026-07-22
  apparatus entries above (registry, probe runner, P(True)/ECE, monitors) superseded it.)

## [0.0.0] - 2026-07-21

Name reservation on PyPI. No source, no functionality.
