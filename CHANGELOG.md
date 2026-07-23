# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/); versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html).

Measured numbers are quoted with the instrument that produced them, because a number without its
instrument is not a result. Where a change altered what gaige *measures*, it says so explicitly.

## [Unreleased] — 0.0.1, in development

Not released. PyPI currently holds a `0.0.0` name-reservation stub only; the metadata for the
first real release is staged in `pypi-stub/`.

### Added

- `gaige analyze` — re-derive AUROC, thresholds, CIs and a full report from scores that already
  exist. No model, no GPU. `run` and `analyze` share one `compute_results`, so a replay cannot
  disagree with the original run.
- **CPU support.** `--device auto|cuda|cpu` with `--quant fp32`; `auto` prefers CUDA and records
  the fallback. First CPU receipt: gpt2/fp32/cpu on hc3-mini(n=50), **AUROC 0.9908**, 100 samples
  in ~11 s.
- **Resumable scoring.** Scores are appended and flushed per sample; `--resume <run-dir>`
  continues an interrupted run. A resume is refused if the corpus, model, quant, `max_tokens`,
  device or library versions changed — continuing would interleave two instruments into one
  report. Verified end-to-end: SIGKILL at 185/300 then resumed, result bit-identical to the
  uninterrupted run.
- Device-aware default model when `--model` is omitted (falcon-7b/4bit on CUDA, gpt2-large/fp32
  on CPU), recorded on the receipt as `model_auto_selected`. Ergonomic default only — which model
  detects best is an empirical question gaige should answer with a published comparison.
- `instrument_mismatches()` — device now counts toward instrument identity alongside library
  versions, so CUDA-calibrated thresholds cannot be silently applied to CPU scores.
- CI: pytest, import hygiene, CLI smoke and `ruff` on **Linux and Windows** across Python
  3.10/3.12/3.13.
- `tools/check_consistency.py` — identity-drift check (version, headers, description alignment,
  required docs, no heavy imports at module scope), run in CI.
- `SECURITY.md` — trust boundaries, the no-persistence property, and an explicit list of what
  gaige does *not* defend against.
- `RUNBOOK.md` — operate gaige with no chat session; exact commands, reference numbers, and the
  traps that have actually bitten.
- Tests: 6 → 47. Including the pinned reference detection instrument (AUROC 0.9720 and both
  thresholds, exact, no GPU needed) so detection cannot rot while other capabilities are built.
- **Conformal thresholds, wired into every report** (2026-07-22, after adversarial review
  against arXiv:2505.05084 — review of record:
  `private-notes/research/conformal-subgroups-review-2026-07-22.md`). Split-conformal order
  statistic verified against the paper's construction (quantile, strict-inequality flag rule,
  minimum-n bound, all exact). The report states the guarantee as it is: **marginal over
  calibration draws, under exchangeability** — and prints the exact conditional dispersion
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
- **scores.csv gains `n_words` and `meta` columns** (derived count + corpus metadata — never
  text), so `gaige analyze` can replay subgroup receipts. Old score sets and partials are
  read tolerantly; their reports say subgroup receipts are unavailable rather than guessing.
- Tests: 47 → 64. The conformal tests assert the STATISTICAL PROPERTY by simulation
  (marginal bound across alphas, conditional-dispersion match to the Beta law, refusal
  boundaries) and their teeth are proven: a deliberately broken order statistic turns the
  suite red (recorded in the review doc), and an in-suite test demonstrates the detection
  margin permanently. Subgroup tests inject a KNOWN length disparity and require the
  interval to bracket the injected truth, and the refusal floor to actually refuse.

### Changed (2026-07-22, analysis-layer performance — an instrument change, stated)

- **Bootstrap vectorized in pure numpy** (no new dependency; the deliberate alternative to a
  numba suggestion, which was rejected: an LLVM dependency against the numpy-only core, and
  parallel reductions reorder floats). Resample index matrices are drawn in one rng call per
  class; a new `calibrate.proportion_ci` fully vectorizes the indicator-mean hot path now
  used for TPR-at-threshold, per-subgroup rates, and probe accuracy. **Measured** (n_boot
  =1000): auroc-CI 0.17s→0.04s at n=200 and 7.93s→0.86s at n=10k (9.2×); proportion path
  0.02s→0.001s (20×). The auroc rows still sort per resample — the next lever at RAID scale.
- **Honesty accounting:** the vectorized draw is a different random stream, so bootstrap CI
  values shift slightly — AUROC and every threshold are deterministic and verified UNCHANGED
  (the re-pin script hard-asserts them before touching the fixture). Reference pin
  re-measured: auroc_ci [0.9458, 0.9938] → **[0.9448, 0.9929]**; both tpr_ci values landed
  identical. The auroc midrank tie-loop was also vectorized — that one is exact math,
  asserted value-identical against the old implementation (ties included) in tests.
- Tests: 87 → 89.

### Added (2026-07-22, apparatus Phase B)

- **The run registry** (`gaige/registry.py`) — the validity backbone: runs land in a series
  keyed by the hash of the instrument identity (provider identity minus attestation prose +
  decoding block + grading version + cutoff + gaige version; identity is order-independent
  and deliberately excludes the probe-file hash). A changed instrument **forks a new
  series**, never mixes; within a series, **vintages are frozen** — an edited vintage label
  is refused by name, new labels are welcome (the longitudinal design). `--register` on
  `gaige probe run`, plus `gaige series list/show`.
- **The Day-0 replicate protocol**: `--replicates k` runs the set k times and the series
  report derives the per-vintage **run-variance bound** (2σ across replicates, measured —
  ±0.0% on the greedy in-process pipeline, printed as a result rather than an assumption).
  Every later run is flagged within-variance or BEYOND the bound: the pre-registered
  movement rule from the longitudinal spec, mechanical.
- The series report carries the fingerprint-constancy statement the spec pre-wrote for the
  chapter: constancy is asserted mechanically, refused rather than compared.
- Tests: 81 → 87 (identity order-independence + decoding sensitivity, fork-not-mix,
  frozen-vintage refusal, new-vintage growth, bound measurement + movement flags).

### Added (2026-07-22, apparatus Phase A)

- **The probe runner** — the acquisition layer the longitudinal spec's M1 metric needs:
  `gaige probe run` takes a dated probe set (JSONL with per-probe provenance:
  source/source_date/vintage) through a model provider to a graded receipt — accuracy per
  vintage with bootstrap CIs (single-class reuse of `calibrate.bootstrap_ci`) and the
  per-vintage **post-cutoff share** against `--cutoff`, so "the probes post-date the model"
  is demonstrated, not asserted. Crash-safe + resumable (runstate pattern, parametrized);
  a resume refuses if ANY pinned fingerprint field changed (provider identity, decoding
  block, grading version, probe-set hash, cutoff).
- **Providers with graded attestation** (`gaige/providers/`): `local-hf` (in-process,
  attestation `verified`, greedy/seeded completion + per-option continuation logprobs for
  the MC control path) and `llamacpp` (OpenAI-compat `/v1`; attestation EARNED —
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
  runner resume, both refusal paths, remote opt-in, capability naming — all via an
  injectable fake provider, no model or network in CI).

### Changed

- **The CPU default is now measured, not asserted** (2026-07-22): six candidates
  (distilgpt2, gpt2, gpt2-medium, gpt2-large, gpt2-xl, gpt-neo-1.3B) on
  hc3-mini(n=100,seed=17), fp32/cpu, fixed protocol. gpt2-large was the only candidate with
  perfect separation (AUROC 1.0000; TPR 100% at 1%-FPR and at conformal α=.01) and did it at
  0.64 s/sample, dominating both larger candidates. `DEFAULT_MODEL` unchanged; its comment
  now cites the measurement. Selection on one corpus under one protocol — not a detector
  ranking; separation was measurably non-monotone in model size (gpt2-large > gpt2-xl here).

### Fixed (2026-07-22, second pass)

- **The reproduce command could silently swap instruments.** A `--device auto` run recorded
  the literal string "auto" in its reproduce command, so re-running the receipt on a
  different machine would resolve to a different device — a different instrument — with no
  warning. Reproduce lines now record the RESOLVED device (verified live: an auto run's
  receipt says `--device cuda`). Policy alongside it in the RUNBOOK: `auto` is for
  exploratory runs; pre-registered runs pin `--device` explicitly.

### Changed (original 0.0.1 notes)

- **Redefined.** "Calibration + receipts for AI-text detectors" → **"calibration and drift
  receipts for AI measurement."** Text detection was the definition while it was the only
  application; it is now one of two, alongside telling whether an observed change is in the system
  measured or in the measuring pipeline. Summary, README, package docstring and every source
  header moved in the same commit, and the consistency check now enforces that they stay together.
- Receipt rendering describes only what was actually recorded — a CPU load cannot count
  `Linear4bit` modules, so it says so rather than raising or implying a verification that never
  happened.
- Detector warnings moved from `print` to `logging`, so a library consumer can control them.

### Fixed

- **Report writing crashed on Windows.** `write_text` used the platform default codec (cp1252),
  which cannot encode the arrow in the caveats section — gaige could not produce a report on
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
- Longitudinal drift measurement (run registry, probe runner, ECE, Page-Hinkley/CUSUM) is **not
  built**. See `private-notes/longitudinal study-requirements-trace.md` for what is HAVE / PLANNED / GAP.

## [0.0.0] — 2026-07-21

Name reservation on PyPI. No source, no functionality.
