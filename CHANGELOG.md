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

### Changed

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

- `conformal.py` and `subgroups.py` are implemented but **unwired and untested** — they reach no
  report. Scheduled for adversarial review before being wired.
- Longitudinal drift measurement (run registry, probe runner, ECE, Page-Hinkley/CUSUM) is **not
  built**. See `private-notes/longitudinal study-requirements-trace.md` for what is HAVE / PLANNED / GAP.

## [0.0.0] — 2026-07-21

Name reservation on PyPI. No source, no functionality.
