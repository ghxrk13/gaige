# gaige runbook

Operate gaige without a chat session. Exact commands, what correct output looks like, and the
traps that have actually bitten. If something here is wrong or unclear, that is a bug in this file.

---

## 0. Which machine for what

| Machine | Role | Why |
|---|---|---|
| **bench** (Linux, RTX 5000 Ada 16 GB) | scoring runs, GPU work, apparatus | the only box with CUDA; the reference instrument lives here |
| **win-seat** (Windows, no GPU) | authoring, `analyze`, reading reports | no CUDA — scoring will fall back to CPU, analysis runs fine |
| **win-node** (Windows, always-on) | long jobs, cross-platform checking | second Windows box; catches Windows-only defects |

`analyze`, `corpora` and the tests need **only numpy + requests**. Scoring needs the GPU extra.

## 1. Install

**bench (the reference environment — do not casually upgrade it):**
```bash
cd ~/personal/gaige
./.venv/bin/python -m pytest tests/ -q          # expect: 64 passed
```
The pinned venv is transformers 4.49 / torch 2.13.0+cu130 / bnb 0.49.2 / cuda 13.0 / py 3.12.3.
**transformers must stay <5** — 5.14.1 was measured silently ignoring 4-bit config and loading
fp16, which changes scores. gaige refuses such a load, but the pin avoids the fight.

**the Windows seats (analysis only, no GPU):**
```
cd %USERPROFILE%\personal\gaige
python -m pip install -e .
python -m pytest tests/ -q                      # expect: 64 passed
```

## 2. Workflow A — calibrate a detector, then score documents

This is the original detection workflow. It is pinned in CI (`tests/test_reference_instrument.py`)
so it cannot rot while other capabilities get built.

### 2a. Calibrate (needs a model; GPU strongly preferred)

```bash
# reference instrument, on bench
./.venv/bin/python -m gaige.cli run --corpus hc3-mini --n 100 --seed 17 \
    --detector fast-detect-gpt --model tiiuae/falcon-7b --quant 4bit --device cuda
```

**Correct output ends with (measured 2026-07-22 on the pinned env):**
```
[receipt] reports/<timestamp>-fast-detect-gpt/report.md
[receipt] AUROC 0.9720 (CI 0.9458-0.9938)
[receipt] @FPR<=1%: thr=2.1229 FPRcal=1.000% (in-sample) TPR=86.0%
[receipt] @FPR<=5%: thr=1.8319 FPRcal=5.000% (in-sample) TPR=91.0%
[receipt] conformal a=0.05: thr=1.8468 TPR=90.0% (marginal FPR guarantee <= 0.05)
[receipt] conformal a=0.01: thr=2.4446 TPR=76.0% (marginal FPR guarantee <= 0.01)
[receipt] conformal a=0.005: refused (alpha=0.005 needs >= 199 human calibration samples, got 100. ...)
```
Those exact numbers are the reference. **If they differ, something about the instrument changed** —
check the fingerprint section of `report.md` before trusting anything downstream.

Read the two threshold families as: `FPRcal` rows are fitted on this sample (in-sample, no
guarantee); `conformal` rows carry a finite-sample guarantee that is **marginal over calibration
draws** — note the α=0.01 conformal threshold is deliberately stricter (2.4446 vs 2.1229) and
catches less (76% vs 86%): that gap is the price of an actual guarantee. The α=0.005 refusal at
n=100 is correct behavior, not an error. `report.md` also now carries per-subgroup rate tables
(rates below n=20 per class are withheld — counts speak instead) and a base-rate section
(`--harm-volume` sets your institution's yearly volume; default is Vanderbilt's published 75,000).

**No GPU?** It still works, with a smaller model:
```bash
python -m gaige.cli run --corpus hc3-mini --n 50 --model gpt2 --quant fp32 --device cpu --max-tokens 512
```
Measured on the bench's CPU: 100 samples in ~11 s, AUROC 0.9908. **This is a different instrument**
than the GPU one — its thresholds are not interchangeable (4.2276 vs 2.1229 at 1% FPR). The
receipt records `device: cpu` and gaige will warn if you try to use one report's thresholds in the
other's environment.

Omit `--model` entirely and gaige picks per device (falcon-7b/4bit on CUDA, gpt2-large/fp32 on CPU)
and records `model_auto_selected` on the receipt. The CPU default is a **measured selection**
(2026-07-22, six candidates on hc3-mini n=100 seed=17, fp32/cpu): gpt2-large was the only
candidate with perfect separation on that corpus (AUROC 1.0000; TPR 100% at 1%-FPR and at
conformal α=.01) at 0.64 s/sample, dominating gpt2-xl and gpt-neo-1.3B on both axes. One
corpus, one protocol — a default-selection receipt, not a detector ranking; each candidate's
run is reproducible with the command above plus `--model <name>`.

**Device policy (decided 2026-07-22):** `--device auto` is for exploratory runs — it prefers
CUDA, falls back to CPU loudly, and the receipt records the fallback. **Pre-registered or
scientific runs pin `--device` explicitly** (the run registry treats device class as
instrument identity, so a mid-series fallback forks the series). Either way the receipt's
reproduce command always carries the RESOLVED device, never "auto" — re-running a receipt can
never silently swap instruments.

### 2b. Score a document against that calibration

```bash
python -m gaige.cli score --report reports/<timestamp>-fast-detect-gpt/ --file draft.md
python -m gaige.cli score --report reports/<ts>-fast-detect-gpt/ --text "some text"
cat draft.md | python -m gaige.cli score --report reports/<ts>-fast-detect-gpt/
```

The document is **never written anywhere** — no log, no cache, no telemetry. That is tested
(`tests/test_privacy.py`), not merely promised.

Read the output as: the score, where it sits against the calibration corpus, and whether it crosses
each threshold. **A crossing is evidence, not a verdict.** Under 50 words the result is
noise-dominated in both directions and says so.

### 2c. Re-analyse without re-scoring (no GPU, any machine)

```bash
python -m gaige.cli analyze --report reports/<ts>-fast-detect-gpt/
python -m gaige.cli analyze --scores path/to/scores.csv --n-boot 2000
```
Recomputes AUROC, thresholds, CIs and a fresh report from scores that already exist. `run` and
`analyze` share one code path, so a replay reproduces the original numbers exactly. Use it to
change `--n-boot`, regenerate a report, or do analysis on a machine with no GPU.

Analysing a bare `scores.csv` with no `env.json` produces a report that says **INSTRUMENT UNKNOWN**
— deliberately, because those thresholds attest to nothing.

## 3. Workflow B — longitudinal drift (NOT BUILT YET)

The longitudinal study apparatus: probe runner, run registry, ECE/P(True), Page-Hinkley/CUSUM, detection
latency. **None of it exists yet.** See `private-notes/longitudinal study-requirements-trace.md` for exactly
what is HAVE / PLANNED / GAP. Do not assume any of it works because this file mentions it.

## 4. Checks you can run any time

```bash
python -m pytest tests/ -q          # 64 passed
python tools/check_consistency.py   # identity drift: version, headers, description, docs, imports
python -m ruff check gaige/ tests/
python -m ruff format --check gaige/ tests/
```
All four also run in CI on Linux and Windows across Python 3.10/3.12/3.13.

## 5. Traps that have actually bitten

- **transformers >=5 with 4-bit** silently loads fp16. gaige aborts rather than emit those scores.
  Keep the pin.
- **Windows default encoding.** Report writing once crashed on Windows because `write_text` used
  cp1252 and the report contains an arrow. All report IO is explicit UTF-8 now, with a test.
- **4-bit on CPU is impossible** (bitsandbytes has no CPU kernel). gaige refuses and tells you to
  use `--quant fp32` with a smaller model.
- **A 7B model on CPU is ~20-36 s/sample** — about two hours for 200 samples. Use a small model.
- **A run that dies mid-way resumes.** Scores are flushed to disk per sample; continue with
  `gaige run ... --resume reports/<dir>` (same corpus + instrument, or it refuses — resuming
  across an instrument change would interleave two instruments into one report). Verified:
  SIGKILL at 185/300, resumed, bit-identical to the uninterrupted run.
- **A co-resident production scorer shares the bench's GPU.** It is the live submission gate — do not run
  heavy GPU work during a production deadline window.
- **Corpus labels are trusted.** gaige validates the *shape* of a corpus, never the correctness of
  its labels. Wrong labels produce confidently wrong thresholds; the sha256 at least makes it
  auditable.

## 6. Where things live

- code + this file: `~/personal/gaige` (bench), `Documents\personal\gaige` (win-seat)
- strategy, design notes, backlog, requirements trace: `private-notes` repo (all three machines)
- start there at `private-notes/gaige-map.md`
