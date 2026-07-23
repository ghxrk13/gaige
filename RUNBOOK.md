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
./.venv/bin/python -m pytest tests/ -q          # expect: 89 passed
```
The pinned venv is transformers 4.49 / torch 2.13.0+cu130 / bnb 0.49.2 / cuda 13.0 / py 3.12.3.
**transformers must stay <5** — 5.14.1 was measured silently ignoring 4-bit config and loading
fp16, which changes scores. gaige refuses such a load, but the pin avoids the fight.

**the Windows seats (analysis only, no GPU):**
```
cd %USERPROFILE%\personal\gaige
python -m pip install -e .
python -m pytest tests/ -q                      # expect: 89 passed
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
[receipt] AUROC 0.9720 (CI 0.9448-0.9929)
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

## 3. Workflow B — longitudinal drift (UNDER CONSTRUCTION; the probe runner is REAL)

### 3a. Run a probe set (built 2026-07-22)

```bash
python -m gaige.cli probe run --probes probes.jsonl --provider local-hf \
    --model <instruct-model> --device cpu --cutoff 2024-06-01
```

Probe JSONL rows: `{"id","prompt","answer","vintage","source","source_date"}` plus optional
`aliases`/`authored` — the loader refuses anything less, naming the row and the remedy.
Output: `reports/<ts>-probes/` with `report.md` (accuracy per vintage with 95% CI, the
per-vintage **post-cutoff share** vs `--cutoff`, full fingerprint incl. attestation +
decoding + grading version), `answers.csv`, and a crash-safe partial while running
(`--resume <dir>` continues; ANY pinned fingerprint change refuses).

**What correct looks like:** greedy is the default (temperature 0, pre-registered); the
provider line prints its attestation (`verified` for local-hf; llamacpp earns
verified/self-reported/opaque — pass `--gguf` to hash the artifact). A NON-local endpoint
refuses to receive text without `--allow-remote-text`; that is a security property.

**Grading is deliberately strict** (normalized exact match + authored aliases, version
`nem-1`): a base model that rambles past the answer grades WRONG — measured on gpt2, which
answered "The capital of France is" with "the capital of the French Republic, and" (graded
False, correctly). Use an instruct model and author prompts that elicit short answers; the
prompt is part of the instrument.

```bash
python -m gaige.cli providers                    # list providers + env config
python -m gaige.cli test-connection --endpoint http://127.0.0.1:8080 [--gguf model.gguf]
```

### 3b. The run registry and series (built 2026-07-22)

```bash
# Day-0: establish the run-variance bound with same-day replicates, registered
python -m gaige.cli probe run --probes probes.jsonl --provider local-hf --model <m> \
    --cutoff 2024-06-01 --register --replicates 3
# Later intervals: single runs into the SAME series (same instrument, frozen vintages)
python -m gaige.cli probe run --probes probes.jsonl ... --register
python -m gaige.cli series list --registry registry
python -m gaige.cli series show <series-id> --registry registry
```

A series is keyed by the instrument identity hash (provider identity + decoding + grading
version + cutoff + gaige version). A changed instrument **forks a new series** — never
mixes. Within a series, a vintage label is **frozen**: re-running an edited "t0" is refused
by name (author a new vintage instead); NEW vintage labels are welcome — that is the
longitudinal design. The series report shows accuracy per vintage per run, the measured
run-variance bound from the replicates (±0.0% on a deterministic pipeline — a result, not
an assumption), and flags each later run's movement as within-variance or BEYOND the bound.

### 3c. M3 — calibration drift (built 2026-07-22)

Add `--ptrue` to any probe run (needs a provider with option_logprobs — local-hf has it;
llamacpp deliberately does not yet):

```bash
python -m gaige.cli probe run --probes probes.jsonl --provider local-hf --model <m> \
    --cutoff 2024-06-01 --ptrue --register
```

Per answer, gaige asks the model whether its own answer is true and reads **P(True) from the
logits** (Kadavath-style; a forward pass, no sampling; the template is hashed into the
fingerprint and FROZEN per series once M3 has run). The receipt gains an M3 table: mean
P(True), accuracy, the **confidence-accuracy gap** (positive = overconfident), and **ECE**
with a bootstrap CI (bin count fixed per series). Toggling `--ptrue` on a later run does NOT
fork the series (the M1 instrument is unchanged) — but resuming a half-finished run with it
toggled refuses, and a changed template refuses at registration.

First live measurement (gpt2 on the toy set): **accuracy 0%, mean P(True) ~79%, gap +79%** —
"fluent and authoritative whilst quietly wrong," demonstrated by the smoke test itself.

### 3d. The real-model apparatus (run live 2026-07-22)

llama.cpp release binary at `~/personal/llamacpp/llama-b10091/` (bench); GGUF weights in
`~/personal/models/`. The full loop, as actually run:

```bash
# serve (CPU is fine for 1-2B; healthy in ~3s)
~/personal/llamacpp/llama-b10091/llama-server \
    -m ~/personal/models/qwen2.5-1.5b-instruct-q4_k_m.gguf --port 8089 --host 127.0.0.1 -t 8 &
# prove the endpoint + earn the attestation BEFORE a long run
python -m gaige.cli test-connection --endpoint http://127.0.0.1:8089 --gguf ~/personal/models/qwen2.5-1.5b-instruct-q4_k_m.gguf
# Day-0: replicates -> measured variance bound, registered
python -m gaige.cli probe run --probes probes/demo.jsonl --provider llamacpp \
    --endpoint http://127.0.0.1:8089 --gguf ~/personal/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
    --cutoff 2023-10-01 --max-new-tokens 12 --register --replicates 3
# later intervals: single runs into the same series
python -m gaige.cli probe run ... --register
```

Measured on the first live series (Qwen2.5-1.5B q4_k_m, the non-gated stand-in for the
gated Llama-3.2-1B the spec names): attestation **verified** by GGUF sha256 + server
identity match · t0 accuracy 75% (n=12), t1 50% (n=8), 100% post-cutoff · replicate bound
**±0.0% — served greedy decoding is deterministic, measured** · follow-up run "within run
variance" · a temperature-0.3 run **forked its own series** rather than mixing. Receipt of
record: `private-notes/research/first-longitudinal-receipt-2026-07-22.md`.

### 3e. M5 — drift monitors over a series (built 2026-07-22)

```bash
python -m gaige.cli series watch <series-id> --registry registry \
    [--vintage t0] [--quantity accuracy|gap] [--alpha 0.2] [--direction down|up]
```

Replays a registered series through the monitor panel — no model touched. Three monitors,
graded honestly:
- **conformal-interval** — per-interval alarm with a **marginal finite-sample false-alarm
  bound** (≤ α per look; expected false alarms = α × looks), calibrated on the Day-0
  replicates. Needs `ceil(1/α)−1` zero-drift reference intervals: **α=0.2 needs 4 — one
  more replicate than the k=3 Day-0 default**, so run Day-0 with `--replicates 4` if you
  want conformal alarms from the start. Refuses honestly below that.
- **page-hinkley** and **cusum** — the drift-literature detectors (Gama/Webb lineage),
  cumulative statistics with tuning constants (δ/λ, k/h) recorded on the receipt and **no
  guarantee claimed** (interval exchangeability does not apply to a cumulative statistic;
  conformal test martingales are the principled extension, future work).

Output prints and lands as `monitors-report.md` in the series directory. `monitors.evaluate`
scores any monitor against a known onset (detection latency + false alarms) — M5's
per-technique scorecard, exercised in tests with injected shifts. Verified live on the
first real series: conformal refused at n=3 reference (correct), PH/CUSUM quiet on a
within-variance interval (correct).

### 3f. Not built yet

M2r (probe-source drift index) awaits an external rescope sign-off; Mondrian conformal and
batched scoring stay banked per the map. Do not assume they work because this file
mentions them.

## 4. Checks you can run any time

```bash
python -m pytest tests/ -q          # 89 passed
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
