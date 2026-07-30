# gaige runbook

Operate gaige without a chat session. Exact commands, what correct output looks like, and the
traps that have actually bitten. If something here is wrong or unclear, that is a bug in this file.

---

## 0. Which machine for what

| Machine | Role | Why |
|---|---|---|
| **gpu bench** (Linux, 16 GB CUDA card) | scoring runs, GPU work, apparatus | the only box with CUDA; the reference instrument lives here |
| **win seat** (Windows, no GPU) | authoring, `analyze`, reading reports | no CUDA; scoring will fall back to CPU, analysis runs fine |
| **win node** (Windows, always-on) | long jobs, cross-platform checking | second Windows box; catches Windows-only defects |

`analyze`, `corpora` and the tests need **only numpy + requests**. Scoring needs the GPU extra.

## 1. Install

**gpu bench (the reference environment: do not casually upgrade it):**
```bash
cd ~/personal/gaige
./.venv/bin/python -m pytest tests/ -q          # expect: all green (182 at 0.0.1)
```
The pinned venv is transformers 4.49 / torch 2.13.0+cu130 / bnb 0.49.2 / cuda 13.0 / py 3.12.3.
**transformers must stay <5**, because 5.14.1 was measured silently ignoring 4-bit config and loading
fp16, which changes scores. gaige refuses such a load, but the pin avoids the fight.

**Windows seats (analysis only, no GPU):**
```
cd %USERPROFILE%\personal\gaige
python -m pip install -e .
python -m pytest tests/ -q                      # expect: all green (182 at 0.0.1)
```

## 2. Workflow A: calibrate a detector, then score documents

This is the original detection workflow. It is pinned in CI (`tests/test_reference_instrument.py`)
so it cannot rot while other capabilities get built.

### 2a. Calibrate (needs a model; GPU strongly preferred)

```bash
# reference instrument, on the gpu bench
./.venv/bin/python -m gaige.cli run --corpus hc3-mini --n 100 --seed 17 \
    --detector fast-detect-gpt --model tiiuae/falcon-7b --quant 4bit --device cuda
```

RAID slices (harder, attack/domain-stratified, added 2026-07-25): prepare first, then run
the produced path. Slices are fetched at preparation time (datasets-server pages; or
`--source csv` against a downloaded RAID csv) and never enter git.

```bash
./.venv/bin/python -m gaige.cli corpus prepare-raid \
    --generators gpt4,mistral-chat --domains abstracts,reddit --attacks none \
    --per-cell 60 --seed 17
./.venv/bin/python -m gaige.cli run --corpus corpora/raid-g2d2a1-n60-s17.jsonl \
    --detector fast-detect-gpt --model tiiuae/falcon-7b --quant 4bit --device cuda
```
The report's subgroup section then stratifies by generator/domain/attack/decoding
automatically (universal meta keys). Slice thresholds describe that slice, nothing more.

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
Those exact numbers are the reference. **If they differ, something about the instrument changed**:
check the fingerprint section of `report.md` before trusting anything downstream.

Read the two threshold families as: `FPRcal` rows are fitted on this sample (in-sample, no
guarantee); `conformal` rows carry a finite-sample guarantee that is **marginal over calibration
draws**; note the α=0.01 conformal threshold is deliberately stricter (2.4446 vs 2.1229) and
catches less (76% vs 86%): that gap is the price of an actual guarantee. The α=0.005 refusal at
n=100 is correct behavior, not an error. `report.md` also now carries per-subgroup rate tables
(rates below n=20 per class are withheld: counts speak instead) and a base-rate section
(`--harm-volume` sets your institution's yearly volume; default is Vanderbilt's published 75,000).

**Detector #2: Binoculars** (built 2026-07-22; two 7B models, effectively GPU-only):

```bash
./.venv/bin/python -m gaige.cli run --corpus hc3-mini --n 100 --seed 17 \
    --detector binoculars --quant 4bit --device cuda
```

Measured on the reference corpus: **AUROC 0.9992**, TPR 97% @1%FPR (thr −0.7829; gaige
emits the NEGATED Binoculars ratio so higher = more AI-like), conformal α=.01 → 95% TPR.
Fingerprint proves BOTH models (256 Linear4bit, ~8.1 GB, the VRAM ceiling beside the
daemon). The paper's global thresholds are deliberately not used: measured on this corpus
they run at **16%** (accuracy-mode) and **3%** (low-FPR-mode) FPR: the receipts gap,
demonstrated. Quant A/B receipts, BOTH scales: gpt2-large fp32-vs-fp16 agree to 4 decimals on thr@1%
while 4-bit moves it ~10%; **falcon-7b: 4-bit shifts thr@1% from 1.9540 (fp16) to 2.1229
(~8.6%)**, measured 2026-07-22 in a clean daemon window (canary pre=post 0.0946, |Δ|=0).
Quantization is an instrument parameter at 0.7B and at 7B.
Detail: the private research notes (burst2b receipts, 2026-07-22).

**No GPU?** It still works, with a smaller model:
```bash
python -m gaige.cli run --corpus hc3-mini --n 50 --model gpt2 --quant fp32 --device cpu --max-tokens 512
```
Measured on the bench's CPU: 100 samples in ~11 s, AUROC 0.9908. **This is a different instrument**
than the GPU one: its thresholds are not interchangeable (4.2276 vs 2.1229 at 1% FPR). The
receipt records `device: cpu` and gaige will warn if you try to use one report's thresholds in the
other's environment.

Omit `--model` entirely and gaige picks per device (falcon-7b/4bit on CUDA, gpt2-large/fp32 on CPU)
and records `model_auto_selected` on the receipt. The CPU default is a **measured selection**
(2026-07-22, six candidates on hc3-mini n=100 seed=17, fp32/cpu): gpt2-large was the only
candidate with perfect separation on that corpus (AUROC 1.0000; TPR 100% at 1%-FPR and at
conformal α=.01) at 0.64 s/sample, dominating gpt2-xl and gpt-neo-1.3B on both axes. One
corpus, one protocol: a default-selection receipt, not a detector ranking; each candidate's
run is reproducible with the command above plus `--model <name>`.

**Device policy (decided 2026-07-22):** `--device auto` is for exploratory runs: it prefers
CUDA, falls back to CPU loudly, and the receipt records the fallback. **Pre-registered or
scientific runs pin `--device` explicitly** (the run registry treats device class as
instrument identity, so a mid-series fallback forks the series). Either way the receipt's
reproduce command always carries the RESOLVED device, never "auto", so re-running a receipt can
never silently swap instruments.

### 2b. Score a document against that calibration

```bash
python -m gaige.cli score --report reports/<timestamp>-fast-detect-gpt/ --file draft.md
python -m gaige.cli score --report reports/<ts>-fast-detect-gpt/ --text "some text"
cat draft.md | python -m gaige.cli score --report reports/<ts>-fast-detect-gpt/
```

The document is **never written anywhere**: no log, no cache, no telemetry. That is tested
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

Analysing a bare `scores.csv` with no `env.json` produces a report that says **INSTRUMENT UNKNOWN**,
deliberately, because those thresholds attest to nothing.

### 2d. Export a receipt as public site data (no GPU, any machine)

```bash
python -m gaige.cli export --report reports/<ts>-<detector>/ --out site-data/
```

Writes `site-data/receipts/<id>.json` (schema `gaige-receipt-export/1`) and rebuilds
`site-data/index.json`: the same numbers, joined with the instrument fingerprint and the
reproduce command, in one self-contained public JSON document. Nothing is recomputed; every
statistic is copied verbatim from `results.json`. Redaction is fail-closed: an absolute path,
an IP address, or a URL off the public allowlist refuses the whole export and names the field.
Reports without `env.json` refuse: INSTRUMENT UNKNOWN receipts are not exportable. Site
placement is a site-repo concern; this command only produces the artifact.

### 2e. Provenance evidence sweep (built 2026-07-29; no model, no GPU, any machine)

```bash
python -m gaige.cli verify photo.png            # C2PA + image watermark carriers
python -m gaige.cli verify document.pdf         # C2PA only (non-image media)
python -m gaige.cli verify --text "some prose"  # keyed text schemes
python -m gaige.cli verify photo.png --json
```

Deterministic evidence, never a score: each scheme reports FOUND / ABSENT / INCONCLUSIVE /
NEEDS_KEYS / UNAVAILABLE / NO_PUBLIC_DETECTOR / ERROR plus what a negative from it means.
The rule that matters: a watermark negative is **ABSENT only when a probe payload embedded
into that same image survives a round trip** — otherwise the carrier can't hold the mark and
the honest answer is INCONCLUSIVE. That rule is structural (`tests/test_provenance.py`): the
overclaiming result cannot be constructed.

The imaging checks need optional libraries; without them the sweep still runs and reports
UNAVAILABLE with the exact install remedy. The dwtDct codec itself is vendored
(`gaige/_dwtdct.py`, bit-format-compatible with invisible-watermark 0.2.0), so the only
optional pieces are PyWavelets + OpenCV: `pip install "gaige[verify]"` covers both arms in
one step. Keyed text schemes (SynthID-Text, red/green-list) are honestly NEEDS_KEYS:
without the deployer's key there is nothing to measure, and a tool claiming otherwise is
checking something else.

Live-arm fixtures for this workflow (the signed/corrupt C2PA pair, the real-encoder
watermark, the honest-negative set) live untracked at `lab/verify-fixtures/` on the bench,
generator scripts beside them; the release acceptance arms use exactly these.

### 2f. Corpus admission: divergence from an accepted baseline (built 2026-07-30)

```bash
# live lane: score the candidate under the baseline's own instrument
python -m gaige.cli admit --baseline reports/<ts>-<detector>/ --candidate new-material.jsonl
# no-GPU lane: bring scores you produced with the matching instrument yourself
python -m gaige.cli admit --baseline reports/<ts>-<detector>/ --candidate-scores scores.csv
# restrict the reference to one labeled side of the baseline
python -m gaige.cli admit --baseline reports/<ts>-<detector>/ --candidate new.jsonl --reference human
```

The baseline is any existing report directory; its env.json fingerprint is the standard
being diverged from (no env.json refuses: no fingerprint, no standard). The candidate is
unlabeled JSONL, rows {id?, text, meta?}; labels present on rows are ignored with a note.
Out comes `reports/<ts>-admit/` with report.md, results.json, env.json, and
candidate-scores.csv (per-document percentile among the reference, a two-sided conformal
p-value, and a short_text flag).

Refusals to expect, all working as intended: alpha rows refuse below ceil(2/alpha)-1
reference scores (39 / 199 / 399 at the defaults); slice statistics withhold below 20
candidate documents (per-document placements still write); strata withhold below 20 per
stratum; the live lane hard-refuses on ANY instrument mismatch (score on the matching
environment, or bring --candidate-scores). A crashed live run leaves scores.partial.csv;
salvage it by passing that file to --candidate-scores. The receipt never says admit or
reject: it measures divergence, and the decision is yours.

## 3. Workflow B: longitudinal drift (UNDER CONSTRUCTION; the probe runner is REAL)

### 3a. Run a probe set (built 2026-07-22)

```bash
python -m gaige.cli probe run --probes probes.jsonl --provider local-hf \
    --model <instruct-model> --device cpu --cutoff 2024-06-01
```

Probe JSONL rows: `{"id","prompt","answer","vintage","source","source_date"}` plus optional
`aliases`/`authored`: the loader refuses anything less, naming the row and the remedy.
Output: `reports/<ts>-probes/` with `report.md` (accuracy per vintage with 95% CI, the
per-vintage **post-cutoff share** vs `--cutoff`, full fingerprint incl. attestation +
decoding + grading version), `answers.csv`, and a crash-safe partial while running
(`--resume <dir>` continues; ANY pinned fingerprint change refuses).

**What correct looks like:** greedy is the default (temperature 0, pre-registered); the
provider line prints its attestation (`verified` for local-hf; llamacpp earns
verified/self-reported/opaque: pass `--gguf` to hash the artifact). A NON-local endpoint
refuses to receive text without `--allow-remote-text`; that is a security property.

**Grading is deliberately strict** (normalized exact match + authored aliases, version
`nem-1`): a base model that rambles past the answer grades WRONG: measured on gpt2, which
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
version + cutoff + gaige version). A changed instrument **forks a new series**, never
mixes. Within a series, a vintage label is **frozen**: re-running an edited "t0" is refused
by name (author a new vintage instead); NEW vintage labels are welcome: that is the
longitudinal design. The series report shows accuracy per vintage per run, the measured
run-variance bound from the replicates (±0.0% on a deterministic pipeline: a result, not
an assumption), and flags each later run's movement as within-variance or BEYOND the bound.

### 3c. M3: calibration drift (built 2026-07-22)

Add `--ptrue` to any probe run (needs a provider with option_logprobs: local-hf has it;
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
fork the series (the M1 instrument is unchanged), but resuming a half-finished run with it
toggled refuses, and a changed template refuses at registration.

First live measurement (gpt2 on the toy set): **accuracy 0%, mean P(True) ~79%, gap +79%**:
"fluent and authoritative whilst quietly wrong," demonstrated by the smoke test itself.

### 3d. The real-model apparatus (run live 2026-07-22)

llama.cpp release binary at `~/personal/llamacpp/llama-b10091/` (the bench); GGUF weights in
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
gated Llama-3.2-1B the longitudinal spec names): attestation **verified** by GGUF sha256 + server
identity match · t0 accuracy 75% (n=12), t1 50% (n=8), 100% post-cutoff · replicate bound
**±0.0%: served greedy decoding is deterministic, measured** · follow-up run "within run
variance" · a temperature-0.3 run **forked its own series** rather than mixing. Receipt of
record: the private research notes (first longitudinal receipt, 2026-07-22).

### 3e. M5: drift monitors over a series (built 2026-07-22)

```bash
python -m gaige.cli series watch <series-id> --registry registry \
    [--vintage t0] [--quantity accuracy|gap] [--alpha 0.2] [--direction down|up]
```

Replays a registered series through the monitor panel: no model touched. Three monitors,
graded honestly:
- **conformal-interval**: per-interval alarm with a **marginal finite-sample false-alarm
  bound** (≤ α per look; expected false alarms = α × looks), calibrated on the Day-0
  replicates. Needs `ceil(1/α)−1` zero-drift reference intervals: **α=0.2 needs 4, one
  more replicate than the k=3 Day-0 default**, so run Day-0 with `--replicates 4` if you
  want conformal alarms from the start. Refuses honestly below that.
- **page-hinkley** and **cusum**: the drift-literature detectors (Gama/Webb lineage),
  cumulative statistics with tuning constants (δ/λ, k/h) recorded on the receipt and **no
  guarantee claimed** (interval exchangeability does not apply to a cumulative statistic;
  conformal test martingales are the principled extension, future work).

Output prints and lands as `monitors-report.md` in the series directory. `monitors.evaluate`
scores any monitor against a known onset (detection latency + false alarms): M5's
per-technique scorecard, exercised in tests with injected shifts. Verified live on the
first real series: conformal refused at n=3 reference (correct), PH/CUSUM quiet on a
within-variance interval (correct).

### 3f. Ollama provider (run live 2026-07-22)

Probe runs against any model an ollama server already serves. COMPLETE only (no stable
full-vocab logprob API, so MC control and P(True) stay on local-hf).

```bash
python -m gaige.cli probe run --probes probes/demo.jsonl --provider ollama \
    --model qwen2.5:7b-instruct --cutoff 2023-10-01 --max-new-tokens 12 --register
```

Attestation is earned per the content-addressed CHAIN: the `/api/tags` digest names the
manifest; the manifest's model-layer digest names the weights blob; gaige re-hashes BOTH →
`verified`. Store unreadable (remote, permissions) → `self-reported` (digest is still
version-shaped identity); no digest → `opaque`; mismatch → loud, never upgraded. Store
roots tried: `$OLLAMA_MODELS`, `~/.ollama/models`, `/usr/share/ollama/.ollama/models`.

Bench specifics: the ollama systemd service stores under `/usr/share/ollama` (mode 700,
service user), so `verified` needed a read-only ACL: granted 2026-07-22 with
`setfacl -R -m u:<you>:rX /usr/share/ollama` (+ default ACL for new blobs); reverse with
`setfacl -R -b /usr/share/ollama` if ever unwanted. Model LOADS still go through
a VRAM-guard wrapper (headroom rules): the provider only talks to a model an
operator already chose to serve. First live series: qwen2.5:7b-instruct, chain
**verified**, 100% both vintages on the demo set, series `10c246457f8d`.

### 3g. `gaige plan`: what can this machine run

```bash
python -m gaige.cli plan
```

Prints the machine (CPUs/RAM/GPU free VRAM/served ollama models/llama-server presence) and
a table of known configurations: fits-now verdict against measured floors, attestation
level, and a measured runtime anchor NAMING its receipt. No quality column, deliberately, because
separation lives in receipts and does not transfer between configurations (the legend on
every table says so; a test enforces it). Bench verdicts while the co-resident production scorer holds its
VRAM: falcon-4bit fits, falcon-fp16 correctly gets a NO (needs 13.7 GB free VRAM with roughly 11 available).

### 3h. Author a probe set (built 2026-07-26)

```bash
python -m gaige.cli probe new --out probes/t0.jsonl --vintage t0 --cutoff 2024-06-01
# ... author the probes and fill in the manifest's control linkage, then:
python -m gaige.cli probe lint --probes probes/t0.jsonl
```

`probe new` scaffolds the JSONL plus a sidecar manifest (`t0.manifest.json`) with the
signed authoring decisions pre-filled: nem grading declaration, greedy temperature-0
decoding block, and a control-linkage stanza (name + sha256 + logprob-argmax scoring) you
must point at the frozen MMLU-subset control. The manifest is a sidecar deliberately, so
editing a declaration never moves the probe-file sha256 or the frozen vintage hashes.

`probe lint` is the mechanical gate: errors for anything violating a signed decision
(missing `authored`, a `source_date` not post-dating the cutoff, `authored` predating its
source, unfilled EDIT-ME placeholders, ungradeable answers, a broken or unhashed control
linkage), warnings for authoring advice (answers over 5 words, redundant aliases,
duplicate prompts). A study set must lint clean, and **`probe run` enforces it**: when a
manifest sits beside the probe file, a set that fails lint refuses to run, and a run
whose temperature contradicts the declared greedy block refuses with the remedy (change
the declaration first, so the fork is visible in history). A manifest-less file (like
`probes/demo.jsonl`) runs as before and says the declarations are unenforced.

### 3i. Not built yet

M2r (probe-source drift index) awaits an external rescope sign-off; Mondrian conformal and
batched scoring stay banked per the map. Do not assume they work because this file
mentions them.

## 4. Checks you can run any time

```bash
python -m pytest tests/ -q          # 278 passed on the no-GPU matrix; the bench's full-deps run adds the torch lanes
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
- **A 7B model on CPU is ~20-36 s/sample**: about two hours for 200 samples. Use a small model.
- **A run that dies mid-way resumes.** Scores are flushed to disk per sample; continue with
  `gaige run ... --resume reports/<dir>` (same corpus + instrument, or it refuses, because resuming
  across an instrument change would interleave two instruments into one report). Verified:
  SIGKILL at 185/300, resumed, bit-identical to the uninterrupted run.
- **A co-resident production scorer can share the GPU.** When one is resident it has priority:
  schedule heavy GPU work around its quiet windows.
- **Corpus labels are trusted.** gaige validates the *shape* of a corpus, never the correctness of
  its labels. Wrong labels produce confidently wrong thresholds; the sha256 at least makes it
  auditable.

## 6. Where things live

- code + this file: `~/personal/gaige` (bench), `Documents\personal\gaige` (Windows seats)
- strategy, design notes, backlog, requirements trace: a private ops repo
