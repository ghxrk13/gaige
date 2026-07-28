# gaige

**Calibration and drift receipts for AI measurement.** Your corpus, your thresholds, honest
error bars, and a fingerprint that proves the instrument hasn't changed underneath you.

[gaige.dev](https://gaige.dev) · [PyPI](https://pypi.org/project/gaige/) ·
[issues](https://github.com/ghxrk13/gaige/issues) ·
[fuel the instrument](https://ko-fi.com/gaigedev)

## Why

Any score you act on comes from an *instrument*, and that instrument's numbers depend on the
model, the quantization, the device, the library versions, and the material it reads. Change
any of those and you have a different instrument, whether or not anyone noticed. Almost
nobody who relies on a score knows its actual error rate on their own material; they trust a
marketing page. When the consequence lands on a real person, "the tool said 87%" is not
evidence.

Point gaige at a corpus or a probe set, give it any pluggable scorer, and what comes back is
a **receipts report**. Inside it: the ROC and AUROC with bootstrap confidence intervals, and
operating thresholds computed two ways: an empirical target-FPR threshold whose in-sample
rate is labeled as exactly that, and a conformal one carrying a finite-sample guarantee,
stated honestly as marginal. Error rates arrive per subgroup with intervals, the base-rate
harm arithmetic is done for you, and the whole thing sits on an instrument fingerprint:
model, device, versions, corpus hash, the exact reproduce command, and quantization
**verified at load time**, because some library combos silently ignore 4-bit and load fp16, which
shifts scores, so gaige refuses to emit numbers from a load it can't prove.

**Two applications of one machine:**

- **Detector calibration**: what is this detector's real false-positive rate on *your*
  material, rather than on a vendor's marketing page. If you use a detector: know its error
  bars before you act. If you are judged by one: these receipts are what an auditable process
  looks like.
- **Instrument drift**: has the system you're measuring changed, or has your measuring
  pipeline? Distinguishing those two is the difference between a finding and an artifact, and
  it is the harder question. gaige is built to answer it with evidence rather than assertion.

Scoring needs a model and ideally a GPU. **Analysis does not**: `gaige analyze` re-derives
thresholds and reports from scores that already exist, so calibration runs on a laptop, or a
CPU-only machine, or an isolated environment with no accelerator at all.

## Quickstart

```bash
pip install gaige                 # analysis only — numpy + requests, no torch
pip install "gaige[gpu]"          # scoring too: torch/transformers/bitsandbytes/accelerate
gaige run --corpus hc3-mini --n 100 --detector fast-detect-gpt
# → reports/<ts>/report.md + scores.csv + roc.json + env.json
gaige run --corpus your-labeled.jsonl   # rows: {"text": ..., "label": "human"|"ai"}
```

On a CPU-only machine the `[gpu]` extra pulls the multi-gigabyte CUDA build of torch by
default. Install the small CPU build first and the extra will leave it alone:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu && pip install "gaige[gpu]"
```

From a source checkout: `pip install -e .`

## Python API

The CLI is the primary interface. For library use, `import gaige` carries the supported
surface: the calibrate/conformal/analyze spine. It never imports torch, so it runs where a
model never could: a laptop, a CI job, an air-gapped review machine.

```python
from pathlib import Path
import numpy as np
import gaige

# Replay an existing report — recompute every statistic from its scores.csv.
rows, corpus, meta = gaige.load_report(Path("reports/2026-07-26T18-02-11Z"))
results = gaige.compute_results(rows)

# Or bring scores from any instrument of your own (gaige refuses under 50 per class —
# CorpusTooSmall — because a threshold fitted on less is a guess).
rng = np.random.default_rng(17)
scores = np.concatenate([rng.normal(0, 1, 100), rng.normal(2, 1, 100)])
labels = np.array(["human"] * 100 + ["ai"] * 100)
gaige.auroc(scores, labels)
gaige.threshold_at_fpr(scores, labels, target_fpr=0.01)   # measured, labeled in-sample
gaige.conformal_threshold(scores[labels == "human"], alpha=0.05)  # finite-sample bound;
# raises InsufficientCalibration when your n cannot support the alpha you asked for.
```

Everything not in `gaige.__all__` is internal and may move without notice.

## What it will never do

- Ship a universal threshold. There isn't one; that's the point.
- Emit a verdict. Scores + measured error rates only.
- Pretend one corpus generalizes. Every report says what it was measured on.

## Where gaige sits

The detection literature of 2024–2026 converged on exactly what gaige does (fixed-FPR
operating points, calibrated in-domain, reported with error bars) while producing steady
evidence that nobody's fixed threshold survives contact with a new domain, a new model, or
even a new decoding setting.

**Default thresholds are the documented failure.** RAID (ACL 2024, arXiv:2405.07940), the
largest shared detection benchmark (6M+ generations), measured open-source detectors at naive
default thresholds running 47–100% false-positive rate on human text (GLTR 100%, LLMDet
97.9%, Fast-DetectGPT 47.3%) while the commercial detectors it tested ship
factory-calibrated at ≤1.7%. Its stated advice is gaige's thesis, published: *"calibrate
detectors on in-domain data before using them."* RAID also fixes and discloses FPR for every
number it reports, and calls itself one of the first shared resources to do so.

**Accuracy does not survive a change of scene.** In a 2026 cross-dataset audit (arXiv
2604.16607), every one of 15 detector variants fell to near-chance (AUROC ≤ 0.60) on at
least one evaluation set (the best worst-case was that 0.60), and 12 of the 15 showed ≥15%
FPR on human text on at least one dataset. An independent black-box evaluation on
deliberately-unseen generators (arXiv 2412.05139) put the best detector it tested at 0.58
TPR at 1% FPR. Rankings reorder across benchmarks and metrics, so a single-benchmark
accuracy claim is not defensible without its corpus disclosed, and detectors also move
under plain configuration changes (sampling strategy, repetition penalty, an unseen model;
measured in RAID) with no "attack" involved.

**Even flagship thresholds are global.** Binoculars (best worst-case in the audit above and
best in the black-box evaluation) ships a single global threshold (0.901) optimized for
accuracy on three reference datasets; its headline 0.01%-FPR figure is a point read off a
ROC curve, not a deployment threshold. Measured here: the paper's global thresholds run at
16% and 3% FPR on our reference corpus (hc3-mini, n=100, seed=17, both scoring models
quant-verified on the receipt). Thresholds do not transfer. That is not a flaw in
Binoculars; it is the reason calibration tooling has to exist.

**FPR-bounded calibration is published methodology with no product behind it.** Conformal
calibration (arXiv 2505.05084) bounds the false-positive rate with a finite-sample
guarantee, and the bound held empirically at α from 0.2 down to 0.005 across seven
detectors, but naive single-quantile conformal costs real TPR, which is why the trade must
be printed, never hidden. gaige implements the single-quantile mode with the trade printed.
Measured on the reference instrument (Fast-DetectGPT, falcon-7b, 4-bit, hc3-mini n=100
seed=17): α=0.01 → threshold 2.4446 at TPR 76%, beside the in-sample 1%-target pair
(2.1229, TPR 86%); α=0.005 correctly refused at n=100: the guarantee's price, measured
and shown.

**Aggregates hide the numbers that matter.** Under a static threshold, short human texts are
flagged at measurably higher rates (arXiv 2502.04528: score distributions differ
significantly by length and style, KS_max 0.3081, p<0.01), and the group-threshold
flagship paper's own tables show its disparity cut bought with mean F1 collapsing
0.51→0.39, invisible in its headline metrics. Our first RAID receipt says the same about
decoding: on a stratified slice (raid g2×d2×a2, per-cell reservoir seed 17; instrument
Fast-DetectGPT, falcon-7b, 4-bit, cuda), aggregate AUROC 0.9285 and TPR@1%FPR 61.5%
(against 86% on the reference corpus, same instrument, harder corpus) concealed a decoding
split of greedy 87.6% vs sampled 39.7% TPR@1%FPR: a 48-point spread inside one aggregate.
Per-subgroup rates with intervals, and refusals where n is too small to say, are the honest
form of the answer.

**So gaige is not a detector, and not a claim that detection works.** It is the measurement
layer this literature keeps asking for: your corpus, your instrument, its actual error
rates, with the fingerprint that proves which instrument produced them, and a report that
refuses to say more than was measured.

## Status / roadmap

v0.0.1 (bursts 1-2a): Fast-DetectGPT (analytic single-model) · HC3 subsample corpus · ROC/AUROC/
threshold receipts with bootstrap CIs · quantization verification · CPU support + `gaige analyze`
replay · resumable scoring · **conformal thresholds** (split conformal per arXiv:2505.05084;
finite-sample bound P(human flagged) ≤ α, stated honestly: marginal over calibration draws, with
the exact conditional Beta dispersion printed beside it) · **subgroup-stratified receipts**
(length buckets always, metadata axes when the corpus carries them; every rate with a bootstrap
interval, rates on fewer than 20 samples withheld: counts speak instead) · **base-rate
arithmetic in every report** (FPR × your volume = wrongly flagged per year, plus PPV at assumed
prevalences: the calculation Vanderbilt published when it disabled its detector) · **Binoculars as detector #2** (released-implementation construction, both models quant-verified on the receipt; the paper's global thresholds measured at 16%/3% FPR on the reference corpus: calibrate in-domain, with receipts) · **quantization A/B receipts** (measured: 4-bit moves the 1%-FPR threshold ~10% where fp32-vs-fp16 agree to four decimals, so quantization is an instrument parameter) · the probe runner, run registry, P(True)/ECE, and drift monitors of the longitudinal apparatus (see RUNBOOK Workflow B) · providers local-hf / llama.cpp / **ollama** (attestation earned per artifact: GGUF sha256; ollama's manifest+weights chain re-hashed by gaige itself) · **`gaige plan`** (what this machine can run, at measured cost, with attestation, but no quality column, deliberately: separation lives in receipts).

Next (research-ranked, see PROGRESS.md): per-subgroup conformal thresholds (Mondrian; the
guarantee-backed version of group-adaptive thresholding, needs larger calibration sets) ·
adversarial degradation panels · TH-Bench corpus adapter (the RAID adapter landed:
`gaige corpus prepare-raid`) · watermark-verifier adapter ·
drift canaries (does your instrument still measure what it measured last term?).

## Licensing and the name

**Code: AGPL-3.0** (`LICENSE`). Free to use, study, modify, and share, including inside an
institution, in research, and in an appeal against a detector's verdict. If you offer a
*modified* version to others as a network service, publish your modifications. A commercial
license removes that obligation: see `COMMERCIAL.md`.

**Name: not licensed.** No open-source license grants trademark rights. You may fork freely,
but under a different name. See `TRADEMARK.md`; the standards there exist so that "a gaige report"
keeps meaning something.

Maintainership is deliberately closed (see `CONTRIBUTING.md`), but issues and receipt-backed
reproductions very welcome; pull requests not accepted at this time.

## Fuel

The fuel link is at [ko-fi.com/gaigedev](https://ko-fi.com/gaigedev). A coffee's worth
genuinely helps, and it goes where [the roadmap](https://gaige.dev/roadmap.html) says it
goes: GPU hours mostly, corpus licensing when one comes up. gaige itself stays AGPL and
free no matter what happens here, and no amount of support has ever changed a number this
tool prints.
