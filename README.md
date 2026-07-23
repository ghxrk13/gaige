# gaige

**Calibration and drift receipts for AI measurement.** Your corpus, your thresholds, honest
error bars — and a fingerprint that proves the instrument hasn't changed underneath you.

## Why

Any score you act on comes from an *instrument*, and that instrument's numbers depend on the
model, the quantization, the device, the library versions, and the material it reads. Change
any of those and you have a different instrument — whether or not anyone noticed. Almost
nobody who relies on a score knows its actual error rate on their own material; they trust a
marketing page. When the consequence lands on a real person, "the tool said 87%" is not
evidence.

gaige takes a corpus or probe set and any pluggable scorer, and emits a **receipts report**:
ROC + AUROC with bootstrap confidence intervals, operating thresholds two ways — empirical
(target FPR with the in-sample rate labeled as exactly that) and conformal (a finite-sample
guarantee, stated honestly as marginal) — per-subgroup error rates with intervals, base-rate
harm arithmetic, and a complete instrument fingerprint — model,
quantization **verified at load time** (some library combos silently ignore 4-bit and load
fp16, which shifts scores; gaige refuses to emit numbers from a load it can't prove), device,
versions, corpus hash, and the exact reproduce command.

**Two applications of one machine:**

- **Detector calibration** — what is this detector's real false-positive rate on *your*
  material, rather than on a vendor's marketing page. If you use a detector: know its error
  bars before you act. If you are judged by one: these receipts are what an auditable process
  looks like.
- **Instrument drift** — has the system you're measuring changed, or has your measuring
  pipeline? Distinguishing those two is the difference between a finding and an artifact, and
  it is the harder question. gaige is built to answer it with evidence rather than assertion.

Scoring needs a model and ideally a GPU. **Analysis does not** — `gaige analyze` re-derives
thresholds and reports from scores that already exist, so calibration runs on a laptop, or a
CPU-only machine, or an isolated environment with no accelerator at all.

## Quickstart

```bash
pip install -e .            # plus torch/transformers/bitsandbytes in your GPU env
gaige run --corpus hc3-mini --n 100 --detector fast-detect-gpt
# → reports/<ts>/report.md + scores.csv + roc.json + env.json
gaige run --corpus your-labeled.jsonl   # rows: {"text": ..., "label": "human"|"ai"}
```

## What it will never do

- Ship a universal threshold. There isn't one; that's the point.
- Emit a verdict. Scores + measured error rates only.
- Pretend one corpus generalizes. Every report says what it was measured on.

## Status / roadmap

v0.0.1 (bursts 1-2a): Fast-DetectGPT (analytic single-model) · HC3 subsample corpus · ROC/AUROC/
threshold receipts with bootstrap CIs · quantization verification · CPU support + `gaige analyze`
replay · resumable scoring · **conformal thresholds** (split conformal per arXiv:2505.05084;
finite-sample bound P(human flagged) ≤ α, stated honestly: marginal over calibration draws, with
the exact conditional Beta dispersion printed beside it) · **subgroup-stratified receipts**
(length buckets always, metadata axes when the corpus carries them; every rate with a bootstrap
interval, rates on fewer than 20 samples withheld — counts speak instead) · **base-rate
arithmetic in every report** (FPR × your volume = wrongly flagged per year, plus PPV at assumed
prevalences — the calculation Vanderbilt published when it disabled its detector) · **Binoculars as detector #2** (released-implementation construction, both models quant-verified on the receipt; the paper's global thresholds measured at 16%/3% FPR on the reference corpus — calibrate in-domain, with receipts) · **quantization A/B receipts** (measured: 4-bit moves the 1%-FPR threshold ~10% where fp32-vs-fp16 agree to four decimals — quantization is an instrument parameter) · the probe runner, run registry, P(True)/ECE, and drift monitors of the longitudinal apparatus (see RUNBOOK Workflow B).

Next (research-ranked — see PROGRESS.md): the falcon-7b fp16 A/B arm (needs a daemon-free
window) · per-subgroup conformal thresholds (Mondrian; the
guarantee-backed version of group-adaptive thresholding, needs larger calibration sets) ·
adversarial degradation panels · RAID/TH-Bench corpus adapters · watermark-verifier adapter ·
drift canaries (does your instrument still measure what it measured last term?).

## Licensing and the name

**Code: AGPL-3.0** (`LICENSE`). Free to use, study, modify, and share — including inside an
institution, in research, and in an appeal against a detector's verdict. If you offer a
*modified* version to others as a network service, publish your modifications. A commercial
license removes that obligation: see `COMMERCIAL.md`.

**Name: not licensed.** No open-source license grants trademark rights. You may fork freely —
under a different name. See `TRADEMARK.md`; the standards there exist so that "a gaige report"
keeps meaning something.

Maintainership is deliberately closed (see `CONTRIBUTING.md`) — issues and receipt-backed
reproductions very welcome; pull requests not accepted at this time.
