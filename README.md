# gaige

**Calibration + receipts for AI-text detectors.** Your corpus, your thresholds, honest error
bars — instead of a vendor's black-box score.

## Why

Every deployed AI-text detector is an *instrument*: its numbers depend on the model, the
quantization, the library versions, and the kind of text it reads. Almost nobody who uses
one knows their actual false-positive rate on their own material — they trust a marketing
page. When the accusation lands on a real person, "the tool said 87%" is not evidence.

gaige takes a labeled corpus (known-human + known-AI text) and any pluggable detector, and
emits a **receipts report**: ROC + AUROC with bootstrap confidence intervals, operating
thresholds at target false-positive rates with *achieved* (measured) rates, and a complete
instrument fingerprint — model, quantization **verified at load time** (some library combos
silently ignore 4-bit and load fp16, which shifts scores; gaige refuses to emit numbers
from a load it can't prove), versions, GPU, corpus hash, and the exact reproduce command.

If you use a detector: know its error bars on text like yours before you act on it.
If you're judged by one: these receipts are what an auditable process looks like.

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

v0.0.1 (burst 1): Fast-DetectGPT (analytic single-model) · HC3 subsample corpus · ROC/AUROC/
threshold receipts with bootstrap CIs · quantization verification.

Next (research-ranked — see PROGRESS.md): conformal FPR-bounded thresholds · subgroup-stratified
receipts (length and style; the disparities that produce real false accusations) · Binoculars as
detector #2 · quantization A/B receipts (bf16 vs 4-bit — measured, not assumed) · adversarial
degradation panels · RAID/TH-Bench corpus adapters · watermark-verifier adapter · base-rate
harm calculator · drift canaries (does your instrument still measure what it measured last term?).

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
