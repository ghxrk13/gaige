# detcal

**Calibration + receipts for AI-text detectors.** Your corpus, your thresholds, honest error
bars — instead of a vendor's black-box score.

## Why

Every deployed AI-text detector is an *instrument*: its numbers depend on the model, the
quantization, the library versions, and the kind of text it reads. Almost nobody who uses
one knows their actual false-positive rate on their own material — they trust a marketing
page. When the accusation lands on a real person, "the tool said 87%" is not evidence.

detcal takes a labeled corpus (known-human + known-AI text) and any pluggable detector, and
emits a **receipts report**: ROC + AUROC with bootstrap confidence intervals, operating
thresholds at target false-positive rates with *achieved* (measured) rates, and a complete
instrument fingerprint — model, quantization **verified at load time** (some library combos
silently ignore 4-bit and load fp16, which shifts scores; detcal refuses to emit numbers
from a load it can't prove), versions, GPU, corpus hash, and the exact reproduce command.

If you use a detector: know its error bars on text like yours before you act on it.
If you're judged by one: these receipts are what an auditable process looks like.

## Quickstart

```bash
pip install -e .            # plus torch/transformers/bitsandbytes in your GPU env
detcal run --corpus hc3-mini --n 100 --detector fast-detect-gpt
# → reports/<ts>/report.md + scores.csv + roc.json + env.json
detcal run --corpus your-labeled.jsonl   # rows: {"text": ..., "label": "human"|"ai"}
```

## What it will never do

- Ship a universal threshold. There isn't one; that's the point.
- Emit a verdict. Scores + measured error rates only.
- Pretend one corpus generalizes. Every report says what it was measured on.

## Status / roadmap

v0.0.1 (burst 1): Fast-DetectGPT (analytic single-model) · HC3 subsample corpus · ROC/AUROC/
threshold receipts with bootstrap CIs · quantization verification.

Next: Binoculars detector · quantization A/B receipts (bf16 vs 4-bit — measured, not assumed) ·
generic REST-API detector adapter (calibrate the commercial tools too) · bring-your-own-corpus
guidance · drift canaries (does your instrument still measure what it measured last term?).

License: Apache-2.0.
