# What the aggregate hides

*A receipted note on one RAID slice. Numbers pinned from the run of 2026-07-25; every
number sits beside the instrument that produced it. Slice thresholds describe that slice,
nothing more.*

A single number over a mixed corpus is a claim that the mixture doesn't matter. We measured
the mixture. It matters more than the number.

## The run

A stratified slice of RAID (Dugan et al., ACL 2024, arXiv:2405.07940) — 2 generators ×
2 domains × 2 attacks, `raid g2×d2×a2`, per-cell reservoir sampling, seed 17 — scored by
one instrument: Fast-DetectGPT, falcon-7b, 4-bit, cuda. The aggregate receipt reads
AUROC 0.9285 and TPR 61.5% at the 1%-FPR operating point. Both numbers are true. Neither
describes any particular text you will score.

## Three things that aggregate hides

**1. The corpus is in the number.** The same instrument — Fast-DetectGPT, falcon-7b,
4-bit — reads TPR 86.0% at 1%-FPR (AUROC 0.9720) on its reference corpus (hc3-mini, n=100,
seed=17). Nothing changed but the corpus, and the operating point moved 24.5 points. An
advertised detection rate is a property of the corpus it was calibrated on. That is RAID's
own recommendation — calibrate on in-domain data before use — reproduced on our instrument
against the raid g2×d2×a2 slice.

**2. The mixture is in the number.** Stratify the same run by decoding strategy and the
61.5% aggregate splits into greedy 87.6% vs sampled 39.7% TPR at 1%-FPR — a 47.9-point
spread inside one aggregate (raid g2×d2×a2 slice, same instrument, same threshold). The
aggregate is an average over a mixture the deployer does not control and usually cannot
see. If the stream you score is sampled-decoding output, the honest headline for this
slice is 39.7, not 61.5 — and nothing on the aggregate row warns you.

**3. Where n runs out, the instrument refuses.** A conformal guarantee at α=.005 needs at
least 199 human calibration rows (the split-conformal construction of arXiv:2505.05084;
gaige refuses below the floor rather than rounding past it). This slice carries at most
120 human rows by construction — humans are sampled per domain, 60 × 2 domains — so the
α=.005 guarantee was refused, out loud, on the run. The refusal is a result: a tighter
guarantee than the data supports would not have been a guarantee.

## Measurement, not a detector complaint

Fast-DetectGPT is not the finding; it is among the strongest zero-shot detectors
published. The finding is about aggregates: any single-threshold instrument scoring a
mixed stream has per-stratum error rates that no aggregate row can show, and the strata
follow axes the deployer rarely sees — decoding here; length and style elsewhere, where
score distributions measurably differ (arXiv 2502.04528, KS_max 0.3081, p<0.01). The
honest form of the answer is per-subgroup rates with intervals, and refusal where n is too
small to say. That is the form a gaige report emits.

## Reproduce

```bash
gaige corpus prepare-raid --generators <...> --domains <...> --attacks <...> \
    --per-cell 60 --seed 17
gaige run --corpus corpora/raid-g2d2a2-n60-s17.jsonl \
    --detector fast-detect-gpt --model tiiuae/falcon-7b --quant 4bit --device cuda
```

The exact generator/domain/attack lists ride in the slice's provenance block and the run's
`env.json` verbatim; the corpus sha256 is the slice's identity, and the report's
fingerprint section is the instrument's. Receipt of record: the run of 2026-07-25, landed
with the adapter in commit 5939940. This note is held to the same bar as a report by the
claims-policy tests: every number beside its instrument, refuted claims blocked, no
verdict language.
