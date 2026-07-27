# Security

gaige is pre-1.0 and single-maintainer. This document says what it defends against, what it
does not, and how to report a problem — because a tool that asks you to trust its numbers owes
you an explicit account of its own trust boundaries.

## Reporting

Email **ghxrk13@proton.me**. Please do not open a public issue for a suspected vulnerability.
Expect an acknowledgement within a week; this is a small unfunded project, and that
limitation is stated rather than papered over.

## What gaige processes, and from where

| Input | Trust | Handling |
|---|---|---|
| Corpus JSONL (`--corpus`) | untrusted | Schema-validated per row; rows lacking `text` + a `human`/`ai` label are rejected with the line number |
| `scores.csv` (`analyze --scores`) | untrusted | Required columns enforced; labels constrained to `human`/`ai`; non-numeric scores raise |
| Report directory (`--report`) | untrusted | JSON parsed; a directory without `scores.csv` is rejected as not-a-report |
| Document to score (`score --file`/stdin) | untrusted | Read as UTF-8 text, tokenized, scored. Never persisted — see below |
| **Model weights** (HuggingFace `--model`) | **TRUSTED — see below** | Loaded via `transformers` without `trust_remote_code` |

## The privacy property

`gaige score` is designed so that **the text you score is never written to disk**: no logging,
no cache, no telemetry, no network egress of document content. The scored document exists in
memory for the duration of the call and nowhere else. This matters because the realistic user
is someone checking their own writing, or an institution checking a student's — the content is
sensitive by construction.

This is a **claim that should be verified, not believed.** gaige emits no network calls during
scoring and writes only to paths you pass explicitly. If you find any path by which scored text
is persisted or transmitted, that is a security bug — report it.

## What gaige does NOT defend against

Stated plainly, because a credibility instrument that overstates its own security is
self-refuting:

- **Malicious model weights.** gaige loads models you name from HuggingFace. `trust_remote_code`
  is not enabled, but model loading is still a supply-chain surface: verify the models you use
  and prefer pinned revisions. gaige cannot vouch for third-party weights.
- **Adversarial text.** Detector scores can be manipulated by a motivated author; this is a
  measured property of every detector in the literature, not a gaige defect. gaige's answer is
  to report error bars and degradation honestly, never to claim robustness it cannot show.
- **Untrusted corpora as ground truth.** gaige validates a corpus's *shape*, not its *labels*.
  A corpus with wrong labels produces confidently wrong thresholds. Provenance is your job;
  gaige records the sha256 so at least it is auditable.
- **Multi-tenant or hostile-local use.** gaige is a local CLI. It has no authentication, no
  sandboxing, and no isolation between invocations. Do not expose it as a network service.
- **Resource exhaustion.** A large corpus or long documents will consume memory and time
  proportionally. There is a VRAM/RAM floor check at load, not a general DoS defence.

## Dependencies

Core runtime is deliberately small — `numpy` and `requests`. The GPU stack (`torch`,
`transformers`, `bitsandbytes`, `accelerate`) is an optional extra, imported lazily, so a
machine without CUDA runs the calibration and reporting layer with a much smaller surface.

`transformers` is pinned `<5`: version 5.14.1 was **measured** silently ignoring 4-bit
quantization config and loading fp16 instead, which changes the score distribution. That is an
instrument-integrity failure, and gaige refuses to emit numbers from a load it cannot verify.
See `PROGRESS.md` for the incident receipt.
