# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Kadavath-style P(True): the model's own confidence in its own answer, read from logits.

Following Kadavath et al., "Language Models (Mostly) Know What They Know": present the
question and the model's proposed answer, ask whether the answer is true, and read
P(True) = softmax over the {True, False} continuations' log-likelihoods. A pure forward
pass — no sampling, no decoding parameters, deterministic.

The prompt template is part of the instrument: its hash enters the fingerprint, and a
changed template forks the series like any other instrument change. Bump PTRUE_VERSION on
ANY change to the template or the option strings.
"""

from __future__ import annotations

import hashlib
import math

from .providers.base import CAP_OPTION_LOGPROBS, Provider, require

PTRUE_VERSION = "ptrue-1"

PTRUE_TEMPLATE = (
    "Question: {prompt}\n"
    "Proposed answer: {answer}\n"
    "Is the proposed answer true? Answer True or False.\n"
    "Answer:"
)

# Leading spaces matter for GPT-family tokenizers; they are part of the pinned instrument.
PTRUE_OPTIONS = {"True": " True", "False": " False"}


def template_fingerprint() -> dict:
    material = PTRUE_TEMPLATE + "||" + repr(sorted(PTRUE_OPTIONS.items()))
    return {
        "version": PTRUE_VERSION,
        "template_sha256": hashlib.sha256(material.encode("utf-8")).hexdigest(),
    }


def ptrue_score(provider: Provider, prompt: str, answer: str) -> float:
    """P(True) in [0, 1] for the provider's own proposed answer to this prompt."""
    require(provider, CAP_OPTION_LOGPROBS)
    lp = provider.option_logprobs(
        PTRUE_TEMPLATE.format(prompt=prompt, answer=answer), dict(PTRUE_OPTIONS)
    )
    lt, lf = lp["True"], lp["False"]
    m = max(lt, lf)  # stable two-way softmax
    et, ef = math.exp(lt - m), math.exp(lf - m)
    return float(et / (et + ef))
