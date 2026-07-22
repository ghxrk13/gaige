# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Deterministic answer grading. The grading rule is part of the instrument.

Free-text grading is normalized exact match against an authored key plus optional aliases —
chosen over semantic similarity (a second model instrument inside the measurement) and over
LLM-as-judge (a judge model drifts, confounding the very thing a drift study measures). The
normalization pipeline is versioned; GRADING_VERSION belongs in the instrument fingerprint,
and changing the pipeline without bumping it is an instrument change nobody would see.

Multiple-choice grading is argmax over per-option log-likelihoods — a pure forward pass with
no decoding parameters, which is what makes it fit for a control that must stay flat.
"""

from __future__ import annotations

import unicodedata
from collections.abc import Mapping, Sequence

# Bump on ANY behavioral change to normalize()/grading. "nem" = normalized exact match.
GRADING_VERSION = "nem-1"

_LEADING_ARTICLES = ("a", "an", "the")


def normalize(text: str) -> str:
    """NFKC -> casefold -> remove punctuation -> collapse whitespace -> drop ONE leading article.

    Punctuation removal deletes characters in Unicode categories P* (no replacement space),
    so "U.S." -> "us" and "well-known" -> "wellknown". Exactly one leading article token is
    dropped ("the the answer" keeps one). Deterministic by construction; every choice here is
    pinned by GRADING_VERSION.
    """
    s = unicodedata.normalize("NFKC", text).casefold()
    s = "".join(c for c in s if not unicodedata.category(c).startswith("P"))
    words = s.split()
    if len(words) > 1 and words[0] in _LEADING_ARTICLES:
        words = words[1:]
    return " ".join(words)


def grade_free_text(answer: str, key: str, aliases: Sequence[str] = ()) -> dict:
    """Normalized exact match against {key} + aliases.

    Returns {"correct", "normalized_answer", "matched"} where matched is the key or alias
    (as authored) whose normalization the answer hit, or None.
    """
    na = normalize(answer)
    for candidate in (key, *aliases):
        if na == normalize(candidate):
            return {"correct": True, "normalized_answer": na, "matched": candidate}
    return {"correct": False, "normalized_answer": na, "matched": None}


def grade_choice(option_logprobs: Mapping[str, float], key: str) -> dict:
    """Argmax over per-option log-likelihoods vs the keyed option.

    An exact tie at the maximum is graded INCORRECT with chosen=None (conservative: a model
    that cannot separate the key from a distractor has not answered), and reported as
    tie=True so a receipt can count how often the instrument was ambiguous.
    """
    if key not in option_logprobs:
        raise ValueError(f"key {key!r} not among options {sorted(option_logprobs)}")
    best = max(option_logprobs.values())
    winners = sorted(k for k, v in option_logprobs.items() if v == best)
    if len(winners) > 1:
        return {"correct": False, "chosen": None, "tie": True}
    return {"correct": winners[0] == key, "chosen": winners[0], "tie": False}
