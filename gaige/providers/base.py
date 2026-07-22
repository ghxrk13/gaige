# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Provider protocol: prompt in, text out — with the attestation graded honestly.

A provider is a weaker instrument than an in-process load unless proven otherwise, and
pretending otherwise would gut the thesis. So every provider carries an ATTESTATION level
on every receipt (design: private-notes/design-byo-ai.md, sharpened 2026-07-22):

  verified       gaige structurally proved the artifact — an in-process load with proof,
                 or a local artifact hash (GGUF sha256 / ollama digest) matching the
                 serving process's reported identity.
  self-reported  the provider reports version-shaped identity (build, digest, quant) with
                 enough resolution that an UNCHANGED report is evidence of an unchanged
                 instrument. Recorded and re-checked every run.
  opaque         a marketing-name model string on an endpoint. Receipts carry a
                 measured-during window; thresholds are valid at most while the endpoint's
                 behavior is unchanged, which nothing at this level can attest; drift
                 attribution is impossible and the receipt says so.

Providers are not interchangeable: detector scoring needs full-vocabulary logprobs, probe
running needs prompt->text, the MC control path needs per-option continuation scoring. So a
provider DECLARES capabilities and gaige refuses a combination it cannot compute, naming
what is missing — never silently substituting an approximation.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

VERIFIED = "verified"
SELF_REPORTED = "self-reported"
OPAQUE = "opaque"

# Capability names. complete: prompt -> generated text. option_logprobs: score each of a
# fixed set of continuations by total log-likelihood (the MC control path; forward pass only).
CAP_COMPLETE = "complete"
CAP_OPTION_LOGPROBS = "option_logprobs"


class MissingCapability(RuntimeError):
    """The chosen provider cannot compute what this run needs. Names what is missing."""


@dataclass(frozen=True)
class Decoding:
    """The decoding block of the instrument fingerprint.

    Greedy (temperature 0) is the pre-registered study default — deterministic generation
    shrinks run variance to numerics level. ANY change to any field is an instrument change:
    the registry forks a series on it, and a resume refuses across it.
    """

    temperature: float = 0.0
    top_p: float = 1.0
    top_k: int = 0
    seed: int | None = None
    max_new_tokens: int = 64

    def fingerprint(self) -> dict:
        return asdict(self)


@runtime_checkable
class Provider(Protocol):
    name: str

    def capabilities(self) -> frozenset[str]: ...

    def complete(self, prompt: str, decoding: Decoding) -> str: ...

    def option_logprobs(self, prompt: str, options: dict[str, str]) -> dict[str, float]:
        """Total log-likelihood of each option's text as a continuation of prompt."""
        ...

    def metadata(self) -> dict:
        """Everything a receipt needs: identity, attestation level, is_local, versions."""
        ...


def require(provider: Provider, *caps: str) -> None:
    missing = sorted(set(caps) - set(provider.capabilities()))
    if missing:
        raise MissingCapability(
            f"provider {provider.name!r} lacks capability(ies) {missing}. "
            f"It declares {sorted(provider.capabilities())}. Choose a provider that can "
            "compute what this run needs; gaige does not substitute approximations."
        )


def require_local_or_optin(provider: Provider, allow_remote_text: bool) -> None:
    """The privacy boundary: text leaves this machine only by explicit opt-in.

    SECURITY.md's no-persistence/no-transmission posture is unconditional for local
    providers; a remote endpoint makes it conditional, and the condition is this flag.
    """
    if provider.metadata().get("is_local", False):
        return
    if not allow_remote_text:
        raise RuntimeError(
            f"provider {provider.name!r} is not local: prompts would leave this machine. "
            "Pass --allow-remote-text to opt in explicitly, or use a local provider. "
            "Nothing is sent without this flag; that is a security property, not a default."
        )
