# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""llama.cpp server provider (OpenAI-compatible /v1), with graded attestation.

Attestation is earned, not assumed:
- verified       — a local GGUF path was supplied, gaige hashed the file (sha256), and the
                   server's reported model identity matches that artifact.
- self-reported  — the server answers identity questions (/props: build, model path) with
                   version-shaped resolution, but gaige cannot reach the artifact.
- opaque         — the endpoint answers completions but no identity endpoint responds.

Capability note (deliberate): this provider declares COMPLETE only. Per-option continuation
scoring over /v1 is server-version-dependent in llama.cpp; until a specific server build's
logprob path is verified against the in-process reference, the MC control runs on the
local-hf provider. Declaring less than what might work beats silently computing something
that looks fine and means nothing.

The decoding block records what gaige REQUESTED. A server build that ignores a field is
precisely why artifact + build identity live in the fingerprint: same request + different
build = different instrument.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests

from .base import CAP_COMPLETE, OPAQUE, SELF_REPORTED, VERIFIED, Decoding

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


@dataclass
class LlamaCpp:
    endpoint: str  # e.g. http://127.0.0.1:8080
    model: str = ""  # model name to request; llama.cpp usually serves exactly one
    gguf_path: str | None = None  # local artifact; supplying it enables `verified`
    timeout: float = 120.0
    name: str = field(init=False, default="llamacpp")
    _identity: dict | None = field(init=False, default=None)

    def capabilities(self) -> frozenset[str]:
        return frozenset({CAP_COMPLETE})

    # -- identity / attestation ----------------------------------------------------------

    def _fetch_props(self) -> dict | None:
        try:
            r = requests.get(f"{self.endpoint.rstrip('/')}/props", timeout=self.timeout)
            r.raise_for_status()
            return r.json()
        except Exception:
            return None

    def connect(self) -> dict:
        """Resolve identity + attestation. Called once per run; re-checks are the point."""
        props = self._fetch_props()
        identity: dict = {"endpoint_host": urlparse(self.endpoint).hostname or self.endpoint}
        attestation, basis = OPAQUE, "no identity endpoint answered"
        if props is not None:
            served_path = props.get("model_path") or (
                props.get("default_generation_settings") or {}
            ).get("model", "")
            identity.update(
                {
                    "server_model_path": served_path,
                    "build_info": props.get("build_info", ""),
                    "n_ctx": (props.get("default_generation_settings") or {}).get("n_ctx"),
                }
            )
            attestation = SELF_REPORTED
            basis = "server /props reports build + model path; artifact not reachable"
        if self.gguf_path:
            p = Path(self.gguf_path)
            if not p.exists():
                raise FileNotFoundError(f"gguf_path {p} does not exist; cannot attest")
            h = hashlib.sha256()
            with open(p, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
            identity["artifact_sha256"] = h.hexdigest()
            served = Path(identity.get("server_model_path") or "").name
            if props is not None and served == p.name:
                attestation = VERIFIED
                basis = "local GGUF sha256 computed; server-reported model file matches it"
            else:
                basis = (
                    "GGUF hashed, but the server's reported model "
                    f"({served or 'unknown'}) does not match {p.name}; staying at "
                    f"{attestation}"
                )
        self._identity = {"attestation": attestation, "attestation_basis": basis, **identity}
        return self._identity

    # -- capabilities --------------------------------------------------------------------

    def complete(self, prompt: str, decoding: Decoding) -> str:
        payload: dict = {
            "model": self.model or "default",
            "prompt": prompt,
            "max_tokens": decoding.max_new_tokens,
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "stream": False,
        }
        if decoding.top_k:
            payload["top_k"] = decoding.top_k
        if decoding.seed is not None:
            payload["seed"] = decoding.seed
        r = requests.post(
            f"{self.endpoint.rstrip('/')}/v1/completions", json=payload, timeout=self.timeout
        )
        r.raise_for_status()
        return r.json()["choices"][0]["text"]

    def option_logprobs(self, prompt: str, options: dict[str, str]) -> dict[str, float]:
        raise NotImplementedError(
            "llamacpp provider does not declare option_logprobs (see module docstring)"
        )

    # -- fingerprint ---------------------------------------------------------------------

    def metadata(self) -> dict:
        if self._identity is None:
            self.connect()
        host = urlparse(self.endpoint).hostname or ""
        return {
            "provider": self.name,
            "is_local": host in _LOCAL_HOSTS,
            "endpoint": self.endpoint,
            "model_requested": self.model or "default",
            **(self._identity or {}),
        }
