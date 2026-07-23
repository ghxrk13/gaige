# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Ollama provider, with attestation earned the artifact way where possible.

Ollama's store is a content-addressed CHAIN: the digest reported by /api/tags names the
MANIFEST file; the manifest declares layer digests; the model layer digest names the
weights blob. So the honest artifact verification hashes the chain with gaige's own hands:
sha256(manifest file) must equal the server digest, and sha256(weights blob) must equal
the manifest's declared model-layer digest — both match → `verified`. Digest reported but
store unreadable (remote endpoint, permissions) → `self-reported` (version-shaped identity
meeting the change-detection criterion). No identity → `opaque`. Any mismatch anywhere is
reported loudly and never upgraded.

Capabilities: COMPLETE only. Ollama exposes no stable full-vocabulary logprob API, so the
MC control path and P(True) stay on local-hf — declaring less than what might work beats
computing something that looks fine and means nothing (same honesty as llamacpp).

Fleet note (bench): model LOADS go through vram-guard.sh per the fleet's VRAM rules; this
provider only talks to a model an operator already chose to serve.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlparse

import requests

from .base import CAP_COMPLETE, OPAQUE, SELF_REPORTED, VERIFIED, Decoding

_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}


def default_store_roots() -> list[Path]:
    """Candidate ollama store roots: env override, user home, Linux service home."""
    import os

    roots = []
    if os.environ.get("OLLAMA_MODELS"):
        roots.append(Path(os.environ["OLLAMA_MODELS"]))
    roots.append(Path.home() / ".ollama" / "models")
    roots.append(Path("/usr/share/ollama/.ollama/models"))
    return roots


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Ollama:
    endpoint: str = "http://127.0.0.1:11434"
    model: str = ""  # required; ollama serves many models by name:tag
    store_roots: list[Path] | None = None
    timeout: float = 300.0
    name: str = field(init=False, default="ollama")
    _identity: dict | None = field(init=False, default=None)

    def capabilities(self) -> frozenset[str]:
        return frozenset({CAP_COMPLETE})

    # -- identity / attestation ----------------------------------------------------------

    def connect(self) -> dict:
        if not self.model:
            raise ValueError("ollama provider requires a model name (e.g. qwen2.5:7b-instruct)")
        identity: dict = {"endpoint_host": urlparse(self.endpoint).hostname or self.endpoint}
        digest = None
        try:
            r = requests.get(f"{self.endpoint.rstrip('/')}/api/tags", timeout=self.timeout)
            r.raise_for_status()
            for m in r.json().get("models", []):
                if m.get("name") == self.model:
                    digest = m.get("digest")
                    identity.update(
                        {
                            "server_digest": digest,
                            "size_bytes": m.get("size"),
                            "modified_at": m.get("modified_at"),
                        }
                    )
                    break
        except Exception:
            pass

        if digest is None:
            attestation, basis = (
                OPAQUE,
                ("endpoint did not report this model's digest; identity rests on the name alone"),
            )
        else:
            attestation, basis = (
                SELF_REPORTED,
                (
                    "server-reported manifest digest (version-shaped identity; an unchanged "
                    "digest is evidence of an unchanged artifact); store not verified"
                ),
            )
            chain = self._verify_chain(digest.split(":")[-1])
            if chain is not None:
                attestation2, basis2, extra = chain
                attestation, basis = attestation2, basis2
                identity.update(extra)
        self._identity = {"attestation": attestation, "attestation_basis": basis, **identity}
        return self._identity

    def _manifest_path(self, root: Path) -> Path:
        # "qwen2.5:7b-instruct" -> manifests/registry.ollama.ai/library/qwen2.5/7b-instruct
        name, _, tag = self.model.partition(":")
        tag = tag or "latest"
        ns = name if "/" in name else f"library/{name}"
        return root / "manifests" / "registry.ollama.ai" / ns / tag

    def _verify_chain(self, digest_hex: str):
        """Hash the manifest against the server digest, then the weights blob against the
        manifest's declared model-layer digest. Returns (attestation, basis, extra) or None
        when no store was readable (stay self-reported silently — remote is normal)."""
        import json as _json

        for root in self.store_roots or default_store_roots():
            mpath = self._manifest_path(root)
            try:
                manifest_hash = _sha256_file(mpath)
            except OSError:
                continue
            if manifest_hash != digest_hex:
                return (
                    SELF_REPORTED,
                    "manifest HASH MISMATCH against the server-reported digest — staying at "
                    "self-reported; investigate the store",
                    {"manifest_sha256_rehashed": manifest_hash},
                )
            try:
                layers = _json.loads(mpath.read_text(encoding="utf-8"))["layers"]
                model_layer = next(ly for ly in layers if ly["mediaType"].endswith("image.model"))
                declared = model_layer["digest"].split(":")[-1]
                blob_hash = _sha256_file(root / "blobs" / f"sha256-{declared}")
            except (OSError, KeyError, StopIteration, ValueError):
                return (
                    SELF_REPORTED,
                    "manifest verified by gaige's own hash, but the weights blob was not "
                    "readable/parseable — staying at self-reported",
                    {"manifest_sha256_rehashed": manifest_hash},
                )
            if blob_hash != declared:
                return (
                    SELF_REPORTED,
                    "weights blob HASH MISMATCH against the manifest's declared layer digest "
                    "— staying at self-reported; investigate the store",
                    {
                        "manifest_sha256_rehashed": manifest_hash,
                        "weights_sha256_rehashed": blob_hash,
                    },
                )
            return (
                VERIFIED,
                "gaige re-hashed the manifest (matches the server digest) AND the weights "
                "blob (matches the manifest's declared layer digest) — chain verified",
                {
                    "manifest_sha256_rehashed": manifest_hash,
                    "weights_sha256_rehashed": blob_hash,
                    "weights_size_bytes": model_layer.get("size"),
                },
            )
        return None

    # -- capabilities --------------------------------------------------------------------

    def complete(self, prompt: str, decoding: Decoding) -> str:
        options: dict = {
            "temperature": decoding.temperature,
            "top_p": decoding.top_p,
            "num_predict": decoding.max_new_tokens,
        }
        if decoding.top_k:
            options["top_k"] = decoding.top_k
        if decoding.seed is not None:
            options["seed"] = decoding.seed
        r = requests.post(
            f"{self.endpoint.rstrip('/')}/api/generate",
            json={"model": self.model, "prompt": prompt, "stream": False, "options": options},
            timeout=self.timeout,
        )
        r.raise_for_status()
        return r.json().get("response", "")

    def option_logprobs(self, prompt: str, options: dict[str, str]) -> dict[str, float]:
        raise NotImplementedError(
            "ollama provider does not declare option_logprobs (no stable full-vocab "
            "logprob API; use local-hf for the MC control and P(True))"
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
            "model_requested": self.model,
            **(self._identity or {}),
        }
