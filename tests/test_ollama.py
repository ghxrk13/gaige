# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Ollama provider tests: the full content-addressed CHAIN (server digest -> manifest ->
weights blob) exercised against a fake store — every tier, every mismatch, no server."""

from __future__ import annotations

import hashlib
import json

import pytest

from gaige.providers import ollama as ollama_mod
from gaige.providers.base import Decoding
from gaige.providers.ollama import Ollama


class _Resp:
    def __init__(self, payload, status=200):
        self._p, self.status_code = payload, status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"http {self.status_code}")

    def json(self):
        return self._p


def build_store(tmp_path, weights=b"pretend gguf weights", corrupt_blob=False, drop_blob=False):
    """A minimal content-addressed store for toy:1b. Returns the manifest digest."""
    wdigest = hashlib.sha256(weights).hexdigest()
    manifest = json.dumps(
        {
            "layers": [
                {
                    "mediaType": "application/vnd.ollama.image.model",
                    "digest": f"sha256:{wdigest}",
                    "size": len(weights),
                },
                {
                    "mediaType": "application/vnd.ollama.image.template",
                    "digest": "sha256:" + "0" * 64,
                    "size": 10,
                },
            ]
        }
    ).encode()
    mdigest = hashlib.sha256(manifest).hexdigest()
    mdir = tmp_path / "manifests" / "registry.ollama.ai" / "library" / "toy"
    mdir.mkdir(parents=True)
    (mdir / "1b").write_bytes(manifest)
    bdir = tmp_path / "blobs"
    bdir.mkdir()
    if not drop_blob:
        (bdir / f"sha256-{wdigest}").write_bytes(b"CORRUPT" if corrupt_blob else weights)
    return mdigest


def patch_tags(monkeypatch, digest):
    monkeypatch.setattr(
        ollama_mod.requests,
        "get",
        lambda url, timeout: _Resp(
            {"models": [{"name": "toy:1b", "digest": f"sha256:{digest}", "size": 1}]}
        ),
    )


def prov(tmp_path):
    return Ollama(model="toy:1b", store_roots=[tmp_path])


def test_full_chain_verified(tmp_path, monkeypatch):
    mdigest = build_store(tmp_path)
    patch_tags(monkeypatch, mdigest)
    ident = prov(tmp_path).connect()
    assert ident["attestation"] == "verified"
    assert "chain verified" in ident["attestation_basis"]
    assert ident["manifest_sha256_rehashed"] == mdigest
    assert len(ident["weights_sha256_rehashed"]) == 64


def test_manifest_mismatch_never_upgrades(tmp_path, monkeypatch):
    build_store(tmp_path)
    patch_tags(monkeypatch, "f" * 64)  # server claims a digest the manifest doesn't hash to
    ident = prov(tmp_path).connect()
    assert ident["attestation"] == "self-reported"
    assert "manifest HASH MISMATCH" in ident["attestation_basis"]


def test_weights_mismatch_never_upgrades(tmp_path, monkeypatch):
    mdigest = build_store(tmp_path, corrupt_blob=True)
    patch_tags(monkeypatch, mdigest)
    ident = prov(tmp_path).connect()
    assert ident["attestation"] == "self-reported"
    assert "weights blob HASH MISMATCH" in ident["attestation_basis"]


def test_missing_blob_stays_self_reported_with_manifest_credit(tmp_path, monkeypatch):
    mdigest = build_store(tmp_path, drop_blob=True)
    patch_tags(monkeypatch, mdigest)
    ident = prov(tmp_path).connect()
    assert ident["attestation"] == "self-reported"
    assert "manifest verified" in ident["attestation_basis"]


def test_unreadable_store_is_plain_self_reported(tmp_path, monkeypatch):
    patch_tags(monkeypatch, "e" * 64)  # digest reported, but tmp store is empty
    ident = prov(tmp_path).connect()
    assert ident["attestation"] == "self-reported"
    assert "store not verified" in ident["attestation_basis"]


def test_opaque_when_model_unknown_to_endpoint(tmp_path, monkeypatch):
    monkeypatch.setattr(ollama_mod.requests, "get", lambda url, timeout: _Resp({"models": []}))
    ident = prov(tmp_path).connect()
    assert ident["attestation"] == "opaque"


def test_model_name_required():
    with pytest.raises(ValueError, match="requires a model name"):
        Ollama().connect()


def test_complete_maps_decoding_to_options(monkeypatch):
    captured = {}

    def fake_post(url, json=None, timeout=None):
        captured.update(json)
        return _Resp({"response": " four"})

    monkeypatch.setattr(ollama_mod.requests, "post", fake_post)
    out = Ollama(model="toy:1b").complete(
        "Two plus two equals", Decoding(temperature=0.0, top_k=5, seed=9, max_new_tokens=8)
    )
    assert out == " four"
    assert captured["model"] == "toy:1b" and captured["stream"] is False
    o = captured["options"]
    assert o["temperature"] == 0.0 and o["top_k"] == 5 and o["seed"] == 9 and o["num_predict"] == 8


def test_metadata_is_local_and_serializable(tmp_path, monkeypatch):
    mdigest = build_store(tmp_path)
    patch_tags(monkeypatch, mdigest)
    meta = prov(tmp_path).metadata()
    assert meta["is_local"] is True and meta["provider"] == "ollama"
    json.dumps(meta)  # the fingerprint must serialize into run.json / series identity
