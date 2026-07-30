# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""`gaige plan`: what can THIS machine run, at what cost, with what attestation.

Feasibility and cost ONLY — deliberately no separation/quality column (the
cross-instrument presentation rules: AUROC/TPR are one-instrument-on-one-corpus properties
that live in receipts, never in a hardware table). Every runtime anchor below is a
MEASURED number with its receipt named; configurations without a receipt say "unmeasured"
instead of guessing.

Works without torch installed (analysis-only machines get the CPU view).
"""

from __future__ import annotations

import os
import shutil

from . import memfloor


def _floor_needs(detector: str, instrument: str, quant: str) -> dict:
    """Detector rows take their needs from the memory-floor table (single source: the
    number `plan` prints and the number the loader enforces cannot disagree)."""
    f = memfloor.MEASURED[(detector, instrument, quant)]
    return {"vram_free_gb": f.gb} if f.kind == "vram" else {"ram_gb": f.gb}


# Measured cost anchors. Provenance: bench receipts, 2026-07-22 (pinned env). A different
# machine will differ — the anchor names its instrument so nobody mistakes it for a spec.
CONFIGS = [
    {
        "config": "detect · fast-detect-gpt · falcon-7b 4bit",
        "needs": _floor_needs("fast-detect-gpt", "tiiuae/falcon-7b", "4bit"),
        "attestation": "verified (Linear4bit count + resident bytes)",
        "anchor": "200 texts in 25 s (~0.13 s/sample) — bench 2026-07-22, report 163959",
    },
    {
        "config": "detect · fast-detect-gpt · falcon-7b fp16",
        "needs": _floor_needs("fast-detect-gpt", "tiiuae/falcon-7b", "fp16"),
        "attestation": "verified (resident bytes)",
        "anchor": "200 texts ≈ 2 min incl. load — bench 2026-07-22, report 221356",
    },
    {
        "config": "detect · fast-detect-gpt · gpt2-large fp32 (CPU)",
        "needs": _floor_needs("fast-detect-gpt", "gpt2-large", "fp32"),
        "attestation": "verified",
        "anchor": "0.64 s/sample — bench 2026-07-22, report 174021 (measured CPU default)",
    },
    {
        "config": "detect · binoculars · falcon-7b + falcon-7b-instruct 4bit",
        "needs": _floor_needs("binoculars", "tiiuae/falcon-7b+tiiuae/falcon-7b-instruct", "4bit"),
        "attestation": "verified (both models proven on one receipt)",
        "anchor": "8.07 GB resident, two forward passes/text — bench 2026-07-22, report 213952",
    },
    {
        "config": "probes · local-hf (any HF causal model; M3-capable)",
        "needs": {"ram_gb": 6.0},
        "attestation": "verified (in-process)",
        "anchor": "gpt2 toy set: 4 probes ≈ 1 s CPU — bench 2026-07-22",
    },
    {
        "config": "probes · llamacpp (GGUF server; complete-only)",
        "needs": {"ram_gb": 4.0},
        "attestation": "verified with --gguf (artifact sha256 = server identity)",
        "anchor": "Qwen2.5-1.5B q4: 20 probes ≈ 30 s CPU — bench 2026-07-22, series 89fb2417c347",
    },
    {
        "config": "probes · ollama (served models; complete-only)",
        "needs": {"ollama": True},
        "attestation": "verified when the store is readable (manifest + weights chain re-hashed)",
        "anchor": "qwen2.5:7b: 20 probes registered — bench 2026-07-22, series 10c246457f8d",
    },
]

LEGEND = (
    "Feasibility and measured cost ONLY — no quality column, deliberately: separation "
    "(AUROC/TPR) is a property of one instrument on one corpus and lives in receipts "
    "(reports/*/report.md), never in a hardware table. Thresholds and scores do not "
    "transfer between configurations."
)


def inspect_environment() -> dict:
    env: dict = {
        "cpus": os.cpu_count(),
        "ram_gb": None,
        "cuda": False,
        "gpu_name": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "ollama_models": None,
        "llama_server_binary": bool(shutil.which("llama-server")),
        "gaige_ai_endpoint": os.environ.get("GAIGE_AI_ENDPOINT"),
    }
    try:
        from .detectors.fast_detect_gpt import _available_ram_gb

        env["ram_gb"] = _available_ram_gb()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            free_b, total_b = torch.cuda.mem_get_info()
            env.update(
                cuda=True,
                gpu_name=torch.cuda.get_device_name(0),
                vram_total_gb=round(total_b / 1e9, 1),
                vram_free_gb=round(free_b / 1e9, 1),
            )
    except Exception:
        pass  # analysis-only machine: the CPU view is still a real answer
    try:
        import requests

        r = requests.get("http://127.0.0.1:11434/api/tags", timeout=2)
        r.raise_for_status()
        env["ollama_models"] = [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return env


def _fits(needs: dict, env: dict) -> str:
    if "ollama" in needs:
        if env.get("ollama_models") is None:
            return "NO — no local ollama endpoint answering"
        n = len(env["ollama_models"])
        return f"fits now ({n} model{'s' if n != 1 else ''} served)"
    if "vram_free_gb" in needs:
        if not env.get("cuda"):
            return "NO — no CUDA device"
        free = env.get("vram_free_gb") or 0.0
        need = needs["vram_free_gb"]
        if free >= need:
            return f"fits now ({free:.1f} GB free ≥ {need:.1f} floor)"
        return f"NO — needs {need:.1f} GB free VRAM, have {free:.1f}"
    if "ram_gb" in needs:
        ram = env.get("ram_gb")
        if ram is None:
            return f"unknown RAM — needs ~{needs['ram_gb']:.0f} GB available"
        if ram >= needs["ram_gb"]:
            return f"fits now ({ram:.0f} GB RAM available)"
        return f"NO — needs ~{needs['ram_gb']:.0f} GB RAM, have {ram:.0f}"
    return "unknown"


def build_plan(env: dict) -> list[dict]:
    return [
        {
            "config": c["config"],
            "fits": _fits(c["needs"], env),
            "attestation": c["attestation"],
            "anchor": c["anchor"],
        }
        for c in CONFIGS
    ]


def render(env: dict, rows: list[dict]) -> str:
    lines = [
        "# gaige plan — what this machine can run, measured",
        "",
        f"machine: {env['cpus']} CPUs · RAM avail {env['ram_gb']:.0f} GB"
        if env.get("ram_gb")
        else f"machine: {env['cpus']} CPUs",
    ]
    if env.get("cuda"):
        lines.append(
            f"gpu: {env['gpu_name']} · {env['vram_free_gb']} GB free of {env['vram_total_gb']} GB"
        )
    else:
        lines.append("gpu: none visible (CPU-only view)")
    if env.get("ollama_models"):
        lines.append(f"ollama: serving {', '.join(env['ollama_models'])}")
    if env.get("llama_server_binary"):
        lines.append("llama-server: on PATH")
    lines += [
        "",
        "| configuration | fits now? | attestation | measured cost (with receipt) |",
        "|---|---|---|---|",
    ]
    for r in rows:
        lines.append(f"| {r['config']} | {r['fits']} | {r['attestation']} | {r['anchor']} |")
    lines += ["", LEGEND]
    return "\n".join(lines)
