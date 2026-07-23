# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""`gaige plan` tests: pure feasibility logic over injected environments — no GPU in CI,
and the presentation rules enforced (legend always present, never a quality column)."""

from __future__ import annotations

from gaige import plan


def env(**kw):
    base = {
        "cpus": 8,
        "ram_gb": 32.0,
        "cuda": False,
        "gpu_name": None,
        "vram_total_gb": None,
        "vram_free_gb": None,
        "ollama_models": None,
        "llama_server_binary": False,
        "gaige_ai_endpoint": None,
    }
    base.update(kw)
    return base


def by_config(rows):
    return {r["config"]: r for r in rows}


def test_cuda_roomy_fits_everything_gpu():
    rows = by_config(
        plan.build_plan(env(cuda=True, gpu_name="X", vram_total_gb=24.0, vram_free_gb=20.0))
    )
    assert rows["detect · fast-detect-gpt · falcon-7b 4bit"]["fits"].startswith("fits now")
    assert rows["detect · fast-detect-gpt · falcon-7b fp16"]["fits"].startswith("fits now")
    assert rows["detect · binoculars · falcon-7b + falcon-7b-instruct 4bit"]["fits"].startswith(
        "fits now"
    )


def test_cuda_tight_excludes_fp16_keeps_4bit():
    """The real bench-beside-the-daemon shape: 11.5 GB free."""
    rows = by_config(
        plan.build_plan(env(cuda=True, gpu_name="X", vram_total_gb=16.4, vram_free_gb=11.5))
    )
    assert rows["detect · fast-detect-gpt · falcon-7b 4bit"]["fits"].startswith("fits now")
    assert "NO — needs 13.7" in rows["detect · fast-detect-gpt · falcon-7b fp16"]["fits"]
    assert rows["detect · binoculars · falcon-7b + falcon-7b-instruct 4bit"]["fits"].startswith(
        "fits now"
    )


def test_cpu_only_machine_gets_honest_nos_and_cpu_yeses():
    rows = by_config(plan.build_plan(env()))
    assert "no CUDA" in rows["detect · fast-detect-gpt · falcon-7b 4bit"]["fits"]
    assert rows["detect · fast-detect-gpt · gpt2-large fp32 (CPU)"]["fits"].startswith("fits now")
    assert "no local ollama" in rows["probes · ollama (served models; complete-only)"]["fits"]


def test_ollama_row_reflects_served_models():
    rows = by_config(plan.build_plan(env(ollama_models=["a:1b", "b:7b"])))
    assert "2 models served" in rows["probes · ollama (served models; complete-only)"]["fits"]


def test_render_carries_legend_and_no_quality_column():
    e = env(cuda=True, gpu_name="X", vram_total_gb=16.0, vram_free_gb=10.0)
    text = plan.render(e, plan.build_plan(e))
    assert "never in a hardware table" in text  # the presentation-rules legend
    assert "AUROC" not in text.split("|")[0]  # no quality column in the table header
    assert "measured" in text and "report 174021" in text  # anchors carry receipts
