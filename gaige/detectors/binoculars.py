# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Binoculars (Hans et al. 2024, arXiv:2401.12070) — detector #2.

Two closely related models look at the same text: an OBSERVER (default falcon-7b) and a
PERFORMER (default falcon-7b-instruct). Following the released reference implementation
(github.com/ahans30/Binoculars, verified 2026-07-22):

    ppl   = mean token cross-entropy of the text under the PERFORMER's logits
    x_ppl = mean cross-entropy of softmax(OBSERVER logits) against the PERFORMER's
            log-probabilities, position-wise
    B     = ppl / x_ppl          # LOWER = more AI-like in the paper's convention

gaige's convention is higher = more AI-like, so `score()` returns **-B** (negation
preserves the ROC exactly); the receipt's score_semantics says so. The paper's published
global thresholds (0.9015 accuracy / 0.8536 low-FPR) are deliberately NOT baked in —
calibration against a corpus is gaige's whole job, and the "receipts gap" around those
global numbers is documented in the research file.

Both models must share a tokenizer (the construction compares distributions over the same
vocabulary at the same positions); a mismatch refuses at load. Quantization is requested
then VERIFIED per model, exactly as for the reference detector — two models means two
proofs on the receipt.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field

from . import base
from .fast_detect_gpt import CUDA_ONLY_QUANT, _available_ram_gb

log = logging.getLogger("gaige.detector")

_TOKENIZER_PROBE = "The 3 quick brown foxes jumped over 2 lazy dogs — naturally."


def _check_tokenizer_match(tok_a, tok_b) -> None:
    """Refuse unless both tokenizers produce identical ids for a probe string."""
    a = tok_a.encode(_TOKENIZER_PROBE)
    b = tok_b.encode(_TOKENIZER_PROBE)
    if a != b:
        raise RuntimeError(
            "Binoculars requires observer and performer to share a tokenizer (the score "
            "compares their distributions position-by-position over one vocabulary); these "
            "two tokenize differently. Pick a related model pair (base + instruct of the "
            "same family)."
        )


@dataclass
class Binoculars:
    observer_id: str = "tiiuae/falcon-7b"
    performer_id: str = "tiiuae/falcon-7b-instruct"
    quant: str = "4bit"  # "4bit" (CUDA only) | "fp16" | "fp32"
    max_tokens: int = 512
    device: str = "auto"
    # None -> the measured floor for this configuration, else a conservative default (9.0:
    # room for TWO 7B loads at 4-bit beside co-resident work); an explicit value is the
    # deliberate escape hatch. See gaige/memfloor.py.
    min_free_gb: float | None = None
    name: str = field(init=False, default="binoculars")
    _loaded: bool = field(init=False, default=False)
    _device: str = field(init=False, default="")
    _device_fallback: bool = field(init=False, default=False)

    # -- device / dtype (mirrors the reference detector's rules) -------------------------

    def resolve_device(self) -> tuple[str, bool]:
        import torch

        if self.device == "cpu":
            return "cpu", False
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device=cuda requested but no CUDA device is available")
            return "cuda", False
        if self.device != "auto":
            raise ValueError(f"unknown device {self.device!r}")
        if torch.cuda.is_available():
            return "cuda", False
        return "cpu", True

    def _effective_quant(self, device: str) -> str:
        if device == "cpu" and self.quant in CUDA_ONLY_QUANT:
            raise RuntimeError(
                "quant=4bit is CUDA-only. On CPU use --quant fp32 — and note TWO 7B models "
                "in fp32 need ~56 GB RAM and are impractically slow; Binoculars is "
                "effectively a GPU detector at 7B scale."
            )
        if device == "cpu" and self.quant == "fp16":
            raise RuntimeError("quant=fp16 on CPU is unsupported; use fp32")
        return self.quant

    # -- load ----------------------------------------------------------------------------

    def _load_one(self, model_id: str, device: str, quant: str):
        import torch
        from transformers import AutoModelForCausalLM

        kwargs: dict = {}
        if device == "cuda":
            kwargs["device_map"] = {"": 0}
            if quant == "4bit":
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
            else:
                kwargs["torch_dtype"] = torch.float16
        else:
            kwargs["torch_dtype"] = torch.float32
        before = torch.cuda.memory_allocated() if device == "cuda" else 0
        with base.mute_torch_dtype_deprecation():
            model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
        if device == "cpu":
            model.to("cpu")
        model.eval()
        if device == "cuda":
            resident = (torch.cuda.memory_allocated() - before) / 1e9
        else:
            resident = sum(p.numel() * p.element_size() for p in model.parameters()) / 1e9
        return model, resident

    def _verify_one(self, model, resident_gb: float, quant: str, role: str) -> dict:
        n4 = 0
        if quant == "4bit":
            import bitsandbytes as bnb

            n4 = sum(1 for m in model.modules() if isinstance(m, bnb.nn.Linear4bit))
            if n4 == 0:
                raise RuntimeError(
                    f"QUANTIZATION SILENTLY IGNORED on the {role}: 4bit requested, zero "
                    f"Linear4bit modules found (resident {resident_gb:.1f} GB). Aborting — "
                    "scores from this load would be receipts fraud."
                )
        return {"linear4bit_modules": n4, "resident_gb": round(resident_gb, 2)}

    def load(self) -> None:
        import torch
        from transformers import AutoTokenizer

        device, fell_back = self.resolve_device()
        quant = self._effective_quant(device)
        self._device, self._device_fallback = device, fell_back
        if fell_back:
            log.warning("no CUDA device; Binoculars on CPU is a DIFFERENT instrument (recorded)")

        from .. import memfloor

        pair = f"{self.observer_id}+{self.performer_id}"
        note = " and Binoculars loads TWO models"
        if device == "cuda":
            free_b, _ = torch.cuda.mem_get_info()
            free = free_b / 1e9
            floor, why = memfloor.effective_floor(self.min_free_gb, self.name, pair, quant, "vram")
            if free < floor:
                raise RuntimeError(memfloor.refusal("free VRAM", free, floor, why, note))
        else:
            avail = _available_ram_gb()
            if avail is not None:
                floor, why = memfloor.effective_floor(
                    self.min_free_gb, self.name, pair, quant, "ram"
                )
                if avail < floor:
                    raise RuntimeError(memfloor.refusal("available RAM", avail, floor, why, note))

        self._tok = AutoTokenizer.from_pretrained(self.observer_id)
        perf_tok = AutoTokenizer.from_pretrained(self.performer_id)
        _check_tokenizer_match(self._tok, perf_tok)

        self._observer, obs_res = self._load_one(self.observer_id, device, quant)
        self._performer, perf_res = self._load_one(self.performer_id, device, quant)
        self._verified = {
            "observer": self._verify_one(self._observer, obs_res, quant, "observer"),
            "performer": self._verify_one(self._performer, perf_res, quant, "performer"),
        }
        self._loaded = True

    # -- score ---------------------------------------------------------------------------

    def score(self, text: str) -> float:
        import torch

        if not self._loaded:
            self.load()
        enc = self._tok(
            text,
            truncation=True,
            max_length=self.max_tokens,
            return_tensors="pt",
            return_token_type_ids=False,
        ).to(self._device)
        labels = enc.input_ids[:, 1:]
        if labels.numel() == 0:
            return 0.0
        with torch.no_grad():
            obs_logits = self._observer(**enc).logits[:, :-1].float()
            perf_logits = self._performer(**enc).logits[:, :-1].float()
        perf_lp = torch.log_softmax(perf_logits, dim=-1)
        # ppl: the text's mean token cross-entropy under the PERFORMER.
        ppl = -perf_lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1).mean()
        # x_ppl: softmax(OBSERVER) cross-entropied against the performer's log-probs.
        x_ppl = -(torch.softmax(obs_logits, dim=-1) * perf_lp).sum(dim=-1).mean()
        b = float((ppl / x_ppl).item())
        return -b  # gaige convention: higher = more AI-like

    # -- fingerprint ---------------------------------------------------------------------

    def metadata(self) -> dict:
        import torch
        import transformers

        versions = {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "cuda": torch.version.cuda,
            "python": platform.python_version(),
        }
        try:
            import bitsandbytes

            versions["bitsandbytes"] = bitsandbytes.__version__
        except Exception:
            pass
        device = self._device or self.device
        compute = {"name": platform.processor() or platform.machine()}
        if device == "cuda":
            compute = {"name": torch.cuda.get_device_name(0)}
        v = getattr(self, "_verified", {"observer": {}, "performer": {}})
        return {
            "detector": self.name,
            "paper": "Hans et al., Binoculars (2024), arXiv:2401.12070; released-implementation construction",
            "model_id": f"{self.observer_id} + {self.performer_id}",
            "observer_id": self.observer_id,
            "performer_id": self.performer_id,
            "quant_requested": self.quant,
            "quant_verified": {
                "linear4bit_modules": sum(x.get("linear4bit_modules", 0) for x in v.values()),
                "resident_gb": round(sum(x.get("resident_gb", 0.0) for x in v.values()), 2),
                "per_model": v,
            },
            "max_tokens": self.max_tokens,
            "versions": versions,
            "device": device,
            "device_requested": self.device,
            "device_fallback": self._device_fallback,
            "model_auto_selected": False,
            "compute": compute,
            "score_semantics": (
                "NEGATED Binoculars ratio -(ppl/x_ppl); higher = more AI-like; RAW criterion "
                "(uncalibrated by design; the paper's global thresholds are deliberately not used)"
            ),
        }
