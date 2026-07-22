# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""In-process transformers provider. Attestation: verified — gaige loaded the weights.

Serves both runner capabilities: greedy/seeded completion, and per-option continuation
log-likelihoods (the MC control path — a pure forward pass with no decoding parameters,
which is exactly why the control stays flat by construction).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .base import CAP_COMPLETE, CAP_OPTION_LOGPROBS, VERIFIED, Decoding


@dataclass
class LocalHF:
    model_id: str
    dtype: str = "fp32"  # fp32 | fp16 (fp16 is CUDA-only, as in the detector)
    device: str = "auto"  # auto | cuda | cpu; auto prefers CUDA, falls back, records it
    name: str = field(init=False, default="local-hf")
    _loaded: bool = field(init=False, default=False)

    def capabilities(self) -> frozenset[str]:
        return frozenset({CAP_COMPLETE, CAP_OPTION_LOGPROBS})

    # -- load ----------------------------------------------------------------------------

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device == "auto":
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._device_fallback = self._device == "cpu"
        elif self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("device=cuda requested but no CUDA device is available")
            self._device, self._device_fallback = "cuda", False
        elif self.device == "cpu":
            self._device, self._device_fallback = "cpu", False
        else:
            raise ValueError(f"unknown device {self.device!r}")
        if self.dtype == "fp16" and self._device == "cpu":
            raise RuntimeError("dtype=fp16 on CPU is unsupported; use fp32")

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self._tok.pad_token_id is None:
            self._tok.pad_token = self._tok.eos_token
        kwargs = {"torch_dtype": torch.float16 if self.dtype == "fp16" else torch.float32}
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self._model.to(self._device)
        self._model.eval()
        self._n_params = sum(p.numel() for p in self._model.parameters())
        self._resident_gb = (
            sum(p.numel() * p.element_size() for p in self._model.parameters()) / 1e9
        )
        self._loaded = True

    def _ensure(self) -> None:
        if not self._loaded:
            self.load()

    # -- capabilities --------------------------------------------------------------------

    def complete(self, prompt: str, decoding: Decoding) -> str:
        import torch

        self._ensure()
        enc = self._tok(prompt, return_tensors="pt", return_token_type_ids=False).to(self._device)
        gen_kwargs: dict = {
            "max_new_tokens": decoding.max_new_tokens,
            "pad_token_id": self._tok.pad_token_id,
        }
        if decoding.temperature == 0.0:
            gen_kwargs["do_sample"] = False
        else:
            gen_kwargs.update(
                do_sample=True,
                temperature=decoding.temperature,
                top_p=decoding.top_p,
                top_k=decoding.top_k,
            )
            if decoding.seed is not None:
                torch.manual_seed(decoding.seed)
        with torch.no_grad():
            out = self._model.generate(**enc, **gen_kwargs)
        new_tokens = out[0, enc["input_ids"].shape[1] :]
        return self._tok.decode(new_tokens, skip_special_tokens=True)

    def option_logprobs(self, prompt: str, options: dict[str, str]) -> dict[str, float]:
        """Total log-likelihood of each option text as a continuation of the prompt."""
        import torch

        self._ensure()
        out: dict[str, float] = {}
        prompt_ids = self._tok(prompt, return_tensors="pt").input_ids
        for label, text in options.items():
            option_ids = self._tok(text, add_special_tokens=False, return_tensors="pt").input_ids
            if option_ids.numel() == 0:
                raise ValueError(f"option {label!r} tokenizes to nothing: {text!r}")
            ids = torch.cat([prompt_ids, option_ids], dim=1).to(self._device)
            with torch.no_grad():
                logits = self._model(input_ids=ids).logits.float()
            lp = torch.log_softmax(logits[:, :-1], dim=-1)
            labels = ids[:, 1:]
            token_lp = lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            n_opt = option_ids.shape[1]
            out[label] = float(token_lp[0, -n_opt:].sum().item())
        return out

    # -- fingerprint ---------------------------------------------------------------------

    def metadata(self) -> dict:
        import platform

        import torch
        import transformers

        self._ensure()
        return {
            "provider": self.name,
            "attestation": VERIFIED,
            "attestation_basis": "in-process load; parameter count and resident bytes measured",
            "is_local": True,
            "model_id": self.model_id,
            "dtype": self.dtype,
            "device": self._device,
            "device_fallback": self._device_fallback,
            "n_params": self._n_params,
            "resident_gb": round(self._resident_gb, 3),
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "python": platform.python_version(),
            },
        }
