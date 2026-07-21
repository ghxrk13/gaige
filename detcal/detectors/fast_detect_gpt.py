"""Fast-DetectGPT (single-model analytic variant).

Score = the analytic sampling discrepancy of Bao et al., "Fast-DetectGPT: Efficient
Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature"
(ICLR 2024), in the sampling==scoring configuration (one forward pass, logits_ref ==
logits_score). Raw criterion only — calibration against a corpus is detcal's job.

Quantization honesty: loading in 4-bit is REQUESTED, then VERIFIED. Some library-version
combinations silently ignore quantization config and load fp16 — which changes both VRAM
and, materially, the score distribution. A detector that cannot prove how it was loaded
does not get to emit numbers here.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field


@dataclass
class FastDetectGPT:
    model_id: str = "tiiuae/falcon-7b"
    quant: str = "4bit"  # "4bit" | "fp16"
    max_tokens: int = 1024
    min_free_gb: float = 8.0
    name: str = field(init=False, default="fast-detect-gpt")
    _loaded: bool = field(init=False, default=False)

    def load(self) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.quant == "4bit" and int(transformers.__version__.split(".")[0]) >= 5:
            # Measured on 2026-07-21 (transformers 5.14.1): BitsAndBytesConfig is misrouted
            # (eetq path selected, zero linear modules quantized) and the load proceeds fp16 —
            # ~13 GB instead of ~5. The verifier below catches it if VRAM allows the load;
            # on smaller GPUs it surfaces as OOM. Known-good: transformers 4.x line.
            print(
                "[detcal][WARN] transformers >=5 detected with quant=4bit — this combo has a "
                "MEASURED silent-quantization failure; expect the load verifier to abort. "
                "Use a transformers 4.x environment for 4-bit runs.",
                flush=True,
            )
        free_b, total_b = torch.cuda.mem_get_info()
        if free_b / 1e9 < self.min_free_gb:
            raise RuntimeError(
                f"refusing to load: {free_b / 1e9:.1f} GB free VRAM < {self.min_free_gb} GB floor "
                "(protects co-resident workloads)"
            )
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        kwargs: dict = {"device_map": {"": 0}}
        if self.quant == "4bit":
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        else:
            kwargs["torch_dtype"] = torch.float16
        before = torch.cuda.memory_allocated()
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except torch.OutOfMemoryError as e:
            raise RuntimeError(
                "OOM during load. If quant=4bit was requested, the library most likely IGNORED "
                "the quantization config and attempted an fp16 load (a measured failure mode of "
                "some transformers versions). This is an instrument-integrity failure, not a "
                "hardware limitation: fix the environment, do not raise the VRAM."
            ) from e
        self._model.eval()
        self._resident_gb = (torch.cuda.memory_allocated() - before) / 1e9
        self._verify_quantization()
        self._loaded = True

    def _verify_quantization(self) -> None:
        """Prove the load matched the request; refuse to score otherwise."""
        if self.quant != "4bit":
            return
        try:
            import bitsandbytes as bnb

            n4 = sum(1 for m in self._model.modules() if isinstance(m, bnb.nn.Linear4bit))
        except Exception as e:  # bnb missing entirely => certainly not quantized
            raise RuntimeError(f"4bit requested but bitsandbytes unusable: {e}") from e
        if n4 == 0:
            raise RuntimeError(
                "QUANTIZATION SILENTLY IGNORED: 4bit requested, zero Linear4bit modules found "
                f"(resident {self._resident_gb:.1f} GB). Version combo likely loads fp16 — "
                "scores from this load would be receipts fraud. Aborting."
            )
        if self._resident_gb > 8.0:
            raise RuntimeError(
                f"4bit requested but resident {self._resident_gb:.1f} GB looks like fp16. Aborting."
            )
        self._n_linear4bit = n4

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
        ).to("cuda")
        labels = enc.input_ids[:, 1:]
        if labels.numel() == 0:
            return 0.0
        with torch.no_grad():
            logits = self._model(**enc).logits[:, :-1].float()
        lab = labels.unsqueeze(-1)
        lprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=lab).squeeze(-1)
        mean_ref = (probs * lprobs).sum(dim=-1)
        var_ref = (probs * torch.square(lprobs)).sum(dim=-1) - torch.square(mean_ref)
        disc = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        return float(disc.mean().item())

    def metadata(self) -> dict:
        import bitsandbytes
        import torch
        import transformers

        return {
            "detector": self.name,
            "paper": "Bao et al., Fast-DetectGPT (ICLR 2024), analytic single-model variant",
            "model_id": self.model_id,
            "quant_requested": self.quant,
            "quant_verified": {
                "linear4bit_modules": getattr(self, "_n_linear4bit", 0),
                "resident_gb": round(getattr(self, "_resident_gb", 0.0), 2),
            },
            "max_tokens": self.max_tokens,
            "versions": {
                "torch": torch.__version__,
                "transformers": transformers.__version__,
                "bitsandbytes": bitsandbytes.__version__,
                "cuda": torch.version.cuda,
                "python": platform.python_version(),
            },
            "gpu": torch.cuda.get_device_name(0),
            "score_semantics": "analytic sampling discrepancy; higher = more AI-like; RAW criterion (uncalibrated by design)",
        }
