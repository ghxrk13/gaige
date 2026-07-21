# gaige — calibration and drift receipts for AI measurement.
# Copyright (C) 2026 ghxrk13. Licensed under AGPL-3.0-only; see LICENSE.
# Commercial licensing available — see COMMERCIAL.md.

"""Fast-DetectGPT (single-model analytic variant).

Score = the analytic sampling discrepancy of Bao et al., "Fast-DetectGPT: Efficient
Zero-Shot Detection of Machine-Generated Text via Conditional Probability Curvature"
(ICLR 2024), in the sampling==scoring configuration (one forward pass, logits_ref ==
logits_score). Raw criterion only — calibration against a corpus is gaige's job.

Quantization honesty: loading in 4-bit is REQUESTED, then VERIFIED. Some library-version
combinations silently ignore quantization config and load fp16 — which changes both VRAM
and, materially, the score distribution. A detector that cannot prove how it was loaded
does not get to emit numbers here.

Device honesty: the same model on CUDA-4bit and on CPU-fp32 is TWO INSTRUMENTS, not one
instrument in two places. Numerics differ, so thresholds calibrated on one do not transfer
to the other. gaige therefore records device and dtype in the fingerprint and refuses to
pretend a fallback was the thing it fell back from. Running without a GPU is fully
supported; running without SAYING you ran without a GPU is not.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass, field

log = logging.getLogger("gaige.detector")

# Quantizations that only exist on CUDA. bitsandbytes 4-bit has no CPU kernel path.
CUDA_ONLY_QUANT = {"4bit"}


def _available_ram_gb() -> float | None:
    """Best-effort available system RAM, or None if it cannot be determined.

    Deliberately dependency-free and deliberately allowed to fail: refusing to run because we
    could not measure memory would be worse than running and letting the OS complain.
    """
    try:
        if platform.system() == "Linux":
            for line in open("/proc/meminfo", encoding="utf-8"):
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1e6  # kB -> GB
        elif platform.system() == "Windows":
            import ctypes

            class _MemStatus(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            st = _MemStatus()
            st.dwLength = ctypes.sizeof(_MemStatus)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(st))
            return st.ullAvailPhys / 1e9
    except Exception:  # measurement is a convenience, never a gate
        return None
    return None


@dataclass
class FastDetectGPT:
    model_id: str = "tiiuae/falcon-7b"
    quant: str = "4bit"  # "4bit" (CUDA only) | "fp16" | "fp32"
    max_tokens: int = 1024
    device: str = "auto"  # "auto" | "cuda" | "cpu"
    min_free_gb: float = 8.0
    name: str = field(init=False, default="fast-detect-gpt")
    _loaded: bool = field(init=False, default=False)
    _device: str = field(init=False, default="")
    _device_fallback: bool = field(init=False, default=False)

    # -- device / dtype resolution -------------------------------------------------------

    def resolve_device(self) -> tuple[str, bool]:
        """Pick the device. Returns (device, fell_back_from_cuda).

        `auto` prefers CUDA and silently uses CPU when there isn't one — but "silently" only
        in the sense that it does not crash. The fallback is recorded and surfaces in the
        receipt, because a CPU run is a different instrument and the reader must know.
        """
        import torch

        if self.device == "cpu":
            return "cpu", False
        if self.device == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "device=cuda requested but no CUDA device is available. Use --device cpu "
                    "(and a smaller --model; see below) or --device auto."
                )
            return "cuda", False
        if self.device != "auto":
            raise ValueError(f"unknown device {self.device!r}: expected auto, cuda, or cpu")
        if torch.cuda.is_available():
            return "cuda", False
        return "cpu", True

    def _effective_quant(self, device: str) -> str:
        """Reconcile the requested quantization with what the device can actually do."""
        if device == "cpu" and self.quant in CUDA_ONLY_QUANT:
            raise RuntimeError(
                f"quant={self.quant!r} is CUDA-only (bitsandbytes has no CPU 4-bit kernel). "
                "On CPU use --quant fp32. Note that a 7B model on CPU is impractically slow "
                "(measured ~20-36 s/sample vs ~0.3-1.5 s on GPU): pair --device cpu with a "
                "small --model such as gpt2-large or EleutherAI/gpt-neo-1.3B. That is a "
                "DIFFERENT INSTRUMENT and needs its own calibration."
            )
        if device == "cpu" and self.quant == "fp16":
            # fp16 matmul on CPU is either unsupported or emulated and slow; fp32 is the
            # honest default. Say so rather than quietly substituting.
            raise RuntimeError(
                "quant=fp16 on CPU is unsupported or badly emulated on most CPUs. "
                "Use --quant fp32 for CPU runs."
            )
        return self.quant

    # -- load ----------------------------------------------------------------------------

    def load(self) -> None:
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer

        device, fell_back = self.resolve_device()
        quant = self._effective_quant(device)
        self._device, self._device_fallback = device, fell_back
        if fell_back:
            log.warning(
                "no CUDA device found; running on CPU. This is a different instrument than a "
                "GPU run - thresholds calibrated on GPU do not transfer. The receipt records it."
            )

        if quant == "4bit" and int(transformers.__version__.split(".")[0]) >= 5:
            # Measured on 2026-07-21 (transformers 5.14.1): BitsAndBytesConfig is misrouted
            # (eetq path selected, zero linear modules quantized) and the load proceeds fp16 —
            # ~13 GB instead of ~5. The verifier below catches it if VRAM allows the load;
            # on smaller GPUs it surfaces as OOM. Known-good: transformers 4.x line.
            log.warning(
                "transformers >=5 with quant=4bit has a MEASURED silent-quantization failure; "
                "expect the load verifier to abort. Use a transformers 4.x environment."
            )

        self._check_memory_floor(device)

        self._tok = AutoTokenizer.from_pretrained(self.model_id)
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
        try:
            self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        except Exception as e:
            if device == "cuda" and isinstance(e, torch.OutOfMemoryError):
                raise RuntimeError(
                    "OOM during load. If quant=4bit was requested, the library most likely "
                    "IGNORED the quantization config and attempted an fp16 load (a measured "
                    "failure mode of some transformers versions). This is an instrument-integrity "
                    "failure, not a hardware limitation: fix the environment, do not raise the VRAM."
                ) from e
            if isinstance(e, MemoryError):
                raise RuntimeError(
                    f"out of system memory loading {self.model_id} on CPU. Use a smaller --model; "
                    "a 7B model in fp32 needs roughly 28 GB of RAM."
                ) from e
            raise

        if device == "cpu":
            self._model.to("cpu")
        self._model.eval()

        if device == "cuda":
            self._resident_gb = (torch.cuda.memory_allocated() - before) / 1e9
        else:
            self._resident_gb = (
                sum(p.numel() * p.element_size() for p in self._model.parameters()) / 1e9
            )

        self._verify_quantization(quant)
        self._loaded = True

    def _check_memory_floor(self, device: str) -> None:
        """Refuse to start a load that would evict co-resident work."""
        import torch

        if device == "cuda":
            free_b, _total_b = torch.cuda.mem_get_info()
            if free_b / 1e9 < self.min_free_gb:
                raise RuntimeError(
                    f"refusing to load: {free_b / 1e9:.1f} GB free VRAM < {self.min_free_gb} GB "
                    "floor (protects co-resident workloads)"
                )
            return
        avail = _available_ram_gb()
        if avail is None:
            log.info("could not determine available RAM; skipping the memory floor check")
        elif avail < self.min_free_gb:
            raise RuntimeError(
                f"refusing to load: {avail:.1f} GB available RAM < {self.min_free_gb} GB floor. "
                "Lower min_free_gb deliberately, or use a smaller --model."
            )

    def _verify_quantization(self, quant: str) -> None:
        """Prove the load matched the request; refuse to score otherwise."""
        self._n_linear4bit = 0
        if quant != "4bit":
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
            logits = self._model(**enc).logits[:, :-1].float()
        lab = labels.unsqueeze(-1)
        lprobs = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        log_likelihood = lprobs.gather(dim=-1, index=lab).squeeze(-1)
        mean_ref = (probs * lprobs).sum(dim=-1)
        var_ref = (probs * torch.square(lprobs)).sum(dim=-1) - torch.square(mean_ref)
        disc = (log_likelihood.sum(dim=-1) - mean_ref.sum(dim=-1)) / var_ref.sum(dim=-1).sqrt()
        return float(disc.mean().item())

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
        try:  # only meaningful when a bitsandbytes path was actually used
            import bitsandbytes

            versions["bitsandbytes"] = bitsandbytes.__version__
        except Exception:
            pass

        device = self._device or self.device
        compute = {"name": platform.processor() or platform.machine()}
        if device == "cuda":
            compute = {"name": torch.cuda.get_device_name(0)}

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
            "versions": versions,
            "device": device,
            "device_requested": self.device,
            "device_fallback": self._device_fallback,
            "compute": compute,
            "score_semantics": "analytic sampling discrepancy; higher = more AI-like; RAW criterion (uncalibrated by design)",
        }
