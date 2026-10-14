"""Inference backends behind one interface.

Add new runtimes here and nowhere else. Two are provided:

* ``HFBackend``       : transformers, manual decode loop so that time to first
                        token and time per output token are measured separately
                        rather than inferred from a total.
* ``LlamaCppBackend`` : llama-cpp-python over GGUF weights, which is the path
                        that matters once you care about quantized on-device
                        footprints.

Both imports are lazy. Nothing in this module should run at import time, so
that ``cli device`` and ``cli predict`` work on a bare checkout.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .registry import ModelSpec


@dataclass
class GenerationTrace:
    """Per-token timing for one generate call."""

    ttft_ms: float
    per_token_ms: list[float] = field(default_factory=list)
    generated_tokens: int = 0

    @property
    def tpot_ms(self) -> float:
        """Mean time per output token, excluding the first (prefill) token."""
        if not self.per_token_ms:
            return 0.0
        return sum(self.per_token_ms) / len(self.per_token_ms)

    @property
    def total_ms(self) -> float:
        return self.ttft_ms + sum(self.per_token_ms)


class Backend:
    """Interface every backend implements."""

    name = "abstract"

    def __init__(self, spec: ModelSpec, quantization: str = "none") -> None:
        self.spec = spec
        self.quantization = quantization

    def load(self) -> None:
        raise NotImplementedError

    def unload(self) -> None:
        pass

    def make_prompt(self, prompt_tokens: int) -> str:
        """A prompt of approximately ``prompt_tokens`` tokens.

        Uses a fixed repeated passage rather than random tokens so that every
        arm sees identical, in-distribution text. Random token ids would change
        both routing behaviour and tokenizer statistics between arms.
        """
        passage = (
            "The question of whether sparse activation reduces the real cost of "
            "inference depends entirely on which resource is scarce. "
        )
        # Roughly 25 tokens per repetition of the passage above.
        repeats = max(1, prompt_tokens // 25)
        return (passage * repeats).strip()

    def generate(self, prompt: str, max_new_tokens: int, concurrency: int = 1) -> GenerationTrace:
        raise NotImplementedError


class HFBackend(Backend):
    """transformers backend with an explicit, instrumented decode loop."""

    name = "hf"

    def __init__(self, spec: ModelSpec, quantization: str = "none", device: str | None = None) -> None:
        super().__init__(spec, quantization)
        self.device = device
        self.model = None
        self.tokenizer = None

    def load(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if self.device is None:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"

        dtype = torch.float16 if self.device != "cpu" else torch.float32
        kwargs: dict = {"dtype": dtype, "low_cpu_mem_usage": True}

        if self.quantization in ("int8", "int4"):
            from transformers import BitsAndBytesConfig

            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=self.quantization == "int8",
                load_in_4bit=self.quantization == "int4",
                bnb_4bit_compute_dtype=dtype,
            )
            kwargs["device_map"] = "auto"

        self.tokenizer = AutoTokenizer.from_pretrained(self.spec.hf_id)
        self.model = AutoModelForCausalLM.from_pretrained(self.spec.hf_id, **kwargs)
        if "device_map" not in kwargs:
            self.model = self.model.to(self.device)
        self.model.eval()

    def unload(self) -> None:
        import gc

        self.model = None
        self.tokenizer = None
        gc.collect()
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass

    def _sync(self) -> None:
        """Accelerator work is asynchronous. Without this every timing is a lie."""
        import torch

        if self.device == "cuda":
            torch.cuda.synchronize()
        elif self.device == "mps":
            torch.mps.synchronize()

    def generate(self, prompt: str, max_new_tokens: int, concurrency: int = 1) -> GenerationTrace:
        import torch

        if self.model is None or self.tokenizer is None:
            raise RuntimeError("call load() before generate()")

        batch = [prompt] * concurrency
        encoded = self.tokenizer(batch, return_tensors="pt", padding=False)
        input_ids = encoded["input_ids"].to(self.model.device)

        with torch.no_grad():
            # Prefill. Everything up to the first sampled token is TTFT.
            self._sync()
            start = time.perf_counter()
            out = self.model(input_ids=input_ids, use_cache=True)
            next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            self._sync()
            ttft_ms = (time.perf_counter() - start) * 1000.0

            past = out.past_key_values
            per_token_ms: list[float] = []
            generated = 1

            # Decode. One forward per token, so each step is timed on its own.
            for _ in range(max_new_tokens - 1):
                step_start = time.perf_counter()
                out = self.model(input_ids=next_ids, past_key_values=past, use_cache=True)
                past = out.past_key_values
                next_ids = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                self._sync()
                per_token_ms.append((time.perf_counter() - step_start) * 1000.0)
                generated += 1

        return GenerationTrace(
            ttft_ms=ttft_ms, per_token_ms=per_token_ms, generated_tokens=generated
        )


class LlamaCppBackend(Backend):
    """llama-cpp-python over GGUF weights.

    This is the realistic on-device path: quantized weights, memory mapped,
    with the option to leave part of the model on disk. That last property is
    exactly what makes a large MoE "run" on a small device while being slow,
    so this backend is where the paper's memory story becomes visible.
    """

    name = "llamacpp"

    def __init__(
        self,
        spec: ModelSpec,
        quantization: str = "q4_k_m",
        n_threads: int | None = None,
        n_gpu_layers: int = 0,
    ) -> None:
        super().__init__(spec, quantization)
        self.n_threads = n_threads
        self.n_gpu_layers = n_gpu_layers
        self.llm = None

    def load(self) -> None:
        from llama_cpp import Llama

        if not self.spec.gguf_repo or not self.spec.gguf_file:
            raise RuntimeError(
                f"{self.spec.key} has no gguf_repo/gguf_file in registry.py. "
                "Add them, or convert the weights yourself with llama.cpp's "
                "convert_hf_to_gguf.py."
            )
        self.llm = Llama.from_pretrained(
            repo_id=self.spec.gguf_repo,
            filename=self.spec.gguf_file,
            n_ctx=4096,
            n_threads=self.n_threads,
            n_gpu_layers=self.n_gpu_layers,
            logits_all=False,
            verbose=False,
        )

    def unload(self) -> None:
        self.llm = None

    def generate(self, prompt: str, max_new_tokens: int, concurrency: int = 1) -> GenerationTrace:
        if self.llm is None:
            raise RuntimeError("call load() before generate()")
        if concurrency != 1:
            raise NotImplementedError(
                "llama.cpp backend measures concurrency 1 only. Use the hf "
                "backend for the concurrency sweep, and keep the two sets of "
                "numbers clearly separated in the paper."
            )

        start = time.perf_counter()
        ttft_ms = 0.0
        per_token_ms: list[float] = []
        previous = start
        generated = 0

        for _ in self.llm(prompt, max_tokens=max_new_tokens, stream=True, echo=False):
            now = time.perf_counter()
            if generated == 0:
                ttft_ms = (now - start) * 1000.0
            else:
                per_token_ms.append((now - previous) * 1000.0)
            previous = now
            generated += 1

        return GenerationTrace(
            ttft_ms=ttft_ms, per_token_ms=per_token_ms, generated_tokens=generated
        )


BACKENDS = {"hf": HFBackend, "llamacpp": LlamaCppBackend}


def build_backend(name: str, spec: ModelSpec, quantization: str = "none", **kwargs) -> Backend:
    if name not in BACKENDS:
        raise KeyError(f"unknown backend {name!r}; available: {sorted(BACKENDS)}")
    return BACKENDS[name](spec, quantization=quantization, **kwargs)
