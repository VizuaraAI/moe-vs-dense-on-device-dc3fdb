"""The measurement loop.

Timing-critical. Two rules that are easy to get wrong and fatal to the paper:

1. **Warm up before measuring.** The first call pays for lazy kernel
   compilation, memory mapping and page faults. Measuring it instead of the
   steady state is the single most common way to produce an impressive but
   meaningless speedup.
2. **Repeat, then report a distribution.** On a thermally limited device the
   fifth run is not the first run. ``repeats`` defaults to 5 and the result
   record carries median and interquartile range, never a single number.
"""

from __future__ import annotations

import time
import traceback

from .backends import Backend, build_backend
from .device_info import DeviceInfo, collect
from .metrics import (
    PeakMemorySampler,
    RunResult,
    peak_accelerator_gb,
    reset_accelerator_peak,
    summarize,
)
from .registry import MatchedPair, ModelSpec, get_pair
from .roofline import TIERS, estimate_decode


def measure_model(
    spec: ModelSpec,
    backend_name: str = "hf",
    quantization: str = "none",
    concurrency: int = 1,
    prompt_tokens: int = 1024,
    max_new_tokens: int = 128,
    repeats: int = 5,
    warmup: int = 1,
    pair_key: str = "",
    device: DeviceInfo | None = None,
    tier: str = "laptop",
    backend_kwargs: dict | None = None,
) -> RunResult:
    """Measure one model in one configuration.

    Always returns a ``RunResult``. Failures (out of memory, missing weights,
    unsupported backend) are recorded with ``status`` set, because a
    configuration that cannot run on a device is one of the paper's findings.
    """
    device = device or collect(measure_bandwidth=False)
    result = RunResult(
        pair=pair_key,
        model_key=spec.key,
        arch=spec.arch,
        backend=backend_name,
        quantization=quantization,
        concurrency=concurrency,
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        repeats=repeats,
        model_resident_gb=round(spec.weight_bytes() / 1e9, 3),
        device=device.to_dict(),
    )

    backend: Backend | None = None
    try:
        backend = build_backend(
            backend_name, spec, quantization=quantization, **(backend_kwargs or {})
        )
        prompt = backend.make_prompt(prompt_tokens)

        reset_accelerator_peak()
        with PeakMemorySampler() as sampler:
            backend.load()

            for _ in range(warmup):
                backend.generate(prompt, max_new_tokens=min(16, max_new_tokens), concurrency=concurrency)

            ttfts: list[float] = []
            tpots: list[float] = []
            for _ in range(repeats):
                trace = backend.generate(prompt, max_new_tokens, concurrency)
                ttfts.append(trace.ttft_ms)
                tpots.append(trace.tpot_ms)

        result.ttft_ms_median, result.ttft_ms_iqr = summarize(ttfts)
        result.tpot_ms_median, result.tpot_ms_iqr = summarize(tpots)
        result.tokens_per_second = (
            (1000.0 / result.tpot_ms_median * concurrency) if result.tpot_ms_median else 0.0
        )
        result.peak_rss_gb = round(sampler.peak_rss_gb, 3)
        result.peak_accel_gb = round(peak_accelerator_gb(), 3)

        # Until bytes moved is instrumented at the platform level, fall back to
        # the analytical estimate so the column is never empty. See CLAUDE.md
        # next-steps item 3.
        estimate = estimate_decode(spec, TIERS.get(tier, TIERS["laptop"]), concurrency)
        result.bytes_moved_per_token = round(estimate.bytes_moved_per_step / concurrency, 1)
        result.bytes_moved_source = "estimated"

    except Exception as exc:  # noqa: BLE001 - a failed config is a data point
        message = f"{type(exc).__name__}: {exc}"
        is_oom = "out of memory" in message.lower() or isinstance(exc, MemoryError)
        result.status = "oom" if is_oom else "error"
        result.error = message[:500]
        result.notes = traceback.format_exc(limit=3)[:900]
    finally:
        if backend is not None:
            backend.unload()

    result.timestamp = time.time()
    return result


def measure_pair(
    pair: MatchedPair | str,
    concurrency: int = 1,
    **kwargs,
) -> list[RunResult]:
    """Measure both arms of a matched pair under identical settings.

    Both arms go through the same code path with the same arguments. That is
    the whole point: any difference in the numbers is a difference between the
    models, not between two measurement scripts.
    """
    pair = get_pair(pair) if isinstance(pair, str) else pair
    device = kwargs.pop("device", None) or collect(measure_bandwidth=False)
    return [
        measure_model(
            spec, concurrency=concurrency, pair_key=pair.key, device=device, **kwargs
        )
        for spec in pair.arms()
    ]


def sweep_pair(
    pair: MatchedPair | str,
    concurrencies: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    **kwargs,
) -> list[RunResult]:
    """Concurrency sweep over both arms, which is what locates the crossover."""
    pair = get_pair(pair) if isinstance(pair, str) else pair
    device = kwargs.pop("device", None) or collect(measure_bandwidth=False)
    results: list[RunResult] = []
    for concurrency in concurrencies:
        results.extend(
            measure_pair(pair, concurrency=concurrency, device=device, **kwargs)
        )
    return results
