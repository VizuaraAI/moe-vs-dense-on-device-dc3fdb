"""Analytical crossover model.

This module answers the paper's central question with arithmetic alone: at what
concurrency does a Mixture-of-Experts model start to beat a dense model?

Decode time per token is modelled as the larger of a compute term and a memory
term, following the arithmetic-intensity analysis of Pope et al. (2022) and the
roofline treatment of Yuan et al. (2024):

    t(C) = max( 2 * P_active * C / F ,  B_moved(C) / W_eff )

* ``P_active`` active parameters, so ``2 * P_active`` FLOPs per token (one
  multiply and one add per weight).
* ``C`` concurrency. Weight loads are shared across a batch, arithmetic is not.
  That asymmetry is the entire story.
* ``F`` achievable FLOP/s.
* ``B_moved`` weight bytes plus key-value cache bytes moved per decode step.
* ``W_eff`` effective bandwidth of whichever tier actually holds the weights.
  When the weights do not fit in RAM, part of the read is served from flash at
  one to two orders of magnitude lower bandwidth.

Keep this module free of torch and of anything that can fail to import on a
phone. It must stay runnable with no downloads, because predicting before
measuring is the habit that makes the project debuggable.
"""

from __future__ import annotations

from dataclasses import dataclass

from .registry import MatchedPair, ModelSpec

# Flash read bandwidth is far below DRAM. Representative orders of magnitude;
# override from config with numbers measured on your own device.
DEFAULT_FLASH_BANDWIDTH_GBPS = 1.5

# Fraction of peak memory bandwidth a routed MoE layer actually achieves.
#
# This is the most important assumption in the whole model, and the thing the
# paper exists to measure. A dense decode reads one large contiguous weight
# matrix per layer and streams close to peak bandwidth. An MoE decode gathers a
# handful of experts out of dozens: the reads are scattered, the per-expert
# GEMMs are small and poorly utilized, and there is routing work on top. The
# result is that MoE moves fewer bytes than an active-parameter-matched dense
# model but moves them more slowly.
#
# 0.55 is a starting guess, not a measurement. Replace it with your own number
# from the bytes-moved instrumentation as soon as you have one, and report the
# measured value in the paper. If it turns out to be close to 1.0, the thesis
# is in trouble at the tiers where the MoE model fits in memory, and that is a
# finding worth knowing in week 2 rather than week 7.
DEFAULT_MOE_BANDWIDTH_EFFICIENCY = 0.55


@dataclass
class HardwareModel:
    """The three hardware numbers the roofline needs."""

    name: str
    memory_bandwidth_gbps: float
    achievable_tflops: float
    memory_capacity_gb: float
    flash_bandwidth_gbps: float = DEFAULT_FLASH_BANDWIDTH_GBPS

    @property
    def bandwidth_bytes_per_s(self) -> float:
        return self.memory_bandwidth_gbps * 1e9

    @property
    def flops_per_s(self) -> float:
        return self.achievable_tflops * 1e12


# Representative tiers, matching the three device tiers in the paper. These are
# starting points: replace with your own measured values from `cli device`.
TIERS: dict[str, HardwareModel] = {
    "phone": HardwareModel(
        name="phone (8 GB unified)",
        memory_bandwidth_gbps=60.0,
        achievable_tflops=1.5,
        memory_capacity_gb=6.0,  # usable budget, not nameplate RAM
    ),
    "laptop": HardwareModel(
        name="laptop (16 GB unified)",
        memory_bandwidth_gbps=200.0,
        achievable_tflops=8.0,
        memory_capacity_gb=12.0,
    ),
    "gpu": HardwareModel(
        name="consumer GPU (8 GB VRAM)",
        memory_bandwidth_gbps=450.0,
        achievable_tflops=40.0,
        memory_capacity_gb=7.5,
        flash_bandwidth_gbps=12.0,  # PCIe host transfer, not literal flash
    ),
}


@dataclass
class DecodeEstimate:
    model_key: str
    concurrency: int
    compute_ms: float
    memory_ms: float
    bytes_moved_per_step: float
    fits_in_memory: bool
    bound_by: str

    @property
    def ms_per_token(self) -> float:
        """Per-token latency for one user, with C users sharing the step."""
        return max(self.compute_ms, self.memory_ms)

    @property
    def tokens_per_second_device(self) -> float:
        """Device-wide throughput, which is what rises with concurrency."""
        step_s = self.ms_per_token / 1000.0
        return (self.concurrency / step_s) if step_s > 0 else 0.0


def effective_bandwidth(
    hardware: HardwareModel, resident_bytes: float
) -> tuple[float, bool]:
    """Blend DRAM and flash bandwidth according to how much of the model fits.

    Returns ``(bytes_per_second, fits_in_memory)``. If the weights fit, this is
    just DRAM bandwidth. If they do not, the read is a weighted harmonic blend:
    the fraction served from flash dominates the time even when it is small,
    which is the mechanism behind the whole on-device MoE penalty.
    """
    capacity_bytes = hardware.memory_capacity_gb * 1e9
    if resident_bytes <= capacity_bytes:
        return hardware.bandwidth_bytes_per_s, True

    in_ram = capacity_bytes / resident_bytes
    from_flash = 1.0 - in_ram
    dram_bps = hardware.bandwidth_bytes_per_s
    flash_bps = hardware.flash_bandwidth_gbps * 1e9
    # Time to read one byte is the fraction-weighted sum of per-byte times.
    seconds_per_byte = in_ram / dram_bps + from_flash / flash_bps
    return 1.0 / seconds_per_byte, False


def estimate_decode(
    model: ModelSpec,
    hardware: HardwareModel,
    concurrency: int = 1,
    context_tokens: int = 1024,
    bits_per_weight: float = 4.5,
    expert_reuse: float = 0.0,
    moe_bandwidth_efficiency: float = DEFAULT_MOE_BANDWIDTH_EFFICIENCY,
) -> DecodeEstimate:
    """Estimate one decode step for ``model`` on ``hardware``.

    ``expert_reuse`` in [0, 1) models an expert cache: the fraction of routed
    expert weight reads that hit memory already warmed by a previous token.
    Dense models have perfect reuse by construction, which is why they are not
    penalised here. Set this from your measured expert hit rate.

    ``moe_bandwidth_efficiency`` is the fraction of peak bandwidth a scattered
    expert gather achieves. See the module constant for why it matters more
    than anything else here.
    """
    if concurrency < 1:
        raise ValueError("concurrency must be at least 1")

    resident_bytes = model.weight_bytes(bits_per_weight)
    bandwidth_bps, fits = effective_bandwidth(hardware, resident_bytes)

    if model.is_moe:
        # Routed experts vary per token, so at concurrency C the union of
        # experts touched grows toward the full set. Reads are shared only to
        # the extent that different sequences route to the same expert.
        union = min(1.0, model.active_params_b / model.total_params_b * concurrency)
        touched_b = model.total_params_b * union
        touched_b *= 1.0 - max(0.0, min(expert_reuse, 0.95))
        weight_bytes = touched_b * 1e9 * bits_per_weight / 8.0
        # Fewer bytes than a matched dense model, but moved more slowly.
        bandwidth_bps *= max(0.05, min(moe_bandwidth_efficiency, 1.0))
    else:
        # A dense model reads every weight once per step, regardless of C.
        weight_bytes = model.active_weight_bytes(bits_per_weight)

    kv_bytes = model.kv_bytes_per_token() * context_tokens * concurrency
    bytes_moved = weight_bytes + kv_bytes

    memory_ms = bytes_moved / bandwidth_bps * 1000.0
    flops = 2.0 * model.active_params_b * 1e9 * concurrency
    compute_ms = flops / hardware.flops_per_s * 1000.0

    return DecodeEstimate(
        model_key=model.key,
        concurrency=concurrency,
        compute_ms=compute_ms,
        memory_ms=memory_ms,
        bytes_moved_per_step=bytes_moved,
        fits_in_memory=fits,
        bound_by="compute" if compute_ms > memory_ms else "memory",
    )


def crossover_concurrency(
    pair: MatchedPair,
    hardware: HardwareModel,
    max_concurrency: int = 1024,
    **kwargs,
) -> int | None:
    """Smallest C at which the MoE arm's device throughput exceeds the dense arm's.

    Returns None if no crossover exists below ``max_concurrency``, which for an
    on-device deployment is the same as "never".
    """
    concurrency = 1
    while concurrency <= max_concurrency:
        dense = estimate_decode(pair.dense, hardware, concurrency, **kwargs)
        moe = estimate_decode(pair.moe, hardware, concurrency, **kwargs)
        if moe.tokens_per_second_device > dense.tokens_per_second_device:
            return concurrency
        concurrency *= 2
    return None


def predict_pair(
    pair: MatchedPair,
    tiers: dict[str, HardwareModel] | None = None,
    concurrencies: tuple[int, ...] = (1, 2, 4, 8, 16, 32, 64),
    **kwargs,
) -> dict:
    """Full prediction for one matched pair across all device tiers."""
    tiers = tiers or TIERS
    out: dict = {"pair": pair.key, "axis": pair.axis, "tiers": {}}
    for tier_key, hardware in tiers.items():
        rows = []
        for concurrency in concurrencies:
            dense = estimate_decode(pair.dense, hardware, concurrency, **kwargs)
            moe = estimate_decode(pair.moe, hardware, concurrency, **kwargs)
            rows.append(
                {
                    "concurrency": concurrency,
                    "dense_ms_per_token": round(dense.ms_per_token, 2),
                    "moe_ms_per_token": round(moe.ms_per_token, 2),
                    "dense_tok_s_device": round(dense.tokens_per_second_device, 1),
                    "moe_tok_s_device": round(moe.tokens_per_second_device, 1),
                    "dense_bound_by": dense.bound_by,
                    "moe_bound_by": moe.bound_by,
                }
            )
        dense_at_one = estimate_decode(pair.dense, hardware, 1, **kwargs)
        moe_at_one = estimate_decode(pair.moe, hardware, 1, **kwargs)
        out["tiers"][tier_key] = {
            "hardware": hardware.name,
            "dense_fits": dense_at_one.fits_in_memory,
            "moe_fits": moe_at_one.fits_in_memory,
            "dense_resident_gb": round(pair.dense.weight_bytes() / 1e9, 2),
            "moe_resident_gb": round(pair.moe.weight_bytes() / 1e9, 2),
            "dense_bytes_at_c1_gb": round(dense_at_one.bytes_moved_per_step / 1e9, 3),
            "moe_bytes_at_c1_gb": round(moe_at_one.bytes_moved_per_step / 1e9, 3),
            "dense_kv_gb_at_c64": round(
                pair.dense.kv_bytes_per_token() * 1024 * 64 / 1e9, 2
            ),
            "moe_kv_gb_at_c64": round(
                pair.moe.kv_bytes_per_token() * 1024 * 64 / 1e9, 2
            ),
            "predicted_crossover": crossover_concurrency(pair, hardware, **kwargs),
            "sweep": rows,
        }
    return out


def format_prediction(prediction: dict) -> str:
    """Human-readable prediction report."""
    lines = [
        f"Predicted crossover for pair '{prediction['pair']}' "
        f"(matched on {prediction['axis']})",
        "=" * 72,
    ]
    for tier_key, tier in prediction["tiers"].items():
        lines.append("")
        lines.append(f"[{tier_key}] {tier['hardware']}")
        lines.append(
            f"  resident weights : dense {tier['dense_resident_gb']} GB "
            f"(fits={tier['dense_fits']}), moe {tier['moe_resident_gb']} GB "
            f"(fits={tier['moe_fits']})"
        )
        lines.append(
            f"  bytes/step C=1   : dense {tier['dense_bytes_at_c1_gb']} GB, "
            f"moe {tier['moe_bytes_at_c1_gb']} GB"
        )
        lines.append(
            f"  KV cache at C=64 : dense {tier['dense_kv_gb_at_c64']} GB, "
            f"moe {tier['moe_kv_gb_at_c64']} GB (1k context)"
        )
        crossover = tier["predicted_crossover"]
        lines.append(
            f"  predicted C*     : {crossover if crossover else 'none below 1024'}"
        )
        lines.append("  C     dense ms/tok   moe ms/tok   dense tok/s   moe tok/s")
        for row in tier["sweep"]:
            lines.append(
                f"  {row['concurrency']:<5} {row['dense_ms_per_token']:>12.2f} "
                f"{row['moe_ms_per_token']:>12.2f} "
                f"{row['dense_tok_s_device']:>13.1f} "
                f"{row['moe_tok_s_device']:>11.1f}"
            )
    lines.append("")
    lines.append(
        "How to read this. If no crossover appears, look at the two diagnostic\n"
        "lines first. Either the MoE arm does not fit in memory (so it is paying\n"
        "flash bandwidth), or its key-value cache is much larger than the dense\n"
        "arm's because it uses fewer grouped-query heads. Both are real effects,\n"
        "but they are different claims and the paper must separate them from the\n"
        "expert-gather effect itself."
    )
    lines.append("")
    lines.append(
        "Write these predictions in your lab log BEFORE measuring, then try to\n"
        "falsify them. A prediction that survives measurement is a paper; a\n"
        "prediction that fails is a more interesting paper."
    )
    return "\n".join(lines)
