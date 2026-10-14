"""Matched dense / Mixture-of-Experts model pairs.

Single source of truth for what gets compared against what, and why. Two
matching axes:

* ``active``  : same active parameters per token, so the same FLOPs per token.
                This is the axis that directly tests the MoE efficiency claim.
* ``quality`` : same downstream accuracy, different architectures. This is the
                axis a product engineer actually faces.

Parameter counts are in billions and come from the model cards / technical
reports. They are used by ``roofline.py`` to predict latency, so keep them
honest: if you change a model id, update the counts too.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ModelSpec:
    """One model, one arm of a comparison."""

    key: str
    hf_id: str
    arch: str  # "dense" or "moe"
    total_params_b: float
    active_params_b: float
    hidden_size: int
    num_layers: int
    num_kv_heads: int
    head_dim: int
    gguf_repo: str | None = None
    gguf_file: str | None = None
    note: str = ""

    @property
    def is_moe(self) -> bool:
        return self.arch == "moe"

    def weight_bytes(self, bits_per_weight: float = 4.5) -> float:
        """Resident weight bytes. This is what the device must hold."""
        return self.total_params_b * 1e9 * bits_per_weight / 8.0

    def active_weight_bytes(self, bits_per_weight: float = 4.5) -> float:
        """Weight bytes touched per token, ignoring cache effects."""
        return self.active_params_b * 1e9 * bits_per_weight / 8.0

    def kv_bytes_per_token(self, dtype_bytes: int = 2) -> float:
        """Key-value cache growth per token, both K and V."""
        return 2.0 * self.num_layers * self.num_kv_heads * self.head_dim * dtype_bytes


@dataclass(frozen=True)
class MatchedPair:
    """A dense arm and an MoE arm that are fair to compare."""

    key: str
    dense: ModelSpec
    moe: ModelSpec
    axis: str  # "active" or "quality"
    rationale: str
    quality_tasks: tuple[str, ...] = field(default=("hellaswag",))

    def arms(self) -> tuple[ModelSpec, ModelSpec]:
        return (self.dense, self.moe)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------

QWEN25_15B = ModelSpec(
    key="qwen2.5-1.5b",
    hf_id="Qwen/Qwen2.5-1.5B-Instruct",
    arch="dense",
    total_params_b=1.54,
    active_params_b=1.54,
    hidden_size=1536,
    num_layers=28,
    num_kv_heads=2,
    head_dim=128,
    gguf_repo="Qwen/Qwen2.5-1.5B-Instruct-GGUF",
    gguf_file="qwen2.5-1.5b-instruct-q4_k_m.gguf",
    note="Dense arm, active-matched against OLMoE.",
)

SMOLLM2_17B = ModelSpec(
    key="smollm2-1.7b",
    hf_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
    arch="dense",
    total_params_b=1.71,
    active_params_b=1.71,
    hidden_size=2048,
    num_layers=24,
    num_kv_heads=32,
    head_dim=64,
    note="Second dense arm. Many KV heads, so a useful KV-cache contrast.",
)

PHI3_MINI = ModelSpec(
    key="phi-3-mini",
    hf_id="microsoft/Phi-3-mini-4k-instruct",
    arch="dense",
    total_params_b=3.82,
    active_params_b=3.82,
    hidden_size=3072,
    num_layers=32,
    num_kv_heads=32,
    head_dim=96,
    note="Dense arm, quality-matched against Mixtral per the Phi-3 report.",
)

OLMOE_1B_7B = ModelSpec(
    key="olmoe-1b-7b",
    hf_id="allenai/OLMoE-1B-7B-0924-Instruct",
    arch="moe",
    total_params_b=6.92,
    active_params_b=1.30,
    hidden_size=2048,
    num_layers=16,
    num_kv_heads=16,
    head_dim=128,
    note="Fully open MoE, 64 experts, top-8. The cleanest small MoE arm.",
)

MIXTRAL_8X7B = ModelSpec(
    key="mixtral-8x7b",
    hf_id="mistralai/Mixtral-8x7B-Instruct-v0.1",
    arch="moe",
    total_params_b=46.7,
    active_params_b=12.9,
    hidden_size=4096,
    num_layers=32,
    num_kv_heads=8,
    head_dim=128,
    note="MoE arm, quality-matched against Phi-3-mini. Will not fit a phone.",
)

MODELS: dict[str, ModelSpec] = {
    m.key: m
    for m in (QWEN25_15B, SMOLLM2_17B, PHI3_MINI, OLMOE_1B_7B, MIXTRAL_8X7B)
}


# --------------------------------------------------------------------------
# Pairs
# --------------------------------------------------------------------------

PAIRS: dict[str, MatchedPair] = {
    "olmoe-vs-qwen15": MatchedPair(
        key="olmoe-vs-qwen15",
        dense=QWEN25_15B,
        moe=OLMOE_1B_7B,
        axis="active",
        rationale=(
            "1.54B dense versus 1.30B active. Near-identical FLOPs per token, "
            "but the MoE arm must hold 4.5x the weights. If the FLOP argument "
            "transferred to devices these two would cost the same. Start here."
        ),
    ),
    "olmoe-vs-smollm2": MatchedPair(
        key="olmoe-vs-smollm2",
        dense=SMOLLM2_17B,
        moe=OLMOE_1B_7B,
        axis="active",
        rationale=(
            "Second active-matched pair with a very different KV-cache shape, "
            "so that no conclusion rests on one dense model's attention layout."
        ),
    ),
    "mixtral-vs-phi3": MatchedPair(
        key="mixtral-vs-phi3",
        dense=PHI3_MINI,
        moe=MIXTRAL_8X7B,
        axis="quality",
        rationale=(
            "The Phi-3 report claims 3.8B dense quality competitive with "
            "Mixtral 8x7B. Equal quality, 12x the resident memory. This is the "
            "pair that matters to anyone shipping a product."
        ),
    ),
}


def get_pair(key: str) -> MatchedPair:
    if key not in PAIRS:
        raise KeyError(f"unknown pair {key!r}; available: {sorted(PAIRS)}")
    return PAIRS[key]


def get_model(key: str) -> ModelSpec:
    if key not in MODELS:
        raise KeyError(f"unknown model {key!r}; available: {sorted(MODELS)}")
    return MODELS[key]
