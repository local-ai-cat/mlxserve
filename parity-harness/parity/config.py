"""Static configuration: binary paths, the model farm, and the tiered
model-architecture set.

Override any path via the matching environment variable so the harness stays
runnable on another machine without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

# --- binaries + model farm (env-overridable) ---------------------------------

NATIVE_BIN = os.environ.get(
    "PARITY_NATIVE_BIN", "/private/tmp/parity-native-baseline/mlxserve-http"
)
OMLX_BIN = os.environ.get(
    "PARITY_OMLX_BIN", "/private/tmp/omlx-sidecar-venv-20260703/bin/omlx"
)
MODEL_FARM = os.environ.get("PARITY_MODEL_FARM", "/private/tmp/omlx-all-models")
OMLX_API_KEY = os.environ.get("PARITY_OMLX_API_KEY", "devkey")

# Token-count agreement tolerance. Chat-template / BOS differences make the two
# servers count prompt tokens off-by-a-few; with a length cap both hit exactly
# max_tokens, so output-token agreement is normally exact.
OUTPUT_TOKEN_TOLERANCE = 2


@dataclass(frozen=True)
class ModelSpec:
    """One model in the architecture matrix.

    `model_id` is the farm symlink basename; it is also omlx's `model` field and
    what native reports in /v1/models. `family` is the short architecture key,
    `family_label` the human string. `reasoning`/`vlm` flag behaviors that change
    how we compare. `milestone` names the MLXServe milestone that will close the
    native gap for this family ("" = native already at parity). `expect_native_gap`
    is the baseline expectation.
    """

    model_id: str
    family: str  # dense | sliding-window | moe | mamba-hybrid | vlm
    family_label: str
    tier: str  # smoke | full
    reasoning: bool = False
    vlm: bool = False
    milestone: str = ""
    expect_native_gap: bool = False


# Milestone that closes each native family gap. Per PLAN.md: full-attention KVCache
# is v1; rotating/sliding-window + hybrid caches and VLM are deferred to M7+.
FAMILY_MILESTONE = {
    "dense": "",  # at parity today
    "moe": "",  # MoE FFN over full attention — expected to work on v1 engine
    "sliding-window": "M7+ (rotating/arrays cache)",
    "mamba-hybrid": "M7+ (hybrid cache)",
    "vlm": "M7+ (VLM out of v1 scope)",
}

# The full registry. SMOKE = small, run every pass. FULL = heavy, on-demand.
ALL_MODELS: list[ModelSpec] = [
    # --- SMOKE ---
    ModelSpec(
        "mlx-community--Qwen3-0.6B-4bit",
        "dense",
        "dense full-attn (Qwen3)",
        "smoke",
        reasoning=True,  # Qwen3 defaults to thinking mode
    ),
    ModelSpec(
        "mlx-community--Llama-3.2-1B-Instruct-4bit",
        "dense",
        "dense full-attn (Llama)",
        "smoke",
    ),
    ModelSpec(
        "mlx-community--DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        "dense",
        "dense full-attn, reasoning (DeepSeek)",
        "smoke",
        reasoning=True,
    ),
    ModelSpec(
        "mlx-community--gemma-4-e2b-it-4bit",
        "sliding-window",
        "sliding-window (Gemma3)",
        "smoke",
        milestone=FAMILY_MILESTONE["sliding-window"],
        expect_native_gap=True,
    ),
    # --- FULL ---
    ModelSpec(
        "mlx-community--gpt-oss-20b-MXFP4-Q8",
        "sliding-window",
        "sliding-window + MoE + harmony (gpt-oss)",
        "full",
        reasoning=True,
        milestone=FAMILY_MILESTONE["sliding-window"],
        expect_native_gap=True,
    ),
    ModelSpec(
        "mlx-community--Qwen3-Coder-30B-A3B-Instruct-4bit",
        "moe",
        "MoE (Qwen3-Coder-30B-A3B)",
        "full",
    ),
    ModelSpec(
        "lmstudio-community--Qwen3.6-27B-MLX-4bit",
        "mamba-hybrid",
        "Mamba-hybrid flagship (Qwen3.6-27B)",
        "full",
        milestone=FAMILY_MILESTONE["mamba-hybrid"],
        expect_native_gap=True,
    ),
    ModelSpec(
        "mlx-community--Qwen2-VL-2B-Instruct-4bit",
        "vlm",
        "VLM (Qwen2-VL-2B)",
        "full",
        vlm=True,
        milestone=FAMILY_MILESTONE["vlm"],
        expect_native_gap=True,
    ),
]

# Conformance axes (schema / semantic / streaming) run on the dense smoke models
# native is known to handle — those axes compare where both servers produce
# output. Sliding-window / hybrid / VLM native gaps are the matrix axis's job.
CONFORMANCE_MODELS: list[ModelSpec] = [
    m for m in ALL_MODELS if m.tier == "smoke" and m.family == "dense"
]

# Error-semantics axis uses one fast dense model.
ERROR_MODEL = "mlx-community--Qwen3-0.6B-4bit"
# Benchmark axis uses the fastest dense model.
BENCH_MODEL = "mlx-community--Qwen3-0.6B-4bit"


def matrix_models(tier: str) -> list[ModelSpec]:
    """Models for the architecture matrix at the requested tier (smoke ⊂ full)."""
    if tier == "full":
        return [m for m in ALL_MODELS if m.tier in ("smoke", "full")]
    return [m for m in ALL_MODELS if m.tier == "smoke"]


def spec_for(model_id: str) -> ModelSpec:
    return next(m for m in ALL_MODELS if m.model_id == model_id)
