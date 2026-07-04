"""Static configuration: binary paths, the model farm, and the tiered
model-architecture set.

Override any path via the matching environment variable so the harness stays
runnable on another machine without editing code.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

# --- binaries + model store (env-overridable) --------------------------------

NATIVE_BIN = os.environ.get(
    "PARITY_NATIVE_BIN", "/private/tmp/parity-native-baseline/mlxserve-http"
)
OMLX_BIN = os.environ.get(
    "PARITY_OMLX_BIN", "/private/tmp/omlx-sidecar-venv-20260703/bin/omlx"
)

# M3 native is multi-model + validating: it is launched ONCE against a directory
# of model subdirs (org/name), serves them on demand, and validates the request
# `model` id (unknown -> 404). Native's discovery derives each id from the LEAF
# dir name (`Qwen3-0.6B-4bit`), recursing at most two levels (<org>/<name>), and
# does NOT follow symlinks. So we point native at the cat's REAL nested store —
# NOT the flattened symlink farm. omlx, pointed at the same real store, derives
# the identical bare-leaf ids, so a single `model` string selects the same model
# on both servers.
MODEL_STORE = os.environ.get(
    "PARITY_MODEL_STORE", str(Path.home() / "Library/Caches/models")
)
# Back-compat alias: older env/callers used PARITY_MODEL_FARM. Both native and
# omlx now use the same store so their discovered ids match.
MODEL_FARM = os.environ.get("PARITY_MODEL_FARM", MODEL_STORE)
OMLX_API_KEY = os.environ.get("PARITY_OMLX_API_KEY", "devkey")

# Token-count agreement tolerance. Chat-template / BOS differences make the two
# servers count prompt tokens off-by-a-few; with a length cap both hit exactly
# max_tokens, so output-token agreement is normally exact.
OUTPUT_TOKEN_TOLERANCE = 2


@dataclass(frozen=True)
class ModelSpec:
    """One model in the architecture matrix.

    `model_id` is the bare leaf dir name in the real store; it is also the
    `model` field both servers accept and report in /v1/models. `family` is the
    short architecture key,
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
        "Qwen3-0.6B-4bit",
        "dense",
        "dense full-attn (Qwen3)",
        "smoke",
        reasoning=True,  # Qwen3 defaults to thinking mode
    ),
    ModelSpec(
        "Llama-3.2-1B-Instruct-4bit",
        "dense",
        "dense full-attn (Llama)",
        "smoke",
    ),
    ModelSpec(
        "DeepSeek-R1-Distill-Qwen-1.5B-4bit",
        "dense",
        "dense full-attn, reasoning (DeepSeek)",
        "smoke",
        reasoning=True,
    ),
    ModelSpec(
        # The real store carries the qat variant of Gemma3-E2B (same
        # sliding-window architecture as the plain -it build).
        "gemma-4-E2B-it-qat-4bit",
        "sliding-window",
        "sliding-window (Gemma3)",
        "smoke",
        milestone=FAMILY_MILESTONE["sliding-window"],
        expect_native_gap=True,
    ),
    # --- FULL ---
    ModelSpec(
        "gpt-oss-20b-MXFP4-Q8",
        "sliding-window",
        "sliding-window + MoE + harmony (gpt-oss)",
        "full",
        reasoning=True,
        milestone=FAMILY_MILESTONE["sliding-window"],
        expect_native_gap=True,
    ),
    ModelSpec(
        "Qwen3-Coder-30B-A3B-Instruct-4bit",
        "moe",
        "MoE (Qwen3-Coder-30B-A3B)",
        "full",
    ),
    ModelSpec(
        "Qwen3.6-27B-MLX-4bit",
        "mamba-hybrid",
        "Mamba-hybrid flagship (Qwen3.6-27B)",
        "full",
        milestone=FAMILY_MILESTONE["mamba-hybrid"],
        expect_native_gap=True,
    ),
    ModelSpec(
        "Qwen2-VL-2B-Instruct-4bit",
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
ERROR_MODEL = "Qwen3-0.6B-4bit"
# Benchmark axis uses the fastest dense model.
BENCH_MODEL = "Qwen3-0.6B-4bit"


def matrix_models(tier: str) -> list[ModelSpec]:
    """Models for the architecture matrix at the requested tier (smoke ⊂ full)."""
    if tier == "full":
        return [m for m in ALL_MODELS if m.tier in ("smoke", "full")]
    return [m for m in ALL_MODELS if m.tier == "smoke"]


def spec_for(model_id: str) -> ModelSpec:
    return next(m for m in ALL_MODELS if m.model_id == model_id)


@lru_cache(maxsize=1)
def _store_leaves() -> dict[str, str]:
    """Map bare-leaf model id -> concrete snapshot path in the real store.

    Mirrors native's discovery: a model dir is one containing config.json, found
    either directly under the store or one level down (<org>/<name>). Symlinks
    are NOT followed for the *directory listing* (native doesn't), matching the
    ids native derives. Cheap filesystem scan; cached for the session.
    """
    root = Path(MODEL_STORE)
    leaves: dict[str, str] = {}
    if not root.is_dir():
        return leaves

    def is_model_dir(path: Path) -> bool:
        return (path / "config.json").exists()

    for child in sorted(root.iterdir()):
        if not child.is_dir() or child.name.startswith("."):
            continue
        if is_model_dir(child):
            leaves.setdefault(child.name, str(child))
            continue
        for grand in sorted(child.iterdir()):
            if grand.is_dir() and not grand.name.startswith(".") and is_model_dir(grand):
                leaves.setdefault(grand.name, str(grand))
    return leaves


def resolve_in_store(model_id: str) -> str | None:
    """Concrete snapshot dir for a bare model id in the real store, or None."""
    return _store_leaves().get(model_id)


def model_present(model_id: str) -> bool:
    """Is this model id discoverable in the real store (present on disk)?"""
    return resolve_in_store(model_id) is not None
