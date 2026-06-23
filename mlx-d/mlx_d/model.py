"""LLaDA model loading via patched LLaMA architecture.

The core insight: LLaDA is structurally identical to LLaMA-3 8B except
for one change — the causal attention mask is removed, giving full
bidirectional attention.  We load LLaDA weights into the standard
mlx_lm LLaMA implementation by:

1. Monkey-patching ``create_attention_mask`` to return ``None``
2. Remapping LLaDA's weight keys to LLaMA's naming convention
3. Quantizing the model structure to match pre-quantized weights
"""

from __future__ import annotations

import glob
import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import AutoTokenizer

from .config import ModelConfig, MASK_TOKEN_ID

log = logging.getLogger(__name__)


# ── Weight key remapping ────────────────────────────────────────────────────

_REMAP_RULES: list[tuple[str, str]] = [
    # Attention projections: layers.X.{q,k,v}_proj → layers.X.self_attn.{q,k,v}_proj
    (r"(layers\.\d+)\.(q_proj|k_proj|v_proj)", r"\1.self_attn.\2"),
    # Attention output: layers.X.attn_out → layers.X.self_attn.o_proj
    (r"(layers\.\d+)\.attn_out", r"\1.self_attn.o_proj"),
    # Input layernorm: layers.X.attn_norm → layers.X.input_layernorm
    (r"(layers\.\d+)\.attn_norm", r"\1.input_layernorm"),
    # Post-attention layernorm: layers.X.ff_norm → layers.X.post_attention_layernorm
    (r"(layers\.\d+)\.ff_norm", r"\1.post_attention_layernorm"),
    # MLP gate: layers.X.ff_proj → layers.X.mlp.gate_proj
    (r"(layers\.\d+)\.ff_proj", r"\1.mlp.gate_proj"),
    # MLP down: layers.X.ff_out → layers.X.mlp.down_proj
    (r"(layers\.\d+)\.ff_out", r"\1.mlp.down_proj"),
    # MLP up: layers.X.up_proj → layers.X.mlp.up_proj  (unchanged path but explicit)
    (r"(layers\.\d+)\.up_proj", r"\1.mlp.up_proj"),
]


def _remap_key(key: str) -> str:
    """Map LLaDA safetensor keys to mlx_lm LLaMA naming convention."""
    key = key.replace("model.model.", "model.")
    key = key.replace("model.lm_head.", "lm_head.")
    for pattern, replacement in _REMAP_RULES:
        key = re.sub(pattern, replacement, key)
    return key


# ── Model loading ───────────────────────────────────────────────────────────

def _load_weights(model_path: str) -> dict[str, mx.array]:
    """Load and remap all safetensor shards from a model directory."""
    shard_files = sorted(glob.glob(f"{model_path}/*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {model_path}")

    weights: dict[str, mx.array] = {}
    for shard in shard_files:
        with safe_open(shard, framework="mlx") as f:
            for key in f.keys():
                weights[_remap_key(key)] = f.get_tensor(key)
    return weights


def load_model(
    config: Optional[ModelConfig] = None,
    model_id: Optional[str] = None,
) -> tuple:
    """Load a LLaDA model with bidirectional attention.

    Returns:
        ``(model, tokenizer, mask_id)`` tuple.

    The model is a standard mlx_lm LLaMA model with the causal mask
    patched out.  The tokenizer handles chat template formatting.
    """
    if config is None:
        config = ModelConfig(model_id=model_id or "mlx-community/LLaDA-8B-Instruct-mlx-4bit")

    t0 = time.perf_counter()

    # Download / locate model weights
    repo = config.model_id
    log.info("downloading/caching model: %s", repo)
    model_path = snapshot_download(
        repo,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    )

    # Load config
    config_path = Path(model_path) / "config.json"
    with open(config_path) as f:
        model_cfg = json.load(f)

    # Patch bidirectional attention: the single architectural delta
    import mlx_lm.models.llama as llama_mod
    llama_mod.create_attention_mask = lambda *a, **kw: None

    # Build model with LLaDA config field names mapped to LLaMA
    args = llama_mod.ModelArgs(
        model_type="llama",
        hidden_size=model_cfg.get("d_model", model_cfg.get("hidden_size", 4096)),
        num_hidden_layers=model_cfg.get("n_layers", model_cfg.get("num_hidden_layers", 32)),
        intermediate_size=model_cfg.get("mlp_hidden_size", model_cfg.get("intermediate_size", 14336)),
        num_attention_heads=model_cfg.get("n_heads", model_cfg.get("num_attention_heads", 32)),
        num_key_value_heads=model_cfg.get("n_kv_heads", model_cfg.get("num_key_value_heads", 8)),
        rms_norm_eps=model_cfg.get("rms_norm_eps", 1e-5),
        vocab_size=model_cfg.get("vocab_size", 128256),
        rope_theta=model_cfg.get("rope_theta", 500000.0),
        tie_word_embeddings=model_cfg.get("weight_tying", model_cfg.get("tie_word_embeddings", False)),
    )

    model = llama_mod.Model(args)

    # Quantize model structure to match pre-quantized weights
    quant_cfg = model_cfg.get("quantization", {})
    if quant_cfg:
        nn.quantize(
            model,
            group_size=quant_cfg.get("group_size", 64),
            bits=quant_cfg.get("bits", config.q_bits),
        )

    # Load remapped weights
    weights = _load_weights(model_path)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Resolve mask token id
    mask_id = model_cfg.get("mask_token_id", MASK_TOKEN_ID)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Warm up Metal shaders (one-time cost)
    if config.compile_model:
        log.info("warming up Metal shaders...")
        dummy = mx.zeros((1, 4), dtype=mx.int32)
        model(dummy)
        mx.eval(model.parameters())

    elapsed = time.perf_counter() - t0
    log.info("model loaded in %.1fs (mask_id=%d)", elapsed, mask_id)

    return model, tokenizer, mask_id
