"""Configuration for LLaDA model loading and diffusion sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal


# ── Model defaults ──────────────────────────────────────────────────────────
# The MLX-community 4-bit quantization is the recommended default.
# It retains ~97% of bf16 MMLU performance at 3.8× lower memory.

DEFAULT_MODEL = "mlx-community/LLaDA-8B-Instruct-mlx-4bit"
MASK_TOKEN_ID = 126_336


@dataclass
class SamplerConfig:
    """Controls the masked-diffusion reverse process.

    Attributes:
        steps: Total denoising steps across all blocks.
        gen_length: Number of tokens to generate (response length).
        block_length: Tokens per semi-autoregressive block.
            Set equal to ``gen_length`` for full-sequence mode.
        temperature: Gumbel noise temperature for categorical sampling.
            0 = argmax (greedy), >0 adds controlled stochasticity.
        remasking: Strategy for choosing which tokens to re-mask.
            ``"low_confidence"`` (default): re-mask uncertain predictions.
            ``"random"``: re-mask uniformly at random.
        cfg_scale: Classifier-free guidance scale. 0 = disabled.
        mask_id: The ``[MASK]`` token id used by LLaDA (126336).
    """

    steps: int = 64
    gen_length: int = 128
    block_length: int = 32
    temperature: float = 0.0
    remasking: Literal["low_confidence", "random"] = "low_confidence"
    cfg_scale: float = 0.0
    mask_id: int = MASK_TOKEN_ID


@dataclass
class ModelConfig:
    """Controls model loading and hardware adaptation.

    Attributes:
        model_id: HuggingFace repo id or local path to LLaDA weights.
        q_bits: Quantization level (4 or 8). Ignored if weights are
            already quantized.
        compile_model: Whether to apply ``mx.compile`` to the forward
            function for throughput gains on repeated calls.
    """

    model_id: str = DEFAULT_MODEL
    q_bits: int = 4
    compile_model: bool = True


@dataclass
class Config:
    """Top-level configuration combining model and sampler settings."""

    model: ModelConfig = field(default_factory=ModelConfig)
    sampler: SamplerConfig = field(default_factory=SamplerConfig)
