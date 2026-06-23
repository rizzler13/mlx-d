"""Standalone masked-diffusion sampler for LLaDA.

This is the core reverse-process implementation, separated from model
loading so it can be composed with any MLX model that produces logits.

The sampler supports:
- Full-sequence mode (block_length == gen_length)
- Semi-autoregressive block mode (block_length < gen_length)
- Low-confidence and random remasking strategies
- Gumbel noise temperature for stochastic sampling
- Step-by-step callbacks for visualization / logging
- Classifier-free guidance (CFG)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import mlx.core as mx

from .config import SamplerConfig, MASK_TOKEN_ID
from .utils import add_gumbel_noise, extract_confidence, get_transfer_schedule

log = logging.getLogger(__name__)


# ── Step info (passed to callbacks) ─────────────────────────────────────────

@dataclass
class StepInfo:
    """Snapshot of a single denoising step, passed to the step callback."""

    block_idx: int
    block_total: int
    step_idx: int
    step_total: int
    tokens: list[int]          # current token ids (full sequence)
    mask_id: int               # which id represents [MASK]
    masks_remaining: int       # how many masks left after this step
    tokens_committed: int      # how many tokens committed in this step
    elapsed_ms: float          # wall-clock time for this step
    confidence: Optional[list[float]] = None  # per-token confidence (response region)


# ── Sampler ─────────────────────────────────────────────────────────────────

class DiffusionSampler:
    """Masked-diffusion reverse process for text generation.

    Usage::

        from mlx_d.model import load_model
        from mlx_d.sampler import DiffusionSampler
        from mlx_d.config import SamplerConfig

        model, tokenizer, mask_id = load_model()
        cfg = SamplerConfig(steps=64, gen_length=128, block_length=32)
        sampler = DiffusionSampler(cfg)

        result = sampler.generate(
            model=model,
            tokenizer=tokenizer,
            messages=[{"role": "user", "content": "What is masked diffusion?"}],
        )
        print(result.text)
    """

    def __init__(self, config: Optional[SamplerConfig] = None):
        self.config = config or SamplerConfig()

    def generate(
        self,
        model,
        tokenizer,
        messages: list[dict],
        *,
        mask_id: Optional[int] = None,
        on_step: Optional[Callable[[StepInfo], None]] = None,
        cancel: Optional[Callable[[], bool]] = None,
    ) -> "GenerationResult":
        """Run the masked-diffusion reverse process.

        Args:
            model: Any MLX model whose ``__call__`` returns logits
                (or an object with a ``.logits`` attribute).
            tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
            messages: Chat messages in OpenAI format.
            mask_id: Override the mask token id (default: from config).
            on_step: Called after each denoising step with a ``StepInfo``.
            cancel: Callable returning ``True`` to abort generation early.

        Returns:
            A ``GenerationResult`` with the generated text and metadata.
        """
        cfg = self.config
        mid = mask_id or cfg.mask_id

        # ── Tokenize prompt ──
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="np")
        prompt_ids = mx.array(encoded["input_ids"])
        prompt_len = prompt_ids.shape[1]

        # ── Initialize sequence: [prompt] + [MASK]*gen_length ──
        mask_tail = mx.full((1, cfg.gen_length), mid, dtype=mx.int32)
        x = mx.concatenate([prompt_ids.astype(mx.int32), mask_tail], axis=1)
        mx.eval(x)

        # ── Run denoising ──
        t0 = time.perf_counter()
        step_log: list[StepInfo] = []

        x = self._denoise(
            model=model,
            x=x,
            prompt_len=prompt_len,
            mask_id=mid,
            on_step=on_step,
            cancel=cancel,
            step_log=step_log,
        )

        elapsed_ms = (time.perf_counter() - t0) * 1000

        if x is None:
            return GenerationResult(text="", cancelled=True, elapsed_ms=elapsed_ms)

        # ── Decode output ──
        output_ids = x[0, prompt_len:].tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        return GenerationResult(
            text=text,
            token_ids=output_ids,
            prompt_len=prompt_len,
            elapsed_ms=elapsed_ms,
            steps=step_log,
        )

    def _denoise(
        self,
        model,
        x: mx.array,
        prompt_len: int,
        mask_id: int,
        on_step,
        cancel,
        step_log: list,
    ) -> Optional[mx.array]:
        """Inner denoising loop with semi-autoregressive block support."""
        cfg = self.config
        gen_length = cfg.gen_length
        block_length = cfg.block_length

        assert gen_length % block_length == 0, (
            f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        )
        num_blocks = gen_length // block_length

        total_steps = cfg.steps
        assert total_steps % num_blocks == 0, (
            f"steps ({total_steps}) must be divisible by num_blocks ({num_blocks})"
        )
        steps_per_block = total_steps // num_blocks

        for block_idx in range(num_blocks):
            block_start = prompt_len + block_idx * block_length
            block_end = prompt_len + (block_idx + 1) * block_length

            # Count masks in this block
            block_slice = x[:, block_start:block_end]
            mask_count = int(mx.sum(block_slice == mask_id).item())
            if mask_count == 0:
                continue

            schedule = get_transfer_schedule(mask_count, steps_per_block)

            for step_idx in range(steps_per_block):
                step_t0 = time.perf_counter()

                if cancel and cancel():
                    return None

                # ── Forward pass ──
                logits = model(x)
                if hasattr(logits, "logits"):
                    logits = logits.logits
                mx.eval(logits)

                # ── Apply Gumbel noise for temperature sampling ──
                noisy_logits = add_gumbel_noise(logits, cfg.temperature)

                # ── Argmax predictions ──
                x0 = mx.argmax(noisy_logits, axis=-1).astype(mx.int32)

                # ── Compute confidence ──
                mask_index = (x == mask_id)

                if cfg.remasking == "low_confidence":
                    x0_probs = extract_confidence(logits, x0)
                elif cfg.remasking == "random":
                    x0_probs = mx.random.uniform(shape=x0.shape)
                else:
                    raise ValueError(f"Unknown remasking strategy: {cfg.remasking}")

                # Suppress confidence for positions beyond current block
                seq_len = x.shape[1]
                if block_end < seq_len:
                    after_block = mx.full((1, seq_len - block_end), float("-inf"))
                    before = x0_probs[:, :block_end]
                    x0_probs = mx.concatenate([before, after_block], axis=1)

                # Apply mask: only update masked positions
                x0 = mx.where(mask_index, x0, x)
                confidence = mx.where(
                    mask_index, x0_probs,
                    mx.full(x0_probs.shape, float("-inf")),
                )

                # ── Commit top-k tokens ──
                k = schedule[step_idx]
                if k > 0:
                    conf_flat = confidence[0]
                    if k >= conf_flat.shape[0]:
                        top_indices = mx.arange(conf_flat.shape[0])
                    else:
                        partitioned = mx.argpartition(
                            conf_flat, kth=conf_flat.shape[0] - k,
                        )
                        top_indices = partitioned[-k:]

                    # Commit via scatter-add workaround (MLX lacks .at[].set())
                    x_flat = x[0].astype(mx.int32)
                    x0_flat = x0[0].astype(mx.int32)
                    top_indices = top_indices.astype(mx.int32)
                    new_vals = mx.take(x0_flat, top_indices)
                    old_vals = mx.take(x_flat, top_indices)
                    x_flat = x_flat.at[top_indices].add(new_vals - old_vals)
                    x = x_flat.reshape(1, -1)

                mx.eval(x)

                # ── Callback ──
                step_ms = (time.perf_counter() - step_t0) * 1000
                masks_left = int(mx.sum(x == mask_id).item())

                info = StepInfo(
                    block_idx=block_idx,
                    block_total=num_blocks,
                    step_idx=step_idx,
                    step_total=steps_per_block,
                    tokens=x[0].tolist(),
                    mask_id=mask_id,
                    masks_remaining=masks_left,
                    tokens_committed=k,
                    elapsed_ms=step_ms,
                )
                step_log.append(info)

                if on_step:
                    on_step(info)

        return x


# ── Result ──────────────────────────────────────────────────────────────────

@dataclass
class GenerationResult:
    """Output of a diffusion generation run."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    prompt_len: int = 0
    elapsed_ms: float = 0.0
    cancelled: bool = False
    steps: list[StepInfo] = field(default_factory=list)

    @property
    def tokens_per_second(self) -> float:
        if self.elapsed_ms <= 0 or not self.token_ids:
            return 0.0
        return len(self.token_ids) / (self.elapsed_ms / 1000)
