"""Shared utilities for masked-diffusion sampling.

Key functions ported from the official LLaDA implementation
(ML-GSAI/LLaDA, generate.py) and adapted for MLX.
"""

from __future__ import annotations

import mlx.core as mx


# ── Transfer schedule ───────────────────────────────────────────────────────

def get_transfer_schedule(mask_count: int, steps: int) -> list[int]:
    """Precompute how many tokens to commit at each denoising step.

    LLaDA uses a linear noise schedule (Eq. 8 in the paper): the expected
    number of tokens unmasked per step should be uniform.  We distribute
    ``mask_count`` evenly across ``steps``, with any remainder front-loaded
    into the earliest steps.

    Returns:
        A list of length ``steps`` summing to ``mask_count``.
    """
    if steps <= 0 or mask_count <= 0:
        return [0] * max(steps, 0)
    base = mask_count // steps
    remainder = mask_count % steps
    schedule = [base + (1 if i < remainder else 0) for i in range(steps)]
    return schedule


# ── Gumbel noise ────────────────────────────────────────────────────────────

def add_gumbel_noise(logits: mx.array, temperature: float) -> mx.array:
    """Apply Gumbel-max trick for stochastic categorical sampling.

    Per arXiv:2409.02908, low-precision Gumbel-max improves perplexity
    but degrades generation quality in MDMs.  We use float32 on MLX
    (MLX lacks float64 support; float32 is the highest available).

    When ``temperature == 0``, returns logits unchanged (greedy argmax).
    """
    if temperature == 0:
        return logits
    logits = logits.astype(mx.float32)
    noise = mx.random.uniform(shape=logits.shape).astype(mx.float32)
    # Clamp noise away from 0 to avoid log(0)
    noise = mx.maximum(noise, mx.array(1e-20))
    gumbel_noise = (-mx.log(noise)) ** temperature
    return mx.exp(logits) / gumbel_noise


# ── Confidence extraction ──────────────────────────────────────────────────

def extract_confidence(
    logits: mx.array,
    x0: mx.array,
) -> mx.array:
    """Compute per-token confidence without materializing the full softmax.

    Uses the logsumexp trick: ``p(x0_i) = exp(logit_i - logsumexp(logits))``.
    This produces a ``(batch, seq_len)`` confidence vector instead of the
    full ``(batch, seq_len, vocab_size)`` softmax tensor (~130 MB per step
    at vocab_size=128256).

    Args:
        logits: Raw model output, shape ``(B, L, V)``.
        x0: Argmax predictions, shape ``(B, L)``.

    Returns:
        Per-position confidence scores, shape ``(B, L)``.
    """
    x0_logits = mx.take_along_axis(
        logits, mx.expand_dims(x0, axis=-1), axis=-1,
    ).squeeze(-1)
    lse = mx.logsumexp(logits, axis=-1)
    return mx.exp(x0_logits - lse)
