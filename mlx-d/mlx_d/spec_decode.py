"""Speculative diffusion decoding for masked-diffusion LLMs.

This module implements the drafter-verifier paradigm adapted for
masked-diffusion models like LLaDA:

1. **Draft phase**: A small/quantized model runs K denoising steps
   to propose candidate tokens for all masked positions in parallel.

2. **Verify phase**: The full-size target model runs a single forward
   pass over the proposed sequence, computing its own confidence scores.

3. **Accept/reject**: Tokens where the verifier's top prediction matches
   the drafter's proposal (and exceeds a confidence threshold) are accepted.
   Rejected tokens are re-masked for the next round.

This is fundamentally different from AR speculative decoding because the
drafter produces all K tokens in a single forward pass (not K sequential
passes).  Published results show up to 8.7× speedup over naive generation.

References:
    - Speculative Diffusion Decoding (arXiv:2408.05636)
    - Self-Speculative Decoding for MDMs (OpenReview, 2025)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Optional

import mlx.core as mx

from .config import SamplerConfig, MASK_TOKEN_ID
from .utils import add_gumbel_noise, extract_confidence, get_transfer_schedule

log = logging.getLogger(__name__)


@dataclass
class SpecDecodeConfig:
    """Configuration for speculative diffusion decoding.

    Attributes:
        draft_steps: Denoising steps the drafter runs per round.
        accept_threshold: Minimum confidence for accepting a drafted token.
        max_rounds: Maximum verification rounds before fallback.
        temperature: Gumbel noise temperature for both models.
    """

    draft_steps: int = 8
    accept_threshold: float = 0.3
    max_rounds: int = 8
    temperature: float = 0.0


@dataclass
class SpecDecodeResult:
    """Output of speculative decoding."""

    text: str
    token_ids: list[int] = field(default_factory=list)
    elapsed_ms: float = 0.0
    draft_calls: int = 0
    verify_calls: int = 0
    tokens_accepted: int = 0
    tokens_rejected: int = 0
    acceptance_rate: float = 0.0


class SpeculativeDiffusionDecoder:
    """Speculative decoding with a drafter-verifier MDM pair.

    The drafter and verifier can be:
    - Different models (e.g., 1B drafter + 8B verifier)
    - Same architecture at different quantizations (e.g., 4-bit draft, 8-bit verify)
    - Self-speculative: same model used for both (fast draft steps + careful verify)

    Usage::

        from mlx_d.model import load_model
        from mlx_d.spec_decode import SpeculativeDiffusionDecoder, SpecDecodeConfig

        # Load drafter (small/quantized) and verifier (full)
        drafter, tok, mask_id = load_model(model_id="mlx-community/LLaDA-8B-Instruct-mlx-4bit")
        verifier, _, _ = load_model(model_id="mlx-community/LLaDA-8B-Instruct-mlx-4bit")

        # For self-speculative: drafter == verifier
        decoder = SpeculativeDiffusionDecoder(
            config=SpecDecodeConfig(draft_steps=8, accept_threshold=0.3),
        )
        result = decoder.generate(
            drafter=drafter,
            verifier=verifier,
            tokenizer=tok,
            messages=[{"role": "user", "content": "Explain masked diffusion."}],
            mask_id=mask_id,
        )
    """

    def __init__(self, config: Optional[SpecDecodeConfig] = None):
        self.config = config or SpecDecodeConfig()

    def generate(
        self,
        drafter,
        verifier,
        tokenizer,
        messages: list[dict],
        *,
        gen_length: int = 128,
        mask_id: int = MASK_TOKEN_ID,
    ) -> SpecDecodeResult:
        """Run speculative diffusion decoding.

        Args:
            drafter: Small/fast model for proposing tokens.
            verifier: Large/accurate model for verification.
            tokenizer: HuggingFace tokenizer with ``apply_chat_template``.
            messages: Chat messages in OpenAI format.
            gen_length: Number of tokens to generate.
            mask_id: The ``[MASK]`` token id.

        Returns:
            A ``SpecDecodeResult`` with generated text and acceptance stats.
        """
        cfg = self.config
        t0 = time.perf_counter()

        # ── Tokenize ──
        prompt_text = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=False,
        )
        encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="np")
        prompt_ids = mx.array(encoded["input_ids"])
        prompt_len = prompt_ids.shape[1]

        # ── Initialize ──
        mask_tail = mx.full((1, gen_length), mask_id, dtype=mx.int32)
        x = mx.concatenate([prompt_ids.astype(mx.int32), mask_tail], axis=1)
        mx.eval(x)

        total_accepted = 0
        total_rejected = 0
        draft_calls = 0
        verify_calls = 0

        for round_idx in range(cfg.max_rounds):
            # Check if all masks are resolved
            masks_remaining = int(mx.sum(x[:, prompt_len:] == mask_id).item())
            if masks_remaining == 0:
                break

            # ── DRAFT PHASE ──
            # Run K denoising steps with the drafter
            x_draft = mx.array(x)  # copy
            draft_mask_count = masks_remaining
            draft_schedule = get_transfer_schedule(draft_mask_count, cfg.draft_steps)

            for step in range(cfg.draft_steps):
                logits = drafter(x_draft)
                if hasattr(logits, "logits"):
                    logits = logits.logits
                mx.eval(logits)

                noisy_logits = add_gumbel_noise(logits, cfg.temperature)
                x0 = mx.argmax(noisy_logits, axis=-1).astype(mx.int32)
                x0_probs = extract_confidence(logits, x0)

                mask_index = (x_draft == mask_id)
                x0 = mx.where(mask_index, x0, x_draft)
                confidence = mx.where(
                    mask_index, x0_probs,
                    mx.full(x0_probs.shape, float("-inf")),
                )

                k = draft_schedule[step]
                if k > 0:
                    conf_flat = confidence[0]
                    if k >= conf_flat.shape[0]:
                        top_indices = mx.arange(conf_flat.shape[0])
                    else:
                        partitioned = mx.argpartition(
                            conf_flat, kth=conf_flat.shape[0] - k,
                        )
                        top_indices = partitioned[-k:]

                    x_flat = x_draft[0].astype(mx.int32)
                    x0_flat = x0[0].astype(mx.int32)
                    top_indices = top_indices.astype(mx.int32)
                    new_vals = mx.take(x0_flat, top_indices)
                    old_vals = mx.take(x_flat, top_indices)
                    x_flat = x_flat.at[top_indices].add(new_vals - old_vals)
                    x_draft = x_flat.reshape(1, -1)

                mx.eval(x_draft)
                draft_calls += 1

            # ── VERIFY PHASE ──
            # Single forward pass with the verifier
            v_logits = verifier(x_draft)
            if hasattr(v_logits, "logits"):
                v_logits = v_logits.logits
            mx.eval(v_logits)
            verify_calls += 1

            v_noisy = add_gumbel_noise(v_logits, cfg.temperature)
            v_x0 = mx.argmax(v_noisy, axis=-1).astype(mx.int32)
            v_probs = extract_confidence(v_logits, v_x0)

            # ── ACCEPT/REJECT ──
            # For positions that were masked in the original x but filled by drafter:
            was_masked = (x[0, prompt_len:] == mask_id)
            draft_tokens = x_draft[0, prompt_len:]
            verifier_tokens = v_x0[0, prompt_len:]
            verifier_conf = v_probs[0, prompt_len:]

            # Accept if: verifier agrees with drafter AND confidence exceeds threshold
            agreement = (draft_tokens == verifier_tokens)
            high_confidence = (verifier_conf > cfg.accept_threshold)
            accept = was_masked & agreement & high_confidence

            # Also accept verifier's own high-confidence predictions for disagreements
            verifier_override = was_masked & (~agreement) & (verifier_conf > cfg.accept_threshold * 1.5)

            n_accepted = int(mx.sum(accept).item()) + int(mx.sum(verifier_override).item())
            n_rejected = int(mx.sum(was_masked).item()) - n_accepted
            total_accepted += n_accepted
            total_rejected += max(0, n_rejected)

            # Build new x: accept drafter where agreed, verifier where overridden, re-mask rest
            response_region = x[0, prompt_len:]
            response_region = mx.where(accept, draft_tokens, response_region)
            response_region = mx.where(verifier_override, verifier_tokens, response_region)

            x = mx.concatenate([
                x[:, :prompt_len],
                response_region.reshape(1, -1),
            ], axis=1)
            mx.eval(x)

            log.debug(
                "round %d: accepted=%d, rejected=%d, remaining=%d",
                round_idx, n_accepted, n_rejected,
                int(mx.sum(x[:, prompt_len:] == mask_id).item()),
            )

        # ── Final cleanup: resolve any remaining masks with verifier ──
        remaining = int(mx.sum(x[:, prompt_len:] == mask_id).item())
        if remaining > 0:
            log.info("resolving %d remaining masks with verifier", remaining)
            logits = verifier(x)
            if hasattr(logits, "logits"):
                logits = logits.logits
            mx.eval(logits)

            x0 = mx.argmax(logits, axis=-1).astype(mx.int32)
            mask_index = (x == mask_id)
            x = mx.where(mask_index, x0, x)
            mx.eval(x)
            verify_calls += 1

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # ── Decode ──
        output_ids = x[0, prompt_len:].tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True).strip()

        total_considered = total_accepted + total_rejected
        acceptance_rate = total_accepted / total_considered if total_considered > 0 else 0.0

        return SpecDecodeResult(
            text=text,
            token_ids=output_ids,
            elapsed_ms=elapsed_ms,
            draft_calls=draft_calls,
            verify_calls=verify_calls,
            tokens_accepted=total_accepted,
            tokens_rejected=total_rejected,
            acceptance_rate=acceptance_rate,
        )
