"""Basic text generation with LLaDA masked diffusion.

Minimal example: load model, generate text, print result.
"""

from mlx_d.config import ModelConfig, SamplerConfig
from mlx_d.model import load_model
from mlx_d.sampler import DiffusionSampler


def main():
    # Load the 4-bit quantized LLaDA-8B-Instruct model (~4.5GB)
    print("Loading LLaDA-8B-Instruct (4-bit)...")
    model, tokenizer, mask_id = load_model()

    # Configure the diffusion sampler
    # - 64 denoising steps
    # - 128-token response
    # - 32-token semi-autoregressive blocks (4 blocks × 16 steps each)
    config = SamplerConfig(
        steps=64,
        gen_length=128,
        block_length=32,
        temperature=0.0,          # greedy (argmax)
        remasking="low_confidence",
    )
    sampler = DiffusionSampler(config)

    # Generate
    messages = [
        {"role": "user", "content": "What makes masked diffusion different from autoregressive generation?"},
    ]

    print("Generating...\n")
    result = sampler.generate(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        mask_id=mask_id,
    )

    print(f"Response: {result.text}")
    print(f"\nStats: {result.elapsed_ms:.0f}ms | {result.tokens_per_second:.1f} tok/s")


if __name__ == "__main__":
    main()
