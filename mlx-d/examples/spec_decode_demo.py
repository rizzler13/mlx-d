"""Speculative diffusion decoding demo.

Demonstrates self-speculative mode: same model used as both drafter
and verifier, with the drafter running fast coarse steps and the
verifier doing a single careful verification pass.
"""

from mlx_d.config import ModelConfig
from mlx_d.model import load_model
from mlx_d.spec_decode import SpeculativeDiffusionDecoder, SpecDecodeConfig


def main():
    print("Loading model (self-speculative: same model for drafter + verifier)...\n")
    model, tokenizer, mask_id = load_model()

    config = SpecDecodeConfig(
        draft_steps=8,         # drafter runs 8 coarse steps per round
        accept_threshold=0.3,  # accept if verifier confidence > 30%
        max_rounds=8,          # up to 8 draft-verify rounds
    )
    decoder = SpeculativeDiffusionDecoder(config)

    prompts = [
        "What is the difference between masked diffusion and autoregressive generation?",
        "Explain why LLaDA doesn't need a KV cache.",
        "What is the Flexibility Trap in diffusion language models?",
    ]

    for prompt in prompts:
        print(f"Prompt: {prompt}")

        messages = [{"role": "user", "content": prompt}]
        result = decoder.generate(
            drafter=model,
            verifier=model,
            tokenizer=tokenizer,
            messages=messages,
            gen_length=128,
            mask_id=mask_id,
        )

        print(f"Response: {result.text}")
        print(
            f"  Stats: {result.elapsed_ms:.0f}ms | "
            f"draft_calls={result.draft_calls} | "
            f"verify_calls={result.verify_calls} | "
            f"acceptance_rate={result.acceptance_rate:.1%}"
        )
        print("─" * 60)
        print()


if __name__ == "__main__":
    main()
