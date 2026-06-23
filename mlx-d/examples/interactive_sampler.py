"""Interactive terminal visualization of the denoising process.

Shows step-by-step token unmasking with color coding:
  - Red: [MASK] tokens (not yet committed)
  - Yellow: newly committed tokens (this step)
  - Green: previously committed tokens
"""

import sys

from mlx_d.config import ModelConfig, SamplerConfig
from mlx_d.model import load_model
from mlx_d.sampler import DiffusionSampler, StepInfo


# ── ANSI colors ──
RED = "\033[91m"
YELLOW = "\033[93m"
GREEN = "\033[92m"
DIM = "\033[2m"
RESET = "\033[0m"
BOLD = "\033[1m"


class TokenVisualizer:
    """Tracks and displays token state across denoising steps."""

    def __init__(self, tokenizer, mask_id: int, prompt_len: int):
        self.tokenizer = tokenizer
        self.mask_id = mask_id
        self.prompt_len = prompt_len
        self.prev_masks: set[int] = set()
        self.step_count = 0

    def on_step(self, info: StepInfo):
        self.step_count += 1
        response_tokens = info.tokens[self.prompt_len:]

        # Find which positions are still masked
        current_masks = {i for i, t in enumerate(response_tokens) if t == self.mask_id}

        # Newly committed = was masked before, now isn't
        newly_committed = self.prev_masks - current_masks if self.prev_masks else set()

        # Build display
        parts = []
        for i, tid in enumerate(response_tokens):
            word = self.tokenizer.decode([tid])
            if tid == self.mask_id:
                parts.append(f"{RED}[M]{RESET}")
            elif i in newly_committed:
                parts.append(f"{YELLOW}{BOLD}{word}{RESET}")
            else:
                parts.append(f"{GREEN}{word}{RESET}")

        # Progress
        total = len(response_tokens)
        resolved = total - len(current_masks)
        pct = 100 * resolved / total

        # Clear line and print
        line = " ".join(parts[:30])  # cap at 30 tokens for readability
        if len(response_tokens) > 30:
            line += f" {DIM}... +{len(response_tokens) - 30} more{RESET}"

        t_label = f"t={1 - (info.step_idx + 1) / info.step_total:.2f}"
        print(
            f"\n  {DIM}block {info.block_idx + 1}/{info.block_total} "
            f"step {info.step_idx + 1:>2}/{info.step_total} "
            f"({t_label}) [{pct:5.1f}%]{RESET} "
            f"{info.elapsed_ms:.0f}ms"
        )
        print(f"  {line}")

        self.prev_masks = current_masks


def main():
    print(f"\n{BOLD}  mlx-d Interactive Sampler{RESET}")
    print(f"  {DIM}Visualizing the masked-diffusion reverse process{RESET}\n")

    # Load model
    print("  Loading model...")
    model, tokenizer, mask_id = load_model()

    # Prompt
    prompt = sys.argv[1] if len(sys.argv) > 1 else "What is the relationship between BERT and LLaDA?"

    print(f"\n  {DIM}Prompt:{RESET} {prompt}")
    print(f"\n  {DIM}Legend: {RED}[M]{RESET} = masked  "
          f"{YELLOW}■{RESET} = newly committed  "
          f"{GREEN}■{RESET} = locked in{RESET}\n")

    config = SamplerConfig(
        steps=32,
        gen_length=64,
        block_length=32,
        temperature=0.0,
        remasking="low_confidence",
    )
    sampler = DiffusionSampler(config)

    messages = [{"role": "user", "content": prompt}]

    # Create visualizer
    # We need prompt_len, so do a quick tokenize first
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False,
    )
    encoded = tokenizer(prompt_text, add_special_tokens=False, return_tensors="np")
    prompt_len = encoded["input_ids"].shape[1]

    viz = TokenVisualizer(tokenizer, mask_id, prompt_len)

    result = sampler.generate(
        model=model,
        tokenizer=tokenizer,
        messages=messages,
        mask_id=mask_id,
        on_step=viz.on_step,
    )

    print(f"\n\n  {BOLD}Final output:{RESET}")
    print(f"  {result.text}")
    print(f"\n  {DIM}[{result.elapsed_ms:.0f}ms | {result.tokens_per_second:.1f} tok/s]{RESET}\n")


if __name__ == "__main__":
    main()
