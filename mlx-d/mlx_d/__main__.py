"""CLI entry point for mlx-d.

Usage:
    mlx-d generate "What is masked diffusion?"
    mlx-d generate --steps 64 --block-length 32 "Explain LLaDA."
    mlx-d benchmark
    mlx-d convert --source GSAI-ML/LLaDA-8B-Instruct --q-bits 4
    mlx-d models
"""

from __future__ import annotations

import argparse
import json
import logging
import sys


def _setup_logging(debug: bool = False):
    level = logging.DEBUG if debug else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s %(name)s %(message)s",
        datefmt="%H:%M:%S",
    )
    for name in ("httpcore", "httpx", "asyncio", "filelock", "huggingface_hub"):
        logging.getLogger(name).setLevel(logging.WARNING)


def cmd_generate(args):
    """Generate text using masked diffusion."""
    from .config import ModelConfig, SamplerConfig
    from .model import load_model
    from .sampler import DiffusionSampler

    model_cfg = ModelConfig(model_id=args.model)
    model, tokenizer, mask_id = load_model(model_cfg)

    sampler_cfg = SamplerConfig(
        steps=args.steps,
        gen_length=args.gen_length,
        block_length=args.block_length,
        temperature=args.temperature,
        remasking=args.remasking,
        mask_id=mask_id,
    )
    sampler = DiffusionSampler(sampler_cfg)

    messages = [{"role": "user", "content": args.prompt}]

    def on_step(info):
        if args.verbose:
            masks = info.masks_remaining
            pct = 100 * (1 - masks / sampler_cfg.gen_length)
            print(
                f"\r  block {info.block_idx + 1}/{info.block_total} "
                f"step {info.step_idx + 1}/{info.step_total} "
                f"[{pct:5.1f}% resolved] "
                f"{info.elapsed_ms:.0f}ms",
                end="", flush=True,
            )

    result = sampler.generate(
        model=model, tokenizer=tokenizer, messages=messages,
        mask_id=mask_id, on_step=on_step if args.verbose else None,
    )

    if args.verbose:
        print()  # newline after progress

    print(f"\n{result.text}")
    print(f"\n  [{result.elapsed_ms:.0f}ms | {result.tokens_per_second:.1f} tok/s]")

    if args.json:
        print(json.dumps({
            "text": result.text,
            "elapsed_ms": result.elapsed_ms,
            "tokens_per_second": result.tokens_per_second,
            "prompt_len": result.prompt_len,
            "gen_length": len(result.token_ids),
        }, indent=2))


def cmd_benchmark(args):
    """Run throughput benchmarks."""
    from .config import ModelConfig
    from .model import load_model
    from .benchmark import run_benchmarks

    model_cfg = ModelConfig(model_id=args.model)
    model, tokenizer, mask_id = load_model(model_cfg)

    suite = run_benchmarks(
        model=model,
        tokenizer=tokenizer,
        mask_id=mask_id,
        model_id=args.model,
        warmup_runs=args.warmup,
        benchmark_runs=args.runs,
    )

    print(suite.to_table())

    if args.json:
        print(suite.to_json())


def cmd_convert(args):
    """Convert model weights."""
    from .convert import convert_model

    path = convert_model(
        source_model=args.source,
        output_dir=args.output,
        q_bits=args.q_bits,
    )
    print(f"Converted model saved to: {path}")


def cmd_models(args):
    """List available models."""
    from .convert import list_models
    print(list_models())


def cmd_spec_decode(args):
    """Generate text using speculative diffusion decoding."""
    from .config import ModelConfig
    from .model import load_model
    from .spec_decode import SpeculativeDiffusionDecoder, SpecDecodeConfig

    print("Loading drafter model...")
    d_cfg = ModelConfig(model_id=args.drafter)
    drafter, tokenizer, mask_id = load_model(d_cfg)

    if args.verifier == args.drafter:
        print("Self-speculative mode (same model for drafter and verifier)")
        verifier = drafter
    else:
        print("Loading verifier model...")
        v_cfg = ModelConfig(model_id=args.verifier)
        verifier, _, _ = load_model(v_cfg)

    spec_cfg = SpecDecodeConfig(
        draft_steps=args.draft_steps,
        accept_threshold=args.threshold,
        max_rounds=args.max_rounds,
    )
    decoder = SpeculativeDiffusionDecoder(spec_cfg)

    messages = [{"role": "user", "content": args.prompt}]
    result = decoder.generate(
        drafter=drafter,
        verifier=verifier,
        tokenizer=tokenizer,
        messages=messages,
        gen_length=args.gen_length,
        mask_id=mask_id,
    )

    print(f"\n{result.text}")
    print(f"\n  [{result.elapsed_ms:.0f}ms | draft_calls={result.draft_calls} | "
          f"verify_calls={result.verify_calls} | "
          f"acceptance={result.acceptance_rate:.1%}]")


def main():
    parser = argparse.ArgumentParser(
        prog="mlx-d",
        description="Masked diffusion language models on Apple Silicon",
    )
    parser.add_argument("--debug", action="store_true")
    sub = parser.add_subparsers(dest="command")

    # ── generate ──
    gen = sub.add_parser("generate", help="Generate text with masked diffusion")
    gen.add_argument("prompt", type=str, help="Input prompt")
    gen.add_argument("--model", default="mlx-community/LLaDA-8B-Instruct-mlx-4bit")
    gen.add_argument("--steps", type=int, default=64)
    gen.add_argument("--gen-length", type=int, default=128)
    gen.add_argument("--block-length", type=int, default=32)
    gen.add_argument("--temperature", type=float, default=0.0)
    gen.add_argument("--remasking", choices=["low_confidence", "random"], default="low_confidence")
    gen.add_argument("--verbose", "-v", action="store_true")
    gen.add_argument("--json", action="store_true", help="Output result as JSON")
    gen.set_defaults(func=cmd_generate)

    # ── benchmark ──
    bench = sub.add_parser("benchmark", help="Run throughput benchmarks")
    bench.add_argument("--model", default="mlx-community/LLaDA-8B-Instruct-mlx-4bit")
    bench.add_argument("--warmup", type=int, default=1)
    bench.add_argument("--runs", type=int, default=3)
    bench.add_argument("--json", action="store_true")
    bench.set_defaults(func=cmd_benchmark)

    # ── convert ──
    conv = sub.add_parser("convert", help="Convert model weights to MLX format")
    conv.add_argument("--source", required=True, help="HuggingFace source model")
    conv.add_argument("--output", default=None, help="Output directory")
    conv.add_argument("--q-bits", type=int, default=4, choices=[4, 8, 16])
    conv.set_defaults(func=cmd_convert)

    # ── models ──
    models = sub.add_parser("models", help="List available models")
    models.set_defaults(func=cmd_models)

    # ── spec-decode ──
    spec = sub.add_parser("spec-decode", help="Speculative diffusion decoding")
    spec.add_argument("prompt", type=str, help="Input prompt")
    spec.add_argument("--drafter", default="mlx-community/LLaDA-8B-Instruct-mlx-4bit")
    spec.add_argument("--verifier", default="mlx-community/LLaDA-8B-Instruct-mlx-4bit")
    spec.add_argument("--gen-length", type=int, default=128)
    spec.add_argument("--draft-steps", type=int, default=8)
    spec.add_argument("--threshold", type=float, default=0.3)
    spec.add_argument("--max-rounds", type=int, default=8)
    spec.set_defaults(func=cmd_spec_decode)

    args = parser.parse_args()
    _setup_logging(args.debug)

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
