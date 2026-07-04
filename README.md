<p align="center">
  <strong style="font-size:2em;">mlx-d</strong>
</p>

<h3 align="center">Masked diffusion language models on Apple Silicon</h3>

<p align="center">
  <a href="https://diffusiononmlx.netlify.app//">Blog</a> ·
  <a href="#quick-start">Quick Start</a> ·
  <a href="#examples">Examples</a> ·
  <a href="#architecture">Architecture</a> ·
  <a href="#benchmarks">Benchmarks</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10%2B-blue" alt="Python 3.10+">
  <img src="https://img.shields.io/badge/MLX-0.24%2B-orange" alt="MLX 0.24+">
  <img src="https://img.shields.io/badge/license-Apache%202.0-green" alt="License">
  <img src="https://img.shields.io/badge/platform-Apple%20Silicon-black" alt="Apple Silicon">
</p>

---

**mlx-d** is a clean, modular implementation of [LLaDA](https://arxiv.org/abs/2502.09992) (Large Language Diffusion with mAsking) for local inference on Apple Silicon via [MLX](https://github.com/ml-explore/mlx). It proves that autoregressive generation isn't the only path to language intelligence.

Instead of generating tokens left-to-right, LLaDA starts with a fully masked sequence and iteratively reveals tokens based on confidence — like a sculptor removing marble to reveal the figure inside.

## Highlights

- **Full LLaDA-8B-Instruct** running locally on Mac (M1/M2/M3/M4)
- **4-bit quantized** — only ~4.5 GB memory, fits on any Apple Silicon Mac
- **Semi-autoregressive block mode** — generates in 32-token blocks for 4× throughput
- **Speculative diffusion decoding** — draft-verify acceleration with configurable acceptance thresholds
- **Step-by-step visualization** — watch tokens unmask in real time via callbacks
- **Low-confidence remasking** — the key trick that makes masked diffusion work (re-mask uncertain predictions, not random ones)
- **No KV cache needed** — bidirectional attention means no growing memory per token

## Quick Start

### Install

```bash
# Clone the repository
git clone https://github.com/rizzler/mlx-d.git
cd mlx-d

# Install with pip
pip install -e .

# Or install with dev dependencies for testing
pip install -e ".[dev]"
```

### Generate text

```bash
# Basic generation (downloads 4-bit model on first run, ~4.5 GB)
mlx-d generate "What is masked diffusion?"

# With custom parameters
mlx-d generate "Explain LLaDA." \
  --steps 64 \
  --gen-length 128 \
  --block-length 32 \
  --temperature 0.0

# Verbose mode — shows step-by-step progress
mlx-d generate -v "How does bidirectional attention help?"
```

### Python API

```python
from mlx_d.model import load_model
from mlx_d.sampler import DiffusionSampler
from mlx_d.config import SamplerConfig

# Load model (~4.5 GB, cached after first download)
model, tokenizer, mask_id = load_model()

# Configure and run
sampler = DiffusionSampler(SamplerConfig(
    steps=64,
    gen_length=128,
    block_length=32,
    temperature=0.0,
    remasking="low_confidence",
))

result = sampler.generate(
    model=model,
    tokenizer=tokenizer,
    messages=[{"role": "user", "content": "What makes masked diffusion different?"}],
    mask_id=mask_id,
)

print(result.text)
print(f"{result.elapsed_ms:.0f}ms | {result.tokens_per_second:.1f} tok/s")
```

## Architecture

```
mlx-d/
├── mlx_d/
│   ├── __init__.py          # Package metadata
│   ├── __main__.py          # CLI entry point (generate, benchmark, convert, spec-decode)
│   ├── config.py            # Dataclass configs: SamplerConfig, ModelConfig, Config
│   ├── model.py             # LLaDA loading via patched LLaMA (bidirectional attention)
│   ├── sampler.py           # Core diffusion sampler with callbacks
│   ├── spec_decode.py       # Speculative diffusion decoding (draft → verify)
│   ├── benchmark.py         # Throughput benchmarks across configurations
│   ├── convert.py           # Weight conversion helper (HF → MLX quantized)
│   └── utils.py             # Transfer schedule, Gumbel noise, confidence extraction
├── examples/
│   ├── basic_generation.py  # Minimal: load → generate → print
│   ├── interactive_sampler.py  # Terminal visualization of denoising steps
│   └── spec_decode_demo.py  # Speculative diffusion decoding demo
├── tests/
│   ├── test_config.py       # 6 tests — config defaults and validation
│   ├── test_sampler.py      # 9 tests — Gumbel noise, confidence, remasking
│   └── test_schedule.py     # 10 tests — transfer schedule edge cases
├── docs/
│   └── index.html           # Technical blog: "The Diffusion Revolt" (GitHub Pages)
├── pyproject.toml
└── LICENSE                  # Apache 2.0
```

### How it works

LLaDA's key insight: take a LLaMA-3 8B model, remove the causal attention mask, and train it as a masked diffusion model. The architecture is identical — only the mask is different.

```
┌─────────────────────────────────────────────────┐
│                   GENERATION                     │
│                                                  │
│  Step 0:  [M] [M] [M] [M] [M] [M] [M] [M] [M]  │
│  Step 16: LLaDA [M] [M] [M] [M] model [M] [M]   │
│  Step 32: LLaDA is [M] masked [M] language model  │
│  Step 48: LLaDA is a masked diffusion language    │
│  Step 64: LLaDA is a masked diffusion LM that ... │
│                                                  │
│  ↑ High-confidence tokens commit first            │
│  ↑ Structure emerges before details               │
└─────────────────────────────────────────────────┘
```

**Loading trick** (in `model.py`):
```python
# The single line that transforms LLaMA → LLaDA
import mlx_lm.models.llama as llama_mod
llama_mod.create_attention_mask = lambda *a, **kw: None
```

**Sampling loop** (in `sampler.py`):
1. Start with `[prompt] + [MASK] × gen_length`
2. Forward pass → get logits for all positions
3. Compute confidence per token (logsumexp trick — no full softmax)
4. Commit the top-k most confident tokens
5. Repeat until all masks are resolved

## Configuration

### `SamplerConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `steps` | `64` | Total denoising steps across all blocks |
| `gen_length` | `128` | Number of response tokens to generate |
| `block_length` | `32` | Tokens per semi-autoregressive block |
| `temperature` | `0.0` | Gumbel noise (0 = greedy argmax) |
| `remasking` | `"low_confidence"` | Strategy: `"low_confidence"` or `"random"` |
| `cfg_scale` | `0.0` | Classifier-free guidance scale (0 = off) |

### `ModelConfig`

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_id` | `mlx-community/LLaDA-8B-Instruct-mlx-4bit` | HuggingFace repo id |
| `q_bits` | `4` | Quantization level (4, 8, or 16) |
| `compile_model` | `True` | Apply `mx.compile` for throughput |

### Block mode explained

Setting `block_length < gen_length` enables semi-autoregressive mode:

```
gen_length=128, block_length=32 → 4 blocks × 16 steps each

Block 1: generate tokens 1-32    (16 denoising steps)
Block 2: generate tokens 33-64   (16 denoising steps, sees block 1)
Block 3: generate tokens 65-96   (16 denoising steps, sees blocks 1-2)
Block 4: generate tokens 97-128  (16 denoising steps, sees blocks 1-3)
```

Each subsequent block can attend to all previously committed tokens, combining the parallelism of diffusion with the coherence of autoregressive generation.

## CLI Reference

```bash
# Generate text
mlx-d generate "Your prompt here" [options]
  --model MODEL_ID          # HuggingFace model (default: 4-bit)
  --steps N                 # Denoising steps (default: 64)
  --gen-length N            # Response length (default: 128)
  --block-length N          # Block size (default: 32)
  --temperature T           # Gumbel noise (default: 0.0)
  --remasking STRATEGY      # low_confidence | random
  -v, --verbose             # Show step-by-step progress
  --json                    # Output as JSON

# Speculative diffusion decoding
mlx-d spec-decode "Your prompt" [options]
  --drafter MODEL_ID        # Draft model
  --verifier MODEL_ID       # Verifier model (same = self-speculative)
  --draft-steps N           # Coarse steps per round (default: 8)
  --threshold T             # Acceptance threshold (default: 0.3)
  --max-rounds N            # Max draft-verify rounds (default: 8)

# Run benchmarks
mlx-d benchmark [options]
  --model MODEL_ID
  --warmup N                # Warmup runs (default: 1)
  --runs N                  # Benchmark runs (default: 3)
  --json                    # Output as JSON

# Convert weights
mlx-d convert --source GSAI-ML/LLaDA-8B-Instruct [options]
  --output DIR              # Output directory
  --q-bits {4,8,16}         # Quantization bits (default: 4)

# List available models
mlx-d models
```

## Examples

### Basic generation

```bash
python examples/basic_generation.py
```

Loads the model, generates a response, prints the result with timing stats.

### Interactive sampler (terminal visualization)

```bash
python examples/interactive_sampler.py "What is the relationship between BERT and LLaDA?"
```

Displays a live, color-coded view of the denoising process:
- 🔴 `[M]` — masked (not yet committed)
- 🟡 **bold** — newly committed this step
- 🟢 text — locked in from previous steps

### Speculative diffusion decoding

```bash
python examples/spec_decode_demo.py
```

Demonstrates self-speculative mode: the same model acts as both drafter (fast, coarse steps) and verifier (single careful pass). Reports acceptance rates and speedup.

### Step callbacks

```python
from mlx_d.sampler import DiffusionSampler, StepInfo

def my_callback(info: StepInfo):
    pct = 100 * (1 - info.masks_remaining / 128)
    print(f"Block {info.block_idx+1} Step {info.step_idx+1}: {pct:.0f}% resolved ({info.elapsed_ms:.0f}ms)")

sampler = DiffusionSampler()
result = sampler.generate(
    model=model, tokenizer=tokenizer,
    messages=[{"role": "user", "content": "Hello!"}],
    mask_id=mask_id,
    on_step=my_callback,
)
```

## Benchmarks

Run benchmarks on your own hardware:

```bash
mlx-d benchmark --runs 5 --json
```

Expected ranges on Apple Silicon (LLaDA-8B-Instruct 4-bit, 128 tokens, 64 steps):

| Chip | Block Size | tok/s (approx) | Memory |
|------|-----------|----------------|--------|
| M1   | 32        | ~8-12          | ~5 GB  |
| M2   | 32        | ~10-15         | ~5 GB  |
| M3 Pro | 32      | ~15-22         | ~5 GB  |
| M4 Pro | 32      | ~20-30         | ~5 GB  |

> **Note**: These are approximate ranges. Actual throughput depends on thermal conditions, system load, and MLX version. Run `mlx-d benchmark` for precise numbers on your hardware.

## Blog

The companion blog post **"The Diffusion Revolt"** covers the full theory, architecture, and implementation in depth:

🔗 **[rizzler.github.io/mlx-d](https://rizzler.github.io/mlx-d/)**

Topics covered:
1. Why masked diffusion models challenge autoregressive dominance
2. LLaDA's architecture — it's literally LLaMA minus the causal mask
3. The forward process (masking) and reverse process (denoising)
4. Semi-autoregressive block generation
5. The confidence trick: logsumexp without full softmax
6. MLX-specific optimizations (Metal shaders, `mx.compile`, `argpartition`)
7. Speculative diffusion decoding
8. Applications: TTS, RL-trained diffusion, infilling

The blog is served as a single static HTML file from the `docs/` directory via GitHub Pages.

## Testing

```bash
# Run all tests (25 tests)
pytest -v

# Run specific test files
pytest tests/test_schedule.py -v    # 10 tests — transfer schedule
pytest tests/test_sampler.py -v     # 9 tests — sampling mechanics
pytest tests/test_config.py -v      # 6 tests — configuration
```

## How LLaDA differs from GPT-style models

| | Autoregressive (GPT) | Masked Diffusion (LLaDA) |
|---|---|---|
| **Generation** | Left-to-right, one token at a time | All positions simultaneously, iteratively refined |
| **Attention** | Causal (triangular mask) | Full bidirectional |
| **KV Cache** | Required (grows with sequence) | Not needed |
| **Parallelism** | Limited (each token depends on previous) | High (all tokens generated in parallel per step) |
| **Infilling** | Requires special training | Native (just mask the gap) |
| **Quality driver** | Exact probability chain | Iterative refinement via confidence |

## Requirements

- **Python** 3.10+
- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **MLX** 0.24+
- **~5 GB** free memory (for 4-bit quantized model)

## Citation

If you use mlx-d in your research, please cite LLaDA:

```bibtex
@article{nie2025llada,
  title={Large Language Diffusion Models},
  author={Nie, Shen and Zhu, Fengqi and You, Chao and Zhang, Xiaojun and Ou, Zhenguo and Li, Jiacheng},
  journal={arXiv preprint arXiv:2502.09992},
  year={2025}
}
```

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

The LLaDA model weights are subject to their own license from [GSAI-ML](https://huggingface.co/GSAI-ML/LLaDA-8B-Instruct).
