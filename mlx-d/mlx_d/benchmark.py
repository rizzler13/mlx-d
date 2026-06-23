"""Throughput and memory benchmarks for LLaDA on MLX.

Measures tokens/second, latency per step, and memory usage across
different configurations (step counts, block sizes, quantizations).
"""

from __future__ import annotations

import json
import logging
import subprocess
import time
from dataclasses import asdict, dataclass, field

import mlx.core as mx

from .config import SamplerConfig
from .sampler import DiffusionSampler

log = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""

    config_name: str
    steps: int
    gen_length: int
    block_length: int
    temperature: float
    remasking: str
    total_ms: float
    tokens_per_second: float
    ms_per_step: float
    peak_memory_gb: float = 0.0


@dataclass
class BenchmarkSuite:
    """Collection of benchmark results."""

    model_id: str
    hardware: str
    results: list[BenchmarkResult] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    def to_table(self) -> str:
        """Format results as a readable table."""
        if not self.results:
            return "No results."

        header = (
            f"{'Config':<28} {'Steps':>5} {'GenLen':>6} {'Block':>5} "
            f"{'tok/s':>7} {'ms/step':>7} {'Total(ms)':>9} {'Mem(GB)':>7}"
        )
        sep = "─" * len(header)
        lines = [f"\n  Model: {self.model_id}", f"  Hardware: {self.hardware}", "", sep, header, sep]

        for r in self.results:
            lines.append(
                f"{r.config_name:<28} {r.steps:>5} {r.gen_length:>6} {r.block_length:>5} "
                f"{r.tokens_per_second:>7.1f} {r.ms_per_step:>7.1f} {r.total_ms:>9.0f} "
                f"{r.peak_memory_gb:>7.1f}"
            )
        lines.append(sep)
        return "\n".join(lines)


def get_hardware_info() -> str:
    """Get Apple Silicon chip info."""
    try:
        out = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True,
        ).strip()
        mem = subprocess.check_output(
            ["sysctl", "-n", "hw.memsize"], text=True,
        )
        mem_gb = int(mem.strip()) / (1024 ** 3)
        return f"{out} ({mem_gb:.0f}GB)"
    except Exception:
        return "Unknown Apple Silicon"


def get_memory_usage_gb() -> float:
    """Approximate current process memory usage."""
    try:
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return usage.ru_maxrss / (1024 ** 3)  # macOS reports in bytes
    except Exception:
        return 0.0


def run_benchmarks(
    model,
    tokenizer,
    mask_id: int,
    model_id: str = "unknown",
    prompt: str = "Explain the concept of masked diffusion in language models.",
    warmup_runs: int = 1,
    benchmark_runs: int = 3,
    configs: list[dict] | None = None,
) -> BenchmarkSuite:
    """Run a suite of benchmarks across different configurations.

    Args:
        model: Loaded MLX model.
        tokenizer: HuggingFace tokenizer.
        mask_id: The [MASK] token id.
        model_id: Model identifier for reporting.
        prompt: Test prompt.
        warmup_runs: Number of warmup runs (not measured).
        benchmark_runs: Number of measured runs per config.
        configs: List of config dicts to test. If None, uses defaults.

    Returns:
        A ``BenchmarkSuite`` with all results.
    """
    if configs is None:
        configs = [
            {"name": "full-seq-16step", "steps": 16, "gen_length": 128, "block_length": 128},
            {"name": "full-seq-32step", "steps": 32, "gen_length": 128, "block_length": 128},
            {"name": "full-seq-64step", "steps": 64, "gen_length": 128, "block_length": 128},
            {"name": "semi-ar-b32-64step", "steps": 64, "gen_length": 128, "block_length": 32},
            {"name": "semi-ar-b16-64step", "steps": 64, "gen_length": 128, "block_length": 16},
            {"name": "short-gen-32tok", "steps": 32, "gen_length": 32, "block_length": 32},
            {"name": "random-remask-64step", "steps": 64, "gen_length": 128, "block_length": 32,
             "remasking": "random"},
            {"name": "temp-0.5-64step", "steps": 64, "gen_length": 128, "block_length": 32,
             "temperature": 0.5},
        ]

    messages = [{"role": "user", "content": prompt}]
    suite = BenchmarkSuite(model_id=model_id, hardware=get_hardware_info())

    for cfg_dict in configs:
        name = cfg_dict.pop("name", "unnamed")
        sampler_cfg = SamplerConfig(
            steps=cfg_dict.get("steps", 64),
            gen_length=cfg_dict.get("gen_length", 128),
            block_length=cfg_dict.get("block_length", 32),
            temperature=cfg_dict.get("temperature", 0.0),
            remasking=cfg_dict.get("remasking", "low_confidence"),
            mask_id=mask_id,
        )
        sampler = DiffusionSampler(sampler_cfg)

        # Warmup
        for _ in range(warmup_runs):
            sampler.generate(model=model, tokenizer=tokenizer, messages=messages, mask_id=mask_id)

        # Benchmark
        timings = []
        for run_idx in range(benchmark_runs):
            result = sampler.generate(
                model=model, tokenizer=tokenizer, messages=messages, mask_id=mask_id,
            )
            timings.append(result.elapsed_ms)
            log.info(
                "  [%s] run %d/%d: %.0fms (%.1f tok/s)",
                name, run_idx + 1, benchmark_runs,
                result.elapsed_ms, result.tokens_per_second,
            )

        avg_ms = sum(timings) / len(timings)
        total_steps = sampler_cfg.steps
        tok_per_sec = sampler_cfg.gen_length / (avg_ms / 1000) if avg_ms > 0 else 0

        suite.results.append(BenchmarkResult(
            config_name=name,
            steps=sampler_cfg.steps,
            gen_length=sampler_cfg.gen_length,
            block_length=sampler_cfg.block_length,
            temperature=sampler_cfg.temperature,
            remasking=sampler_cfg.remasking,
            total_ms=avg_ms,
            tokens_per_second=tok_per_sec,
            ms_per_step=avg_ms / total_steps if total_steps > 0 else 0,
            peak_memory_gb=get_memory_usage_gb(),
        ))

    return suite
