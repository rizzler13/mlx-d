"""Weight conversion and download helper for LLaDA models.

Provides a unified interface for:
1. Downloading official LLaDA weights from HuggingFace
2. Converting to MLX-native safetensors format
3. Quantizing (4-bit, 8-bit) for reduced memory footprint

The MLX-community already hosts pre-converted 4-bit weights at:
    mlx-community/LLaDA-8B-Instruct-mlx-4bit

This module is for users who want to convert from the official
GSAI-ML weights or use a different quantization level.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)


# ── Pre-converted models ────────────────────────────────────────────────────

KNOWN_MODELS = {
    "llada-8b-instruct-4bit": {
        "hf_id": "mlx-community/LLaDA-8B-Instruct-mlx-4bit",
        "description": "LLaDA-8B-Instruct, 4-bit quantized (~4.5GB)",
        "bits": 4,
        "size_gb": 4.5,
    },
    "llada-8b-instruct": {
        "hf_id": "GSAI-ML/LLaDA-8B-Instruct",
        "description": "LLaDA-8B-Instruct, full precision bf16 (~16GB)",
        "bits": 16,
        "size_gb": 16.0,
    },
}


def list_models() -> str:
    """List known LLaDA models with descriptions."""
    lines = ["\n  Available LLaDA models:", "  " + "─" * 60]
    for key, info in KNOWN_MODELS.items():
        lines.append(f"  {key:<30} {info['description']}")
        lines.append(f"  {'':30} HuggingFace: {info['hf_id']}")
        lines.append("")
    return "\n".join(lines)


def download_model(model_id: str) -> str:
    """Download a model from HuggingFace hub.

    Args:
        model_id: HuggingFace repo id (e.g., 'mlx-community/LLaDA-8B-Instruct-mlx-4bit').

    Returns:
        Local path to the downloaded model directory.
    """
    from huggingface_hub import snapshot_download

    log.info("downloading %s ...", model_id)
    path = snapshot_download(
        model_id,
        allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"],
    )
    log.info("downloaded to %s", path)
    return path


def convert_model(
    source_model: str,
    output_dir: Optional[str] = None,
    q_bits: int = 4,
    group_size: int = 64,
) -> str:
    """Convert official LLaDA weights to MLX format with quantization.

    Uses ``mlx_lm.convert`` under the hood. Requires mlx-lm to be installed.

    Args:
        source_model: HuggingFace repo id of the source model.
        output_dir: Where to write converted weights. Defaults to
            ``./mlx_models/{model_name}-mlx-{q_bits}bit``.
        q_bits: Quantization bits (4 or 8). Use 16 for no quantization.
        group_size: Quantization group size (default: 64).

    Returns:
        Path to the converted model directory.
    """
    if output_dir is None:
        model_name = source_model.split("/")[-1]
        output_dir = f"./mlx_models/{model_name}-mlx-{q_bits}bit"

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", source_model,
        "--mlx-path", str(output_path),
    ]

    if q_bits < 16:
        cmd.extend(["--q-bits", str(q_bits)])
        cmd.extend(["--q-group-size", str(group_size)])

    log.info("converting: %s", " ".join(cmd))

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        log.info("conversion complete: %s", output_dir)
        if result.stdout:
            log.debug(result.stdout)
        return str(output_path)
    except subprocess.CalledProcessError as e:
        log.error("conversion failed: %s", e.stderr)
        raise RuntimeError(f"Model conversion failed: {e.stderr}") from e
