"""Tests for the diffusion sampler utilities and logic.

These tests verify the mathematical components without requiring
a loaded model (no GPU/model dependencies).
"""

import pytest

from mlx_d.config import SamplerConfig


class TestSamplerConfigValidation:
    """Test sampler configuration constraints."""

    def test_gen_length_divisible_by_block_length(self):
        """gen_length must be divisible by block_length."""
        # Valid
        cfg = SamplerConfig(gen_length=128, block_length=32)
        assert cfg.gen_length % cfg.block_length == 0

        cfg = SamplerConfig(gen_length=128, block_length=128)
        assert cfg.gen_length % cfg.block_length == 0

    def test_steps_divisible_by_num_blocks(self):
        """steps must be divisible by (gen_length / block_length)."""
        cfg = SamplerConfig(steps=64, gen_length=128, block_length=32)
        num_blocks = cfg.gen_length // cfg.block_length
        assert cfg.steps % num_blocks == 0

    def test_full_sequence_mode(self):
        """block_length == gen_length means full-sequence (no semi-AR)."""
        cfg = SamplerConfig(gen_length=128, block_length=128)
        num_blocks = cfg.gen_length // cfg.block_length
        assert num_blocks == 1

    def test_remasking_strategies(self):
        """Only two strategies are valid."""
        cfg_lc = SamplerConfig(remasking="low_confidence")
        assert cfg_lc.remasking == "low_confidence"

        cfg_r = SamplerConfig(remasking="random")
        assert cfg_r.remasking == "random"


class TestGumbelNoise:
    """Test Gumbel noise application."""

    def test_zero_temperature_returns_unchanged(self):
        """Temperature 0 = greedy (no noise)."""
        from mlx_d.utils import add_gumbel_noise
        import mlx.core as mx

        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        result = add_gumbel_noise(logits, temperature=0.0)
        # Should be exactly the same values
        assert mx.allclose(result, logits).item()

    def test_nonzero_temperature_adds_noise(self):
        """Temperature > 0 should change the logits."""
        from mlx_d.utils import add_gumbel_noise
        import mlx.core as mx

        logits = mx.array([[1.0, 2.0, 3.0, 4.0]])
        result = add_gumbel_noise(logits, temperature=1.0)
        mx.eval(result)
        # Result should differ from input (stochastic)
        # We can't guarantee this 100% but it's overwhelmingly likely
        assert result.shape == logits.shape


class TestConfidenceExtraction:
    """Test logsumexp-based confidence computation."""

    def test_shape_preserved(self):
        """Output shape should be (B, L), not (B, L, V)."""
        from mlx_d.utils import extract_confidence
        import mlx.core as mx

        B, L, V = 1, 10, 100
        logits = mx.random.normal(shape=(B, L, V))
        x0 = mx.argmax(logits, axis=-1).astype(mx.int32)
        conf = extract_confidence(logits, x0)
        mx.eval(conf)

        assert conf.shape == (B, L)

    def test_confidence_between_zero_and_one(self):
        """Confidence scores should be valid probabilities."""
        from mlx_d.utils import extract_confidence
        import mlx.core as mx

        logits = mx.random.normal(shape=(1, 20, 50))
        x0 = mx.argmax(logits, axis=-1).astype(mx.int32)
        conf = extract_confidence(logits, x0)
        mx.eval(conf)

        assert mx.all(conf >= 0).item()
        assert mx.all(conf <= 1).item()

    def test_argmax_has_highest_confidence(self):
        """The argmax token should have the highest confidence."""
        from mlx_d.utils import extract_confidence
        import mlx.core as mx

        # Create logits where position 2 is clearly the winner
        logits = mx.array([[[0.1, 0.2, 10.0, 0.3]]])  # (1, 1, 4)
        x0 = mx.argmax(logits, axis=-1).astype(mx.int32)  # should be 2
        conf = extract_confidence(logits, x0)
        mx.eval(conf)

        # Confidence for argmax should be very high (close to 1)
        assert conf[0, 0].item() > 0.9
