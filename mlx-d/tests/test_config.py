"""Tests for configuration dataclasses."""

from mlx_d.config import Config, ModelConfig, SamplerConfig, MASK_TOKEN_ID


class TestSamplerConfig:
    def test_defaults(self):
        cfg = SamplerConfig()
        assert cfg.steps == 64
        assert cfg.gen_length == 128
        assert cfg.block_length == 32
        assert cfg.temperature == 0.0
        assert cfg.remasking == "low_confidence"
        assert cfg.cfg_scale == 0.0
        assert cfg.mask_id == MASK_TOKEN_ID

    def test_custom_values(self):
        cfg = SamplerConfig(steps=32, gen_length=64, block_length=16, temperature=0.5)
        assert cfg.steps == 32
        assert cfg.gen_length == 64
        assert cfg.block_length == 16
        assert cfg.temperature == 0.5


class TestModelConfig:
    def test_defaults(self):
        cfg = ModelConfig()
        assert "LLaDA" in cfg.model_id
        assert cfg.q_bits == 4
        assert cfg.compile_model is True

    def test_custom_model(self):
        cfg = ModelConfig(model_id="custom/model", q_bits=8)
        assert cfg.model_id == "custom/model"
        assert cfg.q_bits == 8


class TestConfig:
    def test_nested_defaults(self):
        cfg = Config()
        assert isinstance(cfg.model, ModelConfig)
        assert isinstance(cfg.sampler, SamplerConfig)
        assert cfg.model.q_bits == 4
        assert cfg.sampler.steps == 64

    def test_mask_token_id_constant(self):
        """Verify the mask token id matches LLaDA's documented value."""
        assert MASK_TOKEN_ID == 126_336
