"""Tests for runner config validation helpers."""

import pytest

from fragnnet.runner import validate_sampler_config


def _base_config(**overrides) -> dict:
    """Return a minimal config dict for validate_sampler_config tests."""
    cfg = {
        "dynamic_batch_sampler": True,
        "model_type": "frag_gnn",
        "automatic_optimization": False,
        "dynamic_batch_sampler_mode": "frag_node",
        "dynamic_batch_sampler_max": 1000,
    }
    cfg.update(overrides)
    return cfg


class TestValidateSamplerConfig:
    def test_disabled_sampler_always_passes(self):
        # dynamic_batch_sampler=False should skip all checks regardless of other values
        cfg = _base_config(
            dynamic_batch_sampler=False,
            model_type="spectrum",  # would fail if checks ran
            dynamic_batch_sampler_mode="invalid",
            dynamic_batch_sampler_max=None,
        )
        validate_sampler_config(cfg)  # must not raise

    def test_frag_node_mode_accepted(self):
        validate_sampler_config(_base_config(dynamic_batch_sampler_mode="frag_node"))

    def test_frag_edge_mode_accepted(self):
        # NodeMLP (or any frag_gnn) must accept frag_edge budget mode
        validate_sampler_config(_base_config(dynamic_batch_sampler_mode="frag_edge"))

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError, match="frag_node or frag_edge"):
            validate_sampler_config(_base_config(dynamic_batch_sampler_mode="invalid_mode"))

    def test_non_frag_gnn_model_raises(self):
        with pytest.raises(ValueError, match="frag_gnn"):
            validate_sampler_config(_base_config(model_type="spectrum"))

    def test_automatic_optimization_raises(self):
        with pytest.raises(ValueError, match="automatic optimization"):
            validate_sampler_config(_base_config(automatic_optimization=True))

    def test_missing_max_raises(self):
        with pytest.raises(ValueError, match="dynamic_batch_sampler_max"):
            validate_sampler_config(_base_config(dynamic_batch_sampler_max=None))
