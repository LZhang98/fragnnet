"""Tests for checkpoint-hyperparameter compatibility migrations."""

import torch as th

from fragnnet.utils.script_utils import get_config_from_ckpt


def test_legacy_mass_shatter_checkpoint_config_is_migrated(tmp_path) -> None:
    checkpoint_path = tmp_path / "legacy_mass_shatter.ckpt"
    th.save(
        {
            "hyper_parameters": {
                "model_type": "mass_shatter",
                "existing_value": 7,
            }
        },
        checkpoint_path,
    )

    config = get_config_from_ckpt(str(checkpoint_path))

    assert config["existing_value"] == 7
    assert config["mass_shatter_intensity_output"] == "probability"
    assert config["mass_shatter_peak_matching"] == "closest"
    assert config["mass_shatter_intensity_entropy_match_weight"] == 0.0
