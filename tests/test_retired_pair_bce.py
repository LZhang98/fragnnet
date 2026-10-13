"""Compatibility for removal of precursor and CE-switch BCE objectives."""

import pytest
import yaml

from fragnnet.pl_model.spectrum_pl import SpectrumPL
from fragnnet.runner import load_config, merge_wandb_config
from fragnnet.utils.migrations import (
    get_pl_hparams,
    migrate_legacy_hparams,
    retire_pair_bce_hparams,
)


@pytest.mark.parametrize("prefix", ["ce_pair_", "precursor_pair_"])
def test_disabled_legacy_settings_are_removed(prefix, tmp_path):
    legacy = {prefix + "loss_weight": 0.0, prefix + "eligible_inst_types": ["OB"]}
    path = tmp_path / "legacy.yml"
    path.write_text(yaml.safe_dump(legacy))
    config = load_config("config", str(path), None)
    assert not any(key.startswith(prefix) for key in config)
    checkpoint_config = get_pl_hparams({"hyper_parameters": legacy})
    assert not any(key.startswith(prefix) for key in checkpoint_config)
    migrate_legacy_hparams(legacy)
    assert not any(key.startswith(prefix) for key in legacy)


@pytest.mark.parametrize("prefix", ["ce_pair_", "precursor_pair_"])
@pytest.mark.parametrize("weight", [0.1, -1.0, float("nan")])
def test_enabled_legacy_objectives_fail_at_entry_points(prefix, weight, tmp_path):
    legacy = {prefix + "loss_weight": weight}
    path = tmp_path / "legacy.yml"
    path.write_text(yaml.safe_dump(legacy))
    for action in (
        lambda: load_config("config", str(path), None),
        lambda: load_config("config", None, [f"{prefix}loss_weight={weight}"]),
        lambda: migrate_legacy_hparams(dict(legacy)),
        lambda: SpectrumPL(**legacy),
        lambda: get_pl_hparams({"hyper_parameters": legacy}),
        lambda: merge_wandb_config({}, legacy),
    ):
        with pytest.raises(ValueError, match="retired BCE objective"):
            action()


def test_retirement_preserves_unrelated_ce_attention_bias():
    config = {"masster_intensity_ce_pair_bias": True, "ce_pair_loss_weight": 0.0}
    retire_pair_bce_hparams(config)
    assert config == {"masster_intensity_ce_pair_bias": True}
