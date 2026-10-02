"""Backfill of hyperparameters that predate their declaration in ``config/template.yml``.

``SpectrumPL`` and its subclasses used to read these keys through ``self.hparams.get(...)``
or ``getattr(self.hparams, ..., default)``. The keys are declared in the template now, so the
reads are canonical and ``migrate_legacy_hparams`` supplies the old fallbacks for checkpoints
saved before each key existed.

MASSter was named MASShatter until 2026-09-25, so pre-rename configs, checkpoints and W&B runs
spell its model type and keys ``mass_shatter``; ``migrate_legacy_masster_name`` renames them.
"""

from pathlib import Path

import pytest
import yaml

from fragnnet.runner import load_config
from fragnnet.utils.pl_utils import (
    get_pl_hparams,
    migrate_legacy_hparams,
    migrate_legacy_masster_name,
)

TEMPLATE_DIR = Path(__file__).parents[1] / "config"

SHARED_LEGACY_DEFAULTS = {
    "mol_max_degree": None,
    "mol_max_spatial_distance": None,
    "mol_multi_hop_max_distance": None,
    "report_metric_std": False,
    "inst_type_loss_tol": None,
    "eval_target_apply_train_rank_filters": False,
}

FRAGNNET_LEGACY_DEFAULTS = {
    "cc_channel_dropout": 0.0,
    "candidate_ion_head": False,
    "candidate_ion_formula_features": True,
    "candidate_ion_neutral_loss_features": True,
    "candidate_ion_chunk_size": 32768,
    "candidate_ion_checkpoint": True,
    "ion_channel_factorized": False,
    "ion_channel_adduct_gated": False,
    "ion_channel_rank": 32,
}


@pytest.mark.parametrize(("key", "expected"), sorted(SHARED_LEGACY_DEFAULTS.items(), key=str))
def test_shared_keys_backfill_for_every_model_type(key, expected):
    """A checkpoint from any model family gets the shared SpectrumPL keys."""
    config = {"model_type": "masster"}
    migrate_legacy_hparams(config)
    assert config[key] == expected


@pytest.mark.parametrize(("key", "expected"), sorted(FRAGNNET_LEGACY_DEFAULTS.items(), key=str))
def test_fragnnet_keys_backfill_only_for_fragnnet(key, expected):
    """FraGNNet construction flags reach FraGNNet checkpoints and nothing else."""
    fragnnet_config = {"model_type": "frag_gnn"}
    migrate_legacy_hparams(fragnnet_config)
    assert fragnnet_config[key] == expected

    masster_config = {"model_type": "masster"}
    migrate_legacy_hparams(masster_config)
    assert key not in masster_config


def test_explicit_values_are_never_overwritten():
    """An explicit config, checkpoint or CLI value always beats the backfill."""
    config = {
        "model_type": "frag_gnn",
        "report_metric_std": True,
        "ion_channel_rank": 8,
        "mol_max_degree": 64,
        "inst_type_loss_tol": {"ORB": {"rel": 1e-5, "min_mz": 200.0}},
    }
    migrate_legacy_hparams(config)
    assert config["report_metric_std"] is True
    assert config["ion_channel_rank"] == 8
    assert config["mol_max_degree"] == 64
    assert config["inst_type_loss_tol"] == {"ORB": {"rel": 1e-5, "min_mz": 200.0}}


def test_legacy_eval_filter_key_is_renamed():
    """``eval_target_apply_train_peak_filters`` carries its value to the current name."""
    config = {"model_type": "frag_gnn", "eval_target_apply_train_peak_filters": True}
    migrate_legacy_hparams(config)
    assert config["eval_target_apply_train_rank_filters"] is True
    assert "eval_target_apply_train_peak_filters" not in config


def test_current_key_wins_over_legacy_name():
    """When both spellings are present the current one is authoritative."""
    config = {
        "model_type": "frag_gnn",
        "eval_target_apply_train_peak_filters": True,
        "eval_target_apply_train_rank_filters": False,
    }
    migrate_legacy_hparams(config)
    assert config["eval_target_apply_train_rank_filters"] is False
    assert "eval_target_apply_train_peak_filters" not in config


@pytest.mark.parametrize(
    ("spec_params", "expected_nce", "expected_ace"),
    [
        ({"nce": True, "ace": False}, True, False),
        ({"nce": False, "ace": True}, False, True),
        ({}, False, False),
        (None, False, False),
    ],
)
def test_use_nce_and_use_ace_fall_back_to_spec_params(spec_params, expected_nce, expected_ace):
    """These two read the dataset section, not a constant, so the backfill must too."""
    config = {"model_type": "clip", "spec_params": spec_params}
    migrate_legacy_hparams(config)
    assert config["use_nce"] is expected_nce
    assert config["use_ace"] is expected_ace


def test_report_metric_std_resolves_the_conflicting_fallbacks():
    """The two removed fallbacks disagreed (False allocating, True reporting).

    A checkpoint without the key allocated no ``metric_sum_wx2`` dict but reported std,
    so ``_accumulate_running_metrics`` subscripted ``None``. ``False`` is the template
    default and the allocation default, so both sites now agree.
    """
    config = {"model_type": "masster"}
    migrate_legacy_hparams(config)
    assert config["report_metric_std"] is False


def test_get_pl_hparams_applies_the_backfill_and_copies():
    """The inference config path migrates without mutating the checkpoint."""
    ckpt = {"hyper_parameters": {"model_type": "frag_gnn", "lr": 1e-3}}
    hparams = get_pl_hparams(ckpt)
    assert hparams["report_metric_std"] is False
    assert hparams["ion_channel_rank"] == 32
    assert hparams["lr"] == pytest.approx(1e-3)
    assert "report_metric_std" not in ckpt["hyper_parameters"]


def test_migration_is_idempotent():
    """Running the backfill twice changes nothing the second time."""
    config = {"model_type": "frag_gnn", "spec_params": {"nce": True}}
    migrate_legacy_hparams(config)
    once = dict(config)
    migrate_legacy_hparams(config)
    assert config == once


def test_pre_rename_masster_spelling_is_renamed():
    """``mass_shatter`` model type and keys carry their values to the MASSter names."""
    config = {"model_type": "mass_shatter", "mass_shatter_head_dropout": 0.2, "lr": 1e-3}
    migrate_legacy_masster_name(config)
    assert config == {"model_type": "masster", "masster_head_dropout": 0.2, "lr": 1e-3}


def test_current_masster_key_wins_over_pre_rename_key():
    """An explicit current key beats the old spelling of the same key."""
    config = {
        "model_type": "masster",
        "mass_shatter_head_dropout": 0.2,
        "masster_head_dropout": 0.0,
    }
    migrate_legacy_masster_name(config)
    assert config == {"model_type": "masster", "masster_head_dropout": 0.0}


def test_get_pl_hparams_renames_a_pre_rename_checkpoint():
    """Old keys are renamed before the backfill, so a saved value is not replaced by a default."""
    ckpt = {
        "hyper_parameters": {"model_type": "mass_shatter", "mass_shatter_candidate_backend": "cuda"}
    }
    hparams = get_pl_hparams(ckpt)
    assert hparams["model_type"] == "masster"
    assert hparams["masster_candidate_backend"] == "cuda"
    assert "mass_shatter_candidate_backend" not in hparams
    assert ckpt["hyper_parameters"]["model_type"] == "mass_shatter"


def test_load_config_accepts_pre_rename_yaml_and_override(tmp_path):
    """A pre-rename experiment YAML and CLI override resolve to the current MASSter keys."""
    custom = tmp_path / "old_masster.yml"
    old_config = {"model_type": "mass_shatter", "mass_shatter_head_dropout": 0.37}
    custom.write_text(yaml.safe_dump(old_config))
    config = load_config(str(TEMPLATE_DIR), str(custom), ["mass_shatter_max_h_transfer=2"])
    assert config["model_type"] == "masster"
    assert config["masster_head_dropout"] == pytest.approx(0.37)
    assert config["masster_max_h_transfer"] == 2
    assert not any(key.startswith("mass_shatter_") for key in config)
