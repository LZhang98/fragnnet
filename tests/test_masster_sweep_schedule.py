"""Check batch-dependent schedules and the fixed scientific sweep objective."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from fragnnet.pl_model.spectrum_pl import SpectrumPL
from fragnnet.runner import load_config, match_sweep_schedule, merge_wandb_config


@pytest.mark.parametrize(
    "batch,accumulation,total,warmup",
    [(512, 1, 24832, 1000), (256, 1, 49664, 2000), (256, 2, 24832, 1000)],
)
def test_sampled_batch_schedule(batch, accumulation, total, warmup):
    config = load_config("config", "config/masster/massspecgym/msgym_val_loss_sweep_40gb.yml", None)
    merge_wandb_config(config, {"train_batch_size": batch, "accumulate_grad_batches": accumulation})
    match_sweep_schedule(config, 99306)
    assert config["lr_total_steps"] == total
    assert config["lr_warmup_steps"] == warmup
    assert config["max_epochs"] == 128


@pytest.mark.parametrize(
    "override",
    [
        {"drop_last": True},
        {"masster_batch_sampler": True},
        # Curriculum epochs train on a growing subset, so they have fewer batches.
        {"bond_size_curriculum": True},
        {"group_sampler": True},
        {"model_type": "frag_gnn", "dynamic_batch_sampler": True},
        {"model_type": "frag_gnn", "batch_sampler_type": "ce_context"},
    ],
)
def test_schedule_rejects_incompatible_loaders(override):
    config = load_config("config", "config/masster/massspecgym/msgym_val_loss_sweep_40gb.yml", None)
    config.update(override)
    with pytest.raises(ValueError, match="requires"):
        match_sweep_schedule(config, 99306)


def test_sweep_fixed_objective_and_budget():
    config = load_config("config", "config/masster/massspecgym/msgym_val_loss_sweep_40gb.yml", None)
    sweep = yaml.safe_load(Path("sweep/masster/msgym_val_loss_48_40gb.yml").read_text())
    assert sweep["run_cap"] == 48
    assert sweep["metric"] == {"goal": "minimize", "name": "val_primary_loss_best/mean"}
    assert config["checkpoint_metric"] == "val_primary_loss_epoch/mean"
    assert config["checkpoint_metric_mode"] == "min"
    assert config["eval_test_split"] is False
    assert config["masster_unique_fragment_target"] == 256
    assert config["masster_ranked_mask_budget"] == 4096
    assert set(sweep["parameters"]).issubset(config)


def test_primary_loss_tracks_minimum():
    config = load_config("config", None, None)
    config["auxiliary_scores"] = []
    model = SimpleNamespace(
        hparams=SimpleNamespace(**config), metric_names={"loss", "primary_loss"},
        tolerance=1e-5, relative=True, tolerance_min_mz=200.0,
    )
    SpectrumPL._setup_metric_fns(model)
    assert model.metric_min_max["primary_loss"] == "min"
    assert "primary_loss" in model.metric_bests
