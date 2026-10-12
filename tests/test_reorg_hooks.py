"""Per-model hooks (reorganization plan Phase 4).

``registry.get_model_hooks`` resolves each model type's ``fragnnet.hooks.ModelHooks`` subclass
lazily. The behavior the hooks took over is pinned elsewhere: the Phase 0 migration records in
``test_reorg_migrations.py`` and the MASSter backend and RT/CCS tests. This file checks the
dispatch itself.
"""

from __future__ import annotations

import itertools
import json
import subprocess
import sys
import textwrap
import types

import pytest
from reorg_layers import MODEL_TYPES, RETIRED_MODEL_TYPES

from fragnnet import registry
from fragnnet.hooks import ModelHooks

# Model types with their own hooks class; every other type uses the defaults.
OWN_HOOKS = {"frag_gnn": "FraGNNetHooks", "fragnnet": "FraGNNetHooks", "masster": "MassterHooks"}


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_every_model_type_resolves_its_hooks(model_type: str) -> None:
    hooks = registry.get_model_hooks(model_type)
    assert isinstance(hooks, ModelHooks)
    assert type(hooks).__name__ == OWN_HOOKS.get(model_type, "ModelHooks")
    assert registry.get_model_hooks(model_type) is hooks  # one instance per hooks class


@pytest.mark.parametrize("model_type", [None, "not_a_model", "mass_shatter", *RETIRED_MODEL_TYPES])
def test_lenient_lookup_gives_the_defaults(model_type) -> None:
    assert type(registry.get_model_hooks(model_type, strict=False)) is ModelHooks


@pytest.mark.parametrize("model_type", ["not_a_model", *RETIRED_MODEL_TYPES])
def test_strict_lookup_rejects_unknown_and_retired_types(model_type: str) -> None:
    with pytest.raises(ValueError):
        registry.get_model_hooks(model_type)


def test_default_hooks_change_nothing() -> None:
    hooks = ModelHooks()
    config = {"model_type": "neims", "spec_params": {"rt": True}}
    before = json.dumps(config, sort_keys=True)
    hooks.migrate_template_config(config)
    hooks.migrate_checkpoint_hparams(config)
    hooks.log_resolved_settings(object(), config, None)
    assert json.dumps(config, sort_keys=True) == before


def test_migration_and_validation_load_no_model_code() -> None:
    """Config migration and validation use the hooks modules only, never a model or extension.

    Validating any config consults every model's ``reject_foreign_config``, so it loads all
    hooks modules; they import core only.
    """
    code = textwrap.dedent(
        """
        import json, sys
        from fragnnet import registry
        from fragnnet.runner import load_config, validate_sampler_config
        from fragnnet.utils.pl_utils import get_pl_hparams
        get_pl_hparams({"hyper_parameters": {"model_type": "mass_shatter", "spec_params": {}}})
        registry.get_model_hooks("masster").migrate_template_config({})
        for model_type in ("neims", "masster", "frag_gnn"):
            validate_sampler_config(load_config("config", None, None, model_type=model_type))
        print(json.dumps(sorted(m for m in sys.modules if m.startswith("fragnnet"))))
        """
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    loaded = set(json.loads(out.stdout.strip().splitlines()[-1]))
    assert {"fragnnet.frag.masster_hooks", "fragnnet.frag.fragnnet_hooks"} <= loaded
    heavy = {
        "fragnnet.pl_model.masster_pl",
        "fragnnet.model.masster_model",
        "fragnnet.frag.masster_core",
        "fragnnet.pl_model.fragnnet_pl",
        "fragnnet.model.fragnnet_model",
        "fragnnet.utils.frag_utils",
        "fragnnet.frag.compute_frags",
        "fragnnet.frag.multi_cut_bfs",
    }
    assert not loaded & heavy


# The pre-hook inline logic of runner.run_training (Trainer kwargs), runner.init_dataloader
# (DataLoader kwargs) and SpectrumPL._seed_dataloader (advance decision), verbatim. Phase 4
# step 3 moved them into hooks; these tables keep the hooks equal to them.
def _old_trainer_kwargs(config: dict) -> dict:
    kwargs = {}
    if config["automatic_optimization"]:
        if config["num_workers"] > 0 and config["model_type"] != "masster":
            kwargs["reload_dataloaders_every_n_epochs"] = 1
    elif config["dynamic_batch_sampler"] or config["dual_group_dynamic_batch_sampler"]:
        kwargs["reload_dataloaders_every_n_epochs"] = 1
        kwargs["check_val_every_n_epoch"] = 1
    return kwargs


def _old_dataloader_kwargs(config: dict) -> dict:
    if (
        config["num_workers"] > 0
        and config["model_type"] == "masster"
        and config["automatic_optimization"]
    ):
        return {"persistent_workers": True}
    return {}


def _old_advances(hparams) -> bool:
    return bool(
        hparams.dag_size_curriculum
        or hparams.bond_size_curriculum
        or hparams.batch_sampler_type == "ce_context"
        or (hasattr(hparams, "masster_batch_sampler") and hparams.masster_batch_sampler)
    )


FRAGNNET_TYPES = ("frag_gnn", "fragnnet")


def _accepted_by_validation(model_type: str, flags: dict) -> bool:
    """Combinations ``validate_sampler_config`` lets through (the rest never reach the hooks)."""
    fragnnet = model_type in FRAGNNET_TYPES
    if (flags["dynamic"] or flags["dual"]) and (not fragnnet or flags["auto"]):
        return False
    if flags["dynamic"] and flags["dual"]:
        return False
    if (flags["dag"] or flags["ce_context"]) and not (fragnnet and flags["dynamic"]):
        return False
    if flags["dag"] and flags["ce_context"]:
        return False
    if flags["masster_sampler"] and (model_type != "masster" or not flags["auto"]):
        return False
    return not (flags["bond"] and (flags["dynamic"] or flags["dual"]))


FLAG_NAMES = ("auto", "dynamic", "dual", "dag", "bond", "ce_context", "masster_sampler")


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_loader_and_epoch_hooks_match_the_pre_hook_logic(model_type: str) -> None:
    hooks = registry.get_model_hooks(model_type)
    checked = 0
    for values in itertools.product([False, True], repeat=len(FLAG_NAMES)):
        flags = dict(zip(FLAG_NAMES, values))
        if not _accepted_by_validation(model_type, flags):
            continue
        for num_workers in (0, 2):
            config = {
                "model_type": model_type,
                "automatic_optimization": flags["auto"],
                "num_workers": num_workers,
                "dynamic_batch_sampler": flags["dynamic"],
                "dual_group_dynamic_batch_sampler": flags["dual"],
                "dag_size_curriculum": flags["dag"],
                "bond_size_curriculum": flags["bond"],
                "batch_sampler_type": "ce_context" if flags["ce_context"] else "spec_mol_frag",
                "masster_batch_sampler": flags["masster_sampler"],
            }
            assert hooks.trainer_options(config) == _old_trainer_kwargs(config), config
            assert hooks.dataloader_options(config) == _old_dataloader_kwargs(config), config
            hparams = types.SimpleNamespace(**config)
            assert bool(hooks.advances_sampler_each_epoch(hparams)) == _old_advances(hparams)
            checked += 1
    assert checked >= 4


@pytest.mark.parametrize("model_type", [t for t in MODEL_TYPES if t not in FRAGNNET_TYPES])
def test_only_fragnnet_and_masster_build_their_own_batch_samplers(model_type: str) -> None:
    config = {"masster_batch_sampler": False}
    assert (
        registry.get_model_hooks(model_type).build_batch_sampler(None, config, None, None) is None
    )
