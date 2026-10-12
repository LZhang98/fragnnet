"""Per-model hooks (reorganization plan Phase 4).

``registry.get_model_hooks`` resolves each model type's ``fragnnet.hooks.ModelHooks`` subclass
lazily. The behavior the hooks took over is pinned elsewhere: the Phase 0 migration records in
``test_reorg_migrations.py`` and the MASSter backend and RT/CCS tests. This file checks the
dispatch itself.
"""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap

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
