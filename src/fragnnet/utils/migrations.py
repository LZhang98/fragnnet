"""Config and checkpoint-hyperparameter migrations (split out of ``utils/pl_utils.py``).

Renames from the MASSter rename, backfills for keys older checkpoints predate, and
``get_pl_hparams``, which applies them all (plus each model's ``migrate_checkpoint_hparams``
hook) to a checkpoint's hyperparameters.
"""

import copy

from fragnnet import registry

# MASSter was named MASShatter until 2026-09-25. Configs, checkpoints, W&B runs and sweeps from
# before the rename declare `model_type: mass_shatter` and `mass_shatter_*` keys.
LEGACY_MASSTER_MODEL_TYPE = "mass_shatter"
_LEGACY_MASSTER_KEY_PREFIX = "mass_shatter_"


def migrate_legacy_masster_key(key: str) -> str:
    """Return the current name of a config key that may predate the MASSter rename."""
    if key.startswith(_LEGACY_MASSTER_KEY_PREFIX):
        return "masster_" + key.removeprefix(_LEGACY_MASSTER_KEY_PREFIX)
    return key


def migrate_legacy_masster_name(config: dict) -> None:
    """Rename pre-rename MASSter model type and ``mass_shatter_*`` keys at config boundaries.

    Mutates the supplied mapping. The current key takes precedence when both spellings exist
    (e.g. an explicit override on top of an old checkpoint). Only top-level keys are renamed;
    MASSter keys are never nested.

    Args:
        config: Hyperparameter mapping from a config file, checkpoint, or W&B run.
    """
    if config.get("model_type") == LEGACY_MASSTER_MODEL_TYPE:
        config["model_type"] = "masster"
    for old_key in [key for key in config if key.startswith(_LEGACY_MASSTER_KEY_PREFIX)]:
        value = config.pop(old_key)
        config.setdefault(migrate_legacy_masster_key(old_key), value)


def migrate_masster_proposal_method(config: dict) -> None:
    """Normalize the original proposal key/values at configuration boundaries.

    Mutates the supplied mapping. The canonical key takes precedence when both keys
    exist (e.g. an explicit checkpoint override). Missing keys remain missing so fresh
    model configurations still fail validation; legacy loaders supply defaults.
    """
    old_key = "masster_proposal_type"
    new_key = "masster_proposal_method"
    if old_key in config:
        old_value = config.pop(old_key)
        if new_key not in config:
            config[new_key] = old_value
    if new_key in config:
        value = config[new_key]
        if value == "fast_gnn":
            config[new_key] = "connected_fragment_beam"


# Keys that `SpectrumPL` and its subclasses once read through a fallback default instead of
# declaring them. `config/template.yml` declares all of them now, so a fresh config always
# supplies them and a missing key can only come from a checkpoint saved before the key existed.
# Each value below is the fallback the removed `.get(...)`/`getattr(...)` call supplied, so an
# old checkpoint keeps its original behavior.
#
# `report_metric_std` had two disagreeing fallbacks (`False` when allocating the trackers,
# `True` when reporting), which made a checkpoint without the key raise in
# `_accumulate_running_metrics`. `False` is both the template default and the allocation
# default, so it is the value that keeps old checkpoints running.
_LEGACY_HPARAM_DEFAULTS = {
    # Graphormer shape limits: `None` disables the corresponding embedding.
    "mol_max_degree": None,
    "mol_max_spatial_distance": None,
    "mol_multi_hop_max_distance": None,
    # SpectrumPL evaluation and reporting.
    "report_metric_std": False,
    "inst_type_loss_tol": None,
    "eval_target_apply_train_rank_filters": False,
}

# FraGNNet-only construction flags, applied when the checkpoint is a FraGNNet model.
_LEGACY_FRAGNNET_HPARAM_DEFAULTS = {
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

_FRAGNNET_MODEL_TYPES = frozenset({"frag_gnn", "fragnnet"})


def retire_pair_bce_hparams(config: dict) -> None:
    """Discard disabled legacy pair-BCE settings; reject enabled retired objectives."""
    for prefix in ("ce_pair_", "precursor_pair_"):
        key = prefix + "loss_weight"
        if config.get(key, 0.0) != 0.0:
            raise ValueError(
                f"{key} enables a retired BCE objective; use the original revision "
                "to reproduce it, or explicitly set its weight to 0.0"
            )
    for key in list(config):
        if key.startswith(("ce_pair_", "precursor_pair_")):
            del config[key]


def migrate_legacy_hparams(config: dict) -> None:
    """Backfill hyperparameters that predate their declaration in ``config/template.yml``.

    Mutates the supplied mapping in place. Only missing keys are filled, so an explicit
    value in a config, checkpoint or CLI override always wins. Fresh configurations already
    carry every key from the template and are left untouched.

    Args:
        config: Hyperparameter mapping from a config file or a checkpoint.
    """
    retire_pair_bce_hparams(config)

    # 1. Rename the legacy eval-filter key before the shared defaults fill the new one.
    legacy_rank_filter_key = "eval_target_apply_train_peak_filters"
    rank_filter_key = "eval_target_apply_train_rank_filters"
    if legacy_rank_filter_key in config:
        legacy_value = config.pop(legacy_rank_filter_key)
        if rank_filter_key not in config:
            config[rank_filter_key] = legacy_value

    # 2. Shared SpectrumPL keys.
    for key, value in _LEGACY_HPARAM_DEFAULTS.items():
        if key not in config:
            config[key] = value

    # 3. Per-model-family keys.
    model_type = config.get("model_type")
    if model_type in _FRAGNNET_MODEL_TYPES:
        for key, value in _LEGACY_FRAGNNET_HPARAM_DEFAULTS.items():
            if key not in config:
                config[key] = value

    # 4. `use_nce`/`use_ace` fell back to the dataset's `spec_params` rather than a constant,
    # so their backfill has to read that section instead of a table.
    spec_params = config.get("spec_params") or {}
    for key, spec_key in (("use_nce", "nce"), ("use_ace", "ace")):
        if key not in config:
            config[key] = bool(spec_params.get(spec_key, False))


def migrate_rt_ccs_spec_params(hparams: dict) -> None:
    """Disable RT/CCS labels in ``spec_params`` saved before they existed (every model type).

    The datasets read these keys for any model, so the backfill is not tied to MASSter, whose
    RT/CCS head keys ``MassterHooks.migrate_checkpoint_hparams`` adds.
    """
    if "spec_params" in hparams:
        for name in ("rt", "ccs"):
            hparams["spec_params"].setdefault(name, False)
            hparams["spec_params"].setdefault(f"{name}_labels_fp", None)
        hparams["spec_params"].setdefault("rt_ccs", False)
        hparams["spec_params"].setdefault("rt_ccs_labels_fp", None)


def get_pl_hparams(ckpt: dict) -> dict:
    """Extract and return a deep copy of hyperparameters from a checkpoint.

    Args:
        ckpt: Checkpoint dictionary containing 'hyper_parameters' key.

    Returns:
        Deep copy of hyperparameters dictionary.

    Raises:
        ValueError: If the hyperparameters name a retired instrument type.
    """
    hyper_parameters = copy.deepcopy(ckpt["hyper_parameters"])
    retire_pair_bce_hparams(hyper_parameters)
    # Embedding indices follow the sorted label order, which the OB -> ORB rename keeps, but a
    # checkpoint keyed by OB filters out and mis-tolerances every relabelled Orbitrap spectrum.
    retired = find_retired_inst_type_refs(hyper_parameters)
    if retired:
        raise ValueError(
            f"checkpoint hparams name a retired instrument type at {retired}. "
            f"{RETIRED_INST_TYPE_HELP}"
        )
    migrate_legacy_masster_name(hyper_parameters)
    migrate_rt_ccs_spec_params(hyper_parameters)
    migrate_legacy_hparams(hyper_parameters)
    # Old checkpoints retain first-replicate CE context unless explicitly migrated.
    if "spec_params" in hyper_parameters:
        spec_params = hyper_parameters["spec_params"]
        if "merge_ce_aggregation" not in spec_params:
            spec_params["merge_ce_aggregation"] = "legacy"
    # Model-specific backfills. The type may be missing, unknown or retired here; those get the
    # default no-op hooks, and load_config reports them.
    registry.get_model_hooks(
        hyper_parameters.get("model_type"), strict=False
    ).migrate_checkpoint_hparams(hyper_parameters)
    return hyper_parameters


# Retired instrument-type labels. "FT" was split into the two Orbitrap labels on 2026-09-28, and
# those were renamed from "OB" / "OB_F" to "ORB" / "ORB_F" on 2026-10-01. Relabelled data has
# none of them, so a config table or list still keyed by one would silently stop applying to
# Orbitrap spectra; configs, checkpoints and spec_dfs that name one are refused.
RETIRED_INST_TYPES = ("FT", "OB", "OB_F")


RETIRED_INST_TYPE_HELP = (
    "Orbitraps are 'ORB' (Elite, Velos, unnamed) and 'ORB_F' (Orbitrap Fusion). 'OB' / 'OB_F' "
    "were renamed on 2026-10-01: migrate a spec_df.pkl or checkpoint with "
    "preproc_scripts/dataset_migration/rename_orbitrap_inst_type.py. 'FT' was split on "
    "2026-09-28: relabel a spec_df.pkl with "
    "preproc_scripts/dataset_migration/relabel_orbitrap_inst_type.py. See "
    "docs/findings/nist_orbitrap_instrument_ce_scales_2026_09_28.md."
)


def find_retired_inst_type_refs(node: object, path: str = "") -> list[str]:
    """Return the paths at which a config names a retired instrument type.

    Args:
        node: A resolved config or checkpoint hparams, or any part of one.
        path: Dotted path of ``node`` inside the config.

    Returns:
        Paths of every dict key, list item or string value in ``RETIRED_INST_TYPES``.
    """
    if isinstance(node, str):
        return [path] if node in RETIRED_INST_TYPES else []
    found = []
    if isinstance(node, dict):
        for key, value in node.items():
            child = f"{path}.{key}" if path else str(key)
            if key in RETIRED_INST_TYPES:
                found.append(child)
            found += find_retired_inst_type_refs(value, child)
    elif isinstance(node, list):
        for idx, value in enumerate(node):
            found += find_retired_inst_type_refs(value, f"{path}[{idx}]")
    return found
