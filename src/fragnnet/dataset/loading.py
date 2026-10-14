"""Datasets and DataLoaders for a resolved config, and the sampler settings they accept.

Core chooses the index sampler; a model's own batch sampler comes from its hooks
(``fragnnet.hooks.ModelHooks.build_batch_sampler``).
"""

import logging
from collections.abc import Sequence
from multiprocessing import Manager

import torch as th
from torch.utils.data import DataLoader, Dataset, Sampler
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from fragnnet import registry
from fragnnet.dataset.data_sampler import (
    BondCountCurriculumSampler,
    FormulaPrecTypeGroupSampler,
    GroupSampler,
    get_group_sampler,
    validate_bond_size_curriculum,
)
from fragnnet.hooks import ModelHooks
from fragnnet.utils.setup_utils import get_dataset_cls

logger = logging.getLogger(__name__)

# Valid values checked by `validate_sampler_config`.
VALID_SIMPLE_GROUP_SAMPLER_TYPES = frozenset(
    {
        "group",
        "mol",
        "group_mol",
        "adduct_inst_type",
        "adduct_inst_type_group",
        "adduct_inst_type_frag_mode",
        "adduct_inst_type_frag_mode_group",
    }
)
VALID_BATCH_SAMPLER_TYPES = frozenset({"spec_mol_frag", "ce_context"})
# Train samplers that `init_dataloader` chooses between; at most one may be set.
GROUP_SAMPLER_KEYS = ("group_sampler", "simple_group_sampler", "formula_prec_type_group_sampler")


def init_dataset(config_d: dict, splits: Sequence[str] = ("train", "val")) -> tuple:
    """
    Instantiate dataset objects for the requested data splits using the configured model.

    Resolves the dataset class with `get_dataset_cls(config_d["model_type"])`, prepares
    shared manager-backed dicts when `config_d["num_workers"] > 0 and config_d["share_memory"]`,
    and constructs one dataset per split by passing `split=...` plus the prepared data
    dictionaries merged into the dataset constructor call.

    Args:
        config_d (dict): Configuration dictionary. Must include "model_type" and may
            contain keyword arguments consumed by the resolved dataset class constructor.
        splits (tuple[str, ...], optional): Iterable of split names to instantiate.
            Defaults to ("train", "val").

    Returns:
        tuple[Dataset, ...]: Dataset instances in the same order as `splits`.

    Raises:
        ValueError: If a config key has the name of one of the dataset's data dicts.
    """
    dataset_cls = get_dataset_cls(config_d["model_type"])
    data_dict_types = dataset_cls.get_data_dict_types()
    shadowed = sorted(set(data_dict_types).intersection(config_d))
    if shadowed:
        raise ValueError(f"config keys {shadowed} would replace the dataset's data dicts")

    # One manager process serves every split's shared dicts; each proxy keeps it alive.
    manager = Manager() if config_d["num_workers"] > 0 and config_d["share_memory"] else None
    datasets = []
    for split in splits:
        if manager is not None:
            data_sds = {k: manager.dict() for k in data_dict_types}
        else:
            data_sds = {k: {} for k in data_dict_types}
        datasets.append(dataset_cls(split=split, **data_sds, **config_d))
    return tuple(datasets)


def build_train_sampler(
    ds: Dataset, config_d: dict, model_hooks: ModelHooks, generator: th.Generator
) -> Sampler | None:
    """Choose the training index sampler.

    ``validate_sampler_config`` allows at most one group sampler, and no group sampler with the
    bond-size curriculum, so the order below only matters for invalid configs.

    Args:
        ds: Training dataset.
        config_d: Resolved config.
        model_hooks: The model's hooks.
        generator: Generator the samplers share; SpectrumPL reseeds it each epoch.

    Returns:
        The sampler, or None when the model's CE-context batch sampler draws the indices.
    """
    if model_hooks.uses_ce_context_sampler(config_d):
        return None
    if config_d["group_sampler"]:
        return GroupSampler(
            ds,
            sample_k=config_d["group_sampler_max_per_group"],
            generator=generator,
        )
    if config_d["simple_group_sampler"]:
        return get_group_sampler(
            ds,
            config_d["simple_group_sampler_type"],
            config_d["simple_group_sampler_avg_per_group"],
            generator,
        )
    if config_d["formula_prec_type_group_sampler"]:
        return FormulaPrecTypeGroupSampler(
            ds,
            sample_k=config_d["formula_prec_type_group_sampler_max_per_group"],
            key_formula="formula",
            key_adduct="prec_type",
            generator=generator,
        )
    if config_d["bond_size_curriculum"]:
        return BondCountCurriculumSampler(
            ds,
            start_fraction=config_d["bond_size_curriculum_start_fraction"],
            curriculum_epochs=config_d["bond_size_curriculum_epochs"],
            generator=generator,
        )
    return RandomSampler(ds, False, generator=generator)


def init_dataloader(ds: Dataset, config_d: dict) -> DataLoader:
    """
    Initialize a PyTorch DataLoader for a given dataset and configuration.

    This function sets up the appropriate sampler and batching strategy for the dataset,
    depending on the split (train/val/test), group sampling options, and dynamic batch sampler settings.

    Args:
        ds (Dataset): The dataset object (must have a .split attribute).
        config_d (dict): Configuration dictionary with dataloader and sampler options.

    Returns:
        DataLoader: Configured PyTorch DataLoader for the dataset.
    """
    split = ds.split
    # Model-specific batching lives in the model's hooks (fragnnet.hooks.ModelHooks).
    model_hooks = registry.get_model_hooks(config_d["model_type"], strict=False)
    logger.info("init_dataloader for split %s", split)
    dl_param_d = {
        "dataset": ds,
        "num_workers": config_d["num_workers"],
        "collate_fn": ds.get_collate_fn(),
        "pin_memory": config_d["pin_memory"] and (config_d["accelerator"] != "cpu"),
    }
    dl_param_d.update(model_hooks.dataloader_options(config_d))

    # Set up a generator for reproducible sampling (will be overwritten each epoch)
    generator = th.Generator()
    if split == "train":
        sampler = build_train_sampler(ds, config_d, model_hooks, generator)
    else:
        sampler = SequentialSampler(ds)

    batch_sampler = model_hooks.build_batch_sampler(ds, config_d, sampler, generator)
    if batch_sampler is not None:
        dl_param_d["batch_sampler"] = batch_sampler
    else:
        # Standard batching
        dl_param_d["sampler"] = sampler
        if split == "train":
            dl_param_d["batch_size"] = config_d["train_batch_size"]
            dl_param_d["drop_last"] = config_d["drop_last"]
        else:
            dl_param_d["batch_size"] = config_d["eval_batch_size"]
            dl_param_d["drop_last"] = False

    return DataLoader(**dl_param_d)


def validate_sampler_config(config_d: dict) -> None:
    """Validate the sampler settings, then each model's own config keys through its hooks.

    Args:
        config_d: Experiment config dictionary.

    Raises:
        ValueError: If the sampler configuration is invalid.
    """
    # `init_dataloader` takes the first group sampler set, so a second one would be ignored.
    group_samplers = [key for key in GROUP_SAMPLER_KEYS if config_d[key]]
    if len(group_samplers) > 1:
        raise ValueError(f"group samplers are mutually exclusive (got {group_samplers})")

    if (
        config_d["simple_group_sampler"]
        and config_d["simple_group_sampler_type"] not in VALID_SIMPLE_GROUP_SAMPLER_TYPES
    ):
        raise ValueError(
            f"simple_group_sampler_type must be one of {sorted(VALID_SIMPLE_GROUP_SAMPLER_TYPES)} "
            f"(got '{config_d['simple_group_sampler_type']}')"
        )

    # Every model's config carries every key. Another model's settings must stay off, and the
    # model's own hooks check the rest (fragnnet.hooks.ModelHooks).
    model_hooks = registry.get_model_hooks(config_d["model_type"], strict=False)
    cross_ev_weight = config_d["cross_ev_fragment_loss_weight"]
    if (
        not isinstance(cross_ev_weight, (int, float))
        or isinstance(cross_ev_weight, bool)
        or cross_ev_weight < 0
    ):
        raise ValueError("cross_ev_fragment_loss_weight must be a non-negative number")
    for other_hooks in registry.all_model_hooks():
        if type(other_hooks) is not type(model_hooks):
            other_hooks.reject_foreign_config(config_d)
    model_hooks.validate_config(config_d)
    uses_ce_context_sampler = model_hooks.uses_ce_context_sampler(config_d)

    if config_d["batch_sampler_type"] not in VALID_BATCH_SAMPLER_TYPES:
        raise ValueError(
            "batch_sampler_type must be one of "
            f"{sorted(VALID_BATCH_SAMPLER_TYPES)} (got '{config_d['batch_sampler_type']}')"
        )

    if uses_ce_context_sampler:
        if config_d["devices"] != 1:
            raise ValueError("CE-context batch samplers require devices=1")
        spectra_per_context = config_d["ce_context_spectra_per_context"]
        if not isinstance(spectra_per_context, int) or spectra_per_context < 2:
            raise ValueError("ce_context_spectra_per_context must be an integer >= 2")
        ce_key_by_inst = config_d["ce_context_ce_key_by_inst"]
        if not isinstance(ce_key_by_inst, dict) or not ce_key_by_inst:
            raise ValueError("ce_context_ce_key_by_inst must be a non-empty mapping")
        if any(
            not isinstance(inst_type, str) or ce_key not in {"nce", "ace"}
            for inst_type, ce_key in ce_key_by_inst.items()
        ):
            raise ValueError("ce_context_ce_key_by_inst values must be 'nce' or 'ace'")
        missing_inst_types = set(config_d["spec_params"]["inst_types"]).difference(ce_key_by_inst)
        if missing_inst_types:
            raise ValueError(
                "ce_context_ce_key_by_inst is missing configured instrument types "
                f"{sorted(missing_inst_types)}"
            )

    if config_d["bond_size_curriculum"]:
        # A model's own batch sampler ignores the index sampler the curriculum provides.
        if model_hooks.builds_own_batch_sampler(config_d):
            raise ValueError("Bond-size curriculum requires fixed-size batching")
        if group_samplers:
            raise ValueError("Bond-size curriculum does not support group samplers")
        validate_bond_size_curriculum(
            config_d["bond_size_curriculum_start_fraction"],
            config_d["bond_size_curriculum_epochs"],
        )
        if (
            config_d["max_epochs"] > 0
            and config_d["bond_size_curriculum_epochs"] >= config_d["max_epochs"]
        ):
            raise ValueError("bond_size_curriculum_epochs must be less than max_epochs")
