import glob
import logging
import os
import shutil
import tempfile
from multiprocessing import Manager

import lightning as L
import torch as th
import yaml
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler
from pandarallel import pandarallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

import fragnnet.utils.misc_utils as misc_utils
from fragnnet.dataset import (
    BondCountCurriculumSampler,
    CEContextDynamicBatchSampler,
    DualGroupDynamicBatchSampler,
    FormulaPrecTypeGroupSampler,
    GroupSampler,
    MassShatterCEContextDynamicBatchSampler,
    SpecMolFragDynamicBatchSampler,
    get_group_sampler,
)
from fragnnet.dataset.data_sampler import (
    validate_bond_size_curriculum,
    validate_dag_size_curriculum,
)
from fragnnet.utils.misc_utils import deep_update, get_core_count
from fragnnet.utils.nn_utils import nan_backward_hook, nan_forward_hook
from fragnnet.utils.pl_utils import ConsoleLogger, migrate_mass_shatter_proposal_method
from fragnnet.utils.profile_utils import MyPyTorchProfiler
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls

# Every model reads the same `config/template.yml`, so shared code can read any key without
# checking `model_type` first. This set only guards against a typo in `model_type` itself.
VALID_MODEL_TYPES = frozenset(
    {
        "frag_gnn",
        "fragnnet",
        "mass_shatter",
        "neims",
        "massformer",
        "iceberg_gen",
        "iceberg_inten",
        "graff",
        "precursor",
        "gnn",
        "mces_pretraining",
        "clip",
        "spec_mol_clip_pretraining",
    }
)
TEMPLATE_FILENAME = "template.yml"
# What a config that declares no `model_type` means, and what `config/template.yml` declares.
# Only reached by a template that omits the key, e.g. a minimal one saved beside a checkpoint.
DEFAULT_MODEL_TYPE = "frag_gnn"

# Mappings whose sub-keys are data (instrument names, split names), not schema. Custom
# configs may introduce keys inside them, so they are not validated recursively.
OPEN_CONFIG_MAPPINGS = frozenset(
    {
        "spec_params.merge_tolerance_by_inst",
        "spec_params.split_filters",
        "ce_context_ce_key_by_inst",
        "inst_type_loss_tol",
    }
)

# Valid values and model-gated keys checked by `validate_sampler_config`.
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
MASS_SHATTER_SAMPLER_KEYS = (
    "mass_shatter_batch_sampler",
    "mass_shatter_share_proposal_across_ev",
    "mass_shatter_unique_fragment_mode",
)


def resolve_template_path(template_path: str) -> str:
    """Resolve the config template.

    ``template_path`` is normally the config directory holding ``template.yml``. A path to a
    file is used as-is, which is how a template saved inside a checkpoint still resolves.

    Args:
        template_path: Config directory, or a single template file.

    Returns:
        Path to the template to load.

    Raises:
        FileNotFoundError: If no template exists at that path.
    """
    if os.path.isfile(template_path):
        return template_path
    if not os.path.isdir(template_path):
        raise FileNotFoundError(f"no config template at {template_path}")
    resolved_fp = os.path.join(template_path, TEMPLATE_FILENAME)
    if not os.path.isfile(resolved_fp):
        raise FileNotFoundError(f"missing config template: {resolved_fp}")
    return resolved_fp


def find_unknown_config_keys(custom_d: dict, config_d: dict, prefix: str = "") -> list[str]:
    """Collect dotted paths in ``custom_d`` that the resolved config does not declare.

    Recurses into nested sections so a typo inside ``spec_params`` is caught as loudly as one
    at the top level. Sections listed in :data:`OPEN_CONFIG_MAPPINGS` hold data rather than a
    schema and are not descended into.

    Args:
        custom_d: The custom config being validated.
        config_d: The resolved model template.
        prefix: Dotted path of the section currently being checked.

    Returns:
        Sorted dotted paths that are not declared by the resolved config.
    """
    unknown = []
    for key, value in custom_d.items():
        path = f"{prefix}.{key}" if prefix else key
        if key not in config_d:
            unknown.append(path)
            continue
        if (
            path not in OPEN_CONFIG_MAPPINGS
            and isinstance(value, dict)
            and isinstance(config_d[key], dict)
        ):
            unknown.extend(find_unknown_config_keys(value, config_d[key], path))
    return sorted(unknown)


def load_config(
    template_path: str,
    custom_fp: str | None,
    override: list | None,
    model_type: str | None = None,
) -> dict:
    """Resolve the config template, the experiment config, and CLI overrides.

    Every model reads the same ``config/template.yml``. Passing a single template file instead
    of the directory uses that file directly, which is how a template saved inside a checkpoint
    resolves.

    ``model_type`` comes from the custom config, else from the ``model_type`` argument (the type
    recorded in a checkpoint), else from the template.

    Args:
        template_path: Config directory, or a single template file.
        custom_fp: Path to the experiment config YAML, or None.
        override: ``key=value`` strings; dotted keys address nested sections.
        model_type: Model type to fall back on when the custom config does not declare one,
            e.g. the type recorded in a checkpoint.

    Returns:
        The resolved configuration dictionary.

    Raises:
        ValueError: If ``model_type`` is unknown, or the custom config sets undeclared keys.
    """
    custom_d = None
    if custom_fp is not None:
        assert os.path.isfile(custom_fp), custom_fp
        with open(custom_fp) as custom_file:
            custom_d = yaml.load(custom_file, Loader=yaml.FullLoader) or {}

    # 1. Load the template every model shares.
    resolved_template_fp = resolve_template_path(template_path)
    with open(resolved_template_fp) as template_file:
        config_d = yaml.load(template_file, Loader=yaml.FullLoader)
    # A caller-supplied model_type (a checkpoint's) outranks the template default, so an
    # inference config that stays silent about the model does not silently become frag_gnn.
    resolved_model_type = (
        (custom_d or {}).get("model_type")
        or model_type
        or config_d.get("model_type")
        or DEFAULT_MODEL_TYPE
    )
    if resolved_model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"unknown model_type {resolved_model_type!r}; known types: {sorted(VALID_MODEL_TYPES)}"
        )
    config_d["model_type"] = resolved_model_type

    migrate_mass_shatter_proposal_method(config_d)
    # Compatibility boundary for templates saved before candidate backends existed.
    # Add the schema key before custom config validation and CLI overrides.
    if resolved_model_type == "mass_shatter":
        if "mass_shatter_candidate_backend" not in config_d:
            config_d["mass_shatter_candidate_backend"] = "cpu"
        if "mass_shatter_proposal_method" not in config_d:
            config_d["mass_shatter_proposal_method"] = "cut_masks"
        if "mass_shatter_fast_gnn_beam_width" not in config_d:
            config_d["mass_shatter_fast_gnn_beam_width"] = 256
        if "mass_shatter_fast_gnn_max_boundary_cuts" not in config_d:
            config_d["mass_shatter_fast_gnn_max_boundary_cuts"] = 3

    # 2. Validate and merge the custom config. The template declares every model's keys, so a
    # key it does not declare is a typo rather than another model's setting. Validation recurses
    # into nested sections, so a typo inside `spec_params` fails as loudly as one at the top.
    if custom_d is not None:
        migrate_mass_shatter_proposal_method(custom_d)
        unknown = find_unknown_config_keys(custom_d, config_d)
        if unknown:
            raise ValueError(f"{custom_fp} sets keys the template does not declare: {unknown}")
        config_d = deep_update(config_d, custom_d)

    # override
    if override is not None:
        for override_item in override:
            dotted_key, value = override_item.split("=", 1)
            if dotted_key in ("mass_shatter_proposal_type", "mass_shatter_proposal_method"):
                proposal_override = {dotted_key: yaml.safe_load(value)}
                migrate_mass_shatter_proposal_method(proposal_override)
                config_d.update(proposal_override)
                continue
            key_path = dotted_key.split(".")
            section = config_d
            for section_key in key_path[:-1]:
                if section_key in section:
                    section = section[section_key]
                else:
                    msg = f"{section_key} not in config"
                    raise KeyError(msg)

            print(f"Updating {key_path[-1]} from {section[key_path[-1]]} to {value}", flush=True)
            # here is not safe way to load, but I'll skip this issue now
            section[key_path[-1]] = yaml.safe_load(value)

    return config_d


def init_dataset(config_d: dict, splits=("train", "val")) -> tuple:
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
        Exception: Propagates errors raised by `get_dataset_cls` or the dataset constructor.
    """
    # Select dataset class based on model type
    dataset_cls = get_dataset_cls(config_d["model_type"])

    # Get the expected data dict types for this dataset
    data_dict_types = dataset_cls.get_data_dict_types()

    datasets = []
    for split in splits:
        # Use multiprocessing manager dicts if requested, else regular dicts
        if config_d["num_workers"] > 0 and config_d["share_memory"]:
            manager = Manager()
            data_sds = {k: manager.dict() for k in data_dict_types}
        else:
            data_sds = {k: {} for k in data_dict_types}
        # Instantiate the dataset for this split
        ds = dataset_cls(split=split, **{**data_sds, **config_d})
        datasets.append(ds)

    return tuple(datasets)


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
    # `mass_shatter_batch_sampler` is MASShatter's; gating on the model type keeps a historical
    # config that still carries the key from selecting a sampler the model cannot use.
    use_mass_shatter_batch_sampler = (
        config_d["model_type"] == "mass_shatter" and config_d["mass_shatter_batch_sampler"]
    )
    assert not (config_d["group_sampler"] and config_d["simple_group_sampler"]), (
        "Cannot use both group_sampler and simple_group_sampler"
    )
    print(f"> init_dataloader for split {split}")
    dl_param_d = {
        "dataset": ds,
        "num_workers": config_d["num_workers"],
        "collate_fn": ds.get_collate_fn(),
        "pin_memory": config_d["pin_memory"] and (config_d["accelerator"] != "cpu"),
    }
    if (
        config_d["num_workers"] > 0
        and config_d["model_type"] == "mass_shatter"
        and config_d["automatic_optimization"]
    ):
        # Keep workers and their preprocessing caches alive; matches the exemption from
        # reload_dataloaders_every_n_epochs, without which workers respawn every epoch anyway.
        dl_param_d["persistent_workers"] = True

    # Set up a generator for reproducible sampling (will be overwritten each epoch)
    generator = th.Generator()

    # Compute maximum effective batch size (accounts for gradient accumulation)
    # used by dynamic batch samplers to decide return thresholds.

    if split == "train":
        # Choose sampler for training
        if use_mass_shatter_batch_sampler or config_d["batch_sampler_type"] == "ce_context":
            sampler = None
        elif config_d["group_sampler"]:
            sampler = GroupSampler(
                ds,
                sample_k=config_d["group_sampler_max_per_group"],
                generator=generator,
            )
        elif config_d["simple_group_sampler"]:
            sampler = get_group_sampler(
                ds,
                config_d["simple_group_sampler_type"],
                config_d["simple_group_sampler_avg_per_group"],
                generator,
            )
        elif config_d["formula_prec_type_group_sampler"]:
            sampler = FormulaPrecTypeGroupSampler(
                ds,
                sample_k=config_d["formula_prec_type_group_sampler_max_per_group"],
                key_formula="formula",
                key_adduct="prec_type",
                generator=generator,
            )
        elif config_d["bond_size_curriculum"]:
            sampler = BondCountCurriculumSampler(
                ds,
                start_fraction=config_d["bond_size_curriculum_start_fraction"],
                curriculum_epochs=config_d["bond_size_curriculum_epochs"],
                generator=generator,
            )
        else:
            sampler = RandomSampler(ds, False, generator=generator)
    else:
        # Use sequential sampler for validation/test/predict_only
        sampler = SequentialSampler(ds)

    # Dynamic batch sampler (for variable batch sizes, e.g. in frag_gnn)
    # Prefer formula-based hard negative dynamic sampler when enabled (train only)

    if use_mass_shatter_batch_sampler:
        return_batch_at = (
            config_d["train_batch_size"] if split == "train" else config_d["eval_batch_size"]
        )
        batch_sampler = MassShatterCEContextDynamicBatchSampler(
            ds,
            max_num=config_d["mass_shatter_batch_sampler_max"],
            ce_key_by_inst=config_d["ce_context_ce_key_by_inst"],
            spectra_per_context=config_d["ce_context_spectra_per_context"],
            return_batch_at=return_batch_at,
            seed=config_d["seed"],
            skip_too_big=split == "train",
        )
        dl_param_d["batch_sampler"] = batch_sampler
    elif config_d["dynamic_batch_sampler"]:
        if split == "train":
            return_batch_at = config_d["train_batch_size"]
            if not config_d["automatic_optimization"]:
                return_batch_at *= config_d["accumulate_grad_batches"]
        else:
            return_batch_at = config_d["eval_batch_size"]
        if config_d["batch_sampler_type"] == "ce_context":
            batch_sampler = CEContextDynamicBatchSampler(
                ds,
                max_num=config_d["dynamic_batch_sampler_max"],
                ce_key_by_inst=config_d["ce_context_ce_key_by_inst"],
                limited_by=config_d["dynamic_batch_sampler_mode"],
                spectra_per_context=config_d["ce_context_spectra_per_context"],
                return_batch_at=return_batch_at,
                seed=config_d["seed"],
                skip_too_big=split == "train",
            )
        elif split == "train":
            batch_sampler = SpecMolFragDynamicBatchSampler(
                ds,
                max_num=config_d["dynamic_batch_sampler_max"],
                limited_by=config_d["dynamic_batch_sampler_mode"],
                skip_too_big=True,
                return_batch_at=return_batch_at,
                sampler=sampler,
                dag_size_curriculum=config_d["dag_size_curriculum"],
                dag_size_curriculum_start_fraction=config_d["dag_size_curriculum_start_fraction"],
                dag_size_curriculum_epochs=config_d["dag_size_curriculum_epochs"],
            )
        else:
            batch_sampler = SpecMolFragDynamicBatchSampler(
                ds,
                max_num=config_d["dynamic_batch_sampler_max"],
                limited_by=config_d["dynamic_batch_sampler_mode"],
                skip_too_big=False,
                return_batch_at=return_batch_at,
                sampler=sampler,
            )
        dl_param_d["batch_sampler"] = batch_sampler
    elif config_d["dual_group_dynamic_batch_sampler"]:
        if split == "train":
            return_batch_at = config_d["train_batch_size"] * config_d["accumulate_grad_batches"]
            batch_sampler = DualGroupDynamicBatchSampler(
                ds,
                max_num=config_d["dual_group_dynamic_batch_sampler_max"],
                limited_by=config_d["dual_group_dynamic_batch_sampler_mode"],
                skip_too_big=True,
                return_batch_at=return_batch_at,
                sample_k1=config_d["dual_group_sampler_k1"],
                sample_k2=config_d["dual_group_sampler_k2"],
                sample_k1_per_key=config_d["dual_group_sampler_k1_per_key"],
                sample_k2_per_key=config_d["dual_group_sampler_k2_per_key"],
                generator=generator,
            )
        else:
            return_batch_at = config_d["eval_batch_size"]
            batch_sampler = SpecMolFragDynamicBatchSampler(
                ds,
                max_num=config_d["dual_group_dynamic_batch_sampler_max"],
                limited_by=config_d["dual_group_dynamic_batch_sampler_mode"],
                skip_too_big=False,
                return_batch_at=return_batch_at,
                sampler=sampler,
            )
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

    dl = DataLoader(**dl_param_d)
    return dl


def validate_sampler_config(config_d: dict) -> None:
    """Validate dynamic batch sampler config entries.

    Args:
        config_d: Experiment config dictionary.

    Raises:
        ValueError: If the sampler configuration is invalid.
    """

    if (
        config_d.get("simple_group_sampler")
        and config_d.get("simple_group_sampler_type") not in VALID_SIMPLE_GROUP_SAMPLER_TYPES
    ):
        raise ValueError(
            f"simple_group_sampler_type must be one of {sorted(VALID_SIMPLE_GROUP_SAMPLER_TYPES)} "
            f"(got '{config_d['simple_group_sampler_type']}')"
        )

    # The `mass_shatter_*` sampler keys are MASShatter's. Every model's config carries them, so
    # the reads are hard; the model type decides whether they mean anything. For another model a
    # historical config may still set one - inert, but loud if it is actually switched on.
    is_mass_shatter = config_d["model_type"] == "mass_shatter"
    if not is_mass_shatter:
        enabled_elsewhere = [key for key in MASS_SHATTER_SAMPLER_KEYS if config_d.get(key)]
        if enabled_elsewhere:
            raise ValueError(
                f"{enabled_elsewhere} require model_type='mass_shatter' "
                f"(got '{config_d['model_type']}')"
            )
    use_mass_shatter_batch_sampler = is_mass_shatter and config_d["mass_shatter_batch_sampler"]
    share_proposal_across_ev = is_mass_shatter and config_d["mass_shatter_share_proposal_across_ev"]
    use_unique_fragments = is_mass_shatter and config_d["mass_shatter_unique_fragment_mode"]

    if not isinstance(use_mass_shatter_batch_sampler, bool):
        raise ValueError("mass_shatter_batch_sampler must be boolean")
    if not isinstance(share_proposal_across_ev, bool):
        raise ValueError("mass_shatter_share_proposal_across_ev must be boolean")
    num_enabled_batch_samplers = int(use_mass_shatter_batch_sampler) + sum(
        int(config_d[key])
        for key in (
            "dynamic_batch_sampler",
            "dual_group_dynamic_batch_sampler",
        )
    )
    if num_enabled_batch_samplers > 1:
        raise ValueError(
            "mass_shatter_batch_sampler, dynamic_batch_sampler, and "
            "dual_group_dynamic_batch_sampler are mutually exclusive"
        )

    if use_mass_shatter_batch_sampler:
        if not config_d["automatic_optimization"]:
            raise ValueError("mass_shatter_batch_sampler requires automatic_optimization=True")
        mass_shatter_batch_sampler_max = config_d["mass_shatter_batch_sampler_max"]
        if (
            isinstance(mass_shatter_batch_sampler_max, bool)
            or not isinstance(mass_shatter_batch_sampler_max, int)
            or mass_shatter_batch_sampler_max <= 0
        ):
            raise ValueError("mass_shatter_batch_sampler_max must be a positive integer")

    if config_d["dynamic_batch_sampler"]:
        if config_d["model_type"] not in ("frag_gnn", "fragnnet"):
            raise ValueError("Dynamic batch sampler can only be used with frag_gnn model")
        if config_d["automatic_optimization"]:
            raise ValueError(
                f"Dynamic batch sampler cannot be used with automatic optimization "
                f"(got automatic_optimization={config_d['automatic_optimization']})"
            )
        if config_d["dynamic_batch_sampler_mode"] not in ["frag_node", "frag_edge"]:
            raise ValueError(
                "Dynamic batch sampler only supports frag_node or frag_edge mode "
                f"(got '{config_d['dynamic_batch_sampler_mode']}')"
            )
        if config_d["dynamic_batch_sampler_max"] is None:
            raise ValueError("Dynamic batch sampler requires dynamic_batch_sampler_max to be set")

    if config_d["batch_sampler_type"] not in VALID_BATCH_SAMPLER_TYPES:
        raise ValueError(
            "batch_sampler_type must be one of "
            f"{sorted(VALID_BATCH_SAMPLER_TYPES)} (got '{config_d['batch_sampler_type']}')"
        )

    if config_d["batch_sampler_type"] == "ce_context":
        if not config_d["dynamic_batch_sampler"]:
            raise ValueError("batch_sampler_type='ce_context' requires dynamic_batch_sampler=True")
        if config_d["dynamic_batch_sampler_mode"] != "frag_edge":
            raise ValueError(
                "batch_sampler_type='ce_context' requires "
                "dynamic_batch_sampler_mode='frag_edge' for fragment-DAG models"
            )
        if config_d["dag_size_curriculum"]:
            raise ValueError("batch_sampler_type='ce_context' does not support DAG-size curriculum")

    if config_d["batch_sampler_type"] == "ce_context" or use_mass_shatter_batch_sampler:
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

    if share_proposal_across_ev and not use_mass_shatter_batch_sampler:
        raise ValueError(
            "mass_shatter_share_proposal_across_ev requires mass_shatter_batch_sampler=True"
        )

    cross_ev_weight = config_d["cross_ev_fragment_loss_weight"]
    if (
        not isinstance(cross_ev_weight, (int, float))
        or isinstance(cross_ev_weight, bool)
        or cross_ev_weight < 0
    ):
        raise ValueError("cross_ev_fragment_loss_weight must be a non-negative number")
    if cross_ev_weight > 0:
        if not share_proposal_across_ev:
            raise ValueError(
                "cross_ev_fragment_loss_weight > 0 requires "
                "mass_shatter_share_proposal_across_ev=True"
            )
        if not use_unique_fragments:
            raise ValueError(
                "cross_ev_fragment_loss_weight > 0 requires mass_shatter_unique_fragment_mode=True"
            )

    pair_weight = config_d["ce_pair_loss_weight"]
    if (
        not isinstance(pair_weight, (int, float))
        or isinstance(pair_weight, bool)
        or pair_weight < 0
    ):
        raise ValueError("ce_pair_loss_weight must be a non-negative number")
    if pair_weight > 0:
        if not (config_d["batch_sampler_type"] == "ce_context" or use_mass_shatter_batch_sampler):
            raise ValueError("ce_pair_loss_weight > 0 requires a CE-context batch sampler")
        confidence = config_d["ce_pair_min_endpoint_confidence"]
        if not isinstance(confidence, (int, float)) or isinstance(confidence, bool):
            raise ValueError(
                "ce_pair_min_endpoint_confidence must be set when ce_pair_loss_weight > 0"
            )
        if not 0.0 < confidence <= 1.0:
            raise ValueError("ce_pair_min_endpoint_confidence must be in (0, 1]")
        contender_floor = config_d["target_rel_ints_thresh"]
        if (
            not isinstance(contender_floor, (int, float))
            or isinstance(contender_floor, bool)
            or not 0.0 < contender_floor < 1.0
        ):
            raise ValueError(
                "ce_pair_loss_weight > 0 requires target_rel_ints_thresh in (0, 1) "
                "as the temporary observed-contender floor"
            )
        if config_d["ce_pair_min_endpoint_reliability"] is not None:
            raise ValueError(
                "ce_pair_min_endpoint_reliability is reserved until the replicate "
                "reliability table is implemented"
            )
        if config_d["ce_pair_subset"] != "switches":
            raise ValueError("CE pair loss v1 supports ce_pair_subset='switches' only")
        if config_d["train_sample_weight"] != "none":
            raise ValueError("CE pair loss v1 requires train_sample_weight='none'")
        inst_types = set(config_d["spec_params"]["inst_types"])
        prec_types = set(config_d["spec_params"]["prec_types"])
        if not config_d["ce_pair_eligible_inst_types"] or not set(
            config_d["ce_pair_eligible_inst_types"]
        ).issubset(inst_types):
            raise ValueError("ce_pair_eligible_inst_types must be configured instrument types")
        if not config_d["ce_pair_eligible_prec_types"] or not set(
            config_d["ce_pair_eligible_prec_types"]
        ).issubset(prec_types):
            raise ValueError("ce_pair_eligible_prec_types must be configured precursor types")
        tolerance_by_inst = config_d["inst_type_loss_tol"]
        eligible_inst_types = set(config_d["ce_pair_eligible_inst_types"])
        if not isinstance(tolerance_by_inst, dict) or not eligible_inst_types.issubset(
            tolerance_by_inst
        ):
            raise ValueError("inst_type_loss_tol must define every ce_pair_eligible_inst_type")
        for inst_type in eligible_inst_types:
            tolerance = tolerance_by_inst[inst_type]
            if not isinstance(tolerance, dict) or set(tolerance) != {"rel", "min_mz"}:
                raise ValueError(
                    "Each eligible inst_type_loss_tol entry must define rel and min_mz"
                )
            if (
                not isinstance(tolerance["rel"], (int, float))
                or isinstance(tolerance["rel"], bool)
                or tolerance["rel"] <= 0
                or not isinstance(tolerance["min_mz"], (int, float))
                or isinstance(tolerance["min_mz"], bool)
                or tolerance["min_mz"] <= 0
            ):
                raise ValueError(
                    "Eligible inst_type_loss_tol rel and min_mz must be positive numbers"
                )

    precursor_pair_weight = config_d["precursor_pair_loss_weight"]
    if (
        not isinstance(precursor_pair_weight, (int, float))
        or isinstance(precursor_pair_weight, bool)
        or precursor_pair_weight < 0
    ):
        raise ValueError("precursor_pair_loss_weight must be a non-negative number")
    if precursor_pair_weight > 0:
        spec_params = config_d["spec_params"]
        if not spec_params["nce"] and not spec_params["ace"]:
            raise ValueError(
                "precursor_pair_loss_weight > 0 requires spec_params.nce or "
                "spec_params.ace so NCE metadata is loaded"
            )
        if spec_params["merge"]:
            raise ValueError("precursor_pair_loss_weight > 0 requires unmerged spectra")
        nce_min = config_d["precursor_pair_nce_min"]
        nce_max = config_d["precursor_pair_nce_max"]
        if (
            not isinstance(nce_min, (int, float))
            or isinstance(nce_min, bool)
            or not isinstance(nce_max, (int, float))
            or isinstance(nce_max, bool)
            or nce_min < 0
            or nce_max <= nce_min
        ):
            raise ValueError("precursor_pair_nce_min/max must define a non-negative interval")
        absent_floor = config_d["precursor_pair_absent_intensity_floor"]
        if (
            not isinstance(absent_floor, (int, float))
            or isinstance(absent_floor, bool)
            or not 0.0 <= absent_floor < 1.0
        ):
            raise ValueError("precursor_pair_absent_intensity_floor must be in [0, 1)")
        inst_types = set(config_d["spec_params"]["inst_types"])
        prec_types = set(config_d["spec_params"]["prec_types"])
        eligible_inst_types = set(config_d["precursor_pair_eligible_inst_types"])
        eligible_prec_types = set(config_d["precursor_pair_eligible_prec_types"])
        if not eligible_inst_types or not eligible_inst_types.issubset(inst_types):
            raise ValueError(
                "precursor_pair_eligible_inst_types must be configured instrument types"
            )
        if not eligible_prec_types or not eligible_prec_types.issubset(prec_types):
            raise ValueError(
                "precursor_pair_eligible_prec_types must be configured precursor types"
            )
        tolerance_by_inst = config_d["inst_type_loss_tol"]
        if not isinstance(tolerance_by_inst, dict) or not eligible_inst_types.issubset(
            tolerance_by_inst
        ):
            raise ValueError(
                "inst_type_loss_tol must define every precursor-pair eligible instrument"
            )
        for inst_type in eligible_inst_types:
            tolerance = tolerance_by_inst[inst_type]
            if not isinstance(tolerance, dict) or set(tolerance) != {"rel", "min_mz"}:
                raise ValueError(
                    "Each precursor-pair eligible tolerance must define rel and min_mz"
                )
            if (
                not isinstance(tolerance["rel"], (int, float))
                or isinstance(tolerance["rel"], bool)
                or tolerance["rel"] <= 0
                or not isinstance(tolerance["min_mz"], (int, float))
                or isinstance(tolerance["min_mz"], bool)
                or tolerance["min_mz"] <= 0
            ):
                raise ValueError(
                    "Precursor-pair eligible tolerance rel and min_mz must be positive numbers"
                )

    if config_d["dag_size_curriculum"]:
        if not config_d["dynamic_batch_sampler"]:
            raise ValueError("DAG-size curriculum requires dynamic_batch_sampler=True")
        validate_dag_size_curriculum(
            config_d["dag_size_curriculum_start_fraction"],
            config_d["dag_size_curriculum_epochs"],
        )
        if (
            config_d["max_epochs"] > 0
            and config_d["dag_size_curriculum_epochs"] >= config_d["max_epochs"]
        ):
            raise ValueError("dag_size_curriculum_epochs must be less than max_epochs")

    if config_d["bond_size_curriculum"]:
        if config_d["dynamic_batch_sampler"] or config_d["dual_group_dynamic_batch_sampler"]:
            raise ValueError("Bond-size curriculum requires fixed-size batching")
        if (
            config_d["group_sampler"]
            or config_d["simple_group_sampler"]
            or config_d["formula_prec_type_group_sampler"]
        ):
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

    if config_d["dual_group_dynamic_batch_sampler"]:
        if config_d["model_type"] not in ("frag_gnn", "fragnnet"):
            raise ValueError(
                "Dual-group dynamic batch sampler can only be used with frag_gnn model"
            )
        if config_d["automatic_optimization"]:
            raise ValueError(
                f"Dual-group dynamic batch sampler cannot be used with automatic optimization "
                f"(got automatic_optimization={config_d['automatic_optimization']})"
            )
        dual_group_sampler_mode = config_d.get("dual_group_dynamic_batch_sampler_mode")
        if dual_group_sampler_mode not in ["frag_node", "frag_edge"]:
            raise ValueError(
                f"Dual-group dynamic batch sampler only supports frag_node or frag_edge mode "
                f"(got '{dual_group_sampler_mode}')"
            )
        dual_group_sampler_max = config_d.get("dual_group_dynamic_batch_sampler_max")
        if dual_group_sampler_max is None:
            raise ValueError(
                "Dual-group dynamic batch sampler requires dual_group_dynamic_batch_sampler_max to be set"
            )


def log_cuda_info() -> None:
    """Log PyTorch version and CUDA device information."""
    logger = logging.getLogger(__name__)
    logger.info("PyTorch version: %s", th.__version__)
    logger.info("CUDA_VISIBLE_DEVICES: %s", os.environ.get("CUDA_VISIBLE_DEVICES", "(not set)"))
    if th.cuda.is_available():
        logger.info("CUDA version: %s", th.version.cuda)
        logger.info("device_count: %d", th.cuda.device_count())
        for i in range(th.cuda.device_count()):
            props = th.cuda.get_device_properties(i)
            total_gb = props.total_memory / 1024**3
            logger.info(
                "GPU %d: %s | %.1f GB | compute capability %d.%d",
                i,
                props.name,
                total_gb,
                props.major,
                props.minor,
            )
    else:
        logger.info("CUDA not available, running on CPU")


def run_training(
    template_path: str, custom_fp: str, wandb_mode: str, job_id: str, override: list | None
) -> L.LightningModule:
    """
    Main entry point for running a training/validation/testing job.
    Args:
        template_path (str): Template directory, or a single template file.
        custom_fp (str | None): Path to the custom YAML config file (can be None).
        wandb_mode (str): Wandb mode ["online", "offline", "disabled"].
        job_id (str): Unique job identifier for checkpointing and wandb.
        override (list | None): List of key-value pairs to override in the config.
    Returns:
                    model: The trained model (LightningModule).
    """

    # Must be set before the CUDA allocator initialises (first tensor.cuda() call).
    # setdefault lets a SLURM-level export override this if needed.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    # setup logger
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )

    log_cuda_info()

    # load config
    config_d = load_config(template_path, custom_fp, override)

    # set random seeds
    seed_everything(config_d["seed"], workers=True)

    # set torch multiprocessing strategy
    th.multiprocessing.set_sharing_strategy(config_d["mp_sharing_strategy"])

    # setup logging and wandb
    logging.info("setup loggers")
    loggers = []
    console_logger = ConsoleLogger()
    loggers.append(console_logger)

    if wandb_mode != "disabled":
        import wandb

        wandb_init_d = {
            "project": config_d["wandb_project"],
            "name": config_d["wandb_name"],
            "group": config_d["wandb_group"],
            "mode": wandb_mode,
            "entity": config_d["wandb_entity"],
            "tags": config_d["wandb_tags"],
            "resume": "allow",
        }

        if job_id and not config_d["disable_checkpoints"]:
            job_id_fp = os.path.join("job_id", f"{job_id}.id")
            if os.path.isfile(job_id_fp):
                is_resume = True
                with open(job_id_fp) as job_id_file:
                    job_id_fields = job_id_file.read().strip().split(";", 1)
                run_id = job_id_fields[0]
                old_wandb_dp = job_id_fields[1] if len(job_id_fields) > 1 else None
            else:
                is_resume = False
                run_id = None
                old_wandb_dp = None
            wandb_init_d["id"] = run_id
            wandb_init_d["resume"] = "must" if run_id is not None else "allow"
            wandb_init_d["config"] = config_d
        else:
            is_resume = False
            job_id_fp = None
            old_wandb_dp = None
            wandb_init_d["config"] = config_d
        wandb.init(**wandb_init_d)
        assert wandb.run is not None, wandb.run
        wandb_logger_d = {**wandb_init_d, "offline": wandb_mode == "offline"}
        wandb_logger = WandbLogger(**wandb_logger_d)
        if job_id_fp:
            os.makedirs(os.path.dirname(job_id_fp), exist_ok=True)
            with open(job_id_fp, "w+") as job_id_file:
                job_id_file.write(f"{wandb.run.id};{os.path.realpath(wandb.run.dir)}")
        # update config results (primarily for wandb)
        # configs can be nested up to 2 levels
        saved_run_config = dict(wandb.config)
        migrate_mass_shatter_proposal_method(saved_run_config)
        for key in list(config_d.keys()):
            if key in saved_run_config:
                if isinstance(config_d[key], dict):
                    # nested
                    assert isinstance(saved_run_config[key], dict)
                    for sub_key in list(config_d[key].keys()):
                        if sub_key in saved_run_config[key]:
                            if config_d[key][sub_key] != saved_run_config[key][sub_key]:
                                print(
                                    f"> config diff -- {key}->{sub_key}: "
                                    f"{config_d[key][sub_key]} vs {saved_run_config[key][sub_key]}"
                                )
                            config_d[key][sub_key] = saved_run_config[key][sub_key]
                else:
                    if config_d[key] != saved_run_config[key]:
                        print(f"> config diff -- {key}: {config_d[key]} vs {saved_run_config[key]}")
                    config_d[key] = saved_run_config[key]
        loggers.append(wandb_logger)
    else:
        is_resume = False
        job_id_fp = None

    # CUDA and TF32 setup
    if config_d["use_tensor_float32"] and config_d["accelerator"] == "gpu":
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
        th.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        th.backends.cudnn.allow_tf32 = True

    # LOG_ZERO setup
    if config_d["log_zero_fp32"] is not None:
        misc_utils.LOG_ZERO_FP32 = float(config_d["log_zero_fp32"])
    if config_d["log_zero_fp16"] is not None:
        misc_utils.LOG_ZERO_FP16 = float(config_d["log_zero_fp16"])

    # setup pandarallel
    pandarallel.initialize(
        progress_bar=config_d["pandarallel_enable_progress_bar"],
        nb_workers=get_core_count(),
    )

    # setup model
    logging.info("setup model")

    model_cls = get_model_cls(config_d["model_type"])
    model = model_cls(**config_d)
    model.train()
    if config_d["model_type"] == "mass_shatter":
        resolved_proposal = config_d["mass_shatter_proposal_method"]
        resolved_backend = (
            "torch_device"
            if resolved_proposal == "connected_fragment_beam"
            else model.model.candidate_backend
        )
        logging.info("Resolved MASShatter candidate backend: %s", resolved_backend)
        if wandb_mode != "disabled":
            wandb.run.summary["mass_shatter_candidate_backend_resolved"] = resolved_backend
            wandb.run.summary["mass_shatter_proposal_method_resolved"] = resolved_proposal

    # setup callbacks
    callbacks = []
    # Names the on-disk checkpoint and profile directories when wandb is disabled.
    run_name = job_id if job_id else config_d.get("wandb_name", "run")
    if wandb_mode != "disabled":
        ckpt_dp = os.path.realpath(os.path.join(wandb.run.dir, "ckpt"))
    else:
        logging.info(
            "wandb is disabled; checkpoints will be saved to "
            f"ckpts/{run_name} — concurrent runs with the same name will collide."
            "use unique job_id or enable wandb to avoid this issue."
        )
        ckpt_dp = os.path.join("ckpts", run_name)
    os.makedirs(ckpt_dp, exist_ok=True)
    if is_resume and old_wandb_dp is not None:
        assert not config_d["disable_checkpoints"]
        # copy the checkpoint files
        old_ckpt_fps = glob.glob(os.path.join(old_wandb_dp, "ckpt", "*.ckpt"))
        if not old_ckpt_fps:
            logging.warning(
                f"Resume requested but no checkpoints found in {old_wandb_dp}/ckpt/, starting from scratch"
            )
        for old_ckpt_fp in old_ckpt_fps:
            new_ckpt_fp = os.path.join(ckpt_dp, os.path.basename(old_ckpt_fp))
            # modify checkpoint metadata (hacky)
            ckpt_data = th.load(old_ckpt_fp, weights_only=False)
            ckpt_callback_data = None
            for callback_key, callback_state in ckpt_data["callbacks"].items():
                if callback_key.startswith("ModelCheckpoint"):
                    ckpt_callback_data = callback_state
                    break
            assert ckpt_callback_data is not None
            ckpt_path_keys = [
                "best_model_path",
                "last_model_path",
                "dirpath",
                "kth_best_model_path",
            ]
            for path_key in ckpt_path_keys:
                if path_key in ckpt_callback_data:
                    ckpt_callback_data[path_key] = ckpt_callback_data[path_key].replace(
                        os.path.realpath(old_wandb_dp), os.path.realpath(wandb.run.dir)
                    )
            if "best_k_models" in ckpt_callback_data:
                for old_path in list(ckpt_callback_data["best_k_models"].keys()):
                    new_path = old_path.replace(
                        os.path.realpath(old_wandb_dp), os.path.realpath(wandb.run.dir)
                    )
                    ckpt_callback_data["best_k_models"][new_path] = ckpt_callback_data[
                        "best_k_models"
                    ].pop(old_path)
            # save modified checkpoint in new wandb dir
            th.save(ckpt_data, new_ckpt_fp)
    if not config_d["disable_checkpoints"]:
        checkpoint_callback = ModelCheckpoint(
            dirpath=ckpt_dp,
            filename="model-{epoch:03d}",
            monitor=config_d["checkpoint_metric"],
            mode=config_d["checkpoint_metric_mode"],
            save_last=config_d["checkpoint_save_last"],
            every_n_epochs=config_d["checkpoint_every_n_epochs"],
            save_top_k=config_d["checkpoint_save_top_k"],
        )
        callbacks.append(checkpoint_callback)

    # setup profiler
    logging.info("setup profiler")
    if wandb_mode != "disabled":
        profile_dp = os.path.join(wandb.run.dir, "profile")
    else:
        profile_dp = os.path.join("tmp_profile", run_name)
    os.makedirs(profile_dp, exist_ok=True)
    if config_d["profiler"] == "simple":
        profiler = SimpleProfiler(dirpath=profile_dp, filename="profile")
    elif config_d["profiler"] == "advanced":
        profiler = AdvancedProfiler(dirpath=profile_dp, filename="profile")
    elif config_d["profiler"] == "pytorch":
        th.profiler._utils._init_for_cuda_graphs()
        profiler_wait = int(config_d["profiler_wait_steps"])
        profiler_warmup = int(config_d["profiler_warmup_steps"])
        profiler_active = int(config_d["profiler_active_steps"])
        profiler_repeat = int(config_d["profiler_repeat"])
        profiler_skip_first = int(config_d["profiler_skip_first_steps"])
        profiler = MyPyTorchProfiler(
            dirpath=profile_dp,
            filename="profile",
            activities=[
                th.profiler.ProfilerActivity.CPU,
                th.profiler.ProfilerActivity.CUDA,
            ],
            # on_trace_ready=th.profiler.tensorboard_trace_handler(profile_dp),
            profile_memory=True,
            record_shapes=False,  # True,
            with_flops=False,  # True,
            with_stack=True,
            with_modules=False,
            export_to_chrome=False,
            export_to_flame_graph=True,
            experimental_config=th._C._profiler._ExperimentalConfig(verbose=True),
            schedule=th.profiler.schedule(
                wait=profiler_wait,
                warmup=profiler_warmup,
                active=profiler_active,
                repeat=profiler_repeat,
                skip_first=profiler_skip_first,
            ),
        )
    else:
        assert config_d["profiler"] == "none", config_d["profiler"]
        # no profiler
        profiler = None

    # setup datasets — test is intentionally excluded here and loaded after
    # training to avoid holding the test set in memory throughout the run.
    logging.info("setup dataset")
    datasets = init_dataset(config_d, splits=["train", "val"])
    train_ds = datasets[0]
    val_ds = datasets[1]

    # check config for sampler
    validate_sampler_config(config_d)

    # check config for CMF; only the FraGNNet template declares frag_params.
    if (
        "frag_params" in config_d
        and config_d["frag_params"]["include_cmf"]
        and "cmf_h_formulae_idx" not in config_d["frag_params"]["pyg_node_feats"]
    ):
        raise ValueError(
            "include_cmf is True but 'cmf_h_formulae_idx' is missing from pyg_node_feats"
        )

    # setup dataloaders
    logging.info("setup dataloader")
    train_dl = init_dataloader(train_ds, config_d)
    val_dl = init_dataloader(val_ds, config_d)

    # setup trainer
    logging.info("setup trainer")
    if config_d["debug_overfit"]:
        overfit_batches = config_d["debug_overfit_batches"]
    else:
        overfit_batches = 0
    log_every_n_steps = min(len(train_dl), config_d["log_every_n_steps"])

    # check precision support
    if str(config_d["precision"]) in ["bf16-mixed", "bf16"]:
        if not (th.cuda.is_available() and th.cuda.is_bf16_supported()):
            logging.warning("BF16 not supported on this device, falling back to 32 bit precision")
            config_d["precision"] = 32
    elif str(config_d["precision"]) in ["16-mixed", "16"] and not th.cuda.is_available():
        logging.warning("FP16 mixed precision requires CUDA, falling back to 32 bit precision")
        config_d["precision"] = 32

    trainer_param_d = {
        "logger": loggers,
        "callbacks": callbacks,
        "accelerator": config_d["accelerator"],
        "devices": config_d["devices"],
        "min_epochs": config_d["min_epochs"],
        "max_epochs": config_d["max_epochs"],
        "precision": config_d["precision"],
        "log_every_n_steps": log_every_n_steps,
        "detect_anomaly": config_d["detect_anomaly"],
        "overfit_batches": overfit_batches,
        "profiler": profiler,
        "num_sanity_val_steps": config_d["num_sanity_val_steps"],
        "enable_progress_bar": config_d["pl_enable_progress_bar"],
        "enable_checkpointing": not config_d["disable_checkpoints"],
    }

    # this things can only set if automatic_optimization
    if config_d["automatic_optimization"]:
        trainer_param_d["accumulate_grad_batches"] = config_d["accumulate_grad_batches"]
        trainer_param_d["gradient_clip_val"] = config_d["gradient_clip_val"]
        trainer_param_d["gradient_clip_algorithm"] = config_d["gradient_clip_algorithm"]
        if config_d["num_workers"] > 0 and config_d["model_type"] != "mass_shatter":
            # little hack to prevent memory explosion
            # https://discuss.pytorch.org/t/how-to-share-data-among-dataloader-processes-to-save-memory/108772
            # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
            # MASShatter is exempt: it uses SpecMolDataset, which has no frag_pl_sd/frag_pp_sd
            # fragment caches, so worker RSS stays bounded and respawning only discards the
            # spectrum/molecule preprocessing caches.
            trainer_param_d["reload_dataloaders_every_n_epochs"] = 1
    elif config_d["dynamic_batch_sampler"] or config_d["dual_group_dynamic_batch_sampler"]:
        # with out this progress will not track dataset length change
        # this will call train_dataloader and val_dataloader before every epoch
        # not sure why this fixed our problem but it works
        trainer_param_d["reload_dataloaders_every_n_epochs"] = 1
        # Use per-epoch batch indices for validation checks. DAG curriculum training also
        # refreshes Lightning's cached train/validation batch counts after sampler updates.
        trainer_param_d["check_val_every_n_epoch"] = 1

    trainer = L.Trainer(**trainer_param_d)

    # set determinism
    th.use_deterministic_algorithms(config_d["deterministic"], warn_only=True)

    # register nan hook
    if config_d["nan_module_hook"]:
        th.nn.modules.module.register_module_forward_hook(nan_forward_hook)
        th.nn.modules.module.register_module_full_backward_hook(nan_backward_hook)

    # fit
    if config_d["debug_overfit"]:
        assert not is_resume
        logging.info("debug overfit model")
        trainer.fit(model, train_dl)
    else:
        logging.info("fit model")
        if is_resume:
            resume_ckpt_fp = os.path.join(ckpt_dp, "last.ckpt")
            if os.path.isfile(resume_ckpt_fp):
                logging.info(f"resuming from checkpoint: {resume_ckpt_fp}")
            else:
                logging.warning(
                    f"Resume requested but last.ckpt not found in {ckpt_dp}, starting from epoch 0"
                )
                resume_ckpt_fp = None
        else:
            resume_ckpt_fp = None
        trainer.fit(model, train_dl, val_dl, ckpt_path=resume_ckpt_fp)
    logging.info("callback metrics")

    if not trainer.interrupted and config_d["eval_test_split"]:
        logging.info("setup test dataset")
        test_ds = init_dataset(config_d, splits=["test"])[0]
        test_dl = init_dataloader(test_ds, config_d)
        logging.info("test model")
        trainer.test(
            model=model,
            ckpt_path="best" if config_d["min_epochs"] > 0 else None,
            dataloaders=test_dl,
        )

    if config_d["compile"]:
        print(model.dynamo_prof.report())

    # Handle checkpoint cleanup/saving after training
    if not trainer.interrupted and not config_d["disable_checkpoints"]:
        ckpt_fps = glob.glob(os.path.join(ckpt_dp, "*.ckpt"))

        if config_d["upload_checkpoints"]:
            # Upload checkpoints to wandb
            for ckpt_fp in ckpt_fps:
                wandb.save(ckpt_fp, base_path=os.path.dirname(ckpt_dp))

        # Temporarily move checkpoints out of the way before wandb.finish()
        # to prevent wandb from syncing them if we don't want to keep them
        temp_ckpt_tmpdir = None
        if not config_d["upload_checkpoints"] and not config_d["delete_checkpoints"]:
            temp_ckpt_tmpdir = tempfile.TemporaryDirectory()
            for ckpt_fp in ckpt_fps:
                shutil.move(ckpt_fp, os.path.join(temp_ckpt_tmpdir.name, os.path.basename(ckpt_fp)))

    if wandb_mode != "disabled":
        wandb.finish()

    # Final checkpoint cleanup after wandb is finished
    if not trainer.interrupted and not config_d["disable_checkpoints"]:
        if config_d["delete_checkpoints"]:
            # Delete all checkpoints
            ckpt_fps = glob.glob(os.path.join(ckpt_dp, "*.ckpt"))
            for ckpt_fp in ckpt_fps:
                os.remove(ckpt_fp)
        elif temp_ckpt_tmpdir is not None:
            # Restore checkpoints that were temporarily moved
            for ckpt_fp in glob.glob(os.path.join(temp_ckpt_tmpdir.name, "*.ckpt")):
                shutil.move(ckpt_fp, os.path.join(ckpt_dp, os.path.basename(ckpt_fp)))
            temp_ckpt_tmpdir.cleanup()

    # cleanup job tracking file
    if job_id_fp:
        os.remove(job_id_fp)

    return model
