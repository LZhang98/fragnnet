import glob
import logging
import math
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing import Manager
from typing import Any

import lightning as L
import torch as th
import yaml
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import Profiler
from pandarallel import pandarallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import RandomSampler, SequentialSampler

from fragnnet import registry
from fragnnet.dataset import (
    BondCountCurriculumSampler,
    FormulaPrecTypeGroupSampler,
    GroupSampler,
    get_group_sampler,
)
from fragnnet.dataset.data_sampler import (
    validate_bond_size_curriculum,
)
from fragnnet.utils import misc_utils
from fragnnet.utils.migrations import (
    RETIRED_INST_TYPE_HELP,
    find_retired_inst_type_refs,
    migrate_legacy_masster_key,
    migrate_legacy_masster_name,
    migrate_masster_proposal_method,
    retire_pair_bce_hparams,
)
from fragnnet.utils.misc_utils import deep_update, get_core_count
from fragnnet.utils.nn_utils import nan_backward_hook, nan_forward_hook
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls
from fragnnet.utils.trainer_utils import ConsoleLogger, ExponentialMovingAverage, build_profiler

# Every model reads the same `config/template.yml`, so shared code can read any key without
# checking `model_type` first. The registry's set only guards against a typo in `model_type`.
VALID_MODEL_TYPES = registry.VALID_MODEL_TYPES
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
# Train samplers that `init_dataloader` chooses between; at most one may be set.
GROUP_SAMPLER_KEYS = ("group_sampler", "simple_group_sampler", "formula_prec_type_group_sampler")


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
        ValueError: If ``model_type`` is unknown or retired, or the custom config sets undeclared
            keys.
    """
    custom_d = None
    if custom_fp is not None:
        if not os.path.isfile(custom_fp):
            raise FileNotFoundError(f"custom config not found: {custom_fp}")
        with open(custom_fp) as custom_file:
            custom_d = yaml.load(custom_file, Loader=yaml.FullLoader) or {}
        retire_pair_bce_hparams(custom_d)
        migrate_legacy_masster_name(custom_d)

    # 1. Load the template every model shares.
    resolved_template_fp = resolve_template_path(template_path)
    with open(resolved_template_fp) as template_file:
        config_d = yaml.load(template_file, Loader=yaml.FullLoader)
    # A template saved inside a pre-rename checkpoint still spells MASSter `mass_shatter`.
    retire_pair_bce_hparams(config_d)
    migrate_legacy_masster_name(config_d)
    # A caller-supplied model_type (a checkpoint's) outranks the template default, so an
    # inference config that stays silent about the model does not silently become frag_gnn.
    resolved_model_type = (
        (custom_d or {}).get("model_type")
        or model_type
        or config_d.get("model_type")
        or DEFAULT_MODEL_TYPE
    )
    registry.raise_if_retired(resolved_model_type)
    if resolved_model_type not in VALID_MODEL_TYPES:
        raise ValueError(
            f"unknown model_type {resolved_model_type!r}; known types: {sorted(VALID_MODEL_TYPES)}"
        )
    config_d["model_type"] = resolved_model_type

    migrate_masster_proposal_method(config_d)
    registry.get_model_hooks(resolved_model_type).migrate_template_config(config_d)

    # 2. Validate and merge the custom config. The template declares every model's keys, so a
    # key it does not declare is a typo rather than another model's setting. Validation recurses
    # into nested sections, so a typo inside `spec_params` fails as loudly as one at the top.
    if custom_d is not None:
        migrate_masster_proposal_method(custom_d)
        unknown = find_unknown_config_keys(custom_d, config_d)
        if unknown:
            raise ValueError(f"{custom_fp} sets keys the template does not declare: {unknown}")
        config_d = deep_update(config_d, custom_d)

    # override
    if override is not None:
        for override_item in override:
            dotted_key, value = override_item.split("=", 1)
            dotted_key = migrate_legacy_masster_key(dotted_key)
            if dotted_key in ("masster_proposal_type", "masster_proposal_method"):
                proposal_override = {dotted_key: yaml.safe_load(value)}
                migrate_masster_proposal_method(proposal_override)
                config_d.update(proposal_override)
                continue
            if dotted_key.startswith(("ce_pair_", "precursor_pair_")):
                retire_pair_bce_hparams({dotted_key: yaml.safe_load(value)})
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

    # 3. Refuse retired instrument types ("FT", "OB", "OB_F"): relabelled data has none, so a
    # table or list still keyed by one would silently stop applying to Orbitrap spectra.
    retired = find_retired_inst_type_refs(config_d)
    if retired:
        raise ValueError(
            f"config names a retired instrument type at {retired}. {RETIRED_INST_TYPE_HELP}"
        )
    return config_d


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
    # Model-specific batching lives in the model's hooks (fragnnet.hooks.ModelHooks).
    model_hooks = registry.get_model_hooks(config_d["model_type"], strict=False)
    print(f"> init_dataloader for split {split}")
    dl_param_d = {
        "dataset": ds,
        "num_workers": config_d["num_workers"],
        "collate_fn": ds.get_collate_fn(),
        "pin_memory": config_d["pin_memory"] and (config_d["accelerator"] != "cpu"),
    }
    dl_param_d.update(model_hooks.dataloader_options(config_d))

    # Set up a generator for reproducible sampling (will be overwritten each epoch)
    generator = th.Generator()

    # Compute maximum effective batch size (accounts for gradient accumulation)
    # used by dynamic batch samplers to decide return thresholds.

    if split == "train":
        # Choose sampler for training
        if model_hooks.uses_ce_context_sampler(config_d):
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

    dl = DataLoader(**dl_param_d)
    return dl


def validate_sampler_config(config_d: dict) -> None:
    """Validate dynamic batch sampler config entries.

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
        config_d.get("simple_group_sampler")
        and config_d.get("simple_group_sampler_type") not in VALID_SIMPLE_GROUP_SAMPLER_TYPES
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


def hold_checkpoints_from_wandb(ckpt_dp: str) -> str:
    """Move a W&B run's checkpoints out of its uploaded directory until ``wandb.finish()``.

    ``wandb.finish()`` uploads every file under ``wandb.run.dir``, and ``ckpt_dp`` is
    ``wandb.run.dir/ckpt``. The checkpoints wait in a new directory beside ``wandb.run.dir``, in
    the run's own folder: outside the upload, and on the same filesystem, so each move is a
    rename. Copying them through the system temp directory filled a 3.7 GB ``/tmp`` with a 2 GB
    checkpoint (2026-09-27). ``mkdtemp`` leaves them on disk if the process dies before
    ``restore_held_checkpoints``.

    Args:
        ckpt_dp: The run's checkpoint directory, ``<run folder>/files/ckpt``.

    Returns:
        The hold directory to pass to ``restore_held_checkpoints``.
    """
    run_folder = os.path.dirname(os.path.dirname(os.path.realpath(ckpt_dp)))
    hold_dp = tempfile.mkdtemp(prefix="ckpt_hold_", dir=run_folder)
    for ckpt_fp in glob.glob(os.path.join(ckpt_dp, "*.ckpt")):
        os.replace(ckpt_fp, os.path.join(hold_dp, os.path.basename(ckpt_fp)))
    return hold_dp


def restore_held_checkpoints(hold_dp: str, ckpt_dp: str) -> None:
    """Move checkpoints held by ``hold_checkpoints_from_wandb`` back and remove the hold dir."""
    for ckpt_fp in glob.glob(os.path.join(hold_dp, "*.ckpt")):
        os.replace(ckpt_fp, os.path.join(ckpt_dp, os.path.basename(ckpt_fp)))
    os.rmdir(hold_dp)


def merge_wandb_config(config_d: dict, saved_run_config: dict, *, resumed: bool = False) -> None:
    """Take the values W&B holds for this run (sweep or resumed run) into ``config_d``, in place.

    Configs nest at most two levels. A value is replaced only when it differs: W&B stores
    config through JSON and returns an integral float as an int (``1.0`` comes back as ``1``),
    which is equal but would rename the metrics built from it, e.g. ``cos_sim_1.0`` to
    ``cos_sim_1``, away from ``checkpoint_metric``.

    On a new run, a key ``config_d`` does not declare can only come from the sweep, so it is a
    typo and raises, as it does in a custom config. A resumed run's saved config may hold keys
    the template has since dropped; those are logged and skipped.

    Args:
        config_d: Resolved run configuration, updated in place.
        saved_run_config: ``dict(wandb.config)`` after ``wandb.init``.
        resumed: Whether ``saved_run_config`` was saved by an earlier attempt of this job.

    Raises:
        ValueError: If a new run's W&B config sets undeclared keys, a section is not a mapping,
            the merge would change ``model_type``, or the merged config names a retired
            instrument type.
    """
    saved_run_config = dict(saved_run_config)
    retire_pair_bce_hparams(saved_run_config)
    unknown = find_unknown_config_keys(saved_run_config, config_d)
    if unknown and not resumed:
        raise ValueError(f"W&B config sets keys the template does not declare: {unknown}")
    if unknown:
        logging.warning("Skipping keys the template no longer declares: %s", unknown)
    for key in list(config_d.keys()):
        if key not in saved_run_config:
            continue
        if key == "model_type" and config_d[key] != saved_run_config[key]:
            raise ValueError(
                f"W&B config sets model_type {saved_run_config[key]!r}; "
                f"the config resolved {config_d[key]!r}"
            )
        if isinstance(config_d[key], dict):
            if not isinstance(saved_run_config[key], dict):
                raise ValueError(f"W&B config sets section {key!r} to a non-mapping")
            for sub_key in list(config_d[key].keys()):
                if (
                    sub_key in saved_run_config[key]
                    and config_d[key][sub_key] != saved_run_config[key][sub_key]
                ):
                    print(
                        f"> config diff -- {key}->{sub_key}: "
                        f"{config_d[key][sub_key]} vs {saved_run_config[key][sub_key]}"
                    )
                    config_d[key][sub_key] = saved_run_config[key][sub_key]
        elif config_d[key] != saved_run_config[key]:
            print(f"> config diff -- {key}: {config_d[key]} vs {saved_run_config[key]}")
            config_d[key] = saved_run_config[key]
    retired = find_retired_inst_type_refs(config_d)
    if retired:
        raise ValueError(
            f"W&B config names a retired instrument type at {retired}. {RETIRED_INST_TYPE_HELP}"
        )


def match_sweep_schedule(config_d: dict, train_size: int) -> None:
    """Match a simple-loader cosine schedule to the sampled batch size, in place.

    Preserve the base config's warmup fraction and planned epoch count. ``train_size``
    must count spectra after training filters, with no dropped final batch.
    """
    if train_size <= 0:
        raise ValueError("train_size must be positive")
    if not config_d["lr_schedule"] or config_d["lr_schedule_type"] != "cosine":
        raise ValueError("Sweep schedule matching requires a cosine schedule")
    # The bond-size curriculum trains on a growing subset, so its epochs have fewer batches.
    model_hooks = registry.get_model_hooks(config_d["model_type"])
    if (
        any(config_d[key] for key in GROUP_SAMPLER_KEYS)
        or config_d["bond_size_curriculum"]
        or model_hooks.builds_own_batch_sampler(config_d)
        or model_hooks.uses_ce_context_sampler(config_d)
    ):
        raise ValueError("Sweep schedule matching requires a simple spectrum loader")
    if config_d["drop_last"]:
        raise ValueError("Sweep schedule matching requires drop_last=False")
    warmup_fraction = config_d["lr_warmup_steps"] / config_d["lr_total_steps"]
    batches = math.ceil(train_size / config_d["train_batch_size"])
    steps = math.ceil(batches / config_d["accumulate_grad_batches"]) * config_d["max_epochs"]
    config_d["lr_total_steps"] = steps
    config_d["lr_warmup_steps"] = round(warmup_fraction * steps)


def fall_back_precision(config_d: dict) -> None:
    """Fall back to 32-bit precision when the device cannot run the configured one, in place.

    Args:
        config_d: Resolved config; ``precision`` may be replaced.
    """
    if str(config_d["precision"]) in ["bf16-mixed", "bf16"]:
        if not (th.cuda.is_available() and th.cuda.is_bf16_supported()):
            logging.warning("BF16 not supported on this device, falling back to 32 bit precision")
            config_d["precision"] = 32
    elif str(config_d["precision"]) in ["16-mixed", "16"] and not th.cuda.is_available():
        logging.warning("FP16 mixed precision requires CUDA, falling back to 32 bit precision")
        config_d["precision"] = 32


def validate_run_config(config_d: dict, wandb_mode: str) -> None:
    """Check the final run config before any dataset or model is built.

    Args:
        config_d: Final run config (after the W&B merge and derived settings).
        wandb_mode: "online", "offline" or "disabled".

    Raises:
        ValueError: If the configuration is invalid.
    """
    if config_d["upload_checkpoints"] and wandb_mode == "disabled":
        raise ValueError("upload_checkpoints requires W&B; got wandb_mode='disabled'")
    if config_d["ema_switch_every_n_epochs"] is not None and config_d["ema_decay"] is None:
        raise ValueError("ema_switch_every_n_epochs requires ema_decay")
    validate_sampler_config(config_d)


@dataclass
class WandbRunState:
    """The job's W&B run, or the empty state when W&B is disabled.

    Attributes:
        run: The active W&B run, or None when W&B is disabled.
        logger: Lightning logger for ``run``, or None.
        is_resume: Whether a job-tracking file named an earlier attempt of this job.
        job_id_fp: Job-tracking file to remove once the run finishes, or None.
        old_wandb_dp: The earlier attempt's run directory, whose checkpoints carry over.
    """

    run: Any | None = None
    logger: WandbLogger | None = None
    is_resume: bool = False
    job_id_fp: str | None = None
    old_wandb_dp: str | None = None


def setup_wandb(config_d: dict, wandb_mode: str, job_id: str | None) -> WandbRunState:
    """Start or resume the job's W&B run and take the config W&B holds into ``config_d``.

    With a ``job_id`` and checkpoints enabled, ``job_id/<job_id>.id`` records the run ID and run
    directory, so a requeued job resumes the same run and finds its checkpoints.

    Args:
        config_d: Resolved config, updated in place by the merge.
        wandb_mode: "online", "offline" or "disabled".
        job_id: Job identifier, or None.

    Returns:
        The run state; empty when W&B is disabled.

    Raises:
        RuntimeError: If ``wandb.init`` starts no run.
    """
    if wandb_mode == "disabled":
        return WandbRunState()
    import wandb

    wandb_init_d = {
        "project": config_d["wandb_project"],
        "name": config_d["wandb_name"],
        "group": config_d["wandb_group"],
        "mode": wandb_mode,
        "entity": config_d["wandb_entity"],
        "tags": config_d["wandb_tags"],
        "resume": "allow",
        "config": config_d,
    }
    is_resume = False
    job_id_fp = None
    old_wandb_dp = None
    if job_id and not config_d["disable_checkpoints"]:
        job_id_fp = os.path.join("job_id", f"{job_id}.id")
        run_id = None
        if os.path.isfile(job_id_fp):
            is_resume = True
            with open(job_id_fp) as job_id_file:
                job_id_fields = job_id_file.read().strip().split(";", 1)
            run_id = job_id_fields[0]
            old_wandb_dp = job_id_fields[1] if len(job_id_fields) > 1 else None
        wandb_init_d["id"] = run_id
        wandb_init_d["resume"] = "must" if run_id is not None else "allow"
    wandb.init(**wandb_init_d)
    if wandb.run is None:
        raise RuntimeError("wandb.init started no run")
    wandb_logger = WandbLogger(**wandb_init_d, offline=wandb_mode == "offline")
    if job_id_fp:
        os.makedirs(os.path.dirname(job_id_fp), exist_ok=True)
        with open(job_id_fp, "w+") as job_id_file:
            job_id_file.write(f"{wandb.run.id};{os.path.realpath(wandb.run.dir)}")
    # A sweep or an earlier attempt of this job may hold values the files do not.
    saved_run_config = dict(wandb.config)
    migrate_legacy_masster_name(saved_run_config)
    migrate_masster_proposal_method(saved_run_config)
    merge_wandb_config(config_d, saved_run_config, resumed=is_resume)
    return WandbRunState(
        run=wandb.run,
        logger=wandb_logger,
        is_resume=is_resume,
        job_id_fp=job_id_fp,
        old_wandb_dp=old_wandb_dp,
    )


def apply_process_settings(config_d: dict) -> None:
    """Set the process-wide numeric and parallelism settings the config names.

    Args:
        config_d: Final run config.
    """
    if config_d["use_tensor_float32"] and config_d["accelerator"] == "gpu":
        # The flag below controls whether to allow TF32 on matmul. This flag defaults to False in PyTorch 1.12 and later.
        th.backends.cuda.matmul.allow_tf32 = True
        # The flag below controls whether to allow TF32 on cuDNN. This flag defaults to True.
        th.backends.cudnn.allow_tf32 = True
    if config_d["log_zero_fp32"] is not None:
        misc_utils.LOG_ZERO_FP32 = float(config_d["log_zero_fp32"])
    if config_d["log_zero_fp16"] is not None:
        misc_utils.LOG_ZERO_FP16 = float(config_d["log_zero_fp16"])
    pandarallel.initialize(
        progress_bar=config_d["pandarallel_enable_progress_bar"],
        nb_workers=get_core_count(),
    )


def migrate_resume_checkpoints(old_wandb_dp: str, new_wandb_dp: str, ckpt_dp: str) -> None:
    """Copy an earlier attempt's checkpoints into this run, repointing their saved paths.

    ``ModelCheckpoint`` restores its best-k state only when the saved ``dirpath`` matches its
    own, so the paths it stored are rewritten from the old run directory to the new one.

    Args:
        old_wandb_dp: The earlier attempt's W&B run directory.
        new_wandb_dp: This run's W&B run directory.
        ckpt_dp: This run's checkpoint directory.

    Raises:
        ValueError: If a checkpoint holds no ``ModelCheckpoint`` state.
    """
    old_ckpt_fps = glob.glob(os.path.join(old_wandb_dp, "ckpt", "*.ckpt"))
    if not old_ckpt_fps:
        logging.warning(
            f"Resume requested but no checkpoints found in {old_wandb_dp}/ckpt/, starting from scratch"
        )
    old_prefix = os.path.realpath(old_wandb_dp)
    new_prefix = os.path.realpath(new_wandb_dp)
    for old_ckpt_fp in old_ckpt_fps:
        ckpt_data = th.load(old_ckpt_fp, map_location="cpu", weights_only=False)
        ckpt_callback_data = None
        for callback_key, callback_state in ckpt_data["callbacks"].items():
            if callback_key.startswith("ModelCheckpoint"):
                ckpt_callback_data = callback_state
                break
        if ckpt_callback_data is None:
            raise ValueError(f"{old_ckpt_fp} holds no ModelCheckpoint state")
        for path_key in ("best_model_path", "last_model_path", "dirpath", "kth_best_model_path"):
            if path_key in ckpt_callback_data:
                ckpt_callback_data[path_key] = ckpt_callback_data[path_key].replace(
                    old_prefix, new_prefix
                )
        if "best_k_models" in ckpt_callback_data:
            for old_path in list(ckpt_callback_data["best_k_models"].keys()):
                new_path = old_path.replace(old_prefix, new_prefix)
                ckpt_callback_data["best_k_models"][new_path] = ckpt_callback_data[
                    "best_k_models"
                ].pop(old_path)
        th.save(ckpt_data, os.path.join(ckpt_dp, os.path.basename(old_ckpt_fp)))


def build_callbacks(config_d: dict, ckpt_dp: str) -> list[L.Callback]:
    """Build the checkpoint and EMA callbacks the config enables.

    Args:
        config_d: Final run config.
        ckpt_dp: Checkpoint directory.

    Returns:
        The callbacks, checkpointing first.
    """
    callbacks = []
    if not config_d["disable_checkpoints"]:
        callbacks.append(
            ModelCheckpoint(
                dirpath=ckpt_dp,
                filename="model-{epoch:03d}",
                monitor=config_d["checkpoint_metric"],
                mode=config_d["checkpoint_metric_mode"],
                save_last=config_d["checkpoint_save_last"],
                every_n_epochs=config_d["checkpoint_every_n_epochs"],
                save_top_k=config_d["checkpoint_save_top_k"],
            )
        )
    if config_d["ema_decay"] is not None:
        callbacks.append(
            ExponentialMovingAverage(config_d["ema_decay"], config_d["ema_switch_every_n_epochs"])
        )
    return callbacks


def build_trainer(
    config_d: dict,
    loggers: list,
    callbacks: list[L.Callback],
    profiler: Profiler | None,
    train_dl: DataLoader,
) -> L.Trainer:
    """Build the Lightning trainer.

    Args:
        config_d: Final run config.
        loggers: Lightning loggers.
        callbacks: Lightning callbacks.
        profiler: Profiler, or None.
        train_dl: Training loader; caps ``log_every_n_steps`` at one epoch.

    Returns:
        The trainer.
    """
    trainer_param_d = {
        "logger": loggers,
        "callbacks": callbacks,
        "accelerator": config_d["accelerator"],
        "devices": config_d["devices"],
        "min_epochs": config_d["min_epochs"],
        "max_epochs": config_d["max_epochs"],
        "precision": config_d["precision"],
        "log_every_n_steps": min(len(train_dl), config_d["log_every_n_steps"]),
        "detect_anomaly": config_d["detect_anomaly"],
        "overfit_batches": config_d["debug_overfit_batches"] if config_d["debug_overfit"] else 0,
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
    # Dataloader reloading and validation cadence depend on the model's batching.
    model_hooks = registry.get_model_hooks(config_d["model_type"])
    trainer_param_d.update(model_hooks.trainer_options(config_d))
    return L.Trainer(**trainer_param_d)


def finalize_checkpoints(config_d: dict, ckpt_dp: str, wandb_run: Any | None) -> None:
    """Upload or hold checkpoints, finish the W&B run, then restore or delete them.

    ``wandb.finish()`` uploads every file under the run directory, so checkpoints that are not
    meant for upload wait outside it until the run has finished. Without W&B nothing is
    uploaded and the checkpoints stay where they are.

    Args:
        config_d: Final run config.
        ckpt_dp: Checkpoint directory.
        wandb_run: The active W&B run, or None when W&B is disabled.
    """
    held_ckpt_dp = None
    if wandb_run is not None and not config_d["disable_checkpoints"]:
        if config_d["upload_checkpoints"]:
            for ckpt_fp in glob.glob(os.path.join(ckpt_dp, "*.ckpt")):
                wandb_run.save(ckpt_fp, base_path=os.path.dirname(ckpt_dp))
        else:
            held_ckpt_dp = hold_checkpoints_from_wandb(ckpt_dp)
    try:
        if wandb_run is not None:
            wandb_run.finish()
    finally:
        if held_ckpt_dp is not None:
            restore_held_checkpoints(held_ckpt_dp, ckpt_dp)
    if not config_d["disable_checkpoints"] and config_d["delete_checkpoints"]:
        for ckpt_fp in glob.glob(os.path.join(ckpt_dp, "*.ckpt")):
            os.remove(ckpt_fp)


def run_training(
    template_path: str, custom_fp: str, wandb_mode: str, job_id: str, override: list | None,
    *, schedule_train_size: int | None = None,
) -> L.LightningModule:
    """
    Main entry point for running a training/validation/testing job.
    Args:
        template_path (str): Template directory, or a single template file.
        custom_fp (str | None): Path to the custom YAML config file (can be None).
        wandb_mode (str): Wandb mode ["online", "offline", "disabled"].
        job_id (str): Unique job identifier for checkpointing and wandb.
        override (list | None): List of key-value pairs to override in the config.
        schedule_train_size: Filtered training spectrum count for opt-in sweep schedule matching.
    Returns:
                    model: The trained model (LightningModule).
    """
    # 1. Process setup and the config from files and overrides.
    # Must be set before the CUDA allocator initialises (first tensor.cuda() call).
    # setdefault lets a SLURM-level export override this if needed.
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler()],
    )
    log_cuda_info()
    config_d = load_config(template_path, custom_fp, override)
    seed_everything(config_d["seed"], workers=True)
    th.multiprocessing.set_sharing_strategy(config_d["mp_sharing_strategy"])

    # 2. W&B: start or resume the run and merge the config it holds.
    logging.info("setup loggers")
    wandb_state = setup_wandb(config_d, wandb_mode, job_id)
    loggers = [ConsoleLogger()]
    if wandb_state.logger is not None:
        loggers.append(wandb_state.logger)

    # 3. Settings derived from the final config, recorded in W&B, then validation, all before
    # anything expensive is built.
    derived_keys = ("lr_total_steps", "lr_warmup_steps", "precision")
    derived_before = {key: config_d[key] for key in derived_keys}
    if schedule_train_size is not None:
        match_sweep_schedule(config_d, schedule_train_size)
    fall_back_precision(config_d)
    derived_changes = {
        key: config_d[key] for key in derived_keys if config_d[key] != derived_before[key]
    }
    if wandb_state.run is not None and derived_changes:
        wandb_state.run.config.update(derived_changes, allow_val_change=True)
    validate_run_config(config_d, wandb_mode)
    apply_process_settings(config_d)

    # 4. Model.
    logging.info("setup model")
    model = get_model_cls(config_d["model_type"])(**config_d)
    model.train()
    registry.get_model_hooks(config_d["model_type"]).log_resolved_settings(
        model, config_d, wandb_state.run
    )

    # 5. Checkpoint directory (with an earlier attempt's checkpoints), callbacks, profiler.
    # Names the on-disk checkpoint and profile directories when wandb is disabled.
    run_name = job_id if job_id else config_d.get("wandb_name", "run")
    if wandb_state.run is not None:
        ckpt_dp = os.path.realpath(os.path.join(wandb_state.run.dir, "ckpt"))
        profile_dp = os.path.join(wandb_state.run.dir, "profile")
    else:
        logging.info(
            "wandb is disabled; checkpoints will be saved to "
            f"ckpts/{run_name} — concurrent runs with the same name will collide."
            "use unique job_id or enable wandb to avoid this issue."
        )
        ckpt_dp = os.path.join("ckpts", run_name)
        profile_dp = os.path.join("tmp_profile", run_name)
    os.makedirs(ckpt_dp, exist_ok=True)
    if wandb_state.is_resume and wandb_state.old_wandb_dp is not None:
        migrate_resume_checkpoints(wandb_state.old_wandb_dp, wandb_state.run.dir, ckpt_dp)
    callbacks = build_callbacks(config_d, ckpt_dp)
    logging.info("setup profiler")
    os.makedirs(profile_dp, exist_ok=True)
    profiler = build_profiler(config_d, profile_dp)

    # 6. Datasets and loaders. The test set is loaded after training so it is not held in
    # memory throughout the run.
    logging.info("setup dataset")
    train_ds, val_ds = init_dataset(config_d, splits=["train", "val"])
    logging.info("setup dataloader")
    if schedule_train_size is not None and len(train_ds) != schedule_train_size:
        raise ValueError(
            f"Sweep schedule expected {schedule_train_size} train spectra, got {len(train_ds)}"
        )
    train_dl = init_dataloader(train_ds, config_d)
    val_dl = init_dataloader(val_ds, config_d)

    # 7. Trainer and debugging switches.
    logging.info("setup trainer")
    trainer = build_trainer(config_d, loggers, callbacks, profiler, train_dl)
    th.use_deterministic_algorithms(config_d["deterministic"], warn_only=True)
    if config_d["nan_module_hook"]:
        th.nn.modules.module.register_module_forward_hook(nan_forward_hook)
        th.nn.modules.module.register_module_full_backward_hook(nan_backward_hook)

    # 8. Fit, resuming from this run's last checkpoint when the job was requeued.
    if config_d["debug_overfit"]:
        if wandb_state.is_resume:
            raise ValueError("debug_overfit cannot resume an earlier attempt of the job")
        logging.info("debug overfit model")
        trainer.fit(model, train_dl)
    else:
        logging.info("fit model")
        resume_ckpt_fp = None
        if wandb_state.is_resume:
            resume_ckpt_fp = os.path.join(ckpt_dp, "last.ckpt")
            if os.path.isfile(resume_ckpt_fp):
                logging.info(f"resuming from checkpoint: {resume_ckpt_fp}")
            else:
                logging.warning(
                    f"Resume requested but last.ckpt not found in {ckpt_dp}, starting from epoch 0"
                )
                resume_ckpt_fp = None
        trainer.fit(model, train_dl, val_dl, ckpt_path=resume_ckpt_fp)

    # 9. Test the best checkpoint when one was saved, otherwise the final weights.
    if config_d["eval_test_split"]:
        logging.info("setup test dataset")
        test_ds = init_dataset(config_d, splits=["test"])[0]
        test_dl = init_dataloader(test_ds, config_d)
        logging.info("test model")
        checkpoint_callback = trainer.checkpoint_callback
        has_best = checkpoint_callback is not None and bool(checkpoint_callback.best_model_path)
        trainer.test(model=model, ckpt_path="best" if has_best else None, dataloaders=test_dl)

    # Only models built with the compile profiler carry a report.
    dynamo_prof = getattr(model, "dynamo_prof", None)
    if config_d["compile"] and dynamo_prof is not None:
        logging.info("%s", dynamo_prof.report())

    # 10. Checkpoints and W&B, then the job-tracking file.
    finalize_checkpoints(config_d, ckpt_dp, wandb_state.run)
    if wandb_state.job_id_fp:
        os.remove(wandb_state.job_id_fp)

    return model
