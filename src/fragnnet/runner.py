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
    CEContextDynamicBatchSampler,
    DualGroupDynamicBatchSampler,
    FormulaPrecTypeGroupSampler,
    GroupSampler,
    SpecMolFragDynamicBatchSampler,
    get_group_sampler,
)
from fragnnet.dataset.data_sampler import validate_dag_size_curriculum
from fragnnet.utils.misc_utils import deep_update, get_core_count
from fragnnet.utils.nn_utils import nan_backward_hook, nan_forward_hook
from fragnnet.utils.pl_utils import ConsoleLogger
from fragnnet.utils.profile_utils import MyPyTorchProfiler
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls


def load_config(template_fp: str, custom_fp: str | None, override: list | None) -> dict:
    """load config from template and custom yaml files
        if custom_fp is given, overwrite template config with custom config
    Args:
        template_fp (str): Path to the template YAML file.
        custom_fp (str | None): Path to the custom YAML file.
        override (list | None): List of key-value pairs to override in the config.

    Returns:
        dict: configuration dictionary.
    """

    assert os.path.isfile(template_fp), template_fp
    with open(template_fp) as template_file:
        config_d = yaml.load(template_file, Loader=yaml.FullLoader)
    # overwrite parts of the config
    if custom_fp is not None:
        assert os.path.isfile(custom_fp), custom_fp
        with open(custom_fp) as custom_file:
            custom_d = yaml.load(custom_file, Loader=yaml.FullLoader)
        assert all(k in config_d for k in custom_d), set(custom_d.keys()) - set(config_d.keys())
        config_d = deep_update(config_d, custom_d)

    # override
    if override is not None:
        for item in override:
            key, value = item.split("=", 1)
            keys = key.split(".")
            cur = config_d
            for k in keys[:-1]:
                if k in cur:
                    cur = cur[k]
                else:
                    msg = f"{k} not in config"
                    raise KeyError(msg)

            print(f"Updating {keys[-1]} from {cur[keys[-1]]} to {value}", flush=True)
            # here is not safe way to load, but I'll skip this issue now
            cur[keys[-1]] = yaml.safe_load(value)

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

    dses = []
    for split in splits:
        # Use multiprocessing manager dicts if requested, else regular dicts
        if config_d["num_workers"] > 0 and config_d["share_memory"]:
            manager = Manager()
            data_sds = {k: manager.dict() for k in data_dict_types}
        else:
            data_sds = {k: {} for k in data_dict_types}
        # Instantiate the dataset for this split
        ds = dataset_cls(split=split, **{**data_sds, **config_d})
        dses.append(ds)

    return tuple(dses)


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

    # Set up a generator for reproducible sampling (will be overwritten each epoch)
    generator = th.Generator()

    # Compute maximum effective batch size (accounts for gradient accumulation)
    # used by dynamic batch samplers to decide return thresholds.

    if split == "train":
        # Choose sampler for training
        if config_d["batch_sampler_type"] == "ce_context":
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
        else:
            sampler = RandomSampler(ds, False, generator=generator)
    else:
        # Use sequential sampler for validation/test/predict_only
        sampler = SequentialSampler(ds)

    # Dynamic batch sampler (for variable batch sizes, e.g. in frag_gnn)
    # Prefer formula-based hard negative dynamic sampler when enabled (train only)

    if config_d["dynamic_batch_sampler"]:
        if split == "train":
            return_batch_at = config_d["train_batch_size"] * config_d["accumulate_grad_batches"]
        else:
            return_batch_at = config_d["eval_batch_size"]
        if split == "train":
            if config_d["batch_sampler_type"] == "ce_context":
                batch_sampler = CEContextDynamicBatchSampler(
                    ds,
                    max_num=config_d["dynamic_batch_sampler_max"],
                    ce_key_by_inst=config_d["ce_context_ce_key_by_inst"],
                    limited_by=config_d["dynamic_batch_sampler_mode"],
                    spectra_per_context=config_d["ce_context_spectra_per_context"],
                    return_batch_at=return_batch_at,
                    seed=config_d["seed"],
                )
            else:
                batch_sampler = SpecMolFragDynamicBatchSampler(
                    ds,
                    max_num=config_d["dynamic_batch_sampler_max"],
                    limited_by=config_d["dynamic_batch_sampler_mode"],
                    skip_too_big=True,
                    return_batch_at=return_batch_at,
                    sampler=sampler,
                    dag_size_curriculum=config_d["dag_size_curriculum"],
                    dag_size_curriculum_start_fraction=config_d[
                        "dag_size_curriculum_start_fraction"
                    ],
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

    _VALID_SIMPLE_GROUP_SAMPLER_TYPES = {
        "group",
        "mol",
        "group_mol",
        "adduct_inst_type",
        "adduct_inst_type_group",
        "adduct_inst_type_frag_mode",
        "adduct_inst_type_frag_mode_group",
    }
    if (
        config_d.get("simple_group_sampler")
        and config_d.get("simple_group_sampler_type") not in _VALID_SIMPLE_GROUP_SAMPLER_TYPES
    ):
        raise ValueError(
            f"simple_group_sampler_type must be one of {sorted(_VALID_SIMPLE_GROUP_SAMPLER_TYPES)} "
            f"(got '{config_d['simple_group_sampler_type']}')"
        )

    if config_d["dynamic_batch_sampler"] and config_d["dual_group_dynamic_batch_sampler"]:
        raise ValueError(
            "dynamic_batch_sampler and dual_group_dynamic_batch_sampler are mutually exclusive"
        )

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
                f"Dynamic batch sampler only supports frag_node or frag_edge mode "
                f"(got '{config_d['dynamic_batch_sampler_mode']}')"
            )
        if config_d["dynamic_batch_sampler_max"] is None:
            raise ValueError("Dynamic batch sampler requires dynamic_batch_sampler_max to be set")

    valid_batch_sampler_types = {"spec_mol_frag", "ce_context"}
    if config_d["batch_sampler_type"] not in valid_batch_sampler_types:
        raise ValueError(
            "batch_sampler_type must be one of "
            f"{sorted(valid_batch_sampler_types)} (got '{config_d['batch_sampler_type']}')"
        )

    if config_d["batch_sampler_type"] == "ce_context":
        if not config_d["dynamic_batch_sampler"]:
            raise ValueError("batch_sampler_type='ce_context' requires dynamic_batch_sampler=True")
        if config_d["dynamic_batch_sampler_mode"] != "frag_edge":
            raise ValueError(
                "batch_sampler_type='ce_context' requires dynamic_batch_sampler_mode='frag_edge'"
            )
        if config_d["dag_size_curriculum"]:
            raise ValueError("batch_sampler_type='ce_context' does not support DAG-size curriculum")
        if config_d["devices"] != 1:
            raise ValueError("batch_sampler_type='ce_context' v1 requires devices=1")
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

    pair_weight = config_d["ce_pair_loss_weight"]
    if (
        not isinstance(pair_weight, (int, float))
        or isinstance(pair_weight, bool)
        or pair_weight < 0
    ):
        raise ValueError("ce_pair_loss_weight must be a non-negative number")
    if pair_weight > 0:
        if config_d["batch_sampler_type"] != "ce_context":
            raise ValueError("ce_pair_loss_weight > 0 requires batch_sampler_type='ce_context'")
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
        dual_mode = config_d.get("dual_group_dynamic_batch_sampler_mode")
        if dual_mode not in ["frag_node", "frag_edge"]:
            raise ValueError(
                f"Dual-group dynamic batch sampler only supports frag_node or frag_edge mode "
                f"(got '{dual_mode}')"
            )
        dual_max = config_d.get("dual_group_dynamic_batch_sampler_max")
        if dual_max is None:
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


def init_run(
    template_fp: str, custom_fp: str, wandb_mode: str, job_id: str, override: list | None
) -> L.LightningModule:
    """
    Main entry point for initializing and running a training/validation/testing job.
    Args:
        template_fp (str): Path to the template YAML config file.
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
    config_d = load_config(template_fp, custom_fp, override)

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

        wandb_d = {
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
                    parts = job_id_file.read().strip().split(";", 1)
                run_id = parts[0]
                old_wandb_dp = parts[1] if len(parts) > 1 else None
            else:
                is_resume = False
                run_id = None
                old_wandb_dp = None
            wandb_d["id"] = run_id
            wandb_d["resume"] = "must" if run_id is not None else "allow"
            wandb_d["config"] = config_d
        else:
            is_resume = False
            job_id_fp = None
            old_wandb_dp = None
            wandb_d["config"] = config_d
        wandb.init(**wandb_d)
        assert wandb.run is not None, wandb.run
        wandb_d["offline"] = wandb_mode == "offline"
        wandb_logger = WandbLogger(**wandb_d)
        if job_id_fp:
            os.makedirs(os.path.dirname(job_id_fp), exist_ok=True)
            with open(job_id_fp, "w+") as job_id_file:
                job_id_file.write(f"{wandb.run.id};{os.path.realpath(wandb.run.dir)}")
        # update config results (primarily for wandb)
        # configs can be nested up to 2 levels
        for k in list(config_d.keys()):
            if k in wandb.config:
                if isinstance(config_d[k], dict):
                    # nested
                    assert isinstance(wandb.config[k], dict)
                    for kk in list(config_d[k].keys()):
                        if kk in wandb.config[k]:
                            if config_d[k][kk] != wandb.config[k][kk]:
                                print(
                                    f"> config diff -- {k}->{kk}: {config_d[k][kk]} vs {wandb.config[k][kk]}"
                                )
                            config_d[k][kk] = wandb.config[k][kk]
                else:
                    if config_d[k] != wandb.config[k]:
                        print(f"> config diff -- {k}: {config_d[k]} vs {wandb.config[k]}")
                    config_d[k] = wandb.config[k]
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

    # setup callbacks
    callbacks = []
    if wandb_mode != "disabled":
        ckpt_dp = os.path.realpath(os.path.join(wandb.run.dir, "ckpt"))
    else:
        run_name = job_id if job_id else config_d.get("wandb_name", "run")
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
            new_ckpt_data = th.load(old_ckpt_fp, weights_only=False)
            ckpt_callback_data = None
            for k, v in new_ckpt_data["callbacks"].items():
                if k.startswith("ModelCheckpoint"):
                    ckpt_callback_data = v
                    break
            assert ckpt_callback_data is not None
            ckpt_keys = [
                "best_model_path",
                "last_model_path",
                "dirpath",
                "kth_best_model_path",
            ]
            for k in ckpt_keys:
                if k in ckpt_callback_data:
                    ckpt_callback_data[k] = ckpt_callback_data[k].replace(
                        os.path.realpath(old_wandb_dp), os.path.realpath(wandb.run.dir)
                    )
            if "best_k_models" in ckpt_callback_data:
                for k in list(ckpt_callback_data["best_k_models"].keys()):
                    new_k = k.replace(
                        os.path.realpath(old_wandb_dp), os.path.realpath(wandb.run.dir)
                    )
                    ckpt_callback_data["best_k_models"][new_k] = ckpt_callback_data[
                        "best_k_models"
                    ].pop(k)
            # save modified checkpoint in new wandb dir
            th.save(new_ckpt_data, new_ckpt_fp)
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
    dses = init_dataset(config_d, splits=["train", "val"])
    train_ds = dses[0]
    val_ds = dses[1]

    # check config for sampler
    validate_sampler_config(config_d)

    # check config for CMF
    if (
        config_d["frag_params"]["include_cmf"]
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
        if config_d["num_workers"] > 0:
            # little hack to prevent memory explosion
            # https://discuss.pytorch.org/t/how-to-share-data-among-dataloader-processes-to-save-memory/108772
            # https://ppwwyyxx.com/blog/2022/Demystify-RAM-Usage-in-Multiprocess-DataLoader/
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
            ckpt_fp = os.path.join(ckpt_dp, "last.ckpt")
            if os.path.isfile(ckpt_fp):
                logging.info(f"resuming from checkpoint: {ckpt_fp}")
            else:
                logging.warning(
                    f"Resume requested but last.ckpt not found in {ckpt_dp}, starting from epoch 0"
                )
                ckpt_fp = None
        else:
            ckpt_fp = None
        trainer.fit(model, train_dl, val_dl, ckpt_path=ckpt_fp)
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
        temp_ckpt_dir = None
        if not config_d["upload_checkpoints"] and not config_d["delete_checkpoints"]:
            temp_ckpt_dir = tempfile.TemporaryDirectory()
            for ckpt_fp in ckpt_fps:
                shutil.move(ckpt_fp, os.path.join(temp_ckpt_dir.name, os.path.basename(ckpt_fp)))

    if wandb_mode != "disabled":
        wandb.finish()

    # Final checkpoint cleanup after wandb is finished
    if not trainer.interrupted and not config_d["disable_checkpoints"]:
        if config_d["delete_checkpoints"]:
            # Delete all checkpoints
            ckpt_fps = glob.glob(os.path.join(ckpt_dp, "*.ckpt"))
            for ckpt_fp in ckpt_fps:
                os.remove(ckpt_fp)
        elif temp_ckpt_dir is not None:
            # Restore checkpoints that were temporarily moved
            for ckpt_fp in glob.glob(os.path.join(temp_ckpt_dir.name, "*.ckpt")):
                shutil.move(ckpt_fp, os.path.join(ckpt_dp, os.path.basename(ckpt_fp)))
            temp_ckpt_dir.cleanup()

    # cleanup job tracking file
    if job_id_fp:
        os.remove(job_id_fp)

    return model
