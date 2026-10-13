import glob
import logging
import math
import os
import tempfile
from dataclasses import dataclass
from typing import Any

import lightning as L
import torch as th
from lightning import seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import Profiler
from pandarallel import pandarallel
from torch.utils.data import DataLoader

from fragnnet import registry
from fragnnet.config import (  # noqa: F401 -- re-exported; configuration lives there
    DEFAULT_MODEL_TYPE,
    OPEN_CONFIG_MAPPINGS,
    TEMPLATE_FILENAME,
    VALID_MODEL_TYPES,
    find_unknown_config_keys,
    load_config,
    merge_wandb_config,
    resolve_template_path,
)
from fragnnet.dataset.loading import (  # noqa: F401 -- re-exported; data loading lives there
    GROUP_SAMPLER_KEYS,
    VALID_BATCH_SAMPLER_TYPES,
    VALID_SIMPLE_GROUP_SAMPLER_TYPES,
    init_dataloader,
    init_dataset,
    validate_sampler_config,
)
from fragnnet.utils import misc_utils
from fragnnet.utils.misc_utils import get_core_count
from fragnnet.utils.nn_utils import nan_backward_hook, nan_forward_hook
from fragnnet.utils.setup_utils import get_model_cls
from fragnnet.utils.trainer_utils import ConsoleLogger, ExponentialMovingAverage, build_profiler

logger = logging.getLogger(__name__)


def log_cuda_info() -> None:
    """Log PyTorch version and CUDA device information."""
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
            logger.warning("BF16 not supported on this device, falling back to 32 bit precision")
            config_d["precision"] = 32
    elif str(config_d["precision"]) in ["16-mixed", "16"] and not th.cuda.is_available():
        logger.warning("FP16 mixed precision requires CUDA, falling back to 32 bit precision")
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
    merge_wandb_config(config_d, dict(wandb.config), resumed=is_resume)
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
        logger.warning(
            "Resume requested but no checkpoints found in %s/ckpt/, starting from scratch",
            old_wandb_dp,
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
    # Lightning accepts these only under automatic optimization.
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
    # 1. The config from files and overrides. Logging and the CUDA allocator settings are the
    # entry script's to configure (scripts/run_pl_model_fit.py).
    log_cuda_info()
    config_d = load_config(template_path, custom_fp, override)
    seed_everything(config_d["seed"], workers=True)
    th.multiprocessing.set_sharing_strategy(config_d["mp_sharing_strategy"])

    # 2. W&B: start or resume the run and merge the config it holds.
    logger.info("setup loggers")
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
    logger.info("setup model")
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
        logger.info(
            "wandb is disabled; checkpoints will be saved to ckpts/%s — concurrent runs with "
            "the same name will collide. Use a unique job_id or enable wandb to avoid this.",
            run_name,
        )
        ckpt_dp = os.path.join("ckpts", run_name)
        profile_dp = os.path.join("tmp_profile", run_name)
    os.makedirs(ckpt_dp, exist_ok=True)
    if wandb_state.is_resume and wandb_state.old_wandb_dp is not None:
        migrate_resume_checkpoints(wandb_state.old_wandb_dp, wandb_state.run.dir, ckpt_dp)
    callbacks = build_callbacks(config_d, ckpt_dp)
    logger.info("setup profiler")
    os.makedirs(profile_dp, exist_ok=True)
    profiler = build_profiler(config_d, profile_dp)

    # 6. Datasets and loaders. The test set is loaded after training so it is not held in
    # memory throughout the run.
    logger.info("setup dataset")
    train_ds, val_ds = init_dataset(config_d, splits=["train", "val"])
    logger.info("setup dataloader")
    if schedule_train_size is not None and len(train_ds) != schedule_train_size:
        raise ValueError(
            f"Sweep schedule expected {schedule_train_size} train spectra, got {len(train_ds)}"
        )
    train_dl = init_dataloader(train_ds, config_d)
    val_dl = init_dataloader(val_ds, config_d)

    # 7. Trainer and debugging switches.
    logger.info("setup trainer")
    trainer = build_trainer(config_d, loggers, callbacks, profiler, train_dl)
    th.use_deterministic_algorithms(config_d["deterministic"], warn_only=True)
    if config_d["nan_module_hook"]:
        th.nn.modules.module.register_module_forward_hook(nan_forward_hook)
        th.nn.modules.module.register_module_full_backward_hook(nan_backward_hook)

    # 8. Fit, resuming from this run's last checkpoint when the job was requeued.
    if config_d["debug_overfit"]:
        if wandb_state.is_resume:
            raise ValueError("debug_overfit cannot resume an earlier attempt of the job")
        logger.info("debug overfit model")
        trainer.fit(model, train_dl)
    else:
        logger.info("fit model")
        resume_ckpt_fp = None
        if wandb_state.is_resume:
            resume_ckpt_fp = os.path.join(ckpt_dp, "last.ckpt")
            if os.path.isfile(resume_ckpt_fp):
                logger.info("resuming from checkpoint: %s", resume_ckpt_fp)
            else:
                logger.warning(
                    "Resume requested but last.ckpt not found in %s, starting from epoch 0", ckpt_dp
                )
                resume_ckpt_fp = None
        trainer.fit(model, train_dl, val_dl, ckpt_path=resume_ckpt_fp)

    # 9. Test the best checkpoint when one was saved, otherwise the final weights.
    if config_d["eval_test_split"]:
        logger.info("setup test dataset")
        test_ds = init_dataset(config_d, splits=["test"])[0]
        test_dl = init_dataloader(test_ds, config_d)
        logger.info("test model")
        checkpoint_callback = trainer.checkpoint_callback
        has_best = checkpoint_callback is not None and bool(checkpoint_callback.best_model_path)
        trainer.test(model=model, ckpt_path="best" if has_best else None, dataloaders=test_dl)

    # Only models built with the compile profiler carry a report.
    dynamo_prof = getattr(model, "dynamo_prof", None)
    if config_d["compile"] and dynamo_prof is not None:
        logger.info("%s", dynamo_prof.report())

    # 10. Checkpoints and W&B, then the job-tracking file.
    finalize_checkpoints(config_d, ckpt_dp, wandb_state.run)
    if wandb_state.job_id_fp:
        os.remove(wandb_state.job_id_fp)

    return model
