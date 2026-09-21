import copy
import logging

import numpy as np
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only


def migrate_mass_shatter_proposal_method(config: dict) -> None:
    """Normalize the original proposal key/values at configuration boundaries.

    Mutates the supplied mapping. The canonical key takes precedence when both keys
    exist (e.g. an explicit checkpoint override). Missing keys remain missing so fresh
    model configurations still fail validation; legacy loaders supply defaults.
    """
    old_key = "mass_shatter_proposal_type"
    new_key = "mass_shatter_proposal_method"
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

# CLIP and MCES pretraining heads, same story as above.
_LEGACY_CLIP_HPARAM_DEFAULTS = {
    "clip_embed_dim": 256,
    "clip_intensity_transform": "sqrt",
    "clip_loss_type": "clip",
    "spectrum_encoder_params": {},
    "mol_num_heads": 4,
    "ace_max": 100.0,
    "ace_mean": 20.0,
    "ace_std": 15.0,
    "lr_schedule_type": "exponential",
    "lr_total_steps": None,
    "lr_min_ratio": 0.0,
}


def migrate_legacy_hparams(config: dict) -> None:
    """Backfill hyperparameters that predate their declaration in ``config/template.yml``.

    Mutates the supplied mapping in place. Only missing keys are filled, so an explicit
    value in a config, checkpoint or CLI override always wins. Fresh configurations already
    carry every key from the template and are left untouched.

    Args:
        config: Hyperparameter mapping from a config file or a checkpoint.
    """
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
    if model_type in ("clip", "spec_mol_clip_pretraining"):
        for key, value in _LEGACY_CLIP_HPARAM_DEFAULTS.items():
            if key not in config:
                config[key] = value
    if model_type == "mces_pretraining" and "auxiliary_scores" not in config:
        config["auxiliary_scores"] = []

    # 4. `use_nce`/`use_ace` fell back to the dataset's `spec_params` rather than a constant,
    # so their backfill has to read that section instead of a table.
    spec_params = config.get("spec_params") or {}
    for key, spec_key in (("use_nce", "nce"), ("use_ace", "ace")):
        if key not in config:
            config[key] = bool(spec_params.get(spec_key, False))


class ConsoleLogger(Logger):
    """Custom console logger that logs metrics to console via logging module.

    This logger is useful for displaying training metrics to stdout without
    writing to files or external services.
    """

    def __init__(self):
        super().__init__()

    @property
    @rank_zero_experiment
    def name(self) -> str:
        """Return logger name."""
        pass

    @property
    @rank_zero_experiment
    def experiment(self):
        """Return experiment object."""
        pass

    @property
    @rank_zero_experiment
    def version(self) -> str:
        """Return logger version."""
        pass

    @rank_zero_only
    def log_hyperparams(self, params: dict) -> None:
        """Log hyperparameters. Not implemented for console logger.

        Args:
            params: Dictionary of hyperparameters.
        """
        # No need to log hparams
        pass

    @rank_zero_only
    def log_metrics(self, metrics: dict, step: int) -> None:
        """Log metrics to console via logging module.

        Args:
            metrics: Dictionary of metric name to value.
            step: Current training step.
        """
        metrics = copy.deepcopy(metrics)

        epoch_num = "??"
        if "epoch" in metrics:
            epoch_num = metrics.pop("epoch")

        for k, v in metrics.items():
            logging.info(f"Epoch {epoch_num}, step {step}-- {k} : {v}")

    @rank_zero_only
    def finalize(self, status: str) -> None:
        """Finalize the logger.

        Args:
            status: Final status message.
        """
        pass


class PrintGradCallback(Callback):
    """Callback that logs model parameter and gradient norms during training.

    Useful for debugging training dynamics and detecting vanishing/exploding
    gradients.
    """

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        """Log parameter norms at the start of each training batch.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
            batch: Current training batch.
            batch_idx: Index of current batch.
        """
        ps = []
        model_params = pl_module.parameters()
        for p in model_params:
            ps.append(p.norm().item())
        logging.info(ps[:10])
        logging.info("param_norm: %s", np.mean(ps))

    def on_after_backward(self, trainer, pl_module) -> None:
        """Log gradient norms after backward pass.

        Args:
            trainer: PyTorch Lightning trainer.
            pl_module: PyTorch Lightning module.
        """
        p_grads = []
        model_params = pl_module.parameters()
        for p in model_params:
            if p.grad is not None:
                p_grads.append(p.grad.norm().item())
        logging.info(p_grads[:10])
        logging.info("grad_norm: %s", np.mean(p_grads))


def get_pl_hparams(ckpt: dict) -> dict:
    """Extract and return a deep copy of hyperparameters from a checkpoint.

    Args:
        ckpt: Checkpoint dictionary containing 'hyper_parameters' key.

    Returns:
        Deep copy of hyperparameters dictionary.
    """
    hyper_parameters = copy.deepcopy(ckpt["hyper_parameters"])
    migrate_legacy_hparams(hyper_parameters)
    # Old checkpoints retain first-replicate CE context unless explicitly migrated.
    if "spec_params" in hyper_parameters:
        spec_params = hyper_parameters["spec_params"]
        if "merge_ce_aggregation" not in spec_params:
            spec_params["merge_ce_aggregation"] = "legacy"
    # Explicit checkpoint migration: pre-context-input MASShatter weights use the
    # structural-only GNN width. Fresh configs must declare this key via the template.
    if (
        hyper_parameters.get("model_type") == "mass_shatter"
        and "mass_shatter_context_in_gnn" not in hyper_parameters
    ):
        hyper_parameters["mass_shatter_context_in_gnn"] = False
    # Explicit checkpoint migration: pre-attention-topology MASShatter weights carry the
    # induced latent set. Fresh configs must declare this key via the template.
    if (
        hyper_parameters.get("model_type") == "mass_shatter"
        and "mass_shatter_intensity_transformer_attention" not in hyper_parameters
    ):
        hyper_parameters["mass_shatter_intensity_transformer_attention"] = "induced"
    # Backend choice changes construction only; old checkpoints retain CPU semantics.
    if (
        hyper_parameters.get("model_type") == "mass_shatter"
        and "mass_shatter_candidate_backend" not in hyper_parameters
    ):
        hyper_parameters["mass_shatter_candidate_backend"] = "cpu"
    if hyper_parameters.get("model_type") == "mass_shatter":
        migrate_mass_shatter_proposal_method(hyper_parameters)
        if "mass_shatter_proposal_method" not in hyper_parameters:
            hyper_parameters["mass_shatter_proposal_method"] = "cut_masks"
        if "mass_shatter_fast_gnn_beam_width" not in hyper_parameters:
            hyper_parameters["mass_shatter_fast_gnn_beam_width"] = 256
        if "mass_shatter_fast_gnn_max_boundary_cuts" not in hyper_parameters:
            hyper_parameters["mass_shatter_fast_gnn_max_boundary_cuts"] = 3
    return hyper_parameters
