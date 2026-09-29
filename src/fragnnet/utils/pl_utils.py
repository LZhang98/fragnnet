import copy
import logging

import numpy as np
import torch as th
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only

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


class ExponentialMovingAverage(Callback):
    """Validate, select and save an exponential moving average of the module's weights.

    The average covers every floating-point entry of the LightningModule's ``state_dict``, which
    means the parameters and the BatchNorm running statistics. Integer buffers such as
    ``num_batches_tracked`` are not averaged. It is updated once per optimizer step:

    ``a_t = d_t * a_{t-1} + (1 - d_t) * w_t``, with ``d_t = min(decay, (1 + t) / (10 + t))``,

    where ``t`` counts updates. The warm-up term keeps early averages from being dominated by
    the initialization; after about ``10 / (1 - decay)`` updates ``d_t = decay`` and the
    average's mean age is ``decay / (1 - decay)`` steps.

    The averaged weights are copied into the module at the start of every validation epoch,
    and the training weights are put back before the next training batch. So validation
    metrics, ``ModelCheckpoint``'s choice of best checkpoint and ``trainer.test(ckpt_path=...)``
    all see the average. Every saved ``checkpoint["state_dict"]`` holds the average too, so
    inference loads it without knowing about this callback. The training weights are saved
    under ``checkpoint["ema_training_state_dict"]`` and restored on resume, so a requeued run
    continues training from exactly where it stopped. On resume the average itself is read
    back from ``checkpoint["state_dict"]`` rather than stored a second time.

    With ``switch_every_n_epochs`` set (Switch EMA), the average also becomes the training
    weights at the start of every ``switch_every_n_epochs``-th epoch, so training continues from
    it and the average feeds back into the trajectory. The optimizer state is kept. The switch
    runs at the epoch's start rather than its end because checkpoints are written during the
    preceding validation: a run resumed from one switches exactly as an uninterrupted run
    would, and a run resumed mid-epoch does not switch a second time.

    Args:
        decay: Per-optimizer-step decay, strictly between 0 and 1.
        switch_every_n_epochs: Epoch interval between switches, or None to never switch.

    Raises:
        ValueError: If ``decay`` is outside (0, 1) or ``switch_every_n_epochs`` is not a
            positive integer or None.
    """

    def __init__(self, decay: float, switch_every_n_epochs: int | None = None) -> None:
        if not 0.0 < decay < 1.0:
            raise ValueError(f"EMA decay must be in (0, 1), got {decay}")
        if switch_every_n_epochs is not None and (
            isinstance(switch_every_n_epochs, bool)
            or not isinstance(switch_every_n_epochs, int)
            or switch_every_n_epochs < 1
        ):
            raise ValueError(
                f"EMA switch interval must be a positive integer or None, "
                f"got {switch_every_n_epochs!r}"
            )
        self.decay = float(decay)
        self.switch_every_n_epochs = switch_every_n_epochs
        self.num_updates = 0
        # Epoch at whose start the average last became the training weights (-1 = never).
        self._last_switch_epoch = -1
        self._last_global_step = -1
        self._average: dict[str, th.Tensor] | None = None
        # Set while the average is in the module (from a validation epoch to the next batch).
        self._training_weights: dict[str, th.Tensor] | None = None
        # Training weights read from a checkpoint, applied in on_fit_start after restore.
        self._resume_training_weights: dict[str, th.Tensor] | None = None

    @staticmethod
    def _float_state(pl_module) -> dict[str, th.Tensor]:
        # state_dict() tensors share storage with the live parameters and buffers, so an
        # in-place copy_ into them writes the module.
        return {k: v for k, v in pl_module.state_dict().items() if v.is_floating_point()}

    def _swap_in_average(self, pl_module) -> None:
        if self._average is None or self._training_weights is not None:
            return
        state = self._float_state(pl_module)
        with th.no_grad():
            self._training_weights = {k: v.detach().clone() for k, v in state.items()}
            for key, value in state.items():
                value.copy_(self._average[key])

    def _restore_training_weights(self, pl_module) -> None:
        if self._training_weights is None:
            return
        with th.no_grad():
            for key, value in self._float_state(pl_module).items():
                value.copy_(self._training_weights[key])
        self._training_weights = None

    def on_fit_start(self, trainer, pl_module) -> None:
        """Start the average from the current weights, or put resumed training weights back."""
        state = self._float_state(pl_module)
        if self._resume_training_weights is not None:
            if set(self._resume_training_weights) != set(state):
                raise ValueError("EMA training weights in the checkpoint do not match the module")
            with th.no_grad():
                for key, value in state.items():
                    value.copy_(self._resume_training_weights[key])
            self._resume_training_weights = None
        if self._average is None:
            self._average = {k: v.detach().clone() for k, v in state.items()}
        else:
            if set(self._average) != set(state):
                raise ValueError("EMA average in the checkpoint does not match the module")
            self._average = {k: v.to(state[k]) for k, v in self._average.items()}
        self._last_global_step = trainer.global_step

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        """Switch EMA: train on from the average at the start of every interval epoch."""
        epoch = trainer.current_epoch
        if (
            self.switch_every_n_epochs is None
            or epoch == 0
            or epoch % self.switch_every_n_epochs != 0
            # A run resumed mid-epoch already switched at this epoch's start.
            or epoch == self._last_switch_epoch
        ):
            return
        # A validation epoch may have left the average in the module. Either way, the training
        # weights it would have restored are abandoned.
        self._training_weights = None
        with th.no_grad():
            for key, value in self._float_state(pl_module).items():
                value.copy_(self._average[key])
        self._last_switch_epoch = epoch

    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx: int) -> None:
        """Put the training weights back if a validation epoch left the average in place."""
        self._restore_training_weights(pl_module)

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx: int) -> None:
        """Fold the new weights into the average once per optimizer step."""
        # Gradient accumulation and skipped steps leave global_step unchanged.
        if trainer.global_step == self._last_global_step:
            return
        self._last_global_step = trainer.global_step
        self.num_updates += 1
        decay = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        with th.no_grad():
            for key, value in self._float_state(pl_module).items():
                self._average[key].lerp_(value, 1.0 - decay)

    def on_validation_epoch_start(self, trainer, pl_module) -> None:
        """Validate the averaged weights."""
        self._swap_in_average(pl_module)

    def on_fit_end(self, trainer, pl_module) -> None:
        """Leave the average in the module, matching what the checkpoints hold."""
        self._swap_in_average(pl_module)

    def on_save_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:
        """Save the average as the model weights and the training weights beside it."""
        if self._average is None:
            return
        state_dict = checkpoint["state_dict"]
        if self._training_weights is not None:
            # The module, and therefore state_dict, already holds the average.
            training_weights = self._training_weights
        else:
            training_weights = {k: state_dict[k] for k in self._average}
            for key, value in self._average.items():
                state_dict[key] = value.detach().clone()
        checkpoint["ema_training_state_dict"] = {
            k: v.detach().clone() for k, v in training_weights.items()
        }

    def on_load_checkpoint(self, trainer, pl_module, checkpoint: dict) -> None:
        """Read the average and the training weights back from a checkpoint of this callback."""
        training_weights = checkpoint["ema_training_state_dict"]
        self._resume_training_weights = training_weights
        self._average = {k: checkpoint["state_dict"][k].detach().clone() for k in training_weights}

    def state_dict(self) -> dict:
        """Return the scalar state; the tensors ride in the checkpoint itself."""
        return {
            "decay": self.decay,
            "switch_every_n_epochs": self.switch_every_n_epochs,
            "num_updates": self.num_updates,
            "last_switch_epoch": self._last_switch_epoch,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the counters, refusing a checkpoint written with another decay or interval."""
        if "switch_every_n_epochs" not in state_dict:
            # Written before Switch EMA existed; that callback never switched.
            state_dict = {**state_dict, "switch_every_n_epochs": None, "last_switch_epoch": -1}
        if state_dict["decay"] != self.decay:
            raise ValueError(
                f"checkpoint EMA decay {state_dict['decay']} differs from configured {self.decay}"
            )
        if state_dict["switch_every_n_epochs"] != self.switch_every_n_epochs:
            raise ValueError(
                f"checkpoint EMA switch interval {state_dict['switch_every_n_epochs']} differs "
                f"from configured {self.switch_every_n_epochs}"
            )
        self.num_updates = state_dict["num_updates"]
        self._last_switch_epoch = state_dict["last_switch_epoch"]


def get_pl_hparams(ckpt: dict) -> dict:
    """Extract and return a deep copy of hyperparameters from a checkpoint.

    Args:
        ckpt: Checkpoint dictionary containing 'hyper_parameters' key.

    Returns:
        Deep copy of hyperparameters dictionary.
    """
    hyper_parameters = copy.deepcopy(ckpt["hyper_parameters"])
    migrate_legacy_masster_name(hyper_parameters)
    migrate_legacy_hparams(hyper_parameters)
    # Old checkpoints retain first-replicate CE context unless explicitly migrated.
    if "spec_params" in hyper_parameters:
        spec_params = hyper_parameters["spec_params"]
        if "merge_ce_aggregation" not in spec_params:
            spec_params["merge_ce_aggregation"] = "legacy"
    # Explicit checkpoint migration: pre-context-input MASSter weights use the
    # structural-only GNN width. Fresh configs must declare this key via the template.
    if (
        hyper_parameters.get("model_type") == "masster"
        and "masster_context_in_gnn" not in hyper_parameters
    ):
        hyper_parameters["masster_context_in_gnn"] = False
    # Explicit checkpoint migration: pre-attention-topology MASSter weights carry the
    # induced latent set. Fresh configs must declare this key via the template.
    if (
        hyper_parameters.get("model_type") == "masster"
        and "masster_intensity_transformer_attention" not in hyper_parameters
    ):
        hyper_parameters["masster_intensity_transformer_attention"] = "induced"
    # Backend choice changes construction only; old checkpoints retain CPU semantics.
    if (
        hyper_parameters.get("model_type") == "masster"
        and "masster_candidate_backend" not in hyper_parameters
    ):
        hyper_parameters["masster_candidate_backend"] = "cpu"
    if hyper_parameters.get("model_type") == "masster":
        migrate_masster_proposal_method(hyper_parameters)
        if "masster_proposal_method" not in hyper_parameters:
            hyper_parameters["masster_proposal_method"] = "cut_masks"
        if "masster_fast_gnn_beam_width" not in hyper_parameters:
            hyper_parameters["masster_fast_gnn_beam_width"] = 256
        if "masster_fast_gnn_max_boundary_cuts" not in hyper_parameters:
            hyper_parameters["masster_fast_gnn_max_boundary_cuts"] = 3
        # Checkpoints from before marginal fragment selection kept the whole enumerated pool.
        if "masster_fragment_selection" not in hyper_parameters:
            hyper_parameters["masster_fragment_selection"] = "mask_order"
        if "masster_selected_fragment_target" not in hyper_parameters:
            hyper_parameters["masster_selected_fragment_target"] = None
        # Checkpoints from before the neutral-loss position (2026-09-29) had none.
        if "masster_intensity_nl_positional_encoding" not in hyper_parameters:
            hyper_parameters["masster_intensity_nl_positional_encoding"] = False
    return hyper_parameters
