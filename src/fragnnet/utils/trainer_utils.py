"""Lightning pieces the runner attaches to the trainer: the console logger and the exponential
moving average of the weights (split out of ``utils/pl_utils.py``).
"""

import copy
import logging

import torch as th
from lightning.fabric.loggers.logger import rank_zero_experiment
from lightning.pytorch.callbacks import Callback
from lightning.pytorch.loggers import Logger
from lightning.pytorch.utilities import rank_zero_only


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
