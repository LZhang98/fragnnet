"""Tests for the ExponentialMovingAverage Lightning callback."""

from types import SimpleNamespace

import lightning as L
import pytest
import torch as th
from lightning.pytorch.callbacks import Callback, ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from fragnnet.utils.callbacks import ExponentialMovingAverage


class _TinyRegressor(L.LightningModule):
    """Linear -> BatchNorm -> Linear, so the state carries an integer buffer too."""

    def __init__(self) -> None:
        super().__init__()
        th.manual_seed(0)
        self.net = th.nn.Sequential(th.nn.Linear(4, 3), th.nn.BatchNorm1d(3), th.nn.Linear(3, 1))
        self.validation_weights: list[th.Tensor] = []
        self.train_start_weights: dict[str, th.Tensor] | None = None

    def forward(self, x: th.Tensor) -> th.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        return th.nn.functional.mse_loss(self(x), y)

    def validation_step(self, batch, batch_idx):
        x, y = batch
        self.validation_weights.append(self.net[0].weight.detach().clone())
        self.log("val_loss", th.nn.functional.mse_loss(self(x), y))

    def on_train_start(self) -> None:
        self.train_start_weights = {k: v.detach().clone() for k, v in self.state_dict().items()}

    def configure_optimizers(self):
        return th.optim.SGD(self.parameters(), lr=0.2)


def _loaders() -> tuple[DataLoader, DataLoader]:
    generator = th.Generator().manual_seed(1)
    x = th.randn(64, 4, generator=generator)
    y = x.sum(dim=1, keepdim=True) + 0.1 * th.randn(64, 1, generator=generator)
    data = TensorDataset(x, y)
    return DataLoader(data, batch_size=8), DataLoader(data, batch_size=32)


def _trainer(tmp_path, callback: ExponentialMovingAverage, max_epochs: int) -> L.Trainer:
    return L.Trainer(
        max_epochs=max_epochs,
        accelerator="cpu",
        devices=1,
        logger=False,
        enable_progress_bar=False,
        enable_model_summary=False,
        num_sanity_val_steps=0,
        callbacks=[
            callback,
            ModelCheckpoint(dirpath=tmp_path, monitor="val_loss", save_last=True),
        ],
    )


def test_ema_rejects_decay_outside_the_open_unit_interval() -> None:
    for decay in (0.0, 1.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="EMA decay must be in"):
            ExponentialMovingAverage(decay)


def test_ema_update_matches_the_warmed_up_recursion() -> None:
    """a_t = d_t a_{t-1} + (1 - d_t) w_t with d_t = min(decay, (1+t)/(10+t)), once per step."""
    module = th.nn.Linear(2, 1, bias=False)
    th.nn.init.zeros_(module.weight)
    callback = ExponentialMovingAverage(0.9)
    trainer = SimpleNamespace(global_step=0)
    callback.on_fit_start(trainer, module)

    expected = th.zeros(1, 2)
    for step in range(1, 40):
        with th.no_grad():
            module.weight.fill_(float(step))
        trainer.global_step = step
        callback.on_train_batch_end(trainer, module, None, None, 0)
        # A second batch end at the same global step is an accumulation step: no update.
        callback.on_train_batch_end(trainer, module, None, None, 0)
        decay = min(0.9, (1 + step) / (10 + step))
        expected = decay * expected + (1 - decay) * float(step)

    assert callback.num_updates == 39
    assert th.allclose(callback._average["weight"], expected)


def test_ema_validates_and_saves_the_average_and_keeps_the_training_weights(tmp_path) -> None:
    module = _TinyRegressor()
    callback = ExponentialMovingAverage(0.8)
    train_loader, val_loader = _loaders()
    _trainer(tmp_path, callback, max_epochs=3).fit(module, train_loader, val_loader)

    average = callback._average
    training = callback._training_weights
    assert training is not None, "the last validation leaves the average in the module"
    # Validation ran on the average, which has drifted away from the raw training weights.
    assert th.equal(module.validation_weights[-1], average["net.0.weight"])
    assert not th.allclose(training["net.0.weight"], average["net.0.weight"])

    checkpoint = th.load(tmp_path / "last.ckpt", weights_only=False)
    for key, value in average.items():
        assert th.equal(checkpoint["state_dict"][key], value), key
        assert th.equal(checkpoint["ema_training_state_dict"][key], training[key]), key
    # Integer buffers are carried in the weights, never averaged or duplicated.
    assert "net.1.num_batches_tracked" in checkpoint["state_dict"]
    assert "net.1.num_batches_tracked" not in checkpoint["ema_training_state_dict"]
    # The callback state holds scalars only; the tensors ride in the checkpoint itself.
    ema_state = next(
        v for k, v in checkpoint["callbacks"].items() if "ExponentialMovingAverage" in k
    )
    assert ema_state == {
        "decay": 0.8,
        "switch_every_n_epochs": None,
        "num_updates": callback.num_updates,
        "last_switch_epoch": -1,
    }
    assert callback.num_updates == 3 * 8


def test_ema_resume_restores_training_weights_and_the_average(tmp_path) -> None:
    train_loader, val_loader = _loaders()
    first = ExponentialMovingAverage(0.8)
    _trainer(tmp_path, first, max_epochs=2).fit(_TinyRegressor(), train_loader, val_loader)
    checkpoint = th.load(tmp_path / "last.ckpt", weights_only=False)

    resumed_module = _TinyRegressor()
    resumed = ExponentialMovingAverage(0.8)
    _trainer(tmp_path, resumed, max_epochs=3).fit(
        resumed_module, train_loader, val_loader, ckpt_path=str(tmp_path / "last.ckpt")
    )

    # Training continued from the raw training weights, not from the saved average.
    for key, value in checkpoint["ema_training_state_dict"].items():
        assert th.equal(resumed_module.train_start_weights[key], value), key
    assert resumed.num_updates == first.num_updates + 8


def test_ema_refuses_a_checkpoint_written_with_another_decay(tmp_path) -> None:
    train_loader, val_loader = _loaders()
    _trainer(tmp_path, ExponentialMovingAverage(0.8), max_epochs=1).fit(
        _TinyRegressor(), train_loader, val_loader
    )

    with pytest.raises(ValueError, match="differs from configured"):
        _trainer(tmp_path, ExponentialMovingAverage(0.9), max_epochs=2).fit(
            _TinyRegressor(), train_loader, val_loader, ckpt_path=str(tmp_path / "last.ckpt")
        )


class _EpochStartRecorder(_TinyRegressor):
    """Records the float state that each epoch's first optimizer step starts from."""

    def __init__(self) -> None:
        super().__init__()
        self.epoch_start_state: dict[int, dict[str, th.Tensor]] = {}

    def training_step(self, batch, batch_idx):
        if batch_idx == 0:
            self.epoch_start_state[self.current_epoch] = {
                k: v.detach().clone() for k, v in self.state_dict().items() if v.is_floating_point()
            }
        return super().training_step(batch, batch_idx)


class _ValidationRecorder(Callback):
    """Records the EMA callback's average and training weights as each validation ends."""

    def __init__(self, ema: ExponentialMovingAverage) -> None:
        self.ema = ema
        self.average: dict[int, dict[str, th.Tensor]] = {}
        self.training: dict[int, dict[str, th.Tensor]] = {}

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        self.average[epoch] = {k: v.clone() for k, v in self.ema._average.items()}
        self.training[epoch] = {k: v.clone() for k, v in self.ema._training_weights.items()}


def test_ema_rejects_a_switch_interval_that_is_not_a_positive_integer() -> None:
    for interval in (0, -1, 1.5, True, "1"):
        with pytest.raises(ValueError, match="switch interval must be a positive integer"):
            ExponentialMovingAverage(0.9, interval)
    assert ExponentialMovingAverage(0.9, None).switch_every_n_epochs is None
    assert ExponentialMovingAverage(0.9, 3).switch_every_n_epochs == 3


def test_switch_ema_trains_on_from_the_average_every_interval_epoch(tmp_path) -> None:
    """Epochs 2 and 4 start from the previous epoch's average; epochs 1 and 3 do not switch."""
    module = _EpochStartRecorder()
    ema = ExponentialMovingAverage(0.8, switch_every_n_epochs=2)
    recorder = _ValidationRecorder(ema)
    train_loader, val_loader = _loaders()
    trainer = _trainer(tmp_path, ema, max_epochs=5)
    trainer.callbacks.append(recorder)
    trainer.fit(module, train_loader, val_loader)

    for epoch in range(1, 5):
        start = module.epoch_start_state[epoch]
        previous_average = recorder.average[epoch - 1]
        previous_training = recorder.training[epoch - 1]
        # The two candidates differ, so the equality below identifies which one training used.
        assert not th.allclose(previous_average["net.0.weight"], previous_training["net.0.weight"])
        expected = previous_average if epoch % 2 == 0 else previous_training
        for key, value in start.items():
            # BatchNorm running statistics switch together with the parameters.
            assert th.equal(value, expected[key]), (epoch, key)
    assert ema._last_switch_epoch == 4
    # The average keeps its own recursion across switches: one update per optimizer step.
    assert ema.num_updates == 5 * 8


def test_switch_ema_never_switches_at_epoch_zero_or_twice_in_one_epoch() -> None:
    module = th.nn.Linear(2, 1, bias=False)
    ema = ExponentialMovingAverage(0.9, switch_every_n_epochs=1)
    trainer = SimpleNamespace(global_step=0, current_epoch=0)
    ema.on_fit_start(trainer, module)
    with th.no_grad():
        ema._average["weight"].fill_(5.0)
        module.weight.fill_(1.0)

    ema.on_train_epoch_start(trainer, module)
    assert th.equal(module.weight, th.ones(1, 2))

    # A run resumed mid-epoch 3 restores the state saved after epoch 3's switch.
    trainer.current_epoch = 3
    ema._last_switch_epoch = 3
    ema.on_train_epoch_start(trainer, module)
    assert th.equal(module.weight, th.ones(1, 2))

    trainer.current_epoch = 4
    ema.on_train_epoch_start(trainer, module)
    assert th.equal(module.weight, th.full((1, 2), 5.0))
    assert ema._last_switch_epoch == 4


def test_switch_ema_resume_matches_an_uninterrupted_run(tmp_path) -> None:
    train_loader, val_loader = _loaders()
    uninterrupted = ExponentialMovingAverage(0.8, switch_every_n_epochs=1)
    _trainer(tmp_path / "a", uninterrupted, max_epochs=4).fit(
        _TinyRegressor(), train_loader, val_loader
    )

    first = ExponentialMovingAverage(0.8, switch_every_n_epochs=1)
    _trainer(tmp_path / "b", first, max_epochs=2).fit(_TinyRegressor(), train_loader, val_loader)
    resumed = ExponentialMovingAverage(0.8, switch_every_n_epochs=1)
    _trainer(tmp_path / "b", resumed, max_epochs=4).fit(
        _TinyRegressor(), train_loader, val_loader, ckpt_path=str(tmp_path / "b" / "last.ckpt")
    )

    assert resumed.num_updates == uninterrupted.num_updates == 4 * 8
    assert resumed._last_switch_epoch == uninterrupted._last_switch_epoch == 3
    for key, value in uninterrupted._average.items():
        assert th.equal(resumed._average[key], value), key
        assert th.equal(resumed._training_weights[key], uninterrupted._training_weights[key]), key


def test_ema_refuses_a_checkpoint_written_with_another_switch_interval(tmp_path) -> None:
    train_loader, val_loader = _loaders()
    _trainer(tmp_path, ExponentialMovingAverage(0.8), max_epochs=1).fit(
        _TinyRegressor(), train_loader, val_loader
    )

    with pytest.raises(ValueError, match="switch interval None differs from configured 1"):
        _trainer(tmp_path, ExponentialMovingAverage(0.8, 1), max_epochs=2).fit(
            _TinyRegressor(), train_loader, val_loader, ckpt_path=str(tmp_path / "last.ckpt")
        )


def test_ema_reads_callback_state_saved_before_switch_ema() -> None:
    """Checkpoints such as u2's (c8hh9j1b) carry only decay and num_updates, and never switched."""
    legacy_state = {"decay": 0.8, "num_updates": 5}
    plain = ExponentialMovingAverage(0.8)
    plain.load_state_dict(legacy_state)
    assert plain.num_updates == 5
    assert plain._last_switch_epoch == -1

    with pytest.raises(ValueError, match="switch interval None differs from configured 1"):
        ExponentialMovingAverage(0.8, 1).load_state_dict(legacy_state)
