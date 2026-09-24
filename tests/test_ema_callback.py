"""Tests for the ExponentialMovingAverage Lightning callback."""

from types import SimpleNamespace

import lightning as L
import pytest
import torch as th
from lightning.pytorch.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader, TensorDataset

from fragnnet.utils.pl_utils import ExponentialMovingAverage


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
    assert ema_state == {"decay": 0.8, "num_updates": callback.num_updates}
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
