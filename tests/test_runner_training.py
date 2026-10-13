"""Tests for ``runner.run_training`` orchestration: config checks, test-checkpoint choice,
checkpoint handling around ``wandb.finish()`` and the resume checkpoint rewrite.

Model, data and trainer are replaced by fakes, so these run on CPU in seconds.
"""

import os
from types import SimpleNamespace

import pytest
import torch as th

import fragnnet.runner as runner

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "..", "config")


class _FakeDataset:
    split = "train"

    def __len__(self):
        return 1


class _FakeTrainer:
    """Records ``fit`` and ``test`` calls; has no checkpoint callback unless one is given."""

    def __init__(self, checkpoint_callback=None):
        self.checkpoint_callback = checkpoint_callback
        self.calls = []

    def fit(self, model, *dataloaders, ckpt_path=None):
        self.calls.append(("fit", ckpt_path))

    def test(self, model, ckpt_path, dataloaders):
        self.calls.append(("test", ckpt_path))


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """Patch model, data and trainer; return the trainer and the model's constructor kwargs."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(runner, "log_cuda_info", lambda: None)
    monkeypatch.setattr(runner.pandarallel, "initialize", lambda **_kwargs: None)
    model_kwargs = {}

    def model_cls(**kwargs):
        model_kwargs.update(kwargs)
        # No `dynamo_prof`: only FraGNNet and NEIMS built with the compile profiler have one.
        return SimpleNamespace(train=lambda: None)

    monkeypatch.setattr(runner, "get_model_cls", lambda _model_type: model_cls)
    monkeypatch.setattr(
        runner, "init_dataset", lambda _config, splits: tuple(_FakeDataset() for _ in splits)
    )
    monkeypatch.setattr(runner, "init_dataloader", lambda _ds, _config: [None])
    trainer = _FakeTrainer()
    monkeypatch.setattr(runner, "build_trainer", lambda *_args: trainer)
    return trainer, model_kwargs


def _run(*overrides):
    return runner.run_training(CONFIG_DIR, None, "disabled", None, list(overrides))


def test_upload_without_wandb_fails_before_training(fake_run):
    trainer, model_kwargs = fake_run
    with pytest.raises(ValueError, match="upload_checkpoints requires W&B"):
        _run("upload_checkpoints=True")
    assert model_kwargs == {} and trainer.calls == []


def test_test_split_without_checkpoints_uses_final_weights(fake_run):
    """``ckpt_path="best"`` with no ``ModelCheckpoint`` makes Lightning raise after the fit."""
    trainer, _ = fake_run
    _run("disable_checkpoints=True", "eval_test_split=True", "min_epochs=1", "compile=True")
    assert trainer.calls == [("fit", None), ("test", None)]


def test_test_split_uses_the_best_checkpoint_when_one_was_saved(fake_run):
    trainer, _ = fake_run
    trainer.checkpoint_callback = SimpleNamespace(best_model_path="ckpts/model-epoch=000.ckpt")
    _run("eval_test_split=True")
    assert trainer.calls == [("fit", None), ("test", "best")]


def test_precision_fallback_reaches_the_model(fake_run, monkeypatch):
    _, model_kwargs = fake_run
    monkeypatch.setattr(runner.th.cuda, "is_available", lambda: False)
    _run("precision=bf16-mixed")
    assert model_kwargs["precision"] == 32


def test_ema_switch_without_decay_fails_before_the_model(fake_run):
    _, model_kwargs = fake_run
    with pytest.raises(ValueError, match="requires ema_decay"):
        _run("ema_switch_every_n_epochs=2")
    assert model_kwargs == {}


class _FakeWandbRun:
    """Records which checkpoints sit in the run's ``ckpt`` directory when ``finish`` runs."""

    def __init__(self, ckpt_dp, fail=False):
        self.ckpt_dp = ckpt_dp
        self.fail = fail
        self.saved = []
        self.at_finish = None

    def save(self, ckpt_fp, base_path):
        self.saved.append(os.path.relpath(ckpt_fp, base_path))

    def finish(self):
        self.at_finish = sorted(os.listdir(self.ckpt_dp))
        if self.fail:
            raise RuntimeError("upload failed")


@pytest.fixture
def ckpt_dp(tmp_path):
    ckpt_dp = tmp_path / "run" / "files" / "ckpt"
    ckpt_dp.mkdir(parents=True)
    (ckpt_dp / "last.ckpt").write_bytes(b"last")
    return str(ckpt_dp)


def _settings(**overrides):
    return {
        "disable_checkpoints": False,
        "upload_checkpoints": False,
        "delete_checkpoints": False,
        **overrides,
    }


def test_checkpoints_wait_outside_the_upload_and_come_back(ckpt_dp):
    run = _FakeWandbRun(ckpt_dp)
    runner.finalize_checkpoints(_settings(), ckpt_dp, run)
    assert run.at_finish == []
    assert os.listdir(ckpt_dp) == ["last.ckpt"]


def test_deleted_checkpoints_are_not_uploaded_first(ckpt_dp):
    run = _FakeWandbRun(ckpt_dp)
    runner.finalize_checkpoints(_settings(delete_checkpoints=True), ckpt_dp, run)
    assert run.at_finish == []
    assert os.listdir(ckpt_dp) == []


def test_uploaded_checkpoints_stay_for_the_upload(ckpt_dp):
    run = _FakeWandbRun(ckpt_dp)
    runner.finalize_checkpoints(_settings(upload_checkpoints=True), ckpt_dp, run)
    assert run.saved == ["ckpt/last.ckpt"]
    assert run.at_finish == ["last.ckpt"]


def test_failed_finish_still_restores_held_checkpoints(ckpt_dp):
    run = _FakeWandbRun(ckpt_dp, fail=True)
    with pytest.raises(RuntimeError, match="upload failed"):
        runner.finalize_checkpoints(_settings(), ckpt_dp, run)
    assert os.listdir(ckpt_dp) == ["last.ckpt"]
    assert [p for p in os.listdir(os.path.dirname(os.path.dirname(ckpt_dp))) if "hold" in p] == []


def test_without_wandb_checkpoints_stay_put(ckpt_dp):
    runner.finalize_checkpoints(_settings(), ckpt_dp, None)
    assert os.listdir(ckpt_dp) == ["last.ckpt"]


def test_resume_rewrites_checkpoint_paths_to_the_new_run(tmp_path):
    old_dp, new_dp = tmp_path / "old_run", tmp_path / "new_run"
    (old_dp / "ckpt").mkdir(parents=True)
    (new_dp / "ckpt").mkdir(parents=True)
    old_best = str(old_dp / "ckpt" / "model-epoch=003.ckpt")
    th.save(
        {
            "state_dict": {"w": th.ones(2)},
            "callbacks": {
                "ModelCheckpoint{'monitor': 'epoch'}": {
                    "best_model_path": old_best,
                    "dirpath": str(old_dp / "ckpt"),
                    "best_k_models": {old_best: th.tensor(3.0)},
                }
            },
        },
        old_dp / "ckpt" / "last.ckpt",
    )
    runner.migrate_resume_checkpoints(str(old_dp), str(new_dp), str(new_dp / "ckpt"))

    state = th.load(new_dp / "ckpt" / "last.ckpt", weights_only=False)
    callback = state["callbacks"]["ModelCheckpoint{'monitor': 'epoch'}"]
    new_best = str(new_dp / "ckpt" / "model-epoch=003.ckpt")
    assert callback["best_model_path"] == new_best
    assert callback["dirpath"] == str(new_dp / "ckpt")
    assert list(callback["best_k_models"]) == [new_best]
    assert th.equal(state["state_dict"]["w"], th.ones(2))
