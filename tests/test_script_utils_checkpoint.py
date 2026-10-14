"""Checkpoint resolution and complete inference weight restoration."""

import copy

import pytest
import torch as th

from fragnnet.inference import checkpoint
from fragnnet.inference.checkpoint import (
    apply_inference_config,
    get_config_from_ckpt,
    init_model_from_ckpt,
)
from fragnnet.models.fragnnet.fragnnet_pl import FraGNNetPL
from fragnnet.runner import load_config


@pytest.fixture
def checkpoint_model():
    config = load_config("config", None, None)
    config.update(model_type="frag_gnn", compile=False, sparse_cosine_similarity=True)
    th.manual_seed(7)
    return FraGNNetPL(**config).eval(), config


@pytest.mark.parametrize("compilation", ["legacy", "none", "whole", "partial"])
def test_inference_restores_exact_checkpoint_state(checkpoint_model, tmp_path, compilation):
    model, config = checkpoint_model
    expected = copy.deepcopy(model.state_dict())
    if compilation == "whole":
        model.model = th.compile(model.model, backend="eager")
    elif compilation == "partial":
        model.model.formula_module = th.compile(model.model.formula_module, backend="eager")
    if compilation == "legacy":
        config.pop("compile")
    else:
        config["compile"] = compilation != "none"
    path = tmp_path / "model.ckpt"
    th.save({"state_dict": model.state_dict(), "hyper_parameters": config}, path)
    inference_config = get_config_from_ckpt(str(path))
    apply_inference_config(inference_config)
    th.manual_seed(11)
    restored = init_model_from_ckpt(str(path), inference_config, th.device("cpu"))
    assert not restored.training
    assert inference_config["compile"] is False
    assert restored.state_dict().keys() == expected.keys()
    for key, value in restored.state_dict().items():
        th.testing.assert_close(value, expected[key], atol=0, rtol=0)


@pytest.mark.parametrize("defect", ["missing", "unexpected"])
def test_inference_rejects_incomplete_checkpoint(checkpoint_model, tmp_path, defect):
    model, config = checkpoint_model
    state = model.state_dict()
    if defect == "missing":
        state.pop(next(iter(dict(model.named_parameters()))))
    else:
        state["model.nonexistent.weight"] = th.ones(1)
    path = tmp_path / "invalid.ckpt"
    th.save({"state_dict": state, "hyper_parameters": config}, path)
    with pytest.raises(RuntimeError, match="key"):
        init_model_from_ckpt(str(path), config, th.device("cpu"))


def test_get_ckpt_path_uses_existing_local_checkpoint(tmp_path) -> None:
    checkpoint_fp = tmp_path / "model.ckpt"
    checkpoint_fp.touch()

    assert (
        checkpoint.get_ckpt_path(None, str(checkpoint_fp), str(tmp_path / "downloads"))
        == str(checkpoint_fp)
    )


def test_get_ckpt_path_rejects_missing_local_checkpoint(tmp_path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("W&B fallback must not run for a supplied local checkpoint")

    monkeypatch.setattr(checkpoint, "get_best_ckpt_from_wandb", fail_if_called)
    checkpoint_fp = tmp_path / "missing.ckpt"

    with pytest.raises(FileNotFoundError, match=str(checkpoint_fp)):
        checkpoint.get_ckpt_path(None, str(checkpoint_fp), str(tmp_path / "downloads"))


def test_get_ckpt_path_requires_a_checkpoint_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="Either wandb_run_id or model_ckpt"):
        checkpoint.get_ckpt_path(None, None, str(tmp_path / "downloads"))


def test_overlay_preserves_checkpoint_settings_and_nested_fields(tmp_path) -> None:
    config = load_config("config", None, None, model_type="masster")
    config.update(ints_transform="sqrt", mol_hidden_size=128, masster_head_dropout=0.37)
    config["spec_params"]["nce"] = False
    path = tmp_path / "model.ckpt"
    th.save({"hyper_parameters": config}, path)
    baseline = get_config_from_ckpt(str(path))
    overlay = tmp_path / "overlay.yml"
    overlay.write_text("num_workers: 0\nspec_params:\n  ace: true\n")
    resolved = get_config_from_ckpt(str(path), str(overlay), "config")
    expected = {
        **baseline,
        "num_workers": 0,
        "spec_params": {**baseline["spec_params"], "ace": True},
    }
    assert resolved == expected
    assert resolved["ints_transform"] == "sqrt"
    assert resolved["mol_hidden_size"] == 128
    assert resolved["masster_head_dropout"] == pytest.approx(0.37)


@pytest.mark.parametrize(
    "overlay_text, message",
    [
        ("spec_params: {typo: true}", "does not declare"),
        ("spec_params: false", "must be a mapping"),
        ("model_type: neims", "cannot change"),
        ("model_type: iceberg_gen", "retired"),
        ("spec_params: {inst_types: [FT]}", "retired instrument type"),
    ],
)
def test_overlay_rejects_invalid_config(tmp_path, overlay_text, message) -> None:
    path = tmp_path / "model.ckpt"
    th.save({"hyper_parameters": load_config("config", None, None)}, path)
    overlay = tmp_path / "overlay.yml"
    overlay.write_text(overlay_text)
    with pytest.raises(ValueError, match=message):
        get_config_from_ckpt(str(path), str(overlay), "config")


def test_empty_overlay_adds_no_template_defaults_to_legacy_checkpoint(tmp_path) -> None:
    path = tmp_path / "legacy.ckpt"
    th.save({"hyper_parameters": {"model_type": "masster", "compile": False}}, path)
    overlay = tmp_path / "empty.yml"
    overlay.write_text("")
    baseline = get_config_from_ckpt(str(path))
    assert "mol_hidden_size" not in baseline
    assert get_config_from_ckpt(str(path), str(overlay), "config") == baseline


class _CheckpointToy(th.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.model = th.nn.Linear(2, 1)


@pytest.fixture
def wandb_checkpoint(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import wandb

    path = tmp_path / "weights.ckpt"
    remote = SimpleNamespace(
        name="ckpt/model-epoch=001.ckpt",
        download=lambda **kwargs: SimpleNamespace(name=str(path)),
    )
    run = SimpleNamespace(files=lambda **kwargs: [remote], file=lambda name: remote)
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(run=lambda name: run))
    monkeypatch.chdir(tmp_path)
    return path


@pytest.mark.parametrize("compiled", [False, True])
def test_wandb_loader_restores_complete_weights(wandb_checkpoint, compiled) -> None:
    th.manual_seed(7)
    original = _CheckpointToy()
    expected = copy.deepcopy(original.state_dict())
    if compiled:
        original.model = th.compile(original.model, backend="eager")
    th.save(
        {"state_dict": original.state_dict(), "hyper_parameters": {"compile": compiled}},
        wandb_checkpoint,
    )
    th.manual_seed(11)
    restored = checkpoint.init_wandb_ckpt("fixture", False, _CheckpointToy, {"compile": False})
    assert restored.state_dict().keys() == expected.keys()
    for key, value in restored.state_dict().items():
        th.testing.assert_close(value, expected[key], atol=0, rtol=0)


@pytest.mark.parametrize("defect", ["missing", "unexpected"])
def test_wandb_loader_rejects_incomplete_weights(wandb_checkpoint, defect) -> None:
    state = _CheckpointToy().state_dict()
    if defect == "missing":
        state.pop("model.weight")
    else:
        state["nonexistent.weight"] = th.ones(1)
    th.save({"state_dict": state, "hyper_parameters": {"compile": False}}, wandb_checkpoint)
    with pytest.raises(RuntimeError, match="key"):
        checkpoint.init_wandb_ckpt("fixture", False, _CheckpointToy, {"compile": False})
