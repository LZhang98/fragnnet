"""Backend configuration and real Lightning checkpoint compatibility boundaries."""

import copy
import logging
import sys
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

import lightning as L
import pytest
import torch as th
import yaml

import fragnnet.runner as runner
from fragnnet.pl_model import MassShatterPL
from fragnnet.utils.pl_utils import get_pl_hparams

_BACKEND_KEY = "mass_shatter_candidate_backend"
_MISSING = object()


@pytest.fixture(scope="module")
def backend_model() -> tuple[MassShatterPL, dict]:
    template = Path(__file__).parents[1] / "config" / "models_template"
    config = runner.load_config(str(template), None, None, model_type="mass_shatter")
    config.update(
        model_type="mass_shatter",
        compile=False,
        sparse_cosine_similarity=True,
        oos_tolerance_multiple=1,
        gaussian_renormalize=True,
        mol_hidden_size=16,
        mol_num_layers=1,
        mass_shatter_context_dim=8,
        mass_shatter_metadata_embedding_dim=4,
        mass_shatter_head_hidden_dim=16,
        mass_shatter_intensity_transformer_dim=16,
        mass_shatter_intensity_transformer_num_layers=1,
        mass_shatter_intensity_transformer_feedforward_dim=32,
        mass_shatter_intensity_transformer_num_inducing_points=4,
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_selected_mask_limit=2,
        mass_shatter_unique_fragment_target=2,
        mass_shatter_ranked_mask_budget=2,
        mass_shatter_max_h_transfer=1,
    )
    return MassShatterPL(**config), config


def _checkpoint(model: MassShatterPL, saved_backend: object) -> dict:
    hparams = copy.deepcopy(dict(model.hparams))
    if saved_backend is _MISSING:
        hparams.pop(_BACKEND_KEY)
    else:
        hparams[_BACKEND_KEY] = saved_backend
    checkpoint = {
        "pytorch-lightning_version": L.__version__,
        "state_dict": model.state_dict(),
        "hyper_parameters": hparams,
    }
    model.on_save_checkpoint(checkpoint)
    return checkpoint


@pytest.mark.parametrize("storage", ["file", "buffer"])
@pytest.mark.parametrize(
    ("saved_backend", "override", "expected"),
    [
        (_MISSING, {}, "cpu"),
        ("cpu", {}, "cpu"),
        ("cuda", {}, "cuda"),
        ("cuda", {_BACKEND_KEY: "cpu"}, "cpu"),
        ("cpu", {_BACKEND_KEY: "cuda"}, "cuda"),
        (_MISSING, {_BACKEND_KEY: "cuda"}, "cuda"),
        ("invalid", {_BACKEND_KEY: "cpu"}, "cpu"),
    ],
)
def test_lightning_backend_checkpoint_roundtrip(
    backend_model: tuple[MassShatterPL, dict],
    tmp_path: Path,
    storage: str,
    saved_backend: object,
    override: dict,
    expected: str,
) -> None:
    model, _ = backend_model
    checkpoint = _checkpoint(model, saved_backend)
    source = tmp_path / "model.ckpt" if storage == "file" else BytesIO()
    th.save(checkpoint, source)
    if storage == "buffer":
        source.seek(0)

    # Exercise all four supported positional arguments, including strict state loading.
    restored = MassShatterPL.load_from_checkpoint(source, "cpu", None, True, **override)

    assert restored.model.candidate_backend == expected
    assert restored.hparams[_BACKEND_KEY] == expected
    assert restored.hparams_initial[_BACKEND_KEY] == expected
    assert "_legacy_candidate_backend" not in restored.hparams
    assert "_legacy_candidate_backend" not in restored.hparams_initial
    assert restored.state_dict().keys() == model.state_dict().keys()
    for name, tensor in model.state_dict().items():
        th.testing.assert_close(restored.state_dict()[name], tensor, rtol=0, atol=0)


@pytest.mark.parametrize("saved_backend", [None, "auto", "invalid"])
def test_explicit_invalid_checkpoint_backend_is_not_migrated(
    backend_model: tuple[MassShatterPL, dict], saved_backend: object
) -> None:
    model, _ = backend_model
    buffer = BytesIO()
    th.save(_checkpoint(model, saved_backend), buffer)
    buffer.seek(0)
    with pytest.raises(ValueError, match="unknown candidate_backend"):
        MassShatterPL.load_from_checkpoint(buffer, map_location="cpu", strict=True)


def test_checkpoint_loading_remains_strict(backend_model: tuple[MassShatterPL, dict]) -> None:
    model, _ = backend_model
    checkpoint = _checkpoint(model, _MISSING)
    checkpoint["state_dict"] = dict(checkpoint["state_dict"])
    checkpoint["state_dict"].pop(next(iter(dict(model.named_parameters()))))
    buffer = BytesIO()
    th.save(checkpoint, buffer)
    buffer.seek(0)
    with pytest.raises(RuntimeError, match="Missing key"):
        MassShatterPL.load_from_checkpoint(buffer, map_location="cpu", strict=True)


def test_backend_migration_does_not_hide_missing_fresh_hyperparameter(
    backend_model: tuple[MassShatterPL, dict],
) -> None:
    model, config = backend_model
    fresh = copy.deepcopy(config)
    del fresh[_BACKEND_KEY]
    with pytest.raises(AttributeError, match=_BACKEND_KEY):
        MassShatterPL(**fresh)

    checkpoint = _checkpoint(model, _MISSING)
    migrated = get_pl_hparams(checkpoint)
    assert migrated[_BACKEND_KEY] == "cpu"
    assert _BACKEND_KEY not in checkpoint["hyper_parameters"]
    restored = MassShatterPL(**migrated)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.model.candidate_backend == "cpu"


@pytest.mark.parametrize("precision", [32, "32", "32-true"])
def test_cuda_backend_accepts_fp32_setup_without_a_gpu(
    backend_model: tuple[MassShatterPL, dict], precision: object
) -> None:
    _, config = backend_model
    config = {**copy.deepcopy(config), _BACKEND_KEY: "cuda", "precision": precision}
    model = MassShatterPL(**config)
    assert model.model.candidate_backend == "cuda"
    assert next(model.parameters()).device.type == "cpu"


@pytest.mark.parametrize("precision", [16, "16-mixed", "bf16-mixed", "bf16-true", "64-true"])
def test_cuda_backend_rejects_non_fp32_setup(
    backend_model: tuple[MassShatterPL, dict], precision: object
) -> None:
    _, config = backend_model
    config = {**copy.deepcopy(config), _BACKEND_KEY: "cuda", "precision": precision}
    with pytest.raises(ValueError, match="CUDA candidates require precision"):
        MassShatterPL(**config)


@pytest.mark.parametrize("saved_backend", ["cpu", "cuda", "invalid", None])
def test_repository_checkpoint_migration_preserves_explicit_backend(saved_backend: object) -> None:
    checkpoint = {"hyper_parameters": {"model_type": "mass_shatter", _BACKEND_KEY: saved_backend}}
    assert get_pl_hparams(checkpoint)[_BACKEND_KEY] == saved_backend


def test_repository_checkpoint_migration_leaves_other_models_unchanged() -> None:
    checkpoint = {"hyper_parameters": {"model_type": "frag_gnn"}}
    assert _BACKEND_KEY not in get_pl_hparams(checkpoint)


@pytest.mark.parametrize("backend", [_MISSING, "cpu", "cuda", "invalid", None])
def test_legacy_template_migration_preserves_custom_backend(
    tmp_path: Path, backend: object
) -> None:
    template = tmp_path / "template.yml"
    custom = tmp_path / "custom.yml"
    template.write_text("model_type: frag_gnn\n")
    custom_config = {"model_type": "mass_shatter"}
    if backend is not _MISSING:
        custom_config[_BACKEND_KEY] = backend
    custom.write_text(yaml.safe_dump(custom_config))
    resolved = runner.load_config(str(template), str(custom), None)
    assert resolved[_BACKEND_KEY] == ("cpu" if backend is _MISSING else backend)
    overridden = runner.load_config(str(template), str(custom), [f"{_BACKEND_KEY}=cuda"])
    assert overridden[_BACKEND_KEY] == "cuda"


@pytest.mark.parametrize("backend", ["cuda", "invalid", None])
def test_explicit_template_backend_is_not_migrated(tmp_path: Path, backend: object) -> None:
    template = tmp_path / "template.yml"
    template.write_text(yaml.safe_dump({"model_type": "mass_shatter", _BACKEND_KEY: backend}))
    assert runner.load_config(str(template), None, None)[_BACKEND_KEY] == backend


@pytest.mark.parametrize("resume", [False, True])
def test_runner_logs_resolved_backend_after_saved_wandb_config_merge(
    backend_model: tuple[MassShatterPL, dict],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    resume: bool,
) -> None:
    _, config = backend_model
    config = copy.deepcopy(config)
    config.update(disable_checkpoints=False, use_tensor_float32=False)
    template = tmp_path / "template.yml"
    template.write_text(yaml.safe_dump(config))
    run = SimpleNamespace(id="backend-test", dir=str(tmp_path / "run"), summary={})
    wandb = SimpleNamespace(run=run, config={_BACKEND_KEY: "cuda"})
    init_kwargs = {}

    def wandb_init(**kwargs: object) -> None:
        init_kwargs.update(kwargs)

    wandb.init = wandb_init
    monkeypatch.setitem(sys.modules, "wandb", wandb)
    monkeypatch.setattr(runner, "WandbLogger", lambda **_kwargs: object())
    monkeypatch.setattr(runner, "log_cuda_info", lambda: None)
    monkeypatch.setattr(runner.pandarallel, "initialize", lambda **_kwargs: None)
    monkeypatch.chdir(tmp_path)
    if resume:
        (tmp_path / "job_id").mkdir()
        (tmp_path / "job_id" / "test.id").write_text("backend-test")

    def model_cls(**kwargs: object) -> SimpleNamespace:
        return SimpleNamespace(
            model=SimpleNamespace(candidate_backend=kwargs[_BACKEND_KEY]), train=lambda: None
        )

    monkeypatch.setattr(runner, "get_model_cls", lambda _model_type: model_cls)

    class SetupComplete(Exception):
        pass

    def stop_after_model(**_kwargs: object) -> None:
        raise SetupComplete

    monkeypatch.setattr(runner, "ModelCheckpoint", stop_after_model)
    with caplog.at_level(logging.INFO), pytest.raises(SetupComplete):
        runner.init_run(str(template), None, "offline", "test", None)

    assert init_kwargs["resume"] == ("must" if resume else "allow")
    assert run.summary["mass_shatter_candidate_backend_resolved"] == "cuda"
    assert "Resolved MASShatter candidate backend: cuda" in caplog.text
