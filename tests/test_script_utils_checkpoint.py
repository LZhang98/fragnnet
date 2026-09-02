"""Tests for resolving local and W&B checkpoints."""

import pytest

from fragnnet.utils import script_utils


def test_get_ckpt_path_uses_existing_local_checkpoint(tmp_path) -> None:
    checkpoint_fp = tmp_path / "model.ckpt"
    checkpoint_fp.touch()

    assert (
        script_utils.get_ckpt_path(None, str(checkpoint_fp), str(tmp_path / "downloads"))
        == str(checkpoint_fp)
    )


def test_get_ckpt_path_rejects_missing_local_checkpoint(tmp_path, monkeypatch) -> None:
    def fail_if_called(*args, **kwargs):
        raise AssertionError("W&B fallback must not run for a supplied local checkpoint")

    monkeypatch.setattr(script_utils, "get_best_ckpt_from_wandb", fail_if_called)
    checkpoint_fp = tmp_path / "missing.ckpt"

    with pytest.raises(FileNotFoundError, match=str(checkpoint_fp)):
        script_utils.get_ckpt_path(None, str(checkpoint_fp), str(tmp_path / "downloads"))


def test_get_ckpt_path_requires_a_checkpoint_source(tmp_path) -> None:
    with pytest.raises(ValueError, match="Either wandb_run_id or model_ckpt"):
        script_utils.get_ckpt_path(None, None, str(tmp_path / "downloads"))
