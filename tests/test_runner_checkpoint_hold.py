"""Tests for holding checkpoints outside the W&B upload while ``wandb.finish()`` runs."""

import os

import pytest

from fragnnet.runner import hold_checkpoints_from_wandb, restore_held_checkpoints


@pytest.fixture
def run_layout(tmp_path):
    """A cluster-style layout: the job's ``wandb`` is a symlink to the checkout's ``wandb``."""
    persistent_run = tmp_path / "checkout" / "wandb" / "run-20260927_000000-abc123"
    ckpt_dp = persistent_run / "files" / "ckpt"
    ckpt_dp.mkdir(parents=True)
    (ckpt_dp / "model-epoch=041.ckpt").write_bytes(b"best")
    (ckpt_dp / "last.ckpt").write_bytes(b"last")
    (ckpt_dp / "notes.txt").write_text("not a checkpoint")
    job_dp = tmp_path / "job"
    job_dp.mkdir()
    os.symlink(tmp_path / "checkout" / "wandb", job_dp / "wandb")
    job_ckpt_dp = job_dp / "wandb" / persistent_run.name / "files" / "ckpt"
    return persistent_run, str(job_ckpt_dp)


def test_hold_moves_checkpoints_beside_the_uploaded_files_dir(run_layout):
    persistent_run, ckpt_dp = run_layout
    hold_dp = hold_checkpoints_from_wandb(ckpt_dp)

    assert os.path.dirname(hold_dp) == str(persistent_run)
    assert not hold_dp.startswith(str(persistent_run / "files"))
    assert sorted(os.listdir(hold_dp)) == ["last.ckpt", "model-epoch=041.ckpt"]
    assert sorted(os.listdir(ckpt_dp)) == ["notes.txt"]
    assert os.stat(hold_dp).st_dev == os.stat(ckpt_dp).st_dev


def test_restore_returns_the_same_files_and_removes_the_hold_dir(run_layout):
    _, ckpt_dp = run_layout
    hold_dp = hold_checkpoints_from_wandb(ckpt_dp)
    restore_held_checkpoints(hold_dp, ckpt_dp)

    assert not os.path.exists(hold_dp)
    assert sorted(os.listdir(ckpt_dp)) == ["last.ckpt", "model-epoch=041.ckpt", "notes.txt"]
    with open(os.path.join(ckpt_dp, "model-epoch=041.ckpt"), "rb") as ckpt_file:
        assert ckpt_file.read() == b"best"


def test_a_run_without_checkpoints_holds_nothing(tmp_path):
    ckpt_dp = tmp_path / "run" / "files" / "ckpt"
    ckpt_dp.mkdir(parents=True)
    hold_dp = hold_checkpoints_from_wandb(str(ckpt_dp))
    assert os.listdir(hold_dp) == []
    restore_held_checkpoints(hold_dp, str(ckpt_dp))
    assert not os.path.exists(hold_dp)
