"""Inference backend overrides preserve the checkpoint's architecture settings."""

import runpy
import sys
from pathlib import Path

import pandas as pd
import pytest

from fragnnet.inference import checkpoint


class _ConfigurationResolved(Exception):
    pass


@pytest.mark.parametrize("override", [None, "cpu", "cuda"])
def test_cli_backend_override_changes_only_the_resolved_backend(monkeypatch, override):
    config = {
        "model_type": "masster",
        "masster_candidate_backend": "cuda",
        "mol_hidden_size": 137,
        "masster_ranked_mask_budget": 997,
        "spec_params": {"prec_types": ["[M+Na]+"]},
    }
    expected = dict(config)
    if override is not None:
        expected["masster_candidate_backend"] = override
    monkeypatch.setattr(checkpoint, "get_ckpt_path", lambda **_: "checkpoint.ckpt")
    monkeypatch.setattr(checkpoint, "get_config_from_ckpt", lambda **_: config)

    def stop_before_data_loading(*_args, **_kwargs):
        raise _ConfigurationResolved

    monkeypatch.setattr(pd, "read_pickle", stop_before_data_loading)
    argv = ["run_inference.py", "--mol_fp", "mol.pkl", "--spec_fp", "spec.pkl"]
    if override is not None:
        argv.extend(["--masster_candidate_backend", override])
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(_ConfigurationResolved):
        runpy.run_path(
            str(Path(__file__).parents[1] / "scripts/run_inference.py"), run_name="__main__"
        )
    assert config == expected


def test_cli_rejects_candidate_backend_override_for_other_models(monkeypatch, capsys):
    monkeypatch.setattr(checkpoint, "get_ckpt_path", lambda **_: "checkpoint.ckpt")
    monkeypatch.setattr(
        checkpoint, "get_config_from_ckpt", lambda **_: {"model_type": "frag_gnn"}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference.py", "--mol_fp", "mol.pkl", "--spec_fp", "spec.pkl",
            "--masster_candidate_backend", "cpu",
        ],
    )
    with pytest.raises(SystemExit) as error:
        runpy.run_path(
            str(Path(__file__).parents[1] / "scripts/run_inference.py"), run_name="__main__"
        )
    assert error.value.code == 2
    assert "requires a MASSter checkpoint" in capsys.readouterr().err
