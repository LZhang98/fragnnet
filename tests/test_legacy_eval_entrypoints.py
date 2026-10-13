"""Local entrypoint regressions: actual exporters, synthetic data and mocked W&B lookup."""

import runpy
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as th

from fragnnet.inference import checkpoint
from fragnnet.runner import load_config

ROOT = Path(__file__).resolve().parents[1]


class _SpectrumModel:
    metric_names = {"cos_sim_0.01"}

    def __init__(self, seed):
        self.seed = seed
        self.training = True

    def to(self, device):
        assert device.type == "cpu"
        return self

    def eval(self):
        self.training = False

    def cpu(self):
        return self

    def inference_step(self, batch, *, metrics=None):
        assert metrics == "test"
        assert not self.training
        return {
            "pred_mzs": th.tensor([100.0 + self.seed]),
            "pred_logprobs": th.tensor([0.0]),
            "pred_batch_idxs": th.tensor([0]),
            "true_mzs": th.tensor([100.0]),
            "true_logprobs": th.tensor([0.0]),
            "true_batch_idxs": th.tensor([0]),
            "true_oos_prob": th.tensor([0.0]),
            "oos_prob": th.tensor([0.0]),
            "cos_sim_0.01": th.tensor([1.0]),
        }


@pytest.mark.parametrize("script", ["run_save_inference", "run_save_inference_ablations"])
def test_evaluation_exports_every_seed_with_real_pickle_api(tmp_path, monkeypatch, script):
    import fragnnet.runner as runner
    import wandb
    from fragnnet.models.fragnnet import fragnnet_dataset

    configs = []

    def local_config(template_path, custom_fp, override):
        # Keep the caller's path and argument checks, without loading production data.
        assert Path(custom_fp).is_file()
        assert override is None
        configs.append(custom_fp)
        config = load_config(template_path, None, None)
        config.update(accelerator="cpu", num_workers=0)
        return config

    monkeypatch.setattr(runner, "load_config", local_config)
    monkeypatch.setattr(runner, "init_dataloader", lambda ds, config: [{}])
    monkeypatch.setattr(fragnnet_dataset, "SpecMolFragDataset", lambda **kwargs: object())
    monkeypatch.setattr(th.cuda, "is_available", lambda: False)
    runs = [SimpleNamespace(name=f"fixture_s{seed}", id=str(seed)) for seed in range(3)]
    monkeypatch.setattr(wandb, "Api", lambda: SimpleNamespace(runs=lambda **kwargs: runs))
    loaded = []

    def local_checkpoint(run_id, last_ckpt, model_cls, config_d):
        assert last_ckpt is False
        loaded.append(int(run_id))
        return _SpectrumModel(int(run_id))

    monkeypatch.setattr(checkpoint, "init_wandb_ckpt", local_checkpoint)
    argv = [script, "--save_dp", str(tmp_path), "--num_seeds", "3"]
    if script == "run_save_inference":
        argv += ["--split_types", "inchikey", "--wandb_model_types", "fragnnet_d3"]
        output = tmp_path / "inchikey" / "fragnnet_d3"
    else:
        argv += ["--ablation_model_types", "fragnnet_d3_edges"]
        output = tmp_path / "fragnnet_d3_edges"
    monkeypatch.setattr(sys, "argv", argv)
    runpy.run_path(str(ROOT / "scripts" / "eval" / f"{script}.py"), run_name="__main__")
    assert len(configs) == 1
    assert loaded == [0, 1, 2]
    assert {path.name for path in output.iterdir()} == {"s0", "s1", "s2"}
    for seed in range(3):
        values = th.load(output / f"s{seed}" / "test.pkl", weights_only=False)
        th.testing.assert_close(values["pred_mzs"], th.tensor([100.0 + seed]))
        th.testing.assert_close(values["cos_sim_0.01"], th.tensor([1.0]))
        assert values["true_batch_idxs"].tolist() == [0]


@pytest.mark.parametrize(
    "script", ["run_save_inference", "run_save_inference_ablations", "run_save_entropy"]
)
def test_evaluation_help_imports_without_remote_access(script):
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "eval" / f"{script}.py"), "--help"],
        cwd=ROOT, capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    assert "usage:" in result.stdout


def test_entropy_entrypoint_reaches_real_pickle_export(tmp_path, monkeypatch):
    import fragnnet.runner as runner
    import wandb

    configs = []

    def local_config(template_path, custom_fp, override):
        assert Path(custom_fp).is_file()
        assert override is None
        config = load_config(template_path, None, None)
        config.update(accelerator="cpu", num_workers=0)
        configs.append(config)
        return config

    monkeypatch.setattr(runner, "load_config", local_config)
    monkeypatch.setattr(runner, "init_dataset", lambda config, splits: (object(),))
    monkeypatch.setattr(runner, "init_dataloader", lambda ds, config: [{}])
    monkeypatch.setattr(th.cuda, "is_available", lambda: False)
    runs = [SimpleNamespace(name="fixture_s0", id="0")]
    monkeypatch.setattr(wandb, "Api", lambda **kwargs: SimpleNamespace(runs=lambda **kw: runs))
    monkeypatch.setattr(checkpoint, "init_wandb_ckpt", lambda *args, **kwargs: _SpectrumModel(0))

    class ExportComplete(Exception):
        pass

    captured = []

    def capture_save(values, path):
        captured.append((values, path))
        raise ExportComplete()

    monkeypatch.setattr(th, "save", capture_save)
    monkeypatch.setattr(
        sys, "argv", ["entropy", "--save_dp", str(tmp_path), "--split_type", "inchikey"]
    )
    with pytest.raises(ExportComplete):
        runpy.run_path(str(ROOT / "scripts" / "eval" / "run_save_entropy.py"), run_name="__main__")
    assert configs[0]["split_dp"] == "data/split/nist20_inchikey"
    values, path = captured[0]
    assert Path(path).name == "vals.pkl"
    th.testing.assert_close(values["fixture_s0"]["pred_mzs"], th.tensor([100.0]))
    assert values["fixture_s0"]["pred_logprobs"].tolist() == [0.0]
