"""Behavioral checks for accumulation and fixed-input performance measurements."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as th
from torch.utils.data import DataLoader, TensorDataset

SCRIPT = (
    Path(__file__).resolve().parents[1] / "scripts/debug/bench/benchmark_masster_full_training.py"
)
spec = importlib.util.spec_from_file_location("full_training_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class LinearTrainingModel(th.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = th.nn.Parameter(th.tensor(1.0))

    def transfer_batch_to_device(self, batch, device, dataloader_idx):
        return {
            key: value.to(device) if isinstance(value, th.Tensor) else value
            for key, value in batch.items()
        }

    def _common_step(self, batch, split, log):
        return {"mean_loss": (self.weight * batch["x"]).square().mean()}


def _batch(values, offset):
    return {
        "batch_size": len(values),
        "spec_unique_id": th.arange(offset, offset + len(values)),
        "x": th.tensor(values, dtype=th.float32),
    }


def _run(
    model, batches, steps, accumulation, clip_value=None, clip_algorithm="norm", scheduler=False
):
    optimizer = th.optim.SGD(model.parameters(), lr=0.01)
    schedule = (
        th.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5) if scheduler else None
    )
    result = benchmark._train_window(
        model,
        optimizer,
        schedule,
        iter(batches),
        optimizer_steps=steps,
        accumulation=accumulation,
        clip_value=clip_value,
        clip_algorithm=clip_algorithm,
        device=th.device("cpu"),
    )
    return result, optimizer


def test_accumulated_microbatches_match_combined_batches_and_clear_gradients():
    accumulated = LinearTrainingModel()
    combined = LinearTrainingModel()
    small = [_batch([1, 2], 0), _batch([3, 4], 2), _batch([5, 6], 4), _batch([7, 8], 6)]
    large = [_batch([1, 2, 3, 4], 0), _batch([5, 6, 7, 8], 4)]
    small_result, _ = _run(accumulated, small, steps=2, accumulation=2)
    large_result, _ = _run(combined, large, steps=2, accumulation=1)
    assert accumulated.weight.item() == pytest.approx(combined.weight.item())
    assert accumulated.weight.item() == pytest.approx(0.1105, abs=1e-6)
    assert small_result["spectra"] == large_result["spectra"] == 8
    assert small_result["optimizer_steps"] == large_result["optimizer_steps"] == 2
    assert sum(small_result["spec_ids"], []) == sum(large_result["spec_ids"], [])
    assert len(small_result["metrics_per_batch"]["mean_loss"]) == 4
    assert small_result["batch_latency_kind"] == "host_enqueue"


@pytest.mark.parametrize("algorithm", ["norm", "value"])
def test_clipping_occurs_after_accumulation_and_scheduler_steps_after_update(algorithm):
    model = LinearTrainingModel()
    result, optimizer = _run(
        model,
        [_batch([100], index) for index in range(4)],
        steps=2,
        accumulation=2,
        clip_value=1.0,
        clip_algorithm=algorithm,
        scheduler=True,
    )
    assert model.weight.item() == pytest.approx(0.985, abs=1e-6)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.0025)
    assert result["optimizer_steps"] == 2


def test_repeated_loader_preserves_equal_samples_and_loading_settings():
    source = DataLoader(TensorDataset(th.arange(20)), batch_size=2, shuffle=True, num_workers=0)
    repeated, fixed = benchmark._repeated_loader(source, 4, 2, 3, seed=42)
    actual = [batch[0].tolist() for batch in repeated]
    assert actual[:2] == fixed[:2]
    assert actual[2:6] == actual[6:10] == actual[10:14] == fixed
    assert repeated.num_workers == source.num_workers
    assert repeated.pin_memory == source.pin_memory
    assert repeated.collate_fn is source.collate_fn
    second, second_fixed = benchmark._repeated_loader(source, 4, 2, 3, seed=42)
    assert second_fixed == fixed
    assert len(second) == 14


def test_reference_import_uses_frozen_python_and_extension_and_records_adapter(tmp_path):
    core_spec = importlib.util.find_spec("fragnnet.models.masster.path_selection.core")
    core_path = Path(core_spec.origin)
    (tmp_path / core_path.name).symlink_to(core_path)
    (tmp_path / "candidates.py").write_text(
        "from dataclasses import dataclass\n"
        "MARKER = 'frozen reference'\n"
        "@dataclass(frozen=True)\n"
        "class MassterCandidateTimings:\n"
        "    topology_ms: float\n"
        "    assembly_ms: float\n"
        "def build_masster_candidates(value):\n"
        "    return value\n"
    )
    code = """
import importlib.util, json, sys
from pathlib import Path
spec = importlib.util.spec_from_file_location('benchmark', sys.argv[1])
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)
report = benchmark._install_reference(Path(sys.argv[2]))
from fragnnet.models.masster.path_selection import core, candidates
report['marker'] = candidates.MARKER
report['combined_ms'] = candidates.MassterCandidateTimings(2, 3).topology_assembly_ms
report['core_file'] = core.__file__
report['python_file'] = candidates.__file__
report['candidate_identity'] = candidates.build_masster_candidates(
    [1, 2, 3], candidate_backend='cpu')
try:
    candidates.build_masster_candidates([], candidate_backend='cuda')
except ValueError:
    report['cuda_reference_rejected'] = True
try:
    benchmark._install_reference(Path(sys.argv[2]))
except RuntimeError:
    report['late_import_rejected'] = True
print(json.dumps(report))
"""
    result = subprocess.run(
        [sys.executable, "-c", code, str(SCRIPT), str(tmp_path)],
        text=True,
        capture_output=True,
        check=True,
    )
    report = json.loads(result.stdout)
    assert report["marker"] == "frozen reference"
    assert report["combined_ms"] == 5
    assert report["timing_adapter"] is not None
    assert Path(report["python_file"]).parent == tmp_path
    assert Path(report["core_file"]) == core_path.resolve()
    assert report["late_import_rejected"] is True
    assert report["candidate_identity"] == [1, 2, 3]
    assert report["cuda_reference_rejected"] is True
    assert report["backend_adapter"] is not None


def test_staged_source_requires_explicit_provenance_without_git(tmp_path):
    with pytest.raises(ValueError, match="Supply --source-revision and --source-diff-sha256"):
        benchmark._source_provenance(None, None, cwd=tmp_path)


def test_explicit_staged_provenance_never_queries_a_different_checkout(monkeypatch, tmp_path):
    def unexpected_git(*args, **kwargs):
        raise AssertionError("explicit provenance must not invoke Git")

    monkeypatch.setattr(benchmark.subprocess, "check_output", unexpected_git)
    result = benchmark._source_provenance("frozen-revision", "a" * 64, cwd=tmp_path)
    assert result == {
        "git_revision": "frozen-revision",
        "tracked_diff_sha256": "a" * 64,
        "source_revision_origin": "explicit",
        "source_diff_origin": "explicit",
    }
    with pytest.raises(ValueError, match="64-character SHA256"):
        benchmark._source_provenance("frozen-revision", "not-a-digest", cwd=tmp_path)


def test_local_provenance_auto_discovers_git_revision_and_diff_digest():
    result = benchmark._source_provenance(None, None, cwd=SCRIPT.parents[2])
    assert len(result["git_revision"]) == 40
    assert len(result["tracked_diff_sha256"]) == 64
    assert result["source_revision_origin"] == result["source_diff_origin"] == "git"


@pytest.mark.parametrize("backend", ["cuda_topology", "cuda_full"])
def test_benchmark_prototype_wrapper_preserves_inputs_and_supplies_private_statistics(
    monkeypatch, backend
):
    received = {}

    def original(*args, **kwargs):
        received.update(args=args, kwargs=kwargs)
        return "candidate batch"

    module = SimpleNamespace(
        build_masster_candidates=original, build_masster_candidates_cuda_topology=original
    )
    monkeypatch.setattr(benchmark.importlib, "import_module", lambda name: module)
    statistics = []
    benchmark._enable_cuda_prototype(statistics, backend)
    public_backend = "cuda" if backend == "cuda_full" else "cpu"
    assert (
        module.build_masster_candidates(
            "logits", max_masks=256, candidate_backend=public_backend
        )
        == "candidate batch"
    )
    assert received["args"] == ("logits",)
    if backend == "cuda_topology":
        expected = {"max_masks": 256, "statistics": statistics}
    else:
        expected = {
            "max_masks": 256,
            "candidate_backend": public_backend,
            "_cuda_full_stats": statistics,
        }
    assert received["kwargs"] == expected
    with pytest.raises(RuntimeError, match="requires public backend"):
        module.build_masster_candidates("logits", candidate_backend="wrong")


@pytest.mark.parametrize(
    ("backend", "resolved"), [("cpu", "cpu"), ("cuda_topology", "cpu"), ("cuda_full", "cuda")]
)
def test_benchmark_backend_overrides_yaml_and_is_the_logged_resolved_value(backend, resolved):
    config = {"masster_candidate_backend": "cuda", "train_batch_size": 128}
    benchmark._resolve_candidate_backend(config, backend)
    assert config == {"masster_candidate_backend": resolved, "train_batch_size": 128}


@pytest.mark.parametrize("backend", ["cuda_topology", "cuda_full"])
def test_benchmark_rejects_reference_and_cuda_topology_before_loading_data(tmp_path, backend):
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--config",
            "unused.yml",
            "--label",
            "invalid",
            "--output",
            str(tmp_path / "unused.json"),
            "--reference-dir",
            str(tmp_path),
            "--candidate-backend",
            backend,
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "--reference-dir cannot be combined" in result.stderr
    assert not (tmp_path / "unused.json").exists()
