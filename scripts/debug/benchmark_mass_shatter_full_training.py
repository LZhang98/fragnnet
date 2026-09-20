"""Measure MASShatter training windows with loading, transfer and optimizer updates.

Run baseline/new variants in separate processes, alternating their order. A reference
folder can supply frozen candidate Python and a native extension without replacing
working-tree files. This FP32, single-device harness excludes Lightning logging,
validation and checkpointing; use a subsequent training run to measure those costs.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import inspect
import itertools
import json
import logging
import math
import platform
import statistics
import subprocess
import sys
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import psutil
import torch as th
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)
METRICS = (
    "mean_loss",
    "selected_mask_count",
    "component_pathway_count",
    "unique_component_count",
    "candidate_input_transfer_batch_ms",
    "candidate_mask_selection_batch_ms",
    "candidate_topology_batch_ms",
    "candidate_assembly_batch_ms",
    "candidate_topology_assembly_batch_ms",
    "candidate_output_transfer_batch_ms",
    "candidate_total_batch_ms",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _install_reference(directory: Path) -> dict:
    """Load frozen kernels and their Python caller before importing any model."""
    names = ("mass_shatter_core", "mass_shatter_candidates")
    if any(f"fragnnet.frag.{name}" in sys.modules for name in names):
        raise RuntimeError("reference modules must be installed before importing MASShatter")
    extensions = list(directory.glob("mass_shatter_core*.so"))
    if len(extensions) != 1:
        raise ValueError("reference directory must contain exactly one mass_shatter_core*.so")
    paths = (extensions[0], directory / "mass_shatter_candidates.py")
    if not all(path.is_file() for path in paths):
        raise ValueError("reference directory must contain mass_shatter_candidates.py")
    package = importlib.import_module("fragnnet.frag")
    installed = {}
    for name, path in zip(names, paths, strict=True):
        module_name = f"fragnnet.frag.{name}"
        spec = importlib.util.spec_from_file_location(module_name, path.resolve())
        if spec is None or spec.loader is None:
            raise RuntimeError(f"cannot import reference {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        setattr(package, name, module)
        installed[module_name] = str(path.resolve())
    timings = sys.modules["fragnnet.frag.mass_shatter_candidates"].MassShatterCandidateTimings
    adapter = None
    if "topology_assembly_ms" not in timings.__dataclass_fields__:
        timings.topology_assembly_ms = property(lambda self: self.topology_ms + self.assembly_ms)
        adapter = "topology_assembly_ms = topology_ms + assembly_ms (serial reference only)"
    candidates = sys.modules["fragnnet.frag.mass_shatter_candidates"]
    original_builder = candidates.build_mass_shatter_candidates
    backend_adapter = None
    if "candidate_backend" not in inspect.signature(original_builder).parameters:

        def build_reference(*args: Any, **kwargs: Any) -> Any:
            if kwargs.pop("candidate_backend", "cpu") != "cpu":
                raise ValueError("frozen reference candidates support candidate_backend='cpu' only")
            return original_builder(*args, **kwargs)

        candidates.build_mass_shatter_candidates = build_reference
        backend_adapter = "Consume candidate_backend='cpu' for the legacy CPU-only signature."
    return {"modules": installed, "timing_adapter": adapter, "backend_adapter": backend_adapter}


def _enable_cuda_prototype(statistics: list[dict], backend: str) -> None:
    """Collect diagnostics on the public CUDA path or enable topology-only comparison."""
    module = importlib.import_module("fragnnet.model.mass_shatter_model")
    original = module.build_mass_shatter_candidates

    if backend not in ("cuda_topology", "cuda_full"):
        raise ValueError(f"unknown CUDA candidate prototype: {backend}")

    def build_with_cuda_prototype(*args: Any, **kwargs: Any) -> Any:
        expected_backend = "cuda" if backend == "cuda_full" else "cpu"
        if kwargs["candidate_backend"] != expected_backend:
            raise RuntimeError(f"benchmark {backend} requires public backend {expected_backend}")
        kwargs[f"_{backend}_stats"] = statistics
        if backend == "cuda_topology":
            kwargs["_cuda_topology"] = True
        return original(*args, **kwargs)

    module.build_mass_shatter_candidates = build_with_cuda_prototype


def _resolve_candidate_backend(config: dict, backend: str) -> None:
    """Make the benchmark label authoritative even when the input YAML selects CUDA."""
    if backend not in ("cpu", "cuda_topology", "cuda_full"):
        raise ValueError(f"unknown candidate benchmark backend: {backend}")
    config["mass_shatter_candidate_backend"] = "cuda" if backend == "cuda_full" else "cpu"


def _repeated_loader(
    loader: DataLoader, microbatches: int, warmup_microbatches: int, repeats: int, seed: int
) -> tuple[DataLoader, list[list[int]]]:
    """Freeze sampler indices, retaining the production loading/worker configuration."""
    generator = th.Generator().manual_seed(seed)
    for sampler in (getattr(loader.batch_sampler, "sampler", None), loader.batch_sampler):
        if sampler is not None and hasattr(sampler, "generator"):
            sampler.generator = generator
    if hasattr(loader.batch_sampler, "set_epoch"):
        loader.batch_sampler.set_epoch(0)
    fixed = [list(batch) for batch in itertools.islice(loader.batch_sampler, microbatches)]
    if len(fixed) != microbatches:
        raise ValueError("dataset has too few batches; increase --subsample-size or reduce steps")
    schedule = [fixed[index % len(fixed)] for index in range(warmup_microbatches)]
    schedule.extend(fixed * repeats)
    repeated = DataLoader(
        loader.dataset,
        batch_sampler=schedule,
        num_workers=loader.num_workers,
        collate_fn=loader.collate_fn,
        pin_memory=loader.pin_memory,
        timeout=loader.timeout,
        worker_init_fn=loader.worker_init_fn,
        multiprocessing_context=loader.multiprocessing_context,
        generator=loader.generator,
        prefetch_factor=loader.prefetch_factor,
        persistent_workers=loader.persistent_workers,
        pin_memory_device=loader.pin_memory_device,
    )
    return repeated, fixed


def _tensor_bytes(value: Any) -> int:
    """Count logical input tensor bytes, including PyG fields (not transfer traffic)."""
    if isinstance(value, th.Tensor):
        return value.numel() * value.element_size()
    if hasattr(value, "to_dict"):
        return _tensor_bytes(value.to_dict())
    if isinstance(value, dict):
        return sum(_tensor_bytes(item) for item in value.values())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_bytes(item) for item in value)
    return 0


def _synchronize(device: th.device) -> None:
    if device.type == "cuda":
        th.cuda.synchronize(device)


class _ResourceSampler:
    """Sample process-tree RSS and NVML GPU utilization without CUDA synchronization."""

    def __init__(self, device: th.device):
        self.device = device
        self.samples: list[dict[str, float | int]] = []
        self.gpu_error: str | None = None
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        process = psutil.Process()
        while not self.stop.is_set():
            sample: dict[str, float | int] = {"time": time.perf_counter()}
            processes = [process, *process.children(recursive=True)]
            rss = 0
            for child in processes:
                try:
                    rss += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue
            sample["process_tree_rss_bytes"] = rss
            if self.device.type == "cuda" and self.gpu_error is None:
                try:
                    sample["gpu_utilization_percent"] = th.cuda.utilization(self.device)
                except Exception as error:  # Optional NVML support must not abort training.
                    self.gpu_error = f"{type(error).__name__}: {error}"
            self.samples.append(sample)
            self.stop.wait(0.5)

    def summarize(self, start: float, end: float) -> dict:
        samples = [sample for sample in self.samples if start <= sample["time"] <= end]
        usage = [
            sample["gpu_utilization_percent"]
            for sample in samples
            if "gpu_utilization_percent" in sample
        ]
        return {
            "sample_count": len(samples),
            "sample_interval_seconds": 0.5,
            "peak_process_tree_rss_bytes": max(
                (sample["process_tree_rss_bytes"] for sample in samples), default=None
            ),
            "gpu_utilization_mean_percent": statistics.mean(usage) if usage else None,
            "gpu_utilization_samples_percent": usage,
            "gpu_utilization_unavailable_reason": self.gpu_error,
            "rss_note": "Sum of parent and child RSS; shared pages can be counted more than once.",
        }


def _train_window(
    model: Any,
    optimizer: th.optim.Optimizer,
    scheduler: Any,
    batches: Iterator,
    *,
    optimizer_steps: int,
    accumulation: int,
    clip_value: float | None,
    clip_algorithm: str,
    device: th.device,
    synchronize_batches: bool = False,
) -> dict:
    """Run complete accumulation groups using Lightning's loss / accumulation scaling."""
    if optimizer_steps < 1 or accumulation < 1:
        raise ValueError("optimizer_steps and accumulation must be positive")
    if clip_algorithm not in ("norm", "value"):
        raise ValueError("gradient_clip_algorithm must be norm or value")
    metrics: dict[str, list[th.Tensor]] = {key: [] for key in METRICS}
    ids, batch_sizes, latency_ms, loader_wait_ms, transfer_host_ms = [], [], [], [], []
    logical_input_bytes = 0
    if device.type == "cuda":
        th.cuda.reset_peak_memory_stats(device)
    _synchronize(device)
    start = time.perf_counter()
    for _ in range(optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        for microbatch in range(accumulation):
            batch_start = time.perf_counter()
            batch = next(batches)
            loaded = time.perf_counter()
            ids.append(batch["spec_unique_id"].tolist())
            batch_sizes.append(int(batch["batch_size"]))
            logical_input_bytes += _tensor_bytes(batch)
            transfer_start = time.perf_counter()
            batch = model.transfer_batch_to_device(batch, device, 0)
            transferred = time.perf_counter()
            result = model._common_step(batch, "train", log=False)
            (result["mean_loss"] / accumulation).backward()
            for key in METRICS:
                if key in result:
                    metrics[key].append(result[key].detach().float().mean())
            if microbatch + 1 == accumulation:
                if clip_value is not None and clip_value > 0:
                    parameters = [p for p in model.parameters() if p.grad is not None]
                    if clip_algorithm == "norm":
                        th.nn.utils.clip_grad_norm_(parameters, clip_value)
                    else:
                        th.nn.utils.clip_grad_value_(parameters, clip_value)
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
            if synchronize_batches:
                _synchronize(device)
            latency_ms.append(1000 * (time.perf_counter() - batch_start))
            loader_wait_ms.append(1000 * (loaded - batch_start))
            transfer_host_ms.append(1000 * (transferred - transfer_start))
            del result, batch
    _synchronize(device)
    end = time.perf_counter()
    values = {key: th.stack(value).cpu().tolist() for key, value in metrics.items() if value}
    if not all(math.isfinite(loss) for loss in values["mean_loss"]):
        raise ValueError("benchmark produced a nonfinite loss")
    elapsed = end - start
    return {
        "start_monotonic": start,
        "end_monotonic": end,
        "seconds": elapsed,
        "spectra": sum(batch_sizes),
        "optimizer_steps": optimizer_steps,
        "spectra_per_second": sum(batch_sizes) / elapsed,
        "optimizer_steps_per_second": optimizer_steps / elapsed,
        "spec_ids": ids,
        "batch_sizes": batch_sizes,
        "batch_latency_ms": latency_ms,
        "batch_latency_kind": "synchronized" if synchronize_batches else "host_enqueue",
        "loader_wait_ms": loader_wait_ms,
        "transfer_host_ms": transfer_host_ms,
        "logical_input_tensor_bytes": logical_input_bytes,
        "metrics_per_batch": values,
        "metrics_batch_mean": {key: statistics.mean(value) for key, value in values.items()},
        "peak_gpu_allocated_bytes": th.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else None,
        "peak_gpu_reserved_bytes": th.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else None,
    }


def _file_identity(path: Path, *, content_hash: bool) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": _sha256(path) if content_hash else None,
    }


def _source_provenance(
    revision: str | None, diff_sha256: str | None, *, cwd: Path | None = None
) -> dict[str, str]:
    """Use explicit frozen-source metadata or query the current Git checkout."""
    origins = {
        "revision": "explicit" if revision is not None else "git",
        "tracked_diff": "explicit" if diff_sha256 is not None else "git",
    }
    if revision is not None and not revision.strip():
        raise ValueError("--source-revision must not be empty")
    if diff_sha256 is not None and (
        len(diff_sha256) != 64
        or any(character not in "0123456789abcdef" for character in diff_sha256)
    ):
        raise ValueError("--source-diff-sha256 must be a lowercase 64-character SHA256 digest")
    try:
        if revision is None:
            revision = subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=cwd, stderr=subprocess.PIPE, text=True
            ).strip()
        if diff_sha256 is None:
            diff_sha256 = hashlib.sha256(
                subprocess.check_output(["git", "diff", "HEAD"], cwd=cwd, stderr=subprocess.PIPE)
            ).hexdigest()
    except (subprocess.CalledProcessError, FileNotFoundError) as error:
        raise ValueError(
            "Git provenance is unavailable in this directory. Supply --source-revision and "
            "--source-diff-sha256 captured from the frozen source checkout before cluster staging."
        ) from error
    return {
        "git_revision": revision,
        "tracked_diff_sha256": diff_sha256,
        "source_revision_origin": origins["revision"],
        "source_diff_origin": origins["tracked_diff"],
    }


def main() -> None:
    """Benchmark one variant; alternate baseline/new invocations for matched trials."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path)
    parser.add_argument(
        "--candidate-backend", choices=("cpu", "cuda_topology", "cuda_full"), default="cpu"
    )
    parser.add_argument("--source-revision")
    parser.add_argument("--source-diff-sha256")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--accumulation", type=int)
    parser.add_argument("--candidate-threads", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--subsample-size", type=int, default=2048)
    parser.add_argument("--optimizer-steps", type=int, default=4)
    parser.add_argument("--warmup-optimizer-steps", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--synchronize-batches", action="store_true")
    args = parser.parse_args()
    if (
        min(args.num_threads, args.subsample_size, args.optimizer_steps, args.repeats) < 1
        or args.warmup_optimizer_steps < 0
    ):
        parser.error("sizes/steps/repeats must be positive; warmup steps must be nonnegative")
    for value in (args.batch_size, args.accumulation, args.candidate_threads):
        if value is not None and value < 1:
            parser.error("batch size, accumulation and candidate threads must be positive")
    if args.num_workers is not None and args.num_workers < 0:
        parser.error("workers must be nonnegative")
    if args.reference_dir is not None and args.candidate_backend != "cpu":
        parser.error("--reference-dir cannot be combined with a CUDA candidate prototype")
    try:
        provenance = _source_provenance(args.source_revision, args.source_diff_sha256)
    except ValueError as error:
        parser.error(str(error))

    # 1. Install references before model imports and capture resolved configuration.
    setup_start = time.perf_counter()
    reference = _install_reference(args.reference_dir) if args.reference_dir is not None else None
    import lightning as L
    from lightning.fabric.utilities.data import _auto_add_worker_init_fn

    from fragnnet.pl_model import MassShatterPL
    from fragnnet.runner import init_dataloader, init_dataset, load_config

    cuda_candidate_statistics: list[dict] = []
    if args.candidate_backend != "cpu":
        _enable_cuda_prototype(cuda_candidate_statistics, args.candidate_backend)
    config = load_config("config/models_template", args.config, None)
    overrides = {
        "train_batch_size": args.batch_size,
        "accumulate_grad_batches": args.accumulation,
        "mass_shatter_mask_selection_threads": args.candidate_threads,
        "num_workers": args.num_workers,
    }
    config.update({key: value for key, value in overrides.items() if value is not None})
    _resolve_candidate_backend(config, args.candidate_backend)
    config["subsample_params"]["subsample_size"] = args.subsample_size
    # This benchmark reports the candidate-generation stage breakdown, so the speed trace
    # is requested here rather than left to the config's monitoring default.
    config["mass_shatter_candidate_timing_metrics"] = True
    if config["model_type"] != "mass_shatter" or not config["automatic_optimization"]:
        parser.error("requires MASShatter automatic optimization")
    if config["precision"] not in (32, "32", "32-true") or config["devices"] != 1:
        parser.error("only FP32 single-device execution is supported")
    if args.candidate_backend != "cpu" and (
        config["mass_shatter_proposal_num_mixtures"] != 1
        or not config["mass_shatter_unique_fragment_mode"]
        or config["mass_shatter_min_mask_relative_probability"] != 0.0
        or config["mass_shatter_smarts_templates"]
    ):
        parser.error(
            "CUDA prototypes require one mixture, unique mode, zero mask floors and SMARTS off"
        )
    if config["check_gradient_norm"]:
        parser.error("disable check_gradient_norm (requires Lightning trainer hooks)")
    device = (
        th.device("cuda" if th.cuda.is_available() else "cpu")
        if args.device == "auto"
        else th.device(args.device)
    )
    if device.type == "cuda" and not th.cuda.is_available():
        parser.error("CUDA requested but unavailable")
    if args.candidate_backend != "cpu" and device.type != "cuda":
        parser.error("CUDA prototypes require --device cuda or an available CUDA device")
    th.set_num_threads(args.num_threads)
    th.multiprocessing.set_sharing_strategy(config["mp_sharing_strategy"])
    th.set_float32_matmul_precision("high" if config["use_tensor_float32"] else "highest")
    th.use_deterministic_algorithms(config["deterministic"], warn_only=True)
    L.seed_everything(config["seed"], workers=True)
    (dataset,) = init_dataset(config, splits=("train",))
    L.seed_everything(config["seed"], workers=True)
    model = MassShatterPL(**config)
    weights_hash = hashlib.sha256()
    for name, value in model.state_dict().items():
        weights_hash.update(name.encode())
        weights_hash.update(value.detach().cpu().contiguous().numpy().tobytes())
    loader = init_dataloader(dataset, config)
    _auto_add_worker_init_fn(loader, rank=0)
    accumulation = config["accumulate_grad_batches"]
    loader, fixed_indices = _repeated_loader(
        loader,
        args.optimizer_steps * accumulation,
        args.warmup_optimizer_steps * accumulation,
        args.repeats,
        model.train_dl_seeds[0].item(),
    )
    model = model.to(device).train()
    optimizer_config = model.configure_optimizers()
    optimizer = optimizer_config["optimizer"]
    scheduler = optimizer_config["lr_scheduler"]["scheduler"] if config["lr_schedule"] else None
    _synchronize(device)
    setup_seconds = time.perf_counter() - setup_start
    source_paths = {
        "benchmark": Path(__file__),
        "config_template": Path("config/models_template"),
        "config": Path(args.config),
    }
    for name in (
        "fragnnet.frag.mass_shatter_core",
        "fragnnet.frag.mass_shatter_candidates",
        "fragnnet.model.mass_shatter_model",
        "fragnnet.pl_model.mass_shatter_pl",
        "fragnnet.utils.pl_utils",
        "fragnnet.runner",
    ):
        source_paths[name] = Path(importlib.import_module(name).__file__)
    if args.candidate_backend != "cpu":
        for name in (
            "mass_shatter_cuda_candidates",
            "mass_shatter_cuda_topology",
            "mass_shatter_cuda_utils",
        ):
            source_paths[name] = Path(importlib.import_module(f"fragnnet.frag.{name}").__file__)
        source_paths["cuda_topology_kernel"] = source_paths[
            "mass_shatter_cuda_topology"
        ].with_suffix(".cu")
        if args.candidate_backend == "cuda_full":
            for name in ("mass_shatter_cuda_full_candidates", "mass_shatter_cuda_selection"):
                source_paths[name] = Path(importlib.import_module(f"fragnnet.frag.{name}").__file__)
            source_paths["cuda_selection_kernel"] = source_paths[
                "mass_shatter_cuda_selection"
            ].with_suffix(".cu")
    report = {
        "label": args.label,
        "reference": reference,
        "candidate_backend": args.candidate_backend,
        "resolved_candidate_backend": config["mass_shatter_candidate_backend"],
        "cuda_candidate_batches": cuda_candidate_statistics,
        "cuda_candidate_warmup_batches": args.warmup_optimizer_steps * accumulation,
        "command": sys.argv,
        "config": config,
        "source_files": {
            name: _file_identity(path, content_hash=True) for name, path in source_paths.items()
        },
        **provenance,
        "data_files": {
            name: _file_identity(Path(config[name]), content_hash=False)
            for name in ("spec_fp", "mol_fp")
        },
        "dataset_size": len(dataset),
        "fixed_dataset_indices": fixed_indices,
        "initial_state_sha256": weights_hash.hexdigest(),
        "python": sys.version,
        "torch_version": th.__version__,
        "lightning_version": L.__version__,
        "platform": platform.platform(),
        "device": str(device),
        "gpu": th.cuda.get_device_name(device) if device.type == "cuda" else None,
        "cuda_version": th.version.cuda,
        "num_threads": th.get_num_threads(),
        "num_interop_threads": th.get_num_interop_threads(),
        "num_workers": loader.num_workers,
        "pin_memory": loader.pin_memory,
        "worker_init_fn": repr(loader.worker_init_fn),
        "prefetch_factor": loader.prefetch_factor,
        "warmup_optimizer_steps": args.warmup_optimizer_steps,
        "setup_seconds": setup_seconds,
        "cuda_candidate_cold_note": "Lazy native compilation/load occurs in the first candidate call, normally warmup; it is not part of setup_seconds. Native diagnostics record its timing when available.",
        "scope": "Loading, device transfer, loss, backward, accumulation, clipping, optimizer and step scheduler. Excludes logging/validation/checkpoints. Fixed windows repeat in one iterator; weights continue updating.",
        "timing_note": "Candidate phase metrics are existing host timers, not isolated CUDA durations; host enqueue latencies are not completed GPU batch latencies. Input bytes are logical tensor sizes, not measured PCIe traffic; candidate transfer traffic is not instrumented.",
        "data_identity_note": "Dataset files identified by path/size/mtime; contents not hashed. Ordered spectrum IDs are recorded per window.",
        "windows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")

    # 2. Keep one production worker pool alive through warmup and repeated windows.
    resources = _ResourceSampler(device)
    iterator_start = time.perf_counter()
    batches = iter(loader)
    report["iterator_setup_seconds"] = time.perf_counter() - iterator_start
    resources.thread.start()
    window_args = {
        "accumulation": accumulation,
        "clip_value": config["gradient_clip_val"],
        "clip_algorithm": config["gradient_clip_algorithm"],
        "device": device,
        "synchronize_batches": args.synchronize_batches,
    }
    try:
        if args.warmup_optimizer_steps:
            report["warmup"] = _train_window(
                model,
                optimizer,
                scheduler,
                batches,
                optimizer_steps=args.warmup_optimizer_steps,
                **window_args,
            )
        for index in range(args.repeats):
            window = _train_window(
                model,
                optimizer,
                scheduler,
                batches,
                optimizer_steps=args.optimizer_steps,
                **window_args,
            )
            window["resources"] = resources.summarize(
                window["start_monotonic"], window["end_monotonic"]
            )
            if report["windows"] and window["spec_ids"] != report["windows"][0]["spec_ids"]:
                raise RuntimeError("repeated windows produced different spectrum identities")
            report["windows"].append(window)
            report["spectra_per_second"] = sum(item["spectra"] for item in report["windows"]) / sum(
                item["seconds"] for item in report["windows"]
            )
            LOGGER.info(
                "%s window %d: %.2f spectra/s; %.2f optimizer steps/s",
                args.label,
                index + 1,
                window["spectra_per_second"],
                window["optimizer_steps_per_second"],
            )
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    finally:
        resources.stop.set()
        resources.thread.join(timeout=2)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
