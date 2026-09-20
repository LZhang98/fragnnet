"""Compare MASShatter forward/backward cost with and without optional monitoring.

Uses fixed real training batches and freshly initialized weights, without optimizer
updates or W&B logging. Dataset loading, device transfer and validation are untimed.
"""

import argparse
import copy
import gc
import json
import logging
import math
import statistics
import time
from pathlib import Path

import lightning as L
import torch as th

from fragnnet.pl_model import MassShatterPL
from fragnnet.runner import init_dataloader, init_dataset, load_config

LOGGER = logging.getLogger(__name__)


def main() -> None:
    """Measure identical batches, preserving the configured scientific losses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--subsample-size", type=int, default=1024)
    parser.add_argument("--batches", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--num-threads", type=int, default=2)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if min(args.subsample_size, args.batches, args.steps, args.num_threads) <= 0 or args.warmup < 0:
        parser.error("sizes, steps and threads must be positive; warmup must be nonnegative")

    # 1. Load and seed the real pipeline, sampling only for this in-memory benchmark.
    config = load_config("config/models_template", args.config, None)
    if config["model_type"] != "mass_shatter":
        parser.error("the config must select mass_shatter")
    if config["precision"] not in (32, "32", "32-true"):
        parser.error("this benchmark implements float32 execution only")
    if any(
        config[key] != 0
        for key in (
            "mass_shatter_base_ce_weight",
            "mass_shatter_ion_h_support_loss_weight",
            "mass_shatter_ion_mode_proposal_loss_weight",
        )
    ):
        parser.error("this comparison requires monitoring-only base CE and channel attribution")
    if args.batch_size is not None:
        if args.batch_size <= 0:
            parser.error("batch size must be positive")
        config["train_batch_size"] = args.batch_size
    config.update(num_workers=0, share_memory=False, pl_enable_progress_bar=False)
    # This benchmark reports candidate_total_batch_ms, so the speed trace is requested here
    # rather than left to the config's monitoring default.
    config["mass_shatter_candidate_timing_metrics"] = True
    config["subsample_params"]["subsample_size"] = args.subsample_size
    device = (
        th.device("cuda" if th.cuda.is_available() else "cpu")
        if args.device == "auto"
        else th.device(args.device)
    )
    th.set_num_threads(args.num_threads)
    th.set_float32_matmul_precision("high" if config["use_tensor_float32"] else "highest")
    th.use_deterministic_algorithms(config["deterministic"], warn_only=True)
    L.seed_everything(config["seed"], workers=True)
    (dataset,) = init_dataset(config, splits=("train",))
    loader = iter(init_dataloader(dataset, config))
    batches = []
    for _ in range(args.batches):
        try:
            batches.append(next(loader))
        except StopIteration:
            parser.error("the subsample contains fewer batches than requested")
    report = {
        "config": config,
        "device": str(device),
        "gpu": th.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": th.__version__,
        "num_threads": args.num_threads,
        "warmup": args.warmup,
        "steps": args.steps,
        "spec_ids": [batch["spec_unique_id"].tolist() for batch in batches],
        "variants": [],
    }
    reference_losses = None

    # 2. Reseed initialization and every forward so variants share weights and dropout.
    for label, overrides in (
        ("configured", {}),
        (
            "monitoring_off",
            {
                "mass_shatter_channel_diagnostics": False,
                "mass_shatter_log_base_ce": False,
            },
        ),
    ):
        th.manual_seed(config["seed"])
        model = MassShatterPL(**(config | overrides)).to(device).train()
        device_batches = [
            model.transfer_batch_to_device(copy.deepcopy(batch), device, 0) for batch in batches
        ]
        durations, losses, candidate_times = [], [], []
        if device.type == "cuda":
            th.cuda.reset_peak_memory_stats(device)
        for index in range(args.warmup + args.steps):
            th.manual_seed(config["seed"] + index)
            model.zero_grad(set_to_none=True)
            if device.type == "cuda":
                th.cuda.synchronize(device)
            start = time.perf_counter()
            result = model._common_step(device_batches[index % len(batches)], "train", log=False)
            result["mean_loss"].backward()
            if device.type == "cuda":
                th.cuda.synchronize(device)
            elapsed_ms = 1000 * (time.perf_counter() - start)
            if index >= args.warmup:
                durations.append(elapsed_ms)
                losses.append(result["mean_loss"].item())
                candidate_times.append(result["candidate_total_batch_ms"].mean().item())
            del result
        if not all(math.isfinite(loss) for loss in losses):
            raise ValueError("benchmark produced a non-finite training loss")
        if reference_losses is None:
            reference_losses = losses
        measurement = {
            "variant": label,
            "overrides": overrides,
            "samples_ms": durations,
            "median_ms": statistics.median(durations),
            "spectra_per_second": (
                sum(
                    int(batches[index % len(batches)]["batch_size"])
                    for index in range(args.warmup, args.warmup + args.steps)
                )
                / (sum(durations) / 1000.0)
            ),
            "candidate_mean_ms": statistics.mean(candidate_times),
            "max_loss_delta": max(
                abs(x - y) for x, y in zip(losses, reference_losses, strict=True)
            ),
            "peak_memory_bytes": (
                th.cuda.max_memory_allocated(device) if device.type == "cuda" else None
            ),
        }
        report["variants"].append(measurement)
        LOGGER.info("%s", measurement)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        del model, device_batches
        gc.collect()
        if device.type == "cuda":
            th.cuda.empty_cache()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
