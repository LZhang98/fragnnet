"""Evaluate one MASSter checkpoint at several candidate-mask budgets.

The model weights and dataset stay fixed. Only the active selected-mask limit or
unique-fragment target changes, so the output isolates the quality/runtime/memory
tradeoff of a larger candidate set.

Example:
    python scripts/eval/masster_mask_budget_sweep.py \
        --model_ckpt path/to/model.ckpt \
        --wandb_mode online
"""

from __future__ import annotations

import argparse
import copy
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

# Avoid allocator fragmentation during long sequential checkpoint evaluations.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import pandas as pd
import torch as th
from lightning import seed_everything
from tqdm import tqdm

from fragnnet.inference.checkpoint import (
    apply_inference_config,
    get_ckpt_path,
    init_model_from_ckpt,
    setup_device,
)
from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.setup_utils import get_dataset_cls

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_MASK_BUDGETS = (100, 256, 512)
BASE_METRICS = (
    "loss",
    "candidate_wrecall",
    "recall",
    "wrecall",
    "retained_mask_probability",
    "selected_mask_count",
    "component_pathway_count",
)


def validate_mask_budgets(mask_budgets: list[int]) -> list[int]:
    """Validate mask budgets while preserving their requested order."""
    if not mask_budgets:
        raise ValueError("At least one mask budget is required.")
    if any(mask_budget <= 0 for mask_budget in mask_budgets):
        raise ValueError(f"Mask budgets must be positive: {mask_budgets}")
    if len(set(mask_budgets)) != len(mask_budgets):
        raise ValueError(f"Mask budgets must be unique: {mask_budgets}")
    return mask_budgets


def sample_weights(
    mode: str,
    spec_per_group: th.Tensor,
    spec_per_mol: th.Tensor,
    group_per_mol: th.Tensor,
) -> th.Tensor:
    """Match ``SpectrumPL`` sample weighting for epoch-level metric means."""
    if mode == "none":
        return th.ones_like(spec_per_group, dtype=th.float64)
    if mode == "group":
        return 1.0 / spec_per_group.to(dtype=th.float64)
    if mode == "mol":
        return 1.0 / spec_per_mol.to(dtype=th.float64)
    if mode == "group_mol":
        return 1.0 / (
            spec_per_group.to(dtype=th.float64) * group_per_mol.to(dtype=th.float64)
        )
    raise ValueError(f"Unsupported eval_sample_weight: {mode}")


@dataclass
class WeightedMetricAccumulator:
    """Accumulate sample-weighted metric means without retaining predictions."""

    metric_names: tuple[str, ...]
    weighted_sums: dict[str, float] = field(default_factory=dict)
    total_weight: float = 0.0
    sample_count: int = 0

    def update(self, batch_results: dict[str, Any], weight_mode: str) -> None:
        weights = sample_weights(
            weight_mode,
            batch_results["spec_per_group"],
            batch_results["spec_per_mol"],
            batch_results["group_per_mol"],
        ).detach().reshape(-1)
        batch_size = weights.numel()
        for metric_name in self.metric_names:
            values = batch_results[metric_name].detach().to(dtype=th.float64).reshape(-1)
            if values.numel() != batch_size:
                raise ValueError(
                    f"Metric {metric_name} has {values.numel()} values for a "
                    f"{batch_size}-sample batch."
                )
            weighted_sum = th.sum(values * weights.to(device=values.device)).cpu().item()
            self.weighted_sums[metric_name] = (
                self.weighted_sums.get(metric_name, 0.0) + weighted_sum
            )
        self.total_weight += weights.sum().cpu().item()
        self.sample_count += batch_size

    def means(self) -> dict[str, float]:
        if self.total_weight <= 0:
            raise ValueError("No evaluation samples were accumulated.")
        return {
            metric_name: weighted_sum / self.total_weight
            for metric_name, weighted_sum in self.weighted_sums.items()
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template_fp", default="config")
    parser.add_argument(
        "--custom_fp",
        default="config/masster/nist20_merged_inchikey/s0.yml",
    )
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--model_ckpt", help="Local checkpoint path.")
    checkpoint_group.add_argument("--wandb_run_id", help="Run ID containing a checkpoint.")
    parser.add_argument("--model_save_dp", default="saved_ckpts")
    parser.add_argument("--mask_budgets", type=int, nargs="+", default=DEFAULT_MASK_BUDGETS)
    parser.add_argument(
        "--eval_split",
        choices=["train", "val", "test", "secondary"],
        default="val",
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=1,
        help="Evaluation batch size. Keep at 1 for memory-safe 256/512-mask evaluation.",
    )
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument(
        "--batch_cutoff",
        type=int,
        default=-1,
        help="Stop after this many batches; -1 evaluates the full split.",
    )
    parser.add_argument("--device", default=None, help="For example cpu or cuda:0.")
    parser.add_argument("--override", nargs="*", default=[])
    parser.add_argument(
        "--output_fp",
        default="output/eval/mass_shatter_mask_budget_sweep.csv",
    )
    parser.add_argument(
        "--wandb_mode",
        choices=["online", "offline", "disabled"],
        default="online",
    )
    parser.add_argument("--wandb_group", default=None)
    parser.add_argument("--wandb_name_prefix", default=None)
    return parser.parse_args()


def metric_names(config_d: dict[str, Any]) -> tuple[str, ...]:
    """Return the fixed diagnostics plus configured binned cosine metrics."""
    cosine_names = tuple(f"cos_sim_{resolution}" for resolution in config_d["eval_mz_bin_res"])
    return BASE_METRICS + cosine_names


def evaluate_budget(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    split: str,
    metric_names_to_collect: tuple[str, ...],
    weight_mode: str,
    batch_cutoff: int,
) -> dict[str, float | int]:
    """Evaluate one budget with the same loss and auxiliary metrics as validation."""
    accumulator = WeightedMetricAccumulator(metric_names_to_collect)
    skipped_batches = 0
    if device.type == "cuda":
        th.cuda.reset_peak_memory_stats(device)
        th.cuda.synchronize(device)
    start_time = time.perf_counter()

    with th.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(dataloader, desc="checkpoint sweep", unit="batch")
        ):
            if 0 <= batch_cutoff <= batch_index:
                break
            try:
                batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
                batch_results = model._common_step(batch, split=split, log=False)
            except ValueError as error:
                skipped_batches += 1
                logger.warning("Skipping invalid evaluation batch %d: %s", batch_index, error)
                continue
            accumulator.update(batch_results, weight_mode)

    if device.type == "cuda":
        th.cuda.synchronize(device)
    elapsed_seconds = time.perf_counter() - start_time
    results: dict[str, float | int] = accumulator.means()
    results.update(
        {
            "num_spectra": accumulator.sample_count,
            "num_skipped_batches": skipped_batches,
            "elapsed_seconds": elapsed_seconds,
            "spectra_per_second": accumulator.sample_count / elapsed_seconds,
            "gpu_peak_memory_gb": (
                th.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else 0.0
            ),
        }
    )
    return results


def init_wandb_run(
    args: argparse.Namespace,
    config_d: dict[str, Any],
    checkpoint_fp: str,
    mask_budget: int,
    budget_key: str,
) -> Any | None:
    """Create one W&B run per budget, or return None when logging is disabled."""
    if args.wandb_mode == "disabled":
        return None
    import wandb

    group = args.wandb_group or f"{config_d['wandb_group']}_checkpoint_mask_sweep"
    prefix = args.wandb_name_prefix or f"{config_d['wandb_name']}_checkpoint"
    return wandb.init(
        project=config_d["wandb_project"],
        entity=config_d["wandb_entity"],
        group=group,
        name=f"{prefix}_masks_{mask_budget}",
        tags=[
            *[tag for tag in config_d["wandb_tags"] if not tag.startswith("topk_")],
            "checkpoint_sweep",
            f"masks_{mask_budget}",
        ],
        mode=args.wandb_mode,
        config={
            "source_checkpoint": os.path.abspath(checkpoint_fp),
            "source_wandb_run_id": args.wandb_run_id,
            budget_key: mask_budget,
            "eval_split": args.eval_split,
            "batch_cutoff": args.batch_cutoff,
        },
        reinit="finish_previous",
    )


def save_rows(rows: list[dict[str, Any]], output_fp: str) -> None:
    os.makedirs(os.path.dirname(output_fp) or ".", exist_ok=True)
    pd.DataFrame(rows).to_csv(output_fp, index=False)


def main() -> None:
    args = parse_args()
    mask_budgets = validate_mask_budgets(list(args.mask_budgets))

    # 1. Resolve the immutable checkpoint and evaluation data.
    checkpoint_fp = get_ckpt_path(
        wandb_run_id=args.wandb_run_id,
        model_ckpt=args.model_ckpt,
        model_save_dp=args.model_save_dp,
        use_cached=True,
    )
    config_d = load_config(args.template_fp, args.custom_fp, args.override)
    if config_d["model_type"] != "masster":
        raise ValueError(f"Expected model_type=masster, got {config_d['model_type']}")
    # candidate_wrecall is the candidate ceiling this sweep exists to compare, so it is
    # requested here rather than left to the config's monitoring default.
    config_d["masster_candidate_coverage_metrics"] = True
    budget_key = (
        "masster_unique_fragment_target"
        if config_d["masster_unique_fragment_mode"]
        else "masster_selected_mask_limit"
    )
    config_d = apply_inference_config(
        config_d,
        disable_preproc=True,
        eval_batch_size=args.eval_batch_size,
        num_workers=args.num_workers,
    )
    seed_everything(config_d["seed"], workers=True)

    device = setup_device(config_d, device=args.device)
    dataset_cls = get_dataset_cls(config_d["model_type"])
    dataset = dataset_cls(split=args.eval_split, **config_d)
    dataloader = init_dataloader(dataset, config_d)
    metrics = metric_names(config_d)

    # 2. Reload identical weights for each candidate-mask budget and evaluate.
    rows: list[dict[str, Any]] = []
    for mask_budget in mask_budgets:
        budget_config = copy.deepcopy(config_d)
        budget_config[budget_key] = mask_budget
        # The vectorized sparse loss otherwise builds cross-sample peak-pair matrices.
        budget_config["loss_batch_size"] = 1
        logger.info("Evaluating checkpoint with %s=%d", budget_key, mask_budget)
        model = init_model_from_ckpt(checkpoint_fp, budget_config, device)
        wandb_run = None
        try:
            wandb_run = init_wandb_run(
                args,
                budget_config,
                checkpoint_fp,
                mask_budget,
                budget_key,
            )
            result = evaluate_budget(
                model=model,
                dataloader=dataloader,
                device=device,
                split=args.eval_split,
                metric_names_to_collect=metrics,
                weight_mode=budget_config["eval_sample_weight"],
                batch_cutoff=args.batch_cutoff,
            )
            row = {
                budget_key: mask_budget,
                "eval_split": args.eval_split,
                "checkpoint": os.path.abspath(checkpoint_fp),
                **result,
            }
            rows.append(row)
            save_rows(rows, args.output_fp)
            logger.info("Budget %d results: %s", mask_budget, result)
            if wandb_run is not None:
                wandb_run.log({f"{args.eval_split}/{key}": value for key, value in result.items()})
        finally:
            if wandb_run is not None:
                wandb_run.finish()
            model.cpu()
            del model
            gc.collect()
            if device.type == "cuda":
                th.cuda.empty_cache()

    # 3. Leave a compact comparison table on disk.
    logger.info("Saved %d sweep rows to %s", len(rows), args.output_fp)


if __name__ == "__main__":
    main()
