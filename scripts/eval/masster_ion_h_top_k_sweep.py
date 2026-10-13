"""Evaluate one MASSter checkpoint with fragment and ion/H pruning rules.

The model weights, fragment candidates, and dataset stay fixed. The audit compares a
dense reference, MAP-relative fragment thresholds, stable support-ranked top-k channels
per fragment, and spectrum-relative thresholds on final joint fragment/ion/H intensity.
Absolute support-probability thresholds are available with ``--support_absolute_thresholds``.

Example:
    python scripts/eval/masster_ion_h_top_k_sweep.py \
        --model_ckpt saved_ckpts/RUN/checkpoints/best.ckpt \
        --custom_fp config/masster/merged/channel_selection/CONFIG.yml \
        --wandb_mode online
"""

from __future__ import annotations

import argparse
import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any

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

DEFAULT_TOP_KS = tuple(range(10))
DEFAULT_FRAGMENT_RELATIVE_THRESHOLDS = (0.0,)
DEFAULT_JOINT_RELATIVE_THRESHOLDS = (0.001, 0.003, 0.01, 0.03, 0.1)
BASE_METRICS = (
    "loss",
    "primary_loss",
    "candidate_wrecall",
    "fragment_precision",
    "fragment_weighted_precision",
    "fragment_count_before_cutoff",
    "fragment_count_after_cutoff",
    "retained_fragment_probability",
    "fragment_cutoff_pruned_fraction",
    "valid_ion_h_count",
    "retained_ion_h_count",
    "retained_ion_h_probability",
    "retained_joint_channel_intensity_fraction",
    "ion_h_cutoff_pruned_fraction",
    "valid_ion_h_per_fragment_mean",
    "attributed_ion_h_per_fragment_mean",
    "ion_h_supported_fragment_fraction",
    "required_ion_h_top_k_95_mean",
    "required_ion_h_top_k_95_p90",
    "required_ion_h_top_k_95_max",
    "required_ion_h_top_k_99_mean",
    "required_ion_h_top_k_99_p90",
    "required_ion_h_top_k_99_max",
)


def validate_top_ks(top_ks: list[int]) -> list[int]:
    """Validate top-k controls while preserving their requested order."""
    if not top_ks:
        raise ValueError("At least one ion/H top-k value is required.")
    if any(isinstance(top_k, bool) or top_k < 0 for top_k in top_ks):
        raise ValueError(f"Ion/H top-k values must be nonnegative integers: {top_ks}")
    if len(set(top_ks)) != len(top_ks):
        raise ValueError(f"Ion/H top-k values must be unique: {top_ks}")
    if 0 not in top_ks:
        raise ValueError("Ion/H top-k values must include the dense reference 0")
    return top_ks


def validate_fragment_relative_thresholds(thresholds: list[float]) -> list[float]:
    """Validate MAP-relative fragment thresholds with an unpruned reference."""
    if not thresholds:
        raise ValueError("At least one fragment-relative threshold is required.")
    if any(isinstance(threshold, bool) or not 0.0 <= threshold <= 1.0 for threshold in thresholds):
        raise ValueError(f"Fragment-relative thresholds must be in [0, 1]: {thresholds}")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError(f"Fragment-relative thresholds must be unique: {thresholds}")
    if 0.0 not in thresholds:
        raise ValueError("Fragment-relative thresholds must include the unpruned reference 0")
    return thresholds


def validate_joint_relative_thresholds(thresholds: list[float]) -> list[float]:
    """Validate spectrum-relative joint channel thresholds."""
    if any(isinstance(threshold, bool) or not 0.0 < threshold <= 1.0 for threshold in thresholds):
        raise ValueError(f"Joint relative thresholds must be in (0, 1]: {thresholds}")
    if len(set(thresholds)) != len(thresholds):
        raise ValueError(f"Joint relative thresholds must be unique: {thresholds}")
    return thresholds


def build_selections(
    fragment_relative_thresholds: list[float],
    top_ks: list[int],
    joint_relative_thresholds: list[float],
) -> list[tuple[str, float, int, float]]:
    """Build fragment-threshold combinations with each channel pruning rule."""
    selections: list[tuple[str, float, int, float]] = []
    for fragment_threshold in fragment_relative_thresholds:
        fragment_label = (
            "" if fragment_threshold == 0.0 else f"fragment_relative_{fragment_threshold:g}"
        )
        for top_k in top_ks:
            channel_label = "" if top_k == 0 else f"top_k_{top_k}"
            label = "_".join(part for part in (fragment_label, channel_label) if part) or "dense"
            selections.append((label, fragment_threshold, top_k, 0.0))
        for joint_threshold in joint_relative_thresholds:
            channel_label = f"joint_relative_{joint_threshold:g}"
            label = "_".join(part for part in (fragment_label, channel_label) if part)
            selections.append((label, fragment_threshold, 0, joint_threshold))
    return selections


def build_support_selections(
    fragment_relative_thresholds: list[float],
    thresholds: list[float],
) -> list[tuple[str, float, int, float]]:
    """Build support-only threshold rules, without top-k or intensity pruning."""
    validate_joint_relative_thresholds(thresholds)
    return [
        (
            (f"fragment_relative_{fragment:g}_" if fragment else "")
            + f"support_absolute_{threshold:g}",
            fragment,
            0,
            threshold,
        )
        for fragment in fragment_relative_thresholds
        for threshold in thresholds
    ]


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
        return 1.0 / (spec_per_group.to(dtype=th.float64) * group_per_mol.to(dtype=th.float64))
    raise ValueError(f"Unsupported eval_sample_weight: {mode}")


@dataclass
class WeightedMetricAccumulator:
    """Accumulate sample-weighted metric means without retaining predictions."""

    metric_names: tuple[str, ...]
    weighted_sums: dict[str, float] = field(default_factory=dict)
    total_weight: float = 0.0
    sample_count: int = 0

    def update(self, batch_results: dict[str, Any], weight_mode: str) -> None:
        weights = (
            sample_weights(
                weight_mode,
                batch_results["spec_per_group"],
                batch_results["spec_per_mol"],
                batch_results["group_per_mol"],
            )
            .detach()
            .reshape(-1)
        )
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
    parser.add_argument("--custom_fp", required=True)
    checkpoint_group = parser.add_mutually_exclusive_group(required=True)
    checkpoint_group.add_argument("--model_ckpt", help="Local checkpoint path.")
    checkpoint_group.add_argument("--wandb_run_id", help="Run ID containing a checkpoint.")
    parser.add_argument("--model_save_dp", default="saved_ckpts")
    parser.add_argument(
        "--fragment_relative_thresholds",
        type=float,
        nargs="+",
        default=DEFAULT_FRAGMENT_RELATIVE_THRESHOLDS,
    )
    parser.add_argument("--ion_h_top_ks", type=int, nargs="+", default=DEFAULT_TOP_KS)
    parser.add_argument(
        "--support_absolute_thresholds",
        type=float,
        nargs="*",
        default=[],
        help="Absolute sigmoid support floors in (0, 1]; retain one fallback channel per spectrum.",
    )
    parser.add_argument(
        "--joint_relative_thresholds",
        type=float,
        nargs="*",
        default=DEFAULT_JOINT_RELATIVE_THRESHOLDS,
    )
    parser.add_argument(
        "--eval_split",
        choices=["train", "val", "test", "secondary"],
        default="val",
    )
    parser.add_argument("--eval_batch_size", type=int, default=16)
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
        default="output/eval/mass_shatter_ion_h_top_k_sweep.csv",
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
    """Return channel diagnostics plus configured binned cosine metrics."""
    auxiliary_scores = set(config_d["auxiliary_scores"])
    match_names = tuple(
        name
        for name in ("precision", "recall", "wprecision", "wrecall")
        if name in auxiliary_scores
    )
    cosine_names = (
        tuple(f"cos_sim_{resolution}" for resolution in config_d["eval_mz_bin_res"])
        if "cos_sim" in auxiliary_scores
        else ()
    )
    return BASE_METRICS + match_names + cosine_names


def evaluate_selection(
    model: th.nn.Module,
    dataloader: th.utils.data.DataLoader,
    device: th.device,
    split: str,
    metric_names_to_collect: tuple[str, ...],
    weight_mode: str,
    batch_cutoff: int,
) -> dict[str, float | int]:
    """Evaluate one channel-selection rule with validation-equivalent metrics."""
    accumulator = WeightedMetricAccumulator(metric_names_to_collect)
    skipped_batches = 0
    if device.type == "cuda":
        th.cuda.reset_peak_memory_stats(device)
        th.cuda.synchronize(device)
    start_time = time.perf_counter()

    with th.inference_mode():
        for batch_index, batch in enumerate(
            tqdm(dataloader, desc="ion/H selection audit", unit="batch")
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
    label: str,
    fragment_relative_threshold: float,
    top_k: int,
    joint_relative_threshold: float,
) -> Any | None:
    """Create one W&B run per channel-selection rule, or disable logging."""
    if args.wandb_mode == "disabled":
        return None
    import wandb

    group = args.wandb_group or f"{config_d['wandb_group']}_checkpoint_ion_h_selection"
    prefix = args.wandb_name_prefix or f"{config_d['wandb_name']}_checkpoint"
    return wandb.init(
        project=config_d["wandb_project"],
        entity=config_d["wandb_entity"],
        group=group,
        name=f"{prefix}_{label}",
        tags=[*config_d["wandb_tags"], "checkpoint_sweep", label],
        mode=args.wandb_mode,
        config={
            "source_checkpoint": os.path.abspath(checkpoint_fp),
            "source_wandb_run_id": args.wandb_run_id,
            "evaluation_fragment_relative_threshold": fragment_relative_threshold,
            "evaluation_ion_h_top_k": top_k,
            "evaluation_joint_channel_relative_threshold": joint_relative_threshold,
            "evaluation_channel_selection": label,
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
    fragment_relative_thresholds = validate_fragment_relative_thresholds(
        list(args.fragment_relative_thresholds)
    )
    top_ks = validate_top_ks(list(args.ion_h_top_ks))
    joint_relative_thresholds = validate_joint_relative_thresholds(
        list(args.joint_relative_thresholds)
    )
    support_selections = build_support_selections(
        fragment_relative_thresholds, list(args.support_absolute_thresholds)
    )

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
    if not config_d["masster_channel_diagnostics"]:
        raise ValueError("The channel audit requires masster_channel_diagnostics=True")
    if config_d["masster_min_ion_h_relative_probability"] != 0.0:
        raise ValueError("Disable the ion/H probability floor for an isolated channel audit")
    if config_d["masster_min_ion_h_absolute_probability"] != 0.0:
        raise ValueError("Disable the absolute ion/H support floor for an isolated channel audit")
    if support_selections and not (
        config_d["masster_separate_ion_h_support_head"]
        and config_d["masster_ion_h_support_binary_logits"]
    ):
        raise ValueError("Support thresholds require a separate binary support head")
    # candidate_wrecall and the channel support summaries are this audit's reported metrics,
    # so they are requested here rather than left to the config's monitoring defaults.
    config_d["masster_candidate_coverage_metrics"] = True
    config_d["masster_channel_support_metrics"] = True
    config_d["auxiliary_scores"] = list(
        dict.fromkeys(
            [
                *config_d["auxiliary_scores"],
                "precision",
                "recall",
                "wprecision",
                "wrecall",
                "cos_sim",
            ]
        )
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

    # 2. Reload identical weights for each fragment/channel selection rule.
    selections = build_selections(
        fragment_relative_thresholds,
        top_ks,
        joint_relative_thresholds,
    )
    support_threshold_by_label = {row[0]: row[3] for row in support_selections}
    selections.extend(
        (label, fragment, top_k, 0.0) for label, fragment, top_k, _threshold in support_selections
    )
    rows: list[dict[str, Any]] = []
    for label, fragment_relative_threshold, top_k, joint_relative_threshold in selections:
        support_absolute_threshold = support_threshold_by_label.get(label, 0.0)
        logger.info("Evaluating checkpoint channel selection: %s", label)
        seed_everything(config_d["seed"], workers=True)
        selection_config = {
            **config_d,
            "masster_min_fragment_relative_probability": fragment_relative_threshold,
            "masster_min_ion_h_absolute_probability": support_absolute_threshold,
        }
        model = init_model_from_ckpt(checkpoint_fp, selection_config, device)
        model.model.set_evaluation_ion_h_top_k(top_k)
        model.model.set_evaluation_joint_channel_relative_threshold(joint_relative_threshold)
        wandb_run = None
        try:
            wandb_run = init_wandb_run(
                args,
                config_d,
                checkpoint_fp,
                label,
                fragment_relative_threshold,
                top_k,
                joint_relative_threshold,
            )
            if wandb_run is not None:
                wandb_run.config.update(
                    {
                        "evaluation_support_absolute_threshold": support_absolute_threshold,
                    }
                )
            result = evaluate_selection(
                model=model,
                dataloader=dataloader,
                device=device,
                split=args.eval_split,
                metric_names_to_collect=metrics,
                weight_mode=config_d["eval_sample_weight"],
                batch_cutoff=args.batch_cutoff,
            )
            row = {
                "evaluation_channel_selection": label,
                "evaluation_fragment_relative_threshold": fragment_relative_threshold,
                "evaluation_ion_h_top_k": top_k,
                "evaluation_joint_channel_relative_threshold": joint_relative_threshold,
                "evaluation_support_absolute_threshold": support_absolute_threshold,
                "eval_split": args.eval_split,
                "checkpoint": os.path.abspath(checkpoint_fp),
                **result,
            }
            rows.append(row)
            save_rows(rows, args.output_fp)
            logger.info("Selection %s results: %s", label, result)
            if wandb_run is not None:
                wandb_run.log({f"{args.eval_split}/{key}": value for key, value in result.items()})
        except ValueError as error:
            logger.exception("Selection %s failed", label)
            rows.append(
                {
                    "evaluation_channel_selection": label,
                    "evaluation_fragment_relative_threshold": fragment_relative_threshold,
                    "evaluation_ion_h_top_k": top_k,
                    "evaluation_joint_channel_relative_threshold": joint_relative_threshold,
                    "evaluation_support_absolute_threshold": support_absolute_threshold,
                    "eval_split": args.eval_split,
                    "checkpoint": os.path.abspath(checkpoint_fp),
                    "status": "failed",
                    "error": str(error),
                }
            )
            save_rows(rows, args.output_fp)
        finally:
            if wandb_run is not None:
                wandb_run.finish()
            model.cpu()
            del model
            gc.collect()
            if device.type == "cuda":
                th.cuda.empty_cache()

    # 3. Leave a compact dense-versus-pruned comparison table on disk.
    logger.info("Saved %d sweep rows to %s", len(rows), args.output_fp)


if __name__ == "__main__":
    main()
