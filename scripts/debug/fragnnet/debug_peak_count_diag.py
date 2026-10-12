"""Diagnose FraGNNet peak over-prediction.

Compares predicted vs. ground-truth peak counts at multiple min_intensity thresholds,
reporting cos_hun and peak count statistics per threshold. Use the output to decide
whether over-prediction is fixable at inference time (raise --min_intensity) or
requires training-time sparsity regularization (positive formula_entropy_weight).

The threshold column maps directly to the --min_intensity argument in run_inference.py
(0-1000 scale, where 1.0 = 0.1% of base peak).

Usage:
    python scripts/debug/fragnnet/debug_peak_count_diag.py \\
        --model_ckpt saved_ckpts/model.ckpt \\
        --batch_cutoff 200

    # With a W&B checkpoint
    python scripts/debug/fragnnet/debug_peak_count_diag.py \\
        --wandb_run_id abc123 \\
        --batch_cutoff 500 --split val
"""

import argparse
import logging

import numpy as np
import torch as th

from fragnnet.loss.spectrum_distance import sparse_cosine_distance_hungarian
from fragnnet.models.fragnnet.spec_mol_frag_dataset import SpecMolFragDataset
from fragnnet.runner import init_dataloader
from fragnnet.utils.misc_utils import scatter_reduce, to_device
from fragnnet.utils.script_utils import (
    apply_inference_config,
    get_ckpt_path,
    get_config_from_ckpt,
    init_model_from_ckpt,
    setup_device,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

THRESHOLDS = [0.0, 1.0, 5.0, 10.0, 20.0, 50.0]
EPS = 1e-12


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Diagnose FraGNNet peak over-prediction")
    parser.add_argument(
        "--template_fp",
        type=str,
        default="config",
        help="Path to template config file",
    )
    parser.add_argument(
        "--custom_fp",
        type=str,
        default=None,
        help="Optional custom config to overlay on checkpoint config",
    )
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default=None,
        help="W&B run ID to download checkpoint from",
    )
    parser.add_argument(
        "--model_save_dp",
        type=str,
        default="saved_ckpts",
        help="Directory to cache downloaded W&B checkpoints",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Path to local checkpoint file",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--batch_cutoff",
        type=int,
        default=-1,
        help="Max number of batches to process (-1 = all)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device string (e.g. 'cpu', 'cuda:0'); auto-detected if None",
    )
    return parser.parse_args()


def _pred_ints_0_1000(model, pred_logprobs: th.Tensor, pred_batch_idxs: th.Tensor) -> th.Tensor:
    """Convert raw model log-softmax probs to 0-1000 scale.

    Matches exactly what run_inference.py does before applying --min_intensity.
    """
    raw_probs = th.exp(pred_logprobs)
    return model.ints_untransform_func(raw_probs, pred_batch_idxs)


def _filter_and_renorm(
    pred_mzs: th.Tensor,
    pred_probs: th.Tensor,
    pred_batch_idxs: th.Tensor,
    pred_ints: th.Tensor,
    threshold: float,
    batch_size: int,
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Keep peaks above threshold and L1-renormalize surviving peaks per spectrum.

    Args:
        pred_mzs: Predicted m/z values [P].
        pred_probs: L1-normalized probabilities [P].
        pred_batch_idxs: Batch indices [P].
        pred_ints: 0-1000 scale intensities [P] used for thresholding.
        threshold: Minimum intensity on 0-1000 scale.
        batch_size: Number of spectra in batch.

    Returns:
        Filtered (mzs, log_probs, batch_idxs) with renormalized probabilities.
    """
    mask = pred_ints > threshold
    if mask.sum() == 0:
        empty = th.zeros(0, device=pred_mzs.device, dtype=pred_mzs.dtype)
        return empty, empty, th.zeros(0, device=pred_mzs.device, dtype=th.long)

    f_mzs = pred_mzs[mask]
    f_probs = pred_probs[mask]
    f_batch = pred_batch_idxs[mask]

    # Re-normalize so filtered probabilities sum to 1 per spectrum.
    sums = scatter_reduce(f_probs, f_batch, "sum", dim_size=batch_size, default=1.0)
    f_probs = f_probs / (sums[f_batch] + EPS)
    return f_mzs, th.log(f_probs + EPS), f_batch


def main() -> None:
    args = get_args()

    ckpt_fp = get_ckpt_path(args.wandb_run_id, args.model_ckpt, args.model_save_dp)
    config_d = get_config_from_ckpt(
        ckpt_fp, custom_fp=args.custom_fp, template_fp=args.template_fp
    )
    apply_inference_config(
        config_d,
        auxiliary_scores=[],
        skip_extra_losses=True,
        disable_preproc=True,
    )

    ds = SpecMolFragDataset(split=args.split, **config_d)
    dl = init_dataloader(ds, config_d)
    log.info(f"Dataset: {len(ds)} spectra, {len(dl)} batches")

    device = setup_device(config_d, args.device)
    model = init_model_from_ckpt(ckpt_fp, config_d, device)

    agg: dict[float, dict[str, list]] = {
        t: {"n_pred": [], "n_true": [], "cos_hun": []} for t in THRESHOLDS
    }

    n_batches = 0
    model.eval()
    with th.no_grad():
        for batch_idx, batch in enumerate(dl):
            if args.batch_cutoff > 0 and batch_idx >= args.batch_cutoff:
                break

            batch = to_device(batch, device)
            batch_size = int(batch["batch_size"])

            # Ground truth: eval-preprocessed (filtered, not log-transformed, L1-normalized).
            true_mzs, true_probs, true_batch_idxs = model.preproc_spec(
                batch["spec_mzs"],
                batch["spec_ints"],
                batch["spec_batch_idxs"],
                train=False,
                pred=False,
                log_in=False,
                log_out=False,
            )
            true_logprobs = th.log(true_probs + EPS)

            # Raw model predictions.
            pred_d = model.forward(**batch)
            pred_mzs = pred_d["pred_mzs"]
            pred_logprobs = pred_d["pred_logprobs"]
            pred_batch_idxs = pred_d["pred_batch_idxs"]

            # 0-1000 scale (matches run_inference.py --min_intensity threshold semantics).
            pred_ints = _pred_ints_0_1000(model, pred_logprobs, pred_batch_idxs)

            # L1-normalized probs for cos_hun after filtering.
            pred_probs = model.ints_normalize_func(pred_ints.clone(), pred_batch_idxs)

            n_true = th.bincount(true_batch_idxs, minlength=batch_size).cpu().numpy()

            for t in THRESHOLDS:
                f_mzs, f_logprobs, f_batch = _filter_and_renorm(
                    pred_mzs, pred_probs, pred_batch_idxs, pred_ints, t, batch_size
                )

                if f_batch.numel() > 0:
                    n_pred = th.bincount(f_batch, minlength=batch_size).cpu().numpy()
                else:
                    n_pred = np.zeros(batch_size, dtype=int)

                if f_mzs.numel() > 0 and true_mzs.numel() > 0:
                    cos_dists = sparse_cosine_distance_hungarian(
                        true_mzs,
                        true_logprobs,
                        true_batch_idxs,
                        f_mzs,
                        f_logprobs,
                        f_batch,
                        tolerance=model.tolerance,
                        relative=model.relative,
                        tolerance_min_mz=model.tolerance_min_mz,
                    ).cpu().numpy()
                    cos_sim = 1.0 - cos_dists
                else:
                    cos_sim = np.zeros(batch_size)

                agg[t]["n_pred"].extend(n_pred.tolist())
                agg[t]["n_true"].extend(n_true.tolist())
                agg[t]["cos_hun"].extend(cos_sim.tolist())

            n_batches += 1
            if n_batches % 50 == 0:
                log.info(f"  {n_batches} batches done...")

    n_specs = len(agg[THRESHOLDS[0]]["n_pred"])
    log.info(f"Done: {n_specs} spectra across {n_batches} batches")

    n_true_all = np.array(agg[THRESHOLDS[0]]["n_true"])
    print(
        f"\nGround truth peaks — median: {np.median(n_true_all):.1f}, "
        f"mean: {np.mean(n_true_all):.1f}, "
        f"p90: {np.percentile(n_true_all, 90):.1f}\n"
    )

    col_w = 11
    header = (
        f"{'Threshold':>10} | {'n_pred p50':>{col_w}} | {'n_pred p90':>{col_w}} | "
        f"{'n_pred mean':>{col_w}} | {'% over GT':>9} | {'cos_hun':>8}"
    )
    print(header)
    print("-" * len(header))
    for t in THRESHOLDS:
        n_pred = np.array(agg[t]["n_pred"])
        n_true = np.array(agg[t]["n_true"])
        cos_hun = np.array(agg[t]["cos_hun"])
        pct_over = 100.0 * np.mean(n_pred > n_true)
        print(
            f"{t:>10.1f} | {np.median(n_pred):>{col_w}.1f} | "
            f"{np.percentile(n_pred, 90):>{col_w}.1f} | "
            f"{np.mean(n_pred):>{col_w}.2f} | "
            f"{pct_over:>8.1f}% | "
            f"{np.nanmean(cos_hun):>8.4f}"
        )
    print()


if __name__ == "__main__":
    main()
