#!/usr/bin/env python
"""Diagnose whether observed peak misses are node-mass or ion-channel errors."""

from __future__ import annotations

import argparse
import math
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
import torch as th
from tqdm import tqdm

from fragnnet.inference.checkpoint import (
    apply_inference_config,
    get_ckpt_path,
    init_model_from_ckpt,
    setup_device,
)
from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.misc_utils import booltype, to_device
from fragnnet.utils.setup_utils import get_dataset_cls
from fragnnet.utils.spec_similarity import calculate_match_mzs
from fragnnet.utils.spec_utils import primary_ce_to_batch


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each observed validation peak, decide whether the model had mass on "
            "the right fragment node P(n), but failed to rank the matching "
            "H-transfer/formula ion P(f|n), or whether node mass itself was missing."
        )
    )
    parser.add_argument("--template_fp", type=str, default="config")
    parser.add_argument("--custom_fp", type=str, required=True)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--model_save_dp", type=str, default="saved_ckpts")
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument(
        "--eval_split",
        type=str,
        default="val",
        choices=["train", "val", "test", "secondary"],
    )
    parser.add_argument("--out_fp", type=str, default=None)
    parser.add_argument("--eval_batch_size", type=int, default=64)
    parser.add_argument("--train_batch_size", type=int, default=4)
    parser.add_argument("--batch_cutoff", type=int, default=-1)
    parser.add_argument("--force_unmerged", type=booltype, default=False)
    parser.add_argument("--use_dynamic_sampler", type=booltype, default=False)
    parser.add_argument("--dynamic_sampler_max", type=int, default=None)
    parser.add_argument(
        "--dynamic_sampler_mode",
        type=str,
        default=None,
        choices=["frag_node", "frag_edge"],
    )
    parser.add_argument("--node_rank_k", type=int, default=10)
    parser.add_argument("--formula_rank_k", type=int, default=10)
    parser.add_argument(
        "--top_true_peak_k",
        type=int,
        default=-1,
        help="Only classify the top K true peaks per spectrum by intensity (-1 = all).",
    )
    parser.add_argument(
        "--min_true_rel_int",
        type=float,
        default=0.0,
        help="Only classify true peaks with relative intensity >= this threshold.",
    )
    parser.add_argument(
        "--default_tolerance_rel",
        type=float,
        default=None,
        help="Override scalar relative m/z tolerance. Defaults to model loss tolerance.",
    )
    parser.add_argument(
        "--default_tolerance_min_mz",
        type=float,
        default=None,
        help="Override scalar m/z floor. Defaults to model loss tolerance floor.",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Config overrides, e.g. model.lr=1e-4 train.batch_size=64",
    )
    return parser.parse_args()


def _tensor_to_cpu(x: Any) -> Any:
    if isinstance(x, th.Tensor):
        return x.detach().cpu()
    return x


def _split_by_batch(
    values: th.Tensor | None,
    batch_idxs: th.Tensor | None,
    batch_size: int,
) -> tuple[th.Tensor | None, ...]:
    if values is None or batch_idxs is None:
        return tuple(None for _ in range(batch_size))
    batch_idxs = batch_idxs.detach().cpu().long()
    counts = th.bincount(batch_idxs, minlength=batch_size).tolist()
    return th.split(values.detach().cpu(), counts)


def _split_by_sizes(
    values: th.Tensor | None,
    sizes: th.Tensor | None,
    batch_size: int,
) -> tuple[th.Tensor | None, ...]:
    if values is None or sizes is None:
        return tuple(None for _ in range(batch_size))
    sizes = sizes.detach().cpu().long().tolist()
    if len(sizes) != batch_size:
        raise ValueError(f"Expected {batch_size} sizes, got {len(sizes)}")
    return th.split(values.detach().cpu(), sizes)


def _rank_map(idxs: th.Tensor, probs: th.Tensor) -> dict[int, int]:
    order = th.argsort(probs, descending=True)
    return {int(idxs[i].item()): rank + 1 for rank, i in enumerate(order.tolist())}


def _prob_map(idxs: th.Tensor, probs: th.Tensor) -> dict[int, float]:
    return {int(idx.item()): float(prob.item()) for idx, prob in zip(idxs, probs)}


def _safe_min(vals: Sequence[int]) -> float:
    return float(min(vals)) if vals else math.nan


def _safe_max(vals: Sequence[float]) -> float:
    return float(max(vals)) if vals else math.nan


def _batch_items(values: Any, batch_size: int) -> list[Any]:
    if values is None:
        return [None] * batch_size
    if isinstance(values, th.Tensor):
        values = values.detach().cpu()
        if values.ndim == 0:
            return [values.item()]
        return [values[i].item() for i in range(min(batch_size, values.shape[0]))]
    if isinstance(values, np.ndarray):
        return [
            values[i].item() if np.ndim(values[i]) == 0 else values[i] for i in range(batch_size)
        ]
    if isinstance(values, Sequence) and not isinstance(values, str):
        return list(values)[:batch_size]
    return [values] * batch_size


def _inst_strings(batch: dict[str, Any], config_d: dict[str, Any], batch_size: int) -> list[str]:
    idxs = batch.get("spec_inst_type")
    inst_types = config_d.get("spec_params", {}).get("inst_types", [])
    if idxs is None or not inst_types:
        return ["unknown"] * batch_size
    idxs = idxs.detach().cpu().reshape(-1).tolist()
    inst_types_sorted = sorted(inst_types)
    return [
        inst_types_sorted[int(i)] if 0 <= int(i) < len(inst_types_sorted) else "unknown"
        for i in idxs[:batch_size]
    ]


def _ce_values(batch: dict[str, Any], key: str, batch_size: int) -> list[float]:
    if key not in batch:
        return [math.nan] * batch_size
    ce = primary_ce_to_batch(batch[key], batch.get(f"{key}_batch_idxs"), batch_size)
    return [float(v) for v in ce.detach().cpu().reshape(-1).tolist()]


def _sample_tolerance(
    model: Any,
    batch: dict[str, Any],
    b_idx: int,
    args: argparse.Namespace,
) -> tuple[float, bool, float | None]:
    if args.default_tolerance_rel is not None:
        min_mz = (
            args.default_tolerance_min_mz
            if args.default_tolerance_min_mz is not None
            else getattr(model, "tolerance_min_mz", 200.0)
        )
        return float(args.default_tolerance_rel), True, float(min_mz)

    if (
        getattr(model, "inst_tol_lookup", None) is not None
        and "spec_inst_type" in batch
        and bool(getattr(model, "relative", True))
    ):
        inst_idx = int(batch["spec_inst_type"].detach().cpu().reshape(-1)[b_idx].item())
        tol = float(model.inst_tol_lookup.detach().cpu()[inst_idx].item())
        min_mz = float(model.inst_min_mz_lookup.detach().cpu()[inst_idx].item())
        return tol, True, min_mz

    tol = float(getattr(model, "tolerance", 1e-5))
    relative = bool(getattr(model, "relative", True))
    min_mz = getattr(model, "tolerance_min_mz", None)
    if args.default_tolerance_min_mz is not None:
        min_mz = args.default_tolerance_min_mz
    return tol, relative, None if min_mz is None else float(min_mz)


def _classify_peak(
    *,
    matched_formula_idxs: set[int],
    formula_rank: dict[int, int],
    formula_prob: dict[int, float],
    node_rank: dict[int, int],
    node_prob: dict[int, float],
    joint_positions_by_formula: dict[int, list[int]],
    joint_node_idxs: th.Tensor,
    joint_ranks: th.Tensor,
    node_formula_probs: th.Tensor,
    node_rank_k: int,
    formula_rank_k: int,
) -> dict[str, float | str | int]:
    if not matched_formula_idxs:
        return {
            "bottleneck": "coverage",
            "n_matched_formula": 0,
            "p_formula": 0.0,
            "p_node_support": 0.0,
            "support_node_count": 0,
            "best_formula_rank": math.nan,
            "best_node_rank": math.nan,
            "best_joint_rank": math.nan,
            "best_p_f_given_n": math.nan,
            "top_support_node_p_f_given_n": math.nan,
        }

    support_positions = [
        position
        for formula_idx in matched_formula_idxs
        for position in joint_positions_by_formula.get(formula_idx, [])
    ]
    support_position_tensor = th.tensor(support_positions, dtype=th.long)
    support_nodes = joint_node_idxs[support_position_tensor]
    support_node_idxs = sorted(set(support_nodes.tolist()))

    matched_formula_present = [f_idx for f_idx in matched_formula_idxs if f_idx in formula_rank]
    p_formula = sum(formula_prob[f_idx] for f_idx in matched_formula_present)
    best_formula_rank = _safe_min([formula_rank[f_idx] for f_idx in matched_formula_present])

    p_node_support = sum(node_prob.get(n_idx, 0.0) for n_idx in support_node_idxs)
    best_node_rank = _safe_min(
        [node_rank[n_idx] for n_idx in support_node_idxs if n_idx in node_rank]
    )

    if support_positions:
        best_joint_rank = int(joint_ranks[support_position_tensor].min().item())
        best_p_f_given_n = float(node_formula_probs[support_position_tensor].max().item())
    else:
        best_joint_rank = math.nan
        best_p_f_given_n = math.nan

    top_support_node_p_f_given_n = math.nan
    if support_node_idxs:
        top_node = max(support_node_idxs, key=lambda n_idx: node_prob.get(n_idx, 0.0))
        top_mask = support_nodes == top_node
        if top_mask.any():
            top_support_node_p_f_given_n = float(
                node_formula_probs[support_position_tensor][top_mask].max().item()
            )

    node_ok = bool(not math.isnan(best_node_rank) and best_node_rank <= node_rank_k)
    formula_ok = bool(not math.isnan(best_formula_rank) and best_formula_rank <= formula_rank_k)
    if node_ok and not formula_ok:
        bottleneck = "h_formula"
    elif not node_ok and formula_ok:
        bottleneck = "node"
    elif not node_ok and not formula_ok:
        bottleneck = "both_low"
    else:
        bottleneck = "both_ok"

    return {
        "bottleneck": bottleneck,
        "n_matched_formula": len(matched_formula_idxs),
        "p_formula": float(p_formula),
        "p_node_support": float(p_node_support),
        "support_node_count": len(support_node_idxs),
        "best_formula_rank": best_formula_rank,
        "best_node_rank": best_node_rank,
        "best_joint_rank": float(best_joint_rank),
        "best_p_f_given_n": best_p_f_given_n,
        "top_support_node_p_f_given_n": top_support_node_p_f_given_n,
    }


def _summarize(df: pd.DataFrame) -> None:
    if df.empty:
        print("No rows produced.")
        return

    def weighted(group: pd.DataFrame) -> pd.Series:
        denom = group["true_intensity"].sum()
        if denom <= 0:
            return pd.Series(dtype=float)
        return (
            group.groupby("bottleneck")["true_intensity"].sum().sort_values(ascending=False) / denom
        )

    print("\n>>> Bottleneck by true-intensity mass")
    print(weighted(df).to_string(float_format=lambda x: f"{x:.4f}"))

    print("\n>>> Bottleneck by peak count")
    print(
        df["bottleneck"].value_counts(normalize=True).to_string(float_format=lambda x: f"{x:.4f}")
    )

    if "ce_nce" in df and df["ce_nce"].notna().any():
        bins = [-math.inf, 20.0, 40.0, math.inf]
        labels = ["low_nce", "mid_nce", "high_nce"]
        df = df.copy()
        df["nce_bin"] = pd.cut(df["ce_nce"], bins=bins, labels=labels)
        print("\n>>> Bottleneck by NCE bin, true-intensity mass")
        rows = []
        for bin_name, group in df.dropna(subset=["nce_bin"]).groupby("nce_bin", observed=True):
            w = weighted(group)
            row = {"nce_bin": str(bin_name), "true_intensity": group["true_intensity"].sum()}
            row.update(w.to_dict())
            rows.append(row)
        if rows:
            print(pd.DataFrame(rows).fillna(0.0).to_string(index=False))


def main() -> None:
    args = get_args()
    if args.wandb_run_id is None and args.model_ckpt is None:
        raise ValueError("Either --wandb_run_id or --model_ckpt must be provided")

    ckpt_fp = get_ckpt_path(
        wandb_run_id=args.wandb_run_id,
        model_ckpt=args.model_ckpt,
        model_save_dp=args.model_save_dp,
        use_cached=True,
    )

    config_d = load_config(args.template_fp, args.custom_fp, args.override)
    config_d = apply_inference_config(
        config_d,
        auxiliary_scores=[],
        skip_extra_losses=True,
        disable_preproc=True,
        output_formula_str=False,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        dynamic_batch_sampler=args.use_dynamic_sampler,
    )
    config_d["activation_checkpointing"] = False
    if args.force_unmerged:
        config_d["spec_params"]["merge"] = False
        config_d["spec_params"]["merge_keep_ces"] = False
    if args.dynamic_sampler_max is not None:
        config_d["dynamic_batch_sampler_max"] = args.dynamic_sampler_max
    if args.dynamic_sampler_mode is not None:
        config_d["dynamic_batch_sampler_mode"] = args.dynamic_sampler_mode

    ds_cls = get_dataset_cls(config_d["model_type"])
    ds = ds_cls(split=args.eval_split, **config_d)
    dl = init_dataloader(ds, config_d)
    device = setup_device(config_d)
    model = init_model_from_ckpt(ckpt_fp, config_d, device)
    model.eval()

    total_batches = len(dl) if args.batch_cutoff < 0 else min(len(dl), args.batch_cutoff)
    rows: list[dict[str, Any]] = []

    with th.inference_mode():
        for batch_idx, batch in tqdm(
            enumerate(dl),
            total=total_batches,
            desc="Diagnosing node vs H/formula",
        ):
            if batch_idx >= total_batches:
                break
            batch = to_device(batch, device, non_blocking=False)
            batch["return_explainability_tensors"] = True
            out = model.inference_step(batch, metrics="test")
            batch_cpu = {k: _tensor_to_cpu(v) for k, v in batch.items()}
            out_cpu = {k: _tensor_to_cpu(v) for k, v in out.items()}
            batch_size = int(batch_cpu["batch_size"])

            required = [
                "pred_formula_formula_idxs",
                "pred_node_node_idxs",
                "pred_joint_formula_idxs",
                "pred_joint_node_idxs",
                "pred_node_formula_logprobs",
                "pred_joint_logprobs",
            ]
            missing = [k for k in required if out_cpu.get(k) is None]
            if missing:
                raise KeyError(
                    f"Missing explainability tensors: {missing}. "
                    "Check return_explainability_tensors propagation."
                )

            true_mzs_split = _split_by_batch(
                out_cpu["true_mzs"], out_cpu["true_batch_idxs"], batch_size
            )
            true_logprob_split = _split_by_batch(
                out_cpu["true_logprobs"], out_cpu["true_batch_idxs"], batch_size
            )
            peak_mzs_split = _split_by_sizes(
                batch_cpu.get("frag_formula_peak_mzs"),
                batch_cpu.get("frag_formula_peak_sizes"),
                batch_size,
            )
            peak_formula_split = _split_by_sizes(
                batch_cpu.get("frag_formula_peak_idxs"),
                batch_cpu.get("frag_formula_peak_sizes"),
                batch_size,
            )
            formula_idx_split = _split_by_batch(
                out_cpu["pred_formula_formula_idxs"],
                out_cpu["pred_formula_batch_idxs"],
                batch_size,
            )
            formula_prob_split = _split_by_batch(
                out_cpu["pred_formula_logprobs"].exp(),
                out_cpu["pred_formula_batch_idxs"],
                batch_size,
            )
            node_idx_split = _split_by_batch(
                out_cpu["pred_node_node_idxs"],
                out_cpu["pred_node_batch_idxs"],
                batch_size,
            )
            node_prob_split = _split_by_batch(
                out_cpu["pred_node_logprobs"].exp(),
                out_cpu["pred_node_batch_idxs"],
                batch_size,
            )
            joint_formula_split = _split_by_batch(
                out_cpu["pred_joint_formula_idxs"],
                out_cpu["pred_joint_batch_idxs"],
                batch_size,
            )
            joint_node_split = _split_by_batch(
                out_cpu["pred_joint_node_idxs"],
                out_cpu["pred_joint_batch_idxs"],
                batch_size,
            )
            joint_prob_split = _split_by_batch(
                out_cpu["pred_joint_logprobs"].exp(),
                out_cpu["pred_joint_batch_idxs"],
                batch_size,
            )
            node_formula_prob_split = _split_by_batch(
                out_cpu["pred_node_formula_logprobs"].exp(),
                out_cpu["pred_joint_batch_idxs"],
                batch_size,
            )

            inst_strs = _inst_strings(batch_cpu, config_d, batch_size)
            spec_ids = _batch_items(batch_cpu.get("spec_unique_id"), batch_size)
            mol_ids = _batch_items(batch_cpu.get("mol_id"), batch_size)
            ce_nce = _ce_values(batch_cpu, "spec_nce", batch_size)
            ce_ace = _ce_values(batch_cpu, "spec_ace", batch_size)
            ce_primary = _ce_values(batch_cpu, "spec_ce", batch_size)
            formula_offsets = batch_cpu["frag_formula_cumsizes"].long()

            for b_idx in range(batch_size):
                true_mzs = true_mzs_split[b_idx]
                true_probs = true_logprob_split[b_idx].exp()
                peak_mzs = peak_mzs_split[b_idx]
                peak_formula_idxs = peak_formula_split[b_idx]
                if true_mzs is None or peak_mzs is None or true_mzs.numel() == 0:
                    continue

                # 1. Select the true peaks before building the match matrix.
                true_order = th.argsort(true_probs, descending=True)
                true_rank = th.empty_like(true_order)
                true_rank[true_order] = th.arange(1, true_order.numel() + 1)
                max_true = float(true_probs.max().item()) if true_probs.numel() else 0.0
                true_rel_ints = (
                    true_probs / max_true
                    if max_true > 0
                    else th.full_like(true_probs, float("nan"))
                )
                keep = true_rel_ints >= args.min_true_rel_int
                if args.top_true_peak_k > 0:
                    keep &= true_rank <= args.top_true_peak_k
                selected_peak_idxs = th.nonzero(keep, as_tuple=False).reshape(-1)
                if selected_peak_idxs.numel() == 0:
                    continue

                tol, relative, min_mz = _sample_tolerance(model, batch_cpu, b_idx, args)
                match = calculate_match_mzs(
                    true_mzs[selected_peak_idxs],
                    peak_mzs,
                    tolerance=tol,
                    relative=relative,
                    tolerance_min_mz=200.0 if min_mz is None else min_mz,
                )

                # 2. Rank model outputs once per spectrum, not once per true peak.
                formula_rank = _rank_map(formula_idx_split[b_idx], formula_prob_split[b_idx])
                formula_prob = _prob_map(formula_idx_split[b_idx], formula_prob_split[b_idx])
                node_rank = _rank_map(node_idx_split[b_idx], node_prob_split[b_idx])
                node_prob = _prob_map(node_idx_split[b_idx], node_prob_split[b_idx])
                joint_order = th.argsort(joint_prob_split[b_idx], descending=True)
                joint_ranks = th.empty_like(joint_order)
                joint_ranks[joint_order] = th.arange(1, joint_order.numel() + 1)
                joint_positions_by_formula: dict[int, list[int]] = {}
                for position, formula_idx in enumerate(joint_formula_split[b_idx].tolist()):
                    joint_positions_by_formula.setdefault(formula_idx, []).append(position)

                # 3. Classify each retained true peak using the shared ranks.
                for selected_i, peak_i_tensor in enumerate(selected_peak_idxs):
                    peak_i = int(peak_i_tensor.item())
                    rank = int(true_rank[peak_i].item())
                    rel_int = float(true_probs[peak_i].item()) / max_true

                    peak_match = match[selected_i]
                    local_formula_idxs = peak_formula_idxs[peak_match].long()
                    matched_formula_idxs = {
                        int(f.item() + formula_offsets[b_idx].item()) for f in local_formula_idxs
                    }
                    result = _classify_peak(
                        matched_formula_idxs=matched_formula_idxs,
                        formula_rank=formula_rank,
                        formula_prob=formula_prob,
                        node_rank=node_rank,
                        node_prob=node_prob,
                        joint_positions_by_formula=joint_positions_by_formula,
                        joint_node_idxs=joint_node_split[b_idx],
                        joint_ranks=joint_ranks,
                        node_formula_probs=node_formula_prob_split[b_idx],
                        node_rank_k=args.node_rank_k,
                        formula_rank_k=args.formula_rank_k,
                    )
                    rows.append(
                        {
                            "batch_idx": batch_idx,
                            "sample_idx": b_idx,
                            "spec_id": spec_ids[b_idx],
                            "mol_id": mol_ids[b_idx],
                            "inst_type": inst_strs[b_idx],
                            "ce_primary": ce_primary[b_idx],
                            "ce_nce": ce_nce[b_idx],
                            "ce_ace": ce_ace[b_idx],
                            "true_peak_rank": rank,
                            "true_mz": float(true_mzs[peak_i].item()),
                            "true_intensity": float(true_probs[peak_i].item()),
                            "true_rel_intensity": rel_int,
                            "match_tolerance": tol,
                            "match_relative": relative,
                            "match_min_mz": min_mz,
                            **result,
                        }
                    )

    df = pd.DataFrame(rows)
    out_fp = args.out_fp
    if out_fp is None:
        run_label = args.wandb_run_id or os.path.splitext(os.path.basename(args.model_ckpt))[0]
        out_fp = os.path.join(
            "output",
            "diagnostics",
            f"node_vs_h_transfer_{run_label}_{args.eval_split}.csv",
        )
    os.makedirs(os.path.dirname(out_fp), exist_ok=True)
    df.to_csv(out_fp, index=False)
    print(f"\n>>> Saved {len(df):,} peak rows to {out_fp}")
    _summarize(df)


if __name__ == "__main__":
    main()
