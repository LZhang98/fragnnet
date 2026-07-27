"""Compare baseline, molecular-context, and cc-identity inference ablations.

The standard FraGNNet configuration uses ``cc_interstage_type=direct``. It therefore does not
inject the separately pooled whole-molecule vector, but its cc atom embeddings still come from
message passing over the full molecular graph. The ``mol_input_only`` condition bypasses that
message passing while retaining the learned atom-feature projection and the true cc mask.

No checkpoint weights or configuration schemas are changed.
"""

import argparse
import os
import random

import numpy as np

# Hungarian peak matching temporarily allocates large pairwise matrices. Expandable segments let
# PyTorch reuse fragmented reserved memory instead of requiring one new contiguous CUDA block.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch as th
from tqdm import tqdm

from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.misc_utils import booltype, to_device
from fragnnet.utils.script_utils import (
    apply_inference_config,
    get_ckpt_path,
    get_score_types,
    init_model_from_ckpt,
    setup_device,
)
from fragnnet.utils.setup_utils import get_dataset_cls


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template_fp", type=str, default="config/template.yml")
    parser.add_argument("--custom_fp", type=str, required=True)
    parser.add_argument("--wandb_run_id", type=str, default=None)
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--model_save_dp", type=str, default="saved_ckpts")
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "val", "test", "secondary"],
    )
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--batch_cutoff", type=int, default=-1)
    parser.add_argument("--use_dynamic_sampler", type=booltype, default=False)
    parser.add_argument("--dynamic_sampler_max", type=int, default=None)
    parser.add_argument(
        "--dynamic_sampler_mode",
        type=str,
        default=None,
        choices=["frag_node", "frag_edge"],
    )
    parser.add_argument("--cc_corruption_frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--auxiliary_scores",
        type=str,
        nargs="+",
        default=["cos_hun_inst_tol"],
    )
    parser.add_argument("--eval_sqrt", type=booltype, default=False)
    parser.add_argument("--override", nargs="*", default=[])
    return parser.parse_args()


def main() -> None:
    args = get_args()
    if args.wandb_run_id is None and args.model_ckpt is None:
        raise ValueError("Either --wandb_run_id or --model_ckpt must be provided")
    if not 0.0 <= args.cc_corruption_frac <= 1.0:
        raise ValueError("--cc_corruption_frac must be in [0, 1]")
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)
    if th.cuda.is_available():
        th.cuda.manual_seed_all(args.seed)

    ckpt_fp = get_ckpt_path(
        wandb_run_id=args.wandb_run_id,
        model_ckpt=args.model_ckpt,
        model_save_dp=args.model_save_dp,
        use_cached=True,
    )
    config_d = load_config(args.template_fp, args.custom_fp, args.override)
    config_d = apply_inference_config(
        config_d,
        auxiliary_scores=args.auxiliary_scores,
        skip_extra_losses=True,
        disable_preproc=True,
        output_formula_str=False,
        train_batch_size=args.eval_batch_size,
        eval_batch_size=args.eval_batch_size,
        dynamic_batch_sampler=args.use_dynamic_sampler,
    )
    config_d["eval_hun_sqrt"] = args.eval_sqrt
    if args.dynamic_sampler_max is not None:
        config_d["dynamic_batch_sampler_max"] = args.dynamic_sampler_max
    if args.dynamic_sampler_mode is not None:
        config_d["dynamic_batch_sampler_mode"] = args.dynamic_sampler_mode

    ds_cls = get_dataset_cls(config_d["model_type"])
    ds = ds_cls(split=args.eval_split, **config_d)
    dl = init_dataloader(ds, config_d)
    device = setup_device(config_d)
    model = init_model_from_ckpt(ckpt_fp, config_d, device)
    core_model = getattr(model, "model", None)
    if core_model is None or not hasattr(core_model, "mol_context_input_only"):
        raise TypeError("This diagnostic requires a FraGNNet model")

    score_names = get_score_types(config_d)
    conditions = [
        ("baseline", False, 0.0),
        ("mol_input_only", True, 0.0),
        ("cc_corrupt", False, args.cc_corruption_frac),
    ]
    total_batches = len(dl) if args.batch_cutoff < 0 else min(len(dl), args.batch_cutoff)

    print(f"\n{'condition':>16} | " + " | ".join(f"{s:>24}" for s in score_names))
    for condition, mol_input_only, cc_corruption_frac in conditions:
        core_model.mol_context_input_only = mol_input_only
        core_model.cc_corruption_frac = cc_corruption_frac
        sums = dict.fromkeys(score_names, 0.0)
        counts = dict.fromkeys(score_names, 0)

        with th.inference_mode():
            for batch_idx, batch in tqdm(
                enumerate(dl),
                total=total_batches,
                desc=condition,
            ):
                if batch_idx >= total_batches:
                    break
                batch = to_device(batch, device, non_blocking=False)
                # Always use evaluation metric policy, even when reading the training dataset.
                results = model.inference_with_ground_truth_step(batch, split="test")
                for score_name in score_names:
                    values = results[score_name].detach()
                    sums[score_name] += values.sum().item()
                    counts[score_name] += values.numel()

        row = f"{condition:>16} | " + " | ".join(
            f"{sums[name] / max(counts[name], 1):>24.4f}" for name in score_names
        )
        print(row)

    core_model.mol_context_input_only = False
    core_model.cc_corruption_frac = 0.0


if __name__ == "__main__":
    main()
