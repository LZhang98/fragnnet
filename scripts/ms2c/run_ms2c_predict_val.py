"""MS2C prediction validation script for FraGNNet models.

Runs inference on MS2C (mass spectrum to candidate) prediction datasets
using a trained FraGNNet checkpoint.
"""

import argparse
import json
import os
import time

import pandas as pd

from fragnnet.dataset import SpecMolFragDataset
from fragnnet.runner import init_dataloader
from fragnnet.utils.misc_utils import booltype, get_core_count
from fragnnet.utils.script_utils import (
    apply_inference_config,
    get_ckpt_path,
    get_config_from_ckpt,
    init_model_from_ckpt,
    run_inference_tabular,
    setup_device,
)


def get_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        Parsed command line arguments.
    """
    parser = argparse.ArgumentParser(description="MS2C prediction validation script")
    parser.add_argument("--device", type=str, default="cuda:0", help="Device to run inference on")
    parser.add_argument(
        "--save_fp",
        type=str,
        default="./data/ms2c/pubchem/predicted/ms2pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50",
        help="Output file path for predictions",
    )
    parser.add_argument(
        "--proc_dp",
        type=str,
        default="./data/ms2c/pubchem/proc/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50",
        help="Directory containing processed mol_df.pkl.gz and spec_df.pkl.gz",
    )
    parser.add_argument(
        "--frag_dp",
        type=str,
        default="./data/ms2c/pubchem/frags/d3_h4_isoFalse/pubchem_nist20_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50",
        help="Directory containing fragmentation DAG data",
    )

    parser.add_argument("--use_cached_ckpt", type=booltype, default=False)
    parser.add_argument("--batch_size", type=int, default=50)
    parser.add_argument("--model_save_dp", type=str, default="./saved_ckpts")
    parser.add_argument("--model_ckpt", type=str, default=None)
    parser.add_argument("--wandb_run_id", type=str, default="wf1zkb7n")
    parser.add_argument("--custom_fp", type=str, required=False)
    parser.add_argument("--template_fp", type=str, default="./config/template.yml")
    parser.add_argument("--num_workers", type=int, default=get_core_count())
    parser.add_argument(
        "--auxiliary_scores",
        type=str,
        nargs="+",
        default=["cos_sim", "cos_hun", "true_oos_prob", "true_oos_e"],
        help="List of auxiliary scores to compute",
    )
    parser.add_argument(
        "--eval_mz_bin_res",
        type=float,
        nargs="+",
        default=[0.01],
        help="List of m/z bin resolutions for binned metrics (e.g., 0.01 0.1)",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    timestr = time.strftime("%Y%m%d-%H%M%S")
    save_dp = os.path.dirname(args.save_fp)
    os.makedirs(save_dp, exist_ok=True)
    records_dict = vars(args)

    # Get checkpoint path (download from W&B or use local)
    ckpt_fp = get_ckpt_path(
        wandb_run_id=args.wandb_run_id,
        model_ckpt=args.model_ckpt,
        model_save_dp=args.model_save_dp,
        use_cached=args.use_cached_ckpt,
    )
    records_dict["model_ckpt"] = ckpt_fp

    # Save run configuration
    with open(f"{args.save_fp}_{timestr}.json", "w") as f:
        json.dump(records_dict, f, indent=2)

    # Load datasets
    print(f"> Loading datasets from {args.proc_dp}")
    mol_fp = os.path.join(args.proc_dp, "mol_df.pkl.gz")
    spec_fp = os.path.join(args.proc_dp, "spec_df.pkl.gz")

    print("> Preloading mol_df")
    mol_df = pd.read_pickle(mol_fp)

    print("> Preloading spec_df")
    spec_df = pd.read_pickle(spec_fp)

    # Load model and configuration
    print(f"> Loading model from {ckpt_fp}")

    # Get configuration from checkpoint
    config_d = get_config_from_ckpt(
        ckpt_fp=ckpt_fp,
        custom_fp=args.custom_fp,
        template_fp=args.template_fp if args.custom_fp else None,
    )

    # Apply inference-specific configuration
    config_d = apply_inference_config(
        config_d,
        auxiliary_scores=args.auxiliary_scores,
        eval_mz_bin_res=args.eval_mz_bin_res,
        num_workers=args.num_workers,
        eval_batch_size=args.batch_size,
        mol_src=mol_df,
        spec_src=spec_df,
    )

    # Set fragmentation directory
    config_d["frag_dp"] = args.frag_dp

    # Auto-detect sqrt options based on _sqrt suffix in auxiliary_scores
    # e.g., cos_sim_sqrt triggers eval_bin_sqrt, cos_hun_sqrt triggers eval_hun_sqrt
    binned_sqrt_metrics = {"cos_sim_sqrt", "jss_sqrt", "opt_cos_sim_sqrt"}
    hungarian_sqrt_metrics = {"cos_hun_sqrt", "jss_hun_sqrt"}
    requested_scores = set(args.auxiliary_scores)
    config_d["eval_bin_sqrt"] = bool(requested_scores & binned_sqrt_metrics)
    config_d["eval_hun_sqrt"] = bool(requested_scores & hungarian_sqrt_metrics)

    # Setup device
    device = setup_device(config_d, device=args.device)

    # Initialize model
    model = init_model_from_ckpt(
        ckpt_fp=ckpt_fp,
        config_d=config_d,
        device=device,
    )

    # Create dataset and dataloader
    print("> Creating dataset")
    ds = SpecMolFragDataset(split="test", **config_d)
    dl = init_dataloader(ds, config_d)

    # Run inference
    print("> Running inference")
    df = run_inference_tabular(
        model=model,
        dataloader=dl,
        device=device,
        evaluate=True,
    )

    # Save results
    print(f"> Saving results to {args.save_fp}")
    df.to_pickle(args.save_fp)
    print("> Done!")
