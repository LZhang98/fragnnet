"""
FragGNN Inference Evaluation Script

This script runs inference on a trained FragGNN model and evaluates its performance
on various metrics including cosine similarity and Hungarian algorithm matching.
The script saves predictions along with evaluation scores for analysis.

Usage:
    python run_save_inference_eval.py --wandb_run_id <run_id> --eval_split test
"""

import argparse
import os
import sys

from fragnnet.runner import init_dataloader, load_config
from fragnnet.utils.misc_utils import booltype
from fragnnet.utils.script_utils import (
    apply_inference_config,
    get_ckpt_path,
    get_required_config_indices,
    get_score_types,
    init_model_from_ckpt,
    list_output_subset_options,
    prepare_output_subset,
    run_inference_tabular,
    run_inference_hdf5,
    setup_device,
)
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls
import h5py
import numpy as np

def get_args():
    # Set up command line argument parser
    parser = argparse.ArgumentParser(description="Run FragGNN inference evaluation with metrics")

    # Configuration file arguments
    parser.add_argument(
        "--template_fp",
        type=str,
        default="config/template.yml",
        help="Path to the template config file",
    )
    parser.add_argument(
        "--custom_fp",
        type=str,
        default="config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc.yml",
        help="Path to custom config file",
    )

    # Model loading arguments
    parser.add_argument(
        "--wandb_run_id",
        type=str,
        default="32n55f2m",
        help="Weights & Biases run ID for model retrieval",
    )
    parser.add_argument(
        "--model_save_dp",
        type=str,
        default="saved_ckpts",
        help="Directory to save downloaded model checkpoints",
    )
    parser.add_argument(
        "--model_ckpt",
        type=str,
        default=None,
        help="Path to model checkpoint file (if not downloading from wandb)",
    )
    # Evaluation and output arguments
    parser.add_argument(
        "--eval_split",
        type=str,
        default="test",
        choices=["train", "val", "test", "secondary"],
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--save_dp",
        type=str,
        default="output/inference_eval",
        help="Directory to save inference evaluation results",
    )
    parser.add_argument(
        "--list_output_options",
        action="store_true",
        help="List all valid tokens for --output_subset and exit",
    )

    # Evaluation metrics configuration
    parser.add_argument(
        "--auxiliary_scores",
        type=str,
        nargs="+",
        default=[
            "cos_sim",  # Cosine similarity between predicted and true spectra
            "cos_hun",  # Hungarian algorithm cosine similarity
            # "true_oos_prob",  # True out-of-scope probability
            # "true_oos_e"      # True out-of-scope energy
        ],
        help="List of auxiliary evaluation scores to compute",
    )
    parser.add_argument(
        "--skip_extra_losses",
        type=booltype,
        default=False,
        help="Skip computation of extra loss terms",
    )
    # Output control arguments
    parser.add_argument(
        "--output_subset",
        type=str,
        nargs="+",
        default=[
            # "true_mzs",          # True m/z values from ground truth
            # "true_logprobs",     # True log probabilities
            "pred_mzs",  # Predicted m/z values
            "pred_ints",  # Predicted intensities
            # "true_oos_prob",     # True out-of-scope probability
            # "pred_oos_prob",     # Predicted out-of-scope probability
            # "pred_formula_probs", # Predicted formula probabilities
            # "frag_formula_peak_mzs", # Fragment formula peak m/z values
            # "formula_scores",    # Formula scoring metrics
        ],
        help="Subset of outputs to include in results",
    )
    parser.add_argument(
        "--output_everything",
        type=booltype,
        default=False,
        help="Output all available metrics and predictions",
    )
    parser.add_argument(
        "--return_formula_str",
        type=booltype,
        default=True,
        help="Whether to return formula strings",
    )
    parser.add_argument(
        "--return_cc_hash", type=booltype, default=True, help="Whether to return node CC hashes"
    )

    # Batch size arguments
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Training batch size for evaluation"
    )
    parser.add_argument("--eval_batch_size", type=int, default=4, help="Evaluation batch size")
    parser.add_argument(
        "--batch_cutoff", type=int, default=1000, help="Number of batches to evaluate (-1 for all)"
    )
    parser.add_argument(
        "--drop_none_cols",
        type=booltype,
        default=True,
        help="Whether to drop columns with None values in the output",
    )
    parser.add_argument(
        "--use_dynamic_sampler",
        action="store_true",
        help="whether use dynamic sampler",
    )
    parser.add_argument(
        "--override",
        nargs="*",
        default=[],
        help="Override config values, e.g. model.lr=1e-4 train.batch_size=64",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Early check for --list_output_options to avoid required argument validation
    from pandarallel import pandarallel
    pandarallel.initialize(progress_bar=True, nb_workers=min(10,  os.cpu_count()))
    if "--list_output_options" in sys.argv:
        print("Available output_subset options:")
        for opt in list_output_subset_options():
            print(f"  - {opt}")
        sys.exit(0)

    args = get_args()

    # Get checkpoint path (download from W&B or use local)
    ckpt_fp = get_ckpt_path(
        wandb_run_id=args.wandb_run_id,
        model_ckpt=args.model_ckpt,
        model_save_dp=args.model_save_dp,
        use_cached=True,
    )

    # Extract evaluation parameters
    eval_split = args.eval_split
    save_dp = args.save_dp

    # Load and configure the model configuration
    template_fp = args.template_fp
    custom_fp = args.custom_fp

    config_d = load_config(template_fp, custom_fp, args.override)
    config_d = apply_inference_config(
        config_d,
        auxiliary_scores=args.auxiliary_scores,
        skip_extra_losses=args.skip_extra_losses,
        disable_preproc=True,
        output_formula_str=False,
        train_batch_size=args.train_batch_size,
        eval_batch_size=args.eval_batch_size,
        dynamic_batch_sampler=args.use_dynamic_sampler
    )

    # Initialize dataset, dataloader, and model (inline)
    print(">>> Initializing components...")
    ds_cls = get_dataset_cls(config_d["model_type"])
    model_cls = get_model_cls(config_d["model_type"])

    # Initialize dataset and dataloader
    ds = ds_cls(split=eval_split, **config_d)
    dl = init_dataloader(ds, config_d)

    # Prepare device and load model from checkpoint
    device = setup_device(config_d)
    model = init_model_from_ckpt(ckpt_fp, config_d, device)

    # Run inference and collect predictions with evaluation scores
    print(">>> Running inference and evaluation...")
    output_set = prepare_output_subset(
        output_subset=args.output_subset, output_everything=args.output_everything
    )
    # Show which index variables will be required for the chosen output_subset
    if output_set is not None:
        req_batch, req_node, req_formula, req_nb_node = get_required_config_indices(
            output_set, model, nb_iso=False
        )
        print("> Required index variables for output_subset:")
        if req_batch:
            print(f"  Batch indices: {req_batch}")
        if req_node:
            print(f"  Node indices: {req_node}")
        if req_formula:
            print(f"  Formula indices: {req_formula}")

    print(f">>> Output set: {output_set}")
    output_save_dp = os.path.join(save_dp, args.wandb_run_id)
    os.makedirs(output_save_dp, exist_ok=True)
    num_predict = run_inference_hdf5(
        model,
        dl,
        device,
        evaluate=True,
        output_set=output_set,
        batch_cutoff=args.batch_cutoff,
        min_peak_int=1.0,
        return_formula_str=args.return_formula_str or args.output_everything,
        return_cc_hash=args.return_cc_hash or args.output_everything,
        rebase_indices=True,
        save_path=os.path.join(output_save_dp, f"predictions_{eval_split}.hdf5")
    )
    print(f"\n>>> Saved predictions to: {output_save_dp}")
    print(f">>> Total predictions: {num_predict}")
    print(">>> Evaluation complete!")
