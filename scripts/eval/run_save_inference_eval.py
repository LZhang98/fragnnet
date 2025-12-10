"""
FragGNN Inference Evaluation Script

This script runs inference on a trained FragGNN model and evaluates its performance
on various metrics including cosine similarity and Hungarian algorithm matching.
The script saves predictions along with evaluation scores for analysis.

Usage:
    python run_save_inference_eval.py --wandb_run_id <run_id> --eval_split test
"""
from fragnnet.runner import load_config, init_dataloader
from fragnnet.utils.script_utils import (
	get_ckpt_path,
	setup_device,
	apply_inference_config,
	init_model_from_ckpt,
	run_inference_df,
 	prepare_output_subset,
 	get_score_types,
)
from fragnnet.utils.setup_utils import get_model_cls, get_dataset_cls
from fragnnet.utils.misc_utils import booltype
import argparse
import os

if __name__ == "__main__":
	# Set up command line argument parser
	parser = argparse.ArgumentParser(description="Run FragGNN inference evaluation with metrics")
	
	# Configuration file arguments
	parser.add_argument(
		"--template_fp",
		type=str,
		default="config/template.yml",
		help="Path to the template config file"
	)
	parser.add_argument(
		"--custom_fp",
		type=str,
		default="config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20_amolc.yml",
		help="Path to custom config file"
	)
	
	# Model loading arguments
	parser.add_argument(
		"--wandb_run_id",
		type=str,
		default="32n55f2m",
		help="Weights & Biases run ID for model retrieval"
	)
	parser.add_argument(
		"--model_save_dp",
		type=str,
		default='saved_ckpts',
		help="Directory to save downloaded model checkpoints"
	)
	parser.add_argument(
		"--model_ckpt",
		type=str,
		default=None,
		help="Path to model checkpoint file (if not downloading from wandb)"
	)
	# Evaluation and output arguments
	parser.add_argument(
		"--eval_split", 
		type=str, 
		default="test", 
		choices=["train","val","test","secondary"],
		help="Dataset split to evaluate on"
	)
	parser.add_argument(
		"--save_dp", 
		type=str, 
		default="output/inference_eval",
		help="Directory to save inference evaluation results"
	)
	parser.add_argument(
		"--force_overwrite", 
		type=booltype, 
		default=False,
		help="Force overwrite existing results"
	)
	
	# Evaluation metrics configuration
	parser.add_argument(
		"--auxiliary_scores", 
		type=str, 
		nargs="+", 
		default=[
			"cos_sim",      # Cosine similarity between predicted and true spectra
			"cos_hun",      # Hungarian algorithm cosine similarity
			#"true_oos_prob",  # True out-of-scope probability
			#"true_oos_e"      # True out-of-scope energy
		],
		help="List of auxiliary evaluation scores to compute"
	)
	parser.add_argument(
		"--skip_extra_losses", 
		type=booltype, 
		default=False,
		help="Skip computation of extra loss terms"
	)
	# Output control arguments
	parser.add_argument(
		"--output_subset", 
		type=str, 
		nargs="+", 
		default=[
			"true_mzs",          # True m/z values from ground truth
			"true_logprobs",     # True log probabilities
			"true_batch_idxs",   # True batch indices
			"pred_mzs",          # Predicted m/z values
			"pred_logprobs",     # Predicted log probabilities
			"pred_batch_idxs",   # Predicted batch indices
			"true_oos_prob",     # True out-of-scope probability
			"pred_oos_prob",          # Predicted out-of-scope probability
		],
		help="Subset of outputs to include in results"
	)
	parser.add_argument(
		"--output_everything",
		type=booltype,
		default=True,
		help="Output all available metrics and predictions"
	)
	parser.add_argument(
		"--disable_preproc",
		type=booltype,
		default=True,
		help="Disable preprocessing to speed up evaluation"
	)
	
	# Batch size arguments
	parser.add_argument(
		"--train_batch_size",
		type=int,
		default=4,
		help="Training batch size for evaluation"
	)
	parser.add_argument(
		"--eval_batch_size",
		type=int,
		default=4,
		help="Evaluation batch size"
	)
	parser.add_argument(
		"--batch_cutoff",
		type=int,
		default=50,
		help="Number of batches to evaluate (-1 for all)"
	)
	args = parser.parse_args()
	
	# Get checkpoint path (download from W&B or use local)
	ckpt_fp = get_ckpt_path(
		wandb_run_id=args.wandb_run_id,
		model_ckpt=args.model_ckpt,
		model_save_dp=args.model_save_dp,
		use_cached=True
	)
  
	# Extract evaluation parameters
	eval_split = args.eval_split
	save_dp = args.save_dp
	force_overwrite = args.force_overwrite

	# Load and configure the model configuration
	template_fp = args.template_fp
	custom_fp = args.custom_fp

	config_d = load_config(template_fp, custom_fp)
	config_d = apply_inference_config(
		config_d,
		auxiliary_scores=args.auxiliary_scores,
		skip_extra_losses=args.skip_extra_losses,
		disable_preproc=args.disable_preproc,
		output_formula_str=False,
		train_batch_size=args.train_batch_size,
		eval_batch_size=args.eval_batch_size,
	)

	# Initialize dataset, dataloader, and model (inline)
	print(">>> Initializing components...")
	ds_cls = get_dataset_cls(config_d['model_type'])
	model_cls = get_model_cls(config_d['model_type'])

	# Initialize dataset and dataloader
	ds = ds_cls(split=eval_split, **config_d)
	dl = init_dataloader(ds, config_d)

	# Prepare device and load model from checkpoint
	device = setup_device(config_d)
	model = init_model_from_ckpt(ckpt_fp, config_d, device)
 
	# Prepare evaluation score types based on configuration

 
	# Run inference and collect predictions with evaluation scores
	print(">>> Running inference and evaluation...")
	output_set = prepare_output_subset(
		output_subset=args.output_subset,
		output_everything=args.output_everything
	)
	print(f">>> Output set: {output_set}")
	predicted_df = run_inference_df(
		model,
		dl,
		device,
		evaluate=True,
		output_set=output_set,
		batch_cutoff=args.batch_cutoff,
		min_peak_int=1.0,
		return_formula_and_cc=True,
		rebase_indices=True,
	)	
	
	# predicted_df.to_csv("inference_eval_results.csv", index=False)

	# Print evaluation score statistics
	print("\n>>> Evaluation Score Statistics:")
	eval_score_types = get_score_types(config_d=config_d)
	for eval_score_type in eval_score_types:
		mean_score = predicted_df[eval_score_type].mean()
		std_score = predicted_df[eval_score_type].std()
		print(f"{eval_score_type}: mean={mean_score:.4f}, std={std_score:.4f}")
	
	# Save predictions with evaluation scores
	output_save_dp = os.path.join(save_dp, args.wandb_run_id)
	os.makedirs(output_save_dp, exist_ok=True)
	output_path = os.path.join(output_save_dp, f"predictions_{eval_split}.pkl")
	predicted_df.to_pickle(output_path)
	print(f"\n>>> Saved predictions to: {output_path}")
	print(f">>> Total predictions: {len(predicted_df)}")
	print(f">>> Evaluation complete!")