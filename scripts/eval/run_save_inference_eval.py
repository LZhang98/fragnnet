"""
FragGNN Inference Evaluation Script

This script runs inference on a trained FragGNN model and evaluates its performance
on various metrics including cosine similarity and Hungarian algorithm matching.
The script saves predictions along with evaluation scores for analysis.

Usage:
    python run_save_inference_eval.py --wandb_run_id <run_id> --eval_split test
"""

import numpy as np
import pandas as pd
from tqdm import tqdm
from fragnnet.runner import load_config, init_dataloader
from fragnnet.pl_model import FragGNNPL
from fragnnet.dataset import SpecMolFragDataset
from fragnnet.utils.script_utils import run_inference
from fragnnet.utils.setup_utils import get_model_cls, get_dataset_cls
from fragnnet.utils.misc_utils import booltype, get_best_ckpt_from_wandb, to_device
from fragnnet.utils.nn_utils import decompile_jit_ckpt
import torch as th
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
		default="config/exp_fraggnn_ma_mi/fraggnn_d3_ma_mi_nist20v2_amolc.yml",
		help="Path to custom config file"
	)
	
	# Model loading arguments
	parser.add_argument(
		"--wandb_run_id",
		type=str,
		default="ikohsc16",
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
		default='saved_ckpts/nist20_inchikey_fraggnn_d3_ma_mi_all_mol_conditon_v2_ikohsc16.ckpt',
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
			"oos_prob",          # Predicted out-of-scope probability
		],
		help="Subset of outputs to include in results"
	)
	parser.add_argument(
		"--output_everything",
		type=booltype,
		default=False,
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

	args = parser.parse_args()
	
	# Model checkpoint loading logic
	if args.model_ckpt is None:
		# Download model from Weights & Biases if no local checkpoint provided
		os.makedirs(args.model_save_dp, exist_ok=True)	
		model_save_dp = args.model_save_dp
		print(f">> save model to {model_save_dp} from wandb")
		print(f">> wandb run id: {args.wandb_run_id}")
		ckpt_fp = get_best_ckpt_from_wandb(model_save_dp, args.wandb_run_id, use_cached = True)
	else:
		# Use provided local checkpoint
		ckpt_fp = args.model_ckpt
  
	# Extract evaluation parameters
	eval_split = args.eval_split
	save_dp = args.save_dp
	force_overwrite = args.force_overwrite
	
	# Determine output subset for results
	if args.output_everything:
		output_subset = None  # Include everything
	else:
		output_subset = set(args.output_subset)  # Use specified subset


	# Load and configure the model configuration
	template_fp = args.template_fp
	custom_fp = args.custom_fp

	print(f">>> Loading config {custom_fp}")
	config_d = load_config(template_fp, custom_fp)
	
	# Modify config to record relevant metrics and optimize for evaluation
	config_d["auxiliary_scores"] = args.auxiliary_scores  # Set evaluation metrics
	config_d["skip_extra_losses"] = args.skip_extra_losses
	config_d["compile"] = False  # Disable compilation for evaluation
	
	# Disable preprocessing if requested (speeds up evaluation)
	if args.disable_preproc:
		for k in ["spec", "mol", "frag", "magma", "ann"]:
			key = f"{k}_params"
			config_d[key]["preprocess"] = False
			if k in ["frag"]:
				config_d[key]["preload"] = False
	
	# Disable string outputs for efficiency
	config_d["output_formula_str"] = False
	config_d["spec_params"]["prec_type_str"] = False
	config_d["frag_params"]["formula_str"] = False
	
	# Disable various samplers for deterministic evaluation
	config_d["dynamic_batch_sampler"] = False
	config_d["group_sampler"] = False 
	config_d["simple_group_sampler"] = False
	
	# Set batch sizes from command line arguments
	config_d["train_batch_size"] = args.train_batch_size
	config_d["eval_batch_size"] = args.eval_batch_size

	# Initialize dataset and dataloader
	print(">>> Loading dataset")
	ds_cls = get_dataset_cls(config_d['model_type'])
	ds = ds_cls(split=eval_split, **config_d)
	dl = init_dataloader(ds, config_d)
	
	# Initialize model and device
	model_cls = get_model_cls(config_d['model_type']) #FragGNNPL
	device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")

	# Load model checkpoint
	ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)
	
	# Decompile JIT checkpoint if needed
	assert not config_d["compile"]
	ckpt = decompile_jit_ckpt(ckpt)
	
	# Initialize and load model state
	model = model_cls(**config_d)
	try:
		model.load_state_dict(ckpt["state_dict"], strict=True)
	except RuntimeError as e:
		print(f"> error when loading from checkpoint, will try with strict=False")
		model.load_state_dict(ckpt["state_dict"], strict=False)
	
	# Move model to device and set to evaluation mode
	model.to(device)
	model.eval()
 

	# Prepare evaluation score types based on configuration
	eval_score_types = []

	for score_type in config_d['auxiliary_scores']:
		if "_hun" in score_type:
			# Hungarian algorithm scores don't need binning
			eval_score_types.append(score_type)
		else:
			# Other scores need binning by m/z resolution
			for bin_res in config_d['eval_mz_bin_res']:
				eval_score_types.append(f"{score_type}_{bin_res}")
	
	# Run inference and collect predictions
	preds_l = []
	with th.inference_mode():
		for _, batch_input in tqdm(enumerate(dl), total=len(dl), desc=" Running inference"):
			# Extract batch metadata
			spec_ids = batch_input['spec_unique_id']
			mol_ids = batch_input['mol_id']

			# Move batch to device
			batch_input = to_device(batch_input, device)
			
			# Run model forward pass with evaluation metrics
			batch_result = model.inference_with_ground_truth_step(batch_input, split=config_d)
			batch_size = batch_input['batch_size'].detach().cpu()

			# Extract predictions
			pred_mzs = batch_result["pred_mzs"].detach().cpu()
			pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()
			
			# Check for out-of-scope predictions
			has_oos_b = "pred_oos_logprobs" in batch_result
			if has_oos_b:
				pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()
	
			# Convert log probabilities to intensities
			pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
			pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)
			
			# Extract evaluation scores
			eval_scores_d = {}
			for eval_score_type in eval_score_types:
				eval_scores_d[eval_score_type] = batch_result[eval_score_type].detach().cpu()
			
			# Process each spectrum in the batch
			for b_idx in range(batch_size):
				# Get peaks for this spectrum
				b_peak_mask = pred_batch_idxs == b_idx
	
				# Extract prediction data for this spectrum
				b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy().item() if has_oos_b else np.nan
				b_spec_id = spec_ids[b_idx]
				b_mol_id = mol_ids[b_idx]

				b_spec_ce = batch_input['spec_ce'][b_idx].item() if 'spec_ce' in batch_input else np.nan
				b_spec_prec_type = batch_input['spec_prec_type'][b_idx].item() if 'spec_prec_type' in batch_input else np.nan
				b_spec_inst_type = batch_input['spec_inst_type'][b_idx].item() if 'spec_inst_type' in batch_input else np.nan
				b_spec_mzs = pred_mzs[b_peak_mask]
				b_spec_ints = pred_ints[b_peak_mask]

				# Create data row with basic prediction info
				data_row = [b_spec_id.item(), b_mol_id, b_spec_ce, b_spec_prec_type, b_spec_inst_type, 
						   b_spec_mzs.tolist(), b_spec_ints.tolist(), b_pred_oos_logprob]
				
				# Add evaluation scores to the row
				for eval_score_type in eval_score_types:
					data_row.append(eval_scores_d[eval_score_type][b_idx].item())
				preds_l.append(data_row)
			
			# Uncomment for debugging (process only one batch)
			# break

	# Create results DataFrame
	columns = ['spec_id','mol_id','spec_ce','spec_prec_type','spec_inst_type',
			   'pred_mzs','pred_ints','pred_oos'] + eval_score_types
	df = pd.DataFrame(preds_l, columns=columns)
	
	# Save results
	model_save_dp = os.path.join(args.save_dp, args.wandb_run_id)
	os.makedirs(model_save_dp, exist_ok=True)
	
	# Print evaluation score statistics
	print("\n>>> Evaluation Score Statistics:")
	for eval_score_type in eval_score_types:
		mean_score = df[eval_score_type].mean()
		std_score = df[eval_score_type].std()
		print(f"{eval_score_type}: mean={mean_score:.4f}, std={std_score:.4f}")
	
	# Save predictions with evaluation scores
	output_path = os.path.join(model_save_dp, f"predictions_{eval_split}.pkl")
	df.to_pickle(output_path)
	print(f"\n>>> Saved predictions to: {output_path}")
	print(f">>> Total predictions: {len(df)}")
	print(f">>> Evaluation complete!")
