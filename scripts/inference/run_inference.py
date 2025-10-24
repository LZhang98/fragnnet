import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from fragnnet.runner import load_config, init_dataloader
from fragnnet.pl_model import FraGNNetPL
from fragnnet.dataset import SpecMolFragDataset
from fragnnet.utils import data_utils, frag_utils
from fragnnet.utils.script_utils import run_inference
from fragnnet.utils.misc_utils import booltype, get_best_ckpt_from_wandb, to_device
from fragnnet.utils.nn_utils import decompile_jit_ckpt
import torch as th
import argparse
import os
import tempfile

if __name__ == "__main__":
	# Parse command line arguments
	parser = argparse.ArgumentParser()
	
	# Model configuration
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
	# Model checkpoint loading
	parser.add_argument(
		"--wandb_run_id",
		type=str,
		default="ikohsc16",
		help="W&B run ID to download checkpoint from (if model_ckpt not provided)"
	)
	parser.add_argument(
		"--model_save_dp",
		type=str,
		default='saved_ckpts',
		help="Directory to save/cache downloaded W&B checkpoints"
	)
	parser.add_argument(
		"--model_ckpt",
		type=str,
		default='saved_ckpts/nist20_inchikey_fraggnn_d3_ma_mi_all_mol_conditon_v2_ikohsc16.ckpt',
		help="Path to local model checkpoint (if None, will download from W&B using wandb_run_id)"
	)
	
	# Output configuration
	parser.add_argument(
		"--save_dp", 
		type=str, 
		default="output/inference_outputs",
		help="Directory to save inference evaluation results"
	)
	# Inference configuration
	parser.add_argument(
		"--skip_extra_losses", 
		type=booltype, 
		default=False,
		help="Skip computing auxiliary losses during inference for speed"
	)
	parser.add_argument(
		"--disable_preproc",
		type=booltype,
		default=True,
		help="Disable dataset preprocessing (set to True for inference to save memory)"
	)
	
	# Data paths
	parser.add_argument(
		"--proc_dp",
		type=str,
		required=True,
		help="Directory containing mol_df.pkl.gz and spec_df.pkl.gz files"
	)
	parser.add_argument(
		"--frag_dp",
		type=str,
		required=True,
		help="Directory containing precomputed fragment graphs; if None, will auto-generate in temp directory"
	)
	
	# Prediction filtering
	parser.add_argument(
		"--min_intensity",
		type=float,
		default=1.0,
		help="Minimum intensity threshold for peaks (filters out peaks below this value)"
	)
	
	# Performance tuning
	parser.add_argument(
		"--eval_batch_size",
		type=int,
		default=1,
		help="Batch size for evaluation/inference (use 1 for variable-size molecules)"
	)
	parser.add_argument(
		"--save_chunk_size",
		type=int,
		default=10_000,
		help="Number of predictions per chunk file (for resumable inference)"
	)
	parser.add_argument(
		"--save_interval",
		type=int,
		default=1_000,
		help="Write predictions to disk every N predictions (lower = more disk I/O but safer)"
	)
	args = parser.parse_args()
	

	# Load Model Checkpoint

	# Download from W&B if local checkpoint not provided
	if args.model_ckpt is None:
		os.makedirs(args.model_save_dp, exist_ok=True)	
		model_save_dp = args.model_save_dp
		print(f">> Downloading model to {model_save_dp} from W&B")
		print(f">> W&B run id: {args.wandb_run_id}")
		ckpt_fp = get_best_ckpt_from_wandb(model_save_dp, args.wandb_run_id, use_cached = True)
	else:
		ckpt_fp = args.model_ckpt
		print(f">> Using local checkpoint: {ckpt_fp}")
  
	save_dp = args.save_dp
	template_fp = args.template_fp
	custom_fp = args.custom_fp


	# Load and Modify Configuration

	print(f">>> Loading config from {custom_fp}")
	config_d = load_config(template_fp, custom_fp)
	
	# Modify config for inference mode (disable training-specific features)
	config_d["auxiliary_scores"] = []  # Disable auxiliary score computation for speed
	config_d["skip_extra_losses"] = args.skip_extra_losses
	config_d["compile"] = False  # Disable JIT compilation (not needed for inference)
	
	# Disable preprocessing to save memory (data already preprocessed)
	if args.disable_preproc:
		for k in ["spec", "mol", "frag", "magma", "ann"]:
			key = f"{k}_params"
			config_d[key]["preprocess"] = False
			if k in ["frag"]:
				config_d[key]["preload"] = False  # Don't preload all fragments into memory
	
	# Disable string outputs (save memory)
	config_d["output_formula_str"] = False
	config_d["spec_params"]["prec_type_str"] = False
	config_d["frag_params"]["formula_str"] = False
	
 	# Disable samplers (use sequential sampling for inference)
	config_d["dynamic_batch_sampler"] = False
	config_d["group_sampler"] = False 
	config_d["simple_group_sampler"] = False
	
	# Override batch size
	config_d["train_batch_size"] = 1
	config_d["eval_batch_size"] = args.eval_batch_size


	# Load Input Data (molecules and spectra)

	print("> Loading mol_df")
	mol_df = pd.read_pickle(os.path.join(args.proc_dp,'mol_df.pkl.gz'))
	print(f"  num mol: {len(mol_df)}")
	print("> Loading spec_df")
	spec_df = pd.read_pickle(os.path.join(args.proc_dp,'spec_df.pkl.gz'))
	print(f"  num spec: {len(spec_df)}")

	print(mol_df.head(5))
	print(spec_df.head(5))


	# Precompute/Load Fragment Graphs (required for FraGNNet models)

	# Fragment graphs (DAGs) are computationally expensive to generate
	# Options: 1) Use pre-computed fragments, 2) Generate in temporary directory
	temp_frag_dir = None
	if args.frag_dp is not None:
		# Use pre-computed fragments
		config_d['frag_dp'] = args.frag_dp
		print(f"> Using provided fragment directory: {config_d['frag_dp']}")
	else:
		# Auto-generate fragments in temporary directory (only for FraGNNet models)
		temp_frag_dir = tempfile.TemporaryDirectory() if config_d["model_type"] in ["frag_gnn","fragnnet"] else None
		if config_d["model_type"] in ["frag_gnn","fragnnet"]:
			print(f"> Using temporary fragment directory: {temp_frag_dir.name}")
			config_d['frag_dp'] = temp_frag_dir.name		

			# Prepare inputs for parallel fragment graph generation
			dag_feat_inputs_l = []
			for _, row in tqdm(mol_df.iterrows(), total=len(mol_df), desc="Preparing fragment inputs"):
				dag_feat_inputs_l.append([
										row['mol'],                             # RDKit molecule object
										row['mol_id'],                          # Molecule identifier
										config_d['num_depth'] - 1,              # Maximum fragmentation depth
										True,                                   # Use hydrogen prior
										config_d['num_hs'],                     # Maximum hydrogen transfers
										150,                                    # Maximum time for fragmentation (seconds)
										False,                                  # Include isotopes
										False,                                  # Number of isomorphic fragments
										-1,                                     # Random seed (-1 for no seed)
										config_d['frag_dp'],                    # Fragment directory (temporary)
										False,                                  # Don't use cached fragments
										config_d['frag_params']['compressed']   # Compression flag
										])
			
			# Time the fragment generation process
			start = time.time()
			print(f"> Running parallel fragment generation for {len(dag_feat_inputs_l)} molecules...")
			# Generate fragment graphs in parallel across CPU cores
			frag_results = data_utils.par_apply(iter(dag_feat_inputs_l), frag_utils.timed_get_dags, True, return_as_generator=True)
			# Consume the generator to complete all fragment computations
			for res in tqdm(frag_results, total=len(dag_feat_inputs_l), desc="Computing Fragments"):
				continue
			end = time.time()
			time_cost = end - start
			print(f"> Fragment generation completed in {time_cost:.2f}s ({len(dag_feat_inputs_l)/time_cost:.2f} mol/s)")
	
	# Filter input data to only include molecules with successfully generated fragments
	print(">> Filtering molecules without fragments...")
	all_frags = set([f.split('.')[0] for f in os.listdir(config_d['frag_dp'])])
	print(f">> Found {len(all_frags)} fragment files")
	mol_df = mol_df[mol_df['mol_id'].astype(str).isin(all_frags)]
	print(f"> Molecules after filtering: {len(mol_df)}")
	spec_df = spec_df[spec_df['mol_id'].isin(mol_df['mol_id'].to_list())]
	print(f"> Spectra after filtering: {len(spec_df)}")
  

	# Setup Output Directory and Resume from Existing Predictions

	model_save_dp = os.path.join(args.save_dp, args.wandb_run_id)
	os.makedirs(model_save_dp, exist_ok=True)	
	columns = ['spec_id','mol_id','spec_ce','spec_prec_type','spec_inst_type','pred_mzs','pred_ints','pred_oos']

	pkl_chunk_dp = os.path.join(model_save_dp, "chunks")
	os.makedirs(pkl_chunk_dp, exist_ok=True)

	# Check for existing chunk files and skip already-processed spectra (resumable inference)
	num_done = 0
	for file in os.listdir(pkl_chunk_dp):
		if file.endswith(".pkl"):
			chunk_fp = os.path.join(pkl_chunk_dp, file)
			print(f"Loading existing chunk: {chunk_fp}")
			existing_df = pd.read_pickle(chunk_fp)
			existing_spec_ids = set(existing_df['spec_id'])
			num_done += len(existing_spec_ids)
			spec_df = spec_df[~spec_df['spec_id'].isin(existing_spec_ids)]
	print(f">> Spectra to predict: {len(spec_df)} (skipped {num_done} already completed)")
	current_chunk_idx = num_done // args.save_chunk_size
	print(f">> Starting from chunk index: {current_chunk_idx}")

	# Set dataframes in config for dataset initialization
	config_d['mol_fp'] = mol_df
	config_d['spec_fp'] = spec_df
 

	# Initialize Dataset and DataLoader

	print(">>> Initializing dataset...")
	ds_cls = SpecMolFragDataset
	ds = ds_cls(split="predict_only", **config_d)
	dl = init_dataloader(ds, config_d)
	print(f">>> Dataset size: {len(ds)} spectra")
	

	# Load Model and Checkpoint

	model_cls = FraGNNetPL
	device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
	print(f">>> Using device: {device}")

	print(">>> Loading checkpoint...")
	ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)
	
	# Decompile JIT checkpoint if needed
	assert not config_d["compile"], "Compilation should be disabled for inference"
	ckpt = decompile_jit_ckpt(ckpt)
	
	# Initialize model and load weights
	model = model_cls(**config_d)
	try:
		model.load_state_dict(ckpt["state_dict"], strict=True)
		print(">>> Model loaded successfully (strict=True)")
	except RuntimeError as e:
		print(f"> Error loading checkpoint with strict=True, retrying with strict=False...")
		model.load_state_dict(ckpt["state_dict"], strict=False)
		print(">>> Model loaded with strict=False (some weights may be missing/extra)")
	
	model.to(device)
	model.eval()  # Set to evaluation mode


	# Run Inference Loop
	all_rows = []
	total_preds = 0
	pkl_fp = os.path.join(model_save_dp, "chunks", f"chunk_{current_chunk_idx}.pkl")
	print(f"\n>>> Starting inference on {len(dl)} batches...")
	print(f">>> Saving predictions every {args.save_interval} spectra")
	print(f">>> Chunk size: {args.save_chunk_size} spectra per file")
	print(f">>> Min intensity threshold: {args.min_intensity}\n")
 
	with th.inference_mode():
		for batch_idx, batch_input in tqdm(enumerate(dl), total=len(dl), desc="Running inference"):
			try:
				# Extract batch metadata
				spec_ids = batch_input['spec_unique_id']
				mol_ids = batch_input['mol_id']

				# Move batch to device and run inference
				batch_input = to_device(batch_input, device)
				batch_result = model.inference_step(batch_input)
				batch_size = batch_input['batch_size'].detach().cpu()

				# Extract predictions
				pred_mzs = batch_result["pred_mzs"].detach().cpu()
				pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()

				# Extract out-of-spectrum probability if available
				has_oos_b = "pred_oos_logprobs" in batch_result
				if has_oos_b:
					pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()

				# Untransform intensities (e.g., from log space back to linear)
				pred_ints = batch_result["pred_ints"].detach().cpu()
				pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)

				# Process each spectrum in the batch
				for b_idx in range(batch_size):
					# Get peaks for this spectrum
					b_peak_mask = pred_batch_idxs == b_idx

					# Extract metadata
					b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy().item() if has_oos_b else np.nan
					b_spec_id = spec_ids[b_idx]
					b_mol_id = mol_ids[b_idx]
					b_spec_ce = batch_input['spec_ce'][b_idx].item()
					b_spec_prec_type = batch_input['spec_prec_type'][b_idx].item()
					spec_inst_type = batch_input['spec_inst_type'][b_idx].item()
					b_spec_mzs = pred_mzs[b_peak_mask]
					b_spec_ints = pred_ints[b_peak_mask]

					# Apply intensity threshold filtering (remove low-intensity peaks)
					if args.min_intensity > 0:
						intensity_mask = b_spec_ints >= args.min_intensity
						b_spec_mzs = b_spec_mzs[intensity_mask]
						b_spec_ints = b_spec_ints[intensity_mask]

					# Store prediction
					data_row = [b_spec_id.item(), b_mol_id, b_spec_ce, b_spec_prec_type, spec_inst_type, b_spec_mzs.tolist(), b_spec_ints.tolist(), b_pred_oos_logprob]
					all_rows.append(data_row)
					total_preds += 1

				# Periodically save predictions to disk (every save_interval spectra)
				if total_preds % args.save_interval == 0:
					batch_df = pd.DataFrame(all_rows, columns=columns)
					all_rows = []
					
					# Check if we need to move to next chunk file
					if (total_preds + num_done) // args.save_chunk_size > current_chunk_idx:
						current_chunk_idx += 1
						pkl_fp = os.path.join(model_save_dp, "chunks", f"chunk_{current_chunk_idx}.pkl")
						print(f"> Moving to chunk {current_chunk_idx}: {pkl_fp}")
					
					# Append to existing chunk or create new one
					if os.path.exists(pkl_fp):
						existing_df = pd.read_pickle(pkl_fp)
						combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
						combined_df = combined_df.drop_duplicates(subset=["spec_id"])
					else:
						combined_df = batch_df
					combined_df.to_pickle(pkl_fp)
					print(f"Saved {len(combined_df)} predictions to {pkl_fp}")
					
			except RuntimeError as e:
				# Handle runtime errors (e.g., CUDA out of memory)
				print(f"Error at batch {batch_idx}: {e}")
				print("Skipping batch and clearing CUDA cache...")
				th.cuda.empty_cache()
				continue


	# Final Save and Cleanup

	# Save any remaining predictions not yet written
	if all_rows:
		print(f"\n>>> Saving final batch of {len(all_rows)} predictions...")
		batch_df = pd.DataFrame(all_rows, columns=columns)
		
		# Check if we need to move to next chunk
		if (total_preds + num_done) // args.save_chunk_size > current_chunk_idx:
			current_chunk_idx += 1
			pkl_fp = os.path.join(model_save_dp, "chunks", f"chunk_{current_chunk_idx}.pkl")
			print(f"> Moving to chunk {current_chunk_idx}: {pkl_fp}")
		
		# Append to existing or create new
		if os.path.exists(pkl_fp):
			existing_df = pd.read_pickle(pkl_fp)
			combined_df = pd.concat([existing_df, batch_df], ignore_index=True)
			combined_df = combined_df.drop_duplicates(subset=["spec_id"])
		else:
			combined_df = batch_df
		combined_df.to_pickle(pkl_fp)
		print(f"Final save: {len(combined_df)} predictions to {pkl_fp}")

	# Clean up temporary fragment directory if it was created
	if temp_frag_dir is not None:
		print(f"\n>>> Cleaning up temporary fragment directory: {temp_frag_dir.name}")
		temp_frag_dir.cleanup()
	
	print(f"\n>>> Inference complete!")
	print(f">>> Total predictions: {total_preds + num_done}")
	print(f">>> Output directory: {model_save_dp}")