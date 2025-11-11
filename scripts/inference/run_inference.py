import time
import pandas as pd
from tqdm import tqdm
from fragnnet.runner import init_dataloader
from fragnnet.dataset import SpecMolFragDataset
from fragnnet.utils import data_utils, frag_utils
from fragnnet.utils.script_utils import (
	get_ckpt_path,
	get_config_from_ckpt,
	setup_device,
	apply_inference_config,
	init_model_from_ckpt,
	run_inference,
)
from fragnnet.utils.misc_utils import booltype
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
		"--mol_fp",
		type=str,
		required=True,
		help="file path to mol_df.pkl.gz"
	)
	parser.add_argument(
		"--spec_fp",
		type=str,
		required=True,
		help="file path to spec_df.pkl.gz"
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
	

	# Load Model Checkpoint and Configuration

	# Get checkpoint path (download from W&B or use local)
	ckpt_fp = get_ckpt_path(
		wandb_run_id=args.wandb_run_id,
		model_ckpt=args.model_ckpt,
		model_save_dp=args.model_save_dp,
		use_cached=True,
	)
  
	save_dp = args.save_dp
	template_fp = args.template_fp
	custom_fp = args.custom_fp

	# Load model and config from checkpoint
	# This extracts config from checkpoint hparams and applies common inference settings
	print(">>> Loading model and configuration from checkpoint")	
	config_d = get_config_from_ckpt(
		ckpt_fp=ckpt_fp,
		custom_fp=custom_fp,
		template_fp=template_fp,
	)
	
	
	# Load Input Data (molecules and spectra)
	print("> Loading mol_df")
	mol_df = pd.read_pickle(args.mol_fp)
	print(f"  num mol: {len(mol_df)}")
	print("> Loading spec_df")
	spec_df = pd.read_pickle(args.spec_fp)
	print(f"  num spec: {len(spec_df)}")

	print(mol_df.head(5))
	print(spec_df.head(5))


	apply_inference_config(
		config_d,
		skip_extra_losses=args.skip_extra_losses,
		disable_preproc=args.disable_preproc,
		eval_batch_size=args.eval_batch_size,
	)
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
  
	# Set dataframes in config for dataset initialization
	config_d['mol_fp'] = mol_df
	config_d['spec_fp'] = spec_df
 
	# Initialize Dataset and DataLoader
	print(">>> Initializing dataset...")
	ds_cls = SpecMolFragDataset
	ds = ds_cls(split="predict_only", **config_d)
	dl = init_dataloader(ds, config_d)
	print(f">>> Dataset size: {len(ds)} spectra")

	device = setup_device(config_d)
	model = init_model_from_ckpt(ckpt_fp, config_d, device)
	print(f">>> Model loaded and ready on device: {device}")
	print(f">>> Model in eval mode: {not model.training}")
	# Run Inference with Chunked Savings
	total_preds = run_inference(
		model=model,
		dataloader=dl,
		device=device,
		output_dp=os.path.join(args.save_dp, args.wandb_run_id),
		save_chunk_size=args.save_chunk_size,
		save_interval=args.save_interval,
		min_intensity=args.min_intensity,
		resume=True,  # Resume from existing chunks
	)
 
	# Clean up temporary fragment directory if it was created
	if temp_frag_dir is not None:
		print(f"\n>>> Cleaning up temporary fragment directory: {temp_frag_dir.name}")
		temp_frag_dir.cleanup()
