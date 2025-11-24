import multiprocessing
import pandas as pd
import torch as th
import numpy as np
from tqdm import tqdm
import tempfile
import os
import gc
import copy
from typing import Dict, Any, Optional, Set, Type

from fragnnet.runner import load_config
from fragnnet.utils.misc_utils import to_device, scatter_reduce, flatten_lol, LOG_ZERO, get_best_ckpt_from_wandb
from fragnnet.pl_model import FraGNNetPL, NeimsPL, GNNPL
from fragnnet.iceberg.pl_model import IcebergIntenPL, IcebergGenPL
from fragnnet.massformer.pl_model import MassFormerPL
from fragnnet.utils.nn_utils import decompile_jit_ckpt
from fragnnet.utils.pl_utils import get_pl_hparams
from fragnnet.utils.setup_utils import get_model_cls, get_dataset_cls

def init_wandb_ckpt(run_id,last_ckpt,model_cls,config_d):
	import wandb
	# load model
	with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dp:
		api = wandb.Api()
		run = api.run(f"frag-gnn/frag-gnn/{run_id}")
		if last_ckpt:
			ckpt_wandb_fp = "ckpt/last.ckpt"
		else:
			files = run.files(per_page=100)
			best_ckpt_wandb_fp, best_epoch = None, None
			for file in files:
				if file.name.endswith(".ckpt") and "last" not in file.name:
					ckpt_wandb_fp = file.name
					epoch = int(os.path.basename(file.name).removeprefix("model-epoch=").removesuffix(".ckpt"))
					if best_ckpt_wandb_fp is None or epoch > best_epoch:
						best_ckpt_wandb_fp = ckpt_wandb_fp
						best_epoch = epoch
		assert ckpt_wandb_fp is not None
		ckpt_file_local = run.file(ckpt_wandb_fp).download(root=temp_dp,replace=False)
		ckpt_fp = ckpt_file_local.name
		ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)
		# decompile
		assert not config_d["compile"]
		ckpt = decompile_jit_ckpt(ckpt)
		# init
		model = model_cls(**config_d)
		try:
			model.load_state_dict(ckpt["state_dict"], strict=True)
		except RuntimeError as e:
			print(f"> error when loading from checkpoint id {run_id}, will try with strict=False")
			model.load_state_dict(ckpt["state_dict"], strict=False)
	return model

def select_model_vals(model_to_vals, keys, stack_dim=0):

	select_d = {k: [] for k in keys}
	for model, vals in model_to_vals.items():
		for k in keys:
			select_d[k].append(vals[k])
	select_d = {k: th.stack(v, dim=stack_dim) for k,v in select_d.items()}
	return select_d

def log_mean(log_p,dim=0):
	return th.logsumexp(log_p,dim=dim) - np.log(log_p.shape[dim])

def log_std(log_p,dim=0,correction=1):
	support_size = log_p.shape[dim]
	log_mean_p = log_mean(log_p,dim=dim)
	log_mean_p_sq = log_mean(2*log_p,dim=dim)
	log_var = th.log(th.exp(log_mean_p_sq) - th.exp(2*log_mean_p))
	return 0.5*(log_var + np.log(support_size) - np.log(support_size - correction))

# define sparse kl divergence
def sparse_kl(log_p,log_q,b_idx,dim_size):
	p = th.exp(log_p)
	kl = scatter_reduce(
		p * (log_p - th.clamp(log_q,min=LOG_ZERO(log_p.dtype))),
		b_idx,
		reduce="sum",
		dim_size=dim_size
	)
	kl = th.clamp(kl,min=0.0)
	return kl

def pearson_r(x,y,dim=0):
	x_mean = th.mean(x,dim=dim)
	y_mean = th.mean(y,dim=dim)
	x_std = th.std(x,dim=dim)
	y_std = th.std(y,dim=dim)
	r = th.mean((x-x_mean)*(y-y_mean),dim=dim) / (x_std * y_std)
	return r

def setup_device(
	config_d: Dict[str, Any],
	device: Optional[str] = None,
	enable_cpu_optimizations: bool = True,
) -> th.device:
	"""
	Prepare and configure the compute device for inference.
	
	Args:
		config_d: Configuration dictionary (should contain 'accelerator' key)
		device: Device string (e.g., "cpu", "cuda:0"). Auto-detects if None.
		enable_cpu_optimizations: Whether to enable CPU-specific optimizations
		
	Returns:
		Configured torch device
	"""
	if device is None:
		th_device = th.device("cuda:0") if config_d.get("accelerator") == "gpu" else th.device("cpu")
	elif not th.cuda.is_available() and device != "cpu":
		print(f"Warning: CUDA not available, falling back to CPU")
		th_device = th.device("cpu")
	else:
		th_device = th.device(device)

	print(f"> Using device: {th_device}")

	# CPU-specific optimizations
	if th_device.type == "cpu" and enable_cpu_optimizations:
		th.set_num_threads(multiprocessing.cpu_count())
		th.multiprocessing.set_sharing_strategy('file_system')
		th.use_deterministic_algorithms(True)
		print(f"  - CPU threads: {multiprocessing.cpu_count()}")
		print(f"  - Deterministic mode: enabled")

	return th_device

def get_ckpt_path(
	wandb_run_id: Optional[str],
	model_ckpt: Optional[str],
	model_save_dp: str,
	use_cached: bool = True,
) -> str:
	"""
	Get checkpoint path either from local file or download from W&B.
	
	Args:
		wandb_run_id: W&B run ID to download from
		model_ckpt: Local checkpoint path
		model_save_dp: Directory to save downloaded checkpoints
		use_cached: Whether to use cached W&B downloads
	
	Returns:
		Path to checkpoint file
	"""
	if model_ckpt is None:
		# Download from W&B
		os.makedirs(model_save_dp, exist_ok=True)
		print(f">> Downloading model to {model_save_dp} from W&B")
		print(f">> W&B run id: {wandb_run_id}")
		ckpt_fp = get_best_ckpt_from_wandb(model_save_dp, wandb_run_id, use_cached=use_cached)
	else:
		# Use local checkpoint
		ckpt_fp = model_ckpt
		print(f">> Using local checkpoint: {ckpt_fp}")
	
	return ckpt_fp

def init_model_from_ckpt(
	ckpt_fp: str,
	config_d: Dict[str, Any],
	device: th.device,
	map_location: Optional[str] = None,
) -> th.nn.Module:
	"""
	Initialize and load model from checkpoint file.
	Args:
		ckpt_fp: Path to checkpoint file
		config_d: Configuration dictionary
		device: Device to load model on
		map_location: Location to map checkpoint tensors (defaults to device)
		
	Returns:
		Loaded model
	"""
	if map_location is None:
		map_location = str(device)
	
	# Load checkpoint
	print(f">> Loading checkpoint from: {ckpt_fp}")
	ckpt = th.load(ckpt_fp, map_location=map_location, weights_only=False)
	
	# Handle compiled checkpoints
	if config_d.get('compile', False):
		print("> Checkpoint was compiled, decompiling...")
		ckpt = decompile_jit_ckpt(ckpt)
		config_d['compile'] = False
	
	# Get model class and initialize
	model_type = config_d['model_type']
	print(f">> Model type: {model_type}")
	model_cls = get_model_cls(model_type)
	
	model = model_cls(**config_d)
	
	# Load state dict
	try:
		model.load_state_dict(ckpt["state_dict"], strict=True)
		print(">> Model loaded successfully (strict=True)")
	except RuntimeError as e:
		print(f"> Warning: Error loading with strict=True, retrying with strict=False")
		print(f"  Error: {e}")
		model.load_state_dict(ckpt["state_dict"], strict=False)
		print(">> Model loaded with strict=False")
	
	model.to(device)
	model.eval()
	
	return model

  
def get_config_from_ckpt(
	ckpt_fp: str,
	custom_fp: Optional[str] = None,
	template_fp: Optional[str] = None,
) -> Dict[str, Any]:
	"""
	Load model and config from checkpoint file (using checkpoint hparams).
	
	This is the recommended approach for MS2C prediction scripts or when you
	want to use the exact configuration that was used during training.
	
	Args:
		ckpt_fp: Path to checkpoint file
		custom_fp: Path to custom config YAML to override checkpoint config
		template_fp: Path to template config YAML (required if custom_fp provided)
		
	Returns:
		Configuration dictionary
	"""
	# Load checkpoint to extract config
	print(">> Loading checkpoint to extract config")
	ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)
	
	print(">> Extracting config from checkpoint hparams")
	config_d = get_pl_hparams(ckpt)
	
	# Override with custom config if provided
	if custom_fp is not None:
		assert template_fp is not None, "template_fp required when custom_fp is provided"
	# Load and merge template and custom config
	if template_fp is not None:
		print(f">> Loading custom config from {custom_fp}")
		config_d_2 = load_config(template_fp, custom_fp)
		for k, v in config_d_2.items():
			if k in config_d and v != config_d[k]:
				print(f"  - Overwriting: {k} = {v} (was {config_d[k]})")
			elif k not in config_d:
				print(f"  - Adding: {k} = {v}")
			config_d[k] = v
	
	return config_d

# ============================================================================
# Inference Configuration and Setup Utilities
# ============================================================================
# 
# These functions provide modular utilities for inference evaluation scripts.
# They complement load_model_and_init_config() but serve different use cases:
#
# ============================================================================

def apply_inference_config(
	config_d: Dict[str, Any],
	auxiliary_scores: Optional[list] = None,
	skip_extra_losses: Optional[bool] = None,
	disable_preproc: Optional[bool] = False,
	eval_mz_bin_res: Optional[list] = None,
	dynamic_batch_sampler: Optional[bool] = None,
	num_workers: Optional[int] = None,
	output_formula_str: bool = False,
	train_batch_size: Optional[int] = None,
	eval_batch_size: Optional[int] = None,
	model_type: Optional[str] = None,
 	# Set dataframes in config for dataset initialization
	mol_src: Optional[pd.DataFrame | str] = None,
	spec_src: Optional[pd.DataFrame | str] = None,
) -> Dict[str, Any]:
	"""_summary_

	Args:
		config_d (Dict[str, Any]): _description_
		auxiliary_scores (list): _description_
		skip_extra_losses (bool): _description_
		disable_preproc (bool): _description_
		eval_mz_bin_res (Optional[list], optional): _description_. Defaults to None.
		dynamic_batch_sampler (Optional[bool], optional): _description_. Defaults to None.
		num_workers (Optional[int], optional): _description_. Defaults to None.
		output_formula_str (bool, optional): _description_. Defaults to False.
		train_batch_size (Optional[int], optional): _description_. Defaults to None.
		eval_batch_size (Optional[int], optional): _description_. Defaults to None.
		model_type (Optional[str], optional): _description_. Defaults to None.
		mol_df (Optional[pd.DataFrame], optional): _description_. Defaults to None.
		spec_df (Optional[pd.DataFrame], optional): _description_. Defaults to None.

	Returns:
		Dict[str, Any]: _description_
	"""
	# Common configuration modifications
	if auxiliary_scores is not None:
		config_d["auxiliary_scores"] = auxiliary_scores
	if skip_extra_losses is not None:
		config_d["skip_extra_losses"] = skip_extra_losses
	config_d["compile"] = False  # Disable compilation for evaluation
	
	# Disable preprocessing if requested
	if disable_preproc:
		for k in ["spec", "mol", "frag", "magma", "ann"]:
			key = f"{k}_params"
			config_d[key]["preprocess"] = False
			if k in ["frag"]:
				config_d[key]["preload"] = False
	
	# Handle formula string outputs
	config_d["output_formula_str"] = output_formula_str
	
	# Model-specific configurations
	if model_type and "fraggnn" in model_type:
		config_d["spec_params"]["prec_type_str"] = output_formula_str
		config_d["frag_params"]["formula_str"] = output_formula_str
	elif model_type and "iceberg" in model_type:
		if output_formula_str:
			config_d["magma_params"]["adduct_form_deltas"] = True
	elif model_type and "graff" in model_type:
		config_d["ann_params"]["formula_str"] = output_formula_str
	
	# Set batch sizes if provided
	if train_batch_size is not None:
		config_d["train_batch_size"] = train_batch_size
	if eval_batch_size is not None:
		config_d["eval_batch_size"] = eval_batch_size
	
	# config_d['auxiliary_scores'] = auxiliary_scores if auxiliary_scores is not None else []
	if eval_mz_bin_res is not None:
		config_d['eval_mz_bin_res'] = eval_mz_bin_res

	# Apply common inference settings
	config_d['track_datapoint_metrics'] = False
		
	# Disable preprocessing to save memory
	config_d['frag_params']['preload'] = False
	config_d['frag_params']['preprocess'] = False
	config_d['magma_params']['preprocess'] = False
	
	# Disable samplers for deterministic inference
	config_d["dynamic_batch_sampler"] = dynamic_batch_sampler if dynamic_batch_sampler is not None else False
	config_d["group_sampler"] = False
	config_d["simple_group_sampler"] = False
	
	if num_workers is not None:
		config_d['num_workers'] = num_workers
  
	# Set dataframes in config for dataset initialization
	if mol_src is not None:
		config_d['mol_fp'] = mol_src
	if spec_src is not None:
		config_d['spec_fp'] = spec_src
	return config_d

def save_inference_results(
	vals: Any,
	save_dp: str,
	model_type: str,
	seed: int,
	eval_split: str,
	force_overwrite: bool = False,
	split_type: Optional[str] = None,
) -> None:
	"""
	Save inference results to disk with proper directory structure.
	
	Args:
		vals: Results to save (typically a dictionary or DataFrame)
		save_dp: Base directory to save results
		model_type: Model type identifier
		seed: Random seed number
		eval_split: Dataset split name
		force_overwrite: Whether to overwrite existing files
		split_type: Optional split type (e.g., "inchikey", "scaffold")
	"""
	# Build save path
	if split_type is not None:
		seed_save_dp = os.path.join(save_dp, split_type, model_type, f"s{seed}")
	else:
		seed_save_dp = os.path.join(save_dp, model_type, f"s{seed}")
	
	os.makedirs(seed_save_dp, exist_ok=True)
	seed_vals_fp = os.path.join(seed_save_dp, f"{eval_split}.pkl")
	
	# Save with overwrite logic
	if os.path.isfile(seed_vals_fp):
		print(f"> Info: {seed_vals_fp} exists already")
		if force_overwrite:
			print(f"> Warning: overwriting {seed_vals_fp}")
			th.save(vals, seed_vals_fp)
		else:
			print(f"> Skipping save (use --force_overwrite to overwrite)")
	else:
		th.save(vals, seed_vals_fp)
		print(f"> Saved results to {seed_vals_fp}")


def cleanup_model(model: th.nn.Module, use_gpu: bool = True) -> None:
	"""
	Clean up model and free memory. This is useful after inference crashed to avoid memory leaks.
	
	Args:
		model: Model to clean up
		use_gpu: Whether GPU memory should be cleared
	"""
	model.cpu()
	del model
	gc.collect()
	
	if use_gpu:
		with th.no_grad():
			th.cuda.empty_cache()

def prepare_output_subset(
	output_everything: bool,
	output_subset: Optional[Set[str]],
	model_metric_names: Optional[Set[str]] = None,
) -> Optional[Set[str]]:
	"""
	Prepare output subset for inference results.
	
	Args:
		output_everything: Whether to output all metrics
		output_subset: Initial output subset
		model_metric_names: Model-specific metric names to include
	
	Returns:
		Final output subset (None if output_everything is True)
	"""
	if output_everything:
		return None
	
	if output_subset is None:
		output_subset = set()
	else:
		output_subset = set(output_subset)
	
	if model_metric_names is not None:
		output_subset = output_subset.union(model_metric_names)
	
	return output_subset


def get_idx_vars(
	output_subset: Optional[Set[str]],
	model,
	nb_iso: bool = False,
) -> Set[str]:
	"""
	Get final output keys based on subset selection.
	
	Args:
		output_subset: Subset of output keys to include (None = all)
		default_output_keys: Default set of output keys
	
	Returns:
		Final set of output keys
	"""
	# Determine which index variables to track
	fraggnn_model = isinstance(model, FraGNNetPL)
	iceberg_model = isinstance(model, IcebergIntenPL)
 
	if not (fraggnn_model or iceberg_model):
		assert not nb_iso
		BATCH_IDX_VARS = ["pred_batch_idxs", "true_batch_idxs"]
		NODE_IDX_VARS = []
		FORMULA_IDX_VARS = []
		NB_NODE_IDX_VARS = []
	elif iceberg_model:
		assert not nb_iso
		BATCH_IDX_VARS = ["pred_batch_idxs", "true_batch_idxs"]
		if model.model.output_formula_str:
			BATCH_IDX_VARS.extend(["pred_formula_batch_idxs"])
		NODE_IDX_VARS = []
		FORMULA_IDX_VARS = []
		NB_NODE_IDX_VARS = []
	else:  # fraggnn_model
		BATCH_IDX_VARS = [
			"pred_batch_idxs", "true_batch_idxs", "pred_formula_batch_idxs",
			"pred_node_batch_idxs", "pred_joint_batch_idxs"
		]
		NODE_IDX_VARS = ["pred_node_node_idxs", "pred_joint_node_idxs"]
		FORMULA_IDX_VARS = ["pred_formula_formula_idxs", "pred_joint_formula_idxs"]
		NB_NODE_IDX_VARS = []
		if nb_iso:
			BATCH_IDX_VARS.extend([
				"pred_nb_node_batch_idxs", "pred_nb_joint_batch_idxs",
				"pred_nb_node_node_batch_idxs"
			])
			FORMULA_IDX_VARS.extend(["pred_nb_joint_formula_idxs"])
			NB_NODE_IDX_VARS.extend([
				"pred_nb_node_node_idxs", "pred_nb_joint_node_idxs",
				"pred_nb_node_node_node_idxs"
			])
	
	if output_subset is not None:
		BATCH_IDX_VARS = [k for k in BATCH_IDX_VARS if k in output_subset]
		NODE_IDX_VARS = [k for k in NODE_IDX_VARS if k in output_subset]
		FORMULA_IDX_VARS = [k for k in FORMULA_IDX_VARS if k in output_subset]
		NB_NODE_IDX_VARS = [k for k in NB_NODE_IDX_VARS if k in output_subset]
  
	return BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, NB_NODE_IDX_VARS

def get_score_types(config_d):
	"""
	Generate a list of evaluation score types based on the configuration dictionary.
	
	Args:
		config_d: Configuration dictionary containing 'auxiliary_scores' and 'eval_mz_bin_res'
	
	Returns:
		List of evaluation score types
	"""
	eval_score_types = []
	for score_type in config_d['auxiliary_scores']:
		if "_hun" in score_type:
			# Hungarian algorithm scores don't need binning
			eval_score_types.append(score_type)
		else:
			# Other scores need binning by m/z resolution
			for bin_res in config_d['eval_mz_bin_res']:
				eval_score_types.append(f"{score_type}_{bin_res}")
	return eval_score_types

def run_inference_pkl(
	model,
	dataloader,
	device: th.device,
	batch_cutoff: int = -1, # max number of batches to eval (-1 = all)
	nb_iso: bool = False,
	output_set: Optional[Set[str]] = None,
	evaluate: bool = True
) -> Dict[str, Any]:
	"""
	Run inference with optional chunked saving for resumable predictions.
	"""
	# setup accumulators and index vars
	cum_num_datapoints = 0
	cum_num_nodes = 0
	cum_num_formulas = 0
	cum_num_nb_nodes = 0

	is_fragnnet_model = isinstance(model, FraGNNetPL)
	BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, NB_NODE_IDX_VARS = get_idx_vars(
		output_set,
		model,
		nb_iso,
	)


	# Setup for evaluation mode (return dict)
	vals = {}
	# Common setup
	print(f"\n>>> Starting inference on {len(dataloader)} batches...")
	if model.training:
		raise RuntimeError(f"> Warning: model {model} is in training mode!")
	
	# Run inference loops
	with th.inference_mode():
		for batch_idx, batch_input in tqdm(enumerate(dataloader), total=min(len(dataloader), batch_cutoff), 
										  desc="Running inference"):
	  
			# Early exit for batch cutoff
			if batch_idx >= batch_cutoff and batch_cutoff > 0:
				break
			
			# Move batch to device
			batch_input = to_device(batch_input, device, non_blocking=False)
			
			# Run inference (choose method based on mode)
			if evaluate:
				# Use ground truth evaluation method
				b_output = model.inference_with_ground_truth_step(batch_input, split="test")
			else:
				# Use prediction-only method
				b_output = model.inference_step(batch_input)
			
			b_output = to_device(b_output, "cpu", non_blocking=False)
			if output_set is not None:
				b_output = {k: b_output[k] for k in b_output if k in output_set}
	
			# ============================================================
			# EVALUATION MODE: Collect all outputs with index rebatching
			# ============================================================
			output_keys = b_output.keys()
			
			# Calculate batch counts for index rebatching
			if "pred_batch_idxs" in output_keys:
				b_num_datapoints = b_output["pred_batch_idxs"].max()+1
			elif "true_batch_idxs" in output_keys:
				b_num_datapoints = b_output["true_batch_idxs"].max()+1
			else:
				b_num_datapoints = 0
			
			if is_fragnnet_model and "pred_node_node_idxs" in output_keys:
				b_num_nodes = b_output["pred_node_node_idxs"].max()+1
			else:
				b_num_nodes = 0
			
			if is_fragnnet_model and "pred_formula_formula_idxs" in output_keys:
				b_num_formulas = b_output["pred_formula_formula_idxs"].max()+1
			else:
				b_num_formulas = 0
			
			if is_fragnnet_model and nb_iso and "pred_nb_node_node_idxs" in output_keys:
				b_num_nb_nodes = b_output["pred_nb_node_node_idxs"].max()+1
			else:
				b_num_nb_nodes = 0
			
			# Rebatch indices
			for k in BATCH_IDX_VARS:
				if k in b_output:
					b_output[k] = b_output[k] + cum_num_datapoints
			for k in NODE_IDX_VARS:
				if k in b_output:
					b_output[k] = b_output[k] + cum_num_nodes
			for k in FORMULA_IDX_VARS:
				if k in b_output:
					b_output[k] = b_output[k] + cum_num_formulas
			for k in NB_NODE_IDX_VARS:
				if k in b_output:
					b_output[k] = b_output[k] + cum_num_nb_nodes
			
			# Update cumulative counts
			cum_num_datapoints += b_num_datapoints
			cum_num_nodes += b_num_nodes
			cum_num_formulas += b_num_formulas
			cum_num_nb_nodes += b_num_nb_nodes
			
			# Collect outputs
			for k in b_output.keys():
				if b_output[k] is None:
					continue
				if k not in vals.keys():
					assert batch_idx == 0
					vals[k] = [b_output[k]]
				else:
					assert batch_idx > 0
					vals[k].append(b_output[k])
				
	# Consolidate collected outputs
	print(f"> num_datapoints = {cum_num_datapoints}")
	print(f"> num_nodes = {cum_num_nodes}")
	print(f"> num_formulas = {cum_num_formulas}")
	print(f"> num_nb_nodes = {cum_num_nb_nodes}")
	
	for k in vals.keys():
		if isinstance(vals[k][0], list):
			vals[k] = flatten_lol(vals[k])
		elif isinstance(vals[k][0], th.Tensor):
			if vals[k][0].dim() == 0:
				vals[k] = th.stack(vals[k], dim=0)
			else:
				vals[k] = th.cat(vals[k], dim=0)
		elif isinstance(vals[k][0], np.ndarray):
			if vals[k][0].ndim == 0:
				vals[k] = np.stack(vals[k], axis=0)
			else:
				vals[k] = np.concatenate(vals[k], axis=0)
		else:
			raise ValueError(f"unexpected type {type(vals[k][0])}")
	
	return vals

def run_inference_df(
	model,
	dataloader,
	device: th.device,
	batch_cutoff: int = -1, # max number of batches to eval (-1 = all)
	evaluate: bool = False,
	output_set: Optional[set[str]] = None,
	nb_iso: bool = False,
	min_peak_int: float = 0.0,
	min_frag_prob: float = 0.001,
) -> pd.DataFrame:
	"""
	Run inference with optional chunked saving for resumable predictions.
	"""
	# Common setup
	print(f"\n>>> Starting inference on {min(len(dataloader),batch_cutoff if batch_cutoff > 0 else len(dataloader))} batches...")
	if model.training:
		raise RuntimeError(f"> Warning: model {model} is in training mode!")
	
	BATCH_IDX_VARS, NODE_IDX_VARS, FORMULA_IDX_VARS, NB_NODE_IDX_VARS = get_idx_vars(
		output_set,
		model,
		nb_iso,
	)

	preds_l = []
 	#print(">> set model to this to eval")
	model.eval()

	with th.inference_mode():
		for batch_idx, batch_input in tqdm(enumerate(dataloader), total=min(len(dataloader), batch_cutoff), desc="Running inference"):
			# Early exit for batch cutoff
			if batch_idx >= batch_cutoff and batch_cutoff > 0:
				print(f"> reached batch cutoff of {batch_cutoff}, stopping inference")
				break

			spec_ids = batch_input['spec_unique_id']
			mol_ids = batch_input['mol_id']

			batch_input = to_device(batch_input, device)
			# use existing evaluate to compute cos sim and etc
			if evaluate:
				batch_result = model._common_step(batch_input, split="test", log=False)
			else:
				batch_result = model.predict_step(**batch_input)
	
			# batch_input['batch_size'] is a tensor -> convert to Python int
			batch_size = int(batch_input['batch_size'].detach().cpu().item())

			pred_mzs = batch_result["pred_mzs"].detach().cpu()
			pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()
			
			has_oos_b = "pred_oos_logprobs" in batch_result
			if has_oos_b:
				pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()
	
			pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
			pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)

			# eval_scores_d = {}
			#for eval_score_type in eval_score_types:
			#	eval_scores_d[eval_score_type] = batch_result[eval_score_type].detach().cpu()
			#print(eval_score_type, eval_scores_d[eval_score_type])

			# check forumla logprobs
			if "pred_formula_logprobs" in batch_result:
				pred_formula_logprobs = batch_result['pred_formula_logprobs']
				pred_formula_probs = th.exp(pred_formula_logprobs)
			else:
				pred_formula_probs = None

			# node forumla p(n|f)
			if "pred_formula_node_logprobs" in batch_result:
				pred_formula_node_logprobs = batch_result['pred_formula_node_logprobs']
			else:
				pred_formula_node_logprobs = None

			# node prob index to node index mapping
			# node prob at i of pred_formula_node_logprobs belongs to node pred_joint_node_idxs[i] 
			if "pred_joint_node_idxs" in NODE_IDX_VARS:
				pred_joint_node_idxs = batch_result['pred_joint_node_idxs']
			else:
				pred_joint_node_idxs = None

			# node prob index to formula index mapping
			# node prob at i of pred_formula_node_logprobs belongs to forumla pred_joint_formula_idxs[i] 
			if "pred_joint_formula_idxs" in FORMULA_IDX_VARS:
				pred_joint_formula_idxs = batch_result['pred_joint_formula_idxs']
			else:
				pred_joint_formula_idxs = None

			# forumla prob index -> forumla index, forumla f[i] forumla i
			# forumla prob at i of pred_formula_logprobs is for formula pred_formusla_formula_idxs[i]
			if "pred_formula_formula_idxs" in FORMULA_IDX_VARS:
				pred_formula_formula_idxs  = batch_result['pred_formula_formula_idxs']
			else:
				pred_formula_formula_idxs = None

			# forumla index -> peak index,  eg, forumla f[i] has peak i
			if "frag_formula_peak_idxs" in batch_result:
				pred_frag_formula_peak_idxs = batch_result['frag_formula_peak_idxs']
				pred_frag_formula_peak_mzs  = batch_result['frag_formula_peak_mzs'] 
			else:
				pred_frag_formula_peak_idxs = None
				pred_frag_formula_peak_mzs = None
	
			if "pred_joint_batch_idxs" in BATCH_IDX_VARS:
				# joint_batch_idxs maps each entry in the joint prediction arrays (like pred_formula_node_logprobs)
				# to its corresponding batch index (i.e., which spectrum/molecule in the batch it belongs to).
				# This is crucial for filtering predictions per sample in the batch loop below.
				joint_batch_idxs = batch_result['pred_joint_batch_idxs']
			else:
				joint_batch_idxs = None
			
			if "pred_formula_batch_idxs" in BATCH_IDX_VARS:
				pred_formula_batch_idxs = batch_result['pred_formula_batch_idxs']
			else:
				pred_formula_batch_idxs = None

			for b_idx in range(batch_size):
				# per-spectrum masks and basic fields
				b_peak_mask = pred_batch_idxs == b_idx
				b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy().item() if has_oos_b else np.nan
				b_spec_id = spec_ids[b_idx]
				b_mol_id = mol_ids[b_idx]
				b_spec_mzs = pred_mzs[b_peak_mask]
				b_spec_ints = pred_ints[b_peak_mask]

				if joint_batch_idxs is not None and pred_formula_node_logprobs is not None and pred_joint_formula_idxs is not None and pred_joint_node_idxs is not None:
					b_pred_formula_node_mask = (joint_batch_idxs == b_idx)
					b_pred_formula_node_logprobs = pred_formula_node_logprobs[b_pred_formula_node_mask]
					b_pred_joint_formula_idxs = pred_joint_formula_idxs[b_pred_formula_node_mask]
					b_pred_joint_node_idxs = pred_joint_node_idxs[b_pred_formula_node_mask]

				if pred_formula_batch_idxs is not None and pred_formula_formula_idxs is not None and pred_formula_probs is not None:
					b_pred_formula_mask = (pred_formula_batch_idxs == b_idx)
					b_pred_formula_formula_idxs = pred_formula_formula_idxs[b_pred_formula_mask]
					b_pred_formula_prob_idx = th.argsort(pred_formula_probs[b_pred_formula_mask], descending = True)
				else:
					b_pred_formula_prob_idx = []
    
				pred_formulas = []
				pred_peak_positions = []
				peak_labels = {}
				peak_id = 1
				for pf_idx in b_pred_formula_prob_idx:
					# get formula idx
					f_idx = b_pred_formula_formula_idxs[pf_idx]
					
					# find peak index for this formula
					peak_matches = th.nonzero(pred_frag_formula_peak_idxs == f_idx)
					if len(peak_matches) == 0:
						continue
					peak_idx = peak_matches[0].item()

					# in case peak is created by more then one forumla
					if peak_idx not in peak_labels:
						peak_labels[peak_idx] = peak_id
						peak_id += 1
						# store peak metadata (mz, int)
						pred_peak_positions.append({
							"mz": pred_mzs[peak_idx].item(),
							"int": pred_ints[peak_idx].item(),
							"peak_id": peak_labels[peak_idx]
						})

					# skip low intensity peaks
					if pred_ints[peak_idx] < min_peak_int:
						continue
  
					bf_pred_formula_mask = (b_pred_formula_formula_idxs == f_idx)
					f_mz = pred_frag_formula_peak_mzs[pred_frag_formula_peak_idxs == f_idx].item()
					# formula = idx_to_formula[f_idx.item()]
					f_pred_prob = pred_formula_probs[b_pred_formula_mask][bf_pred_formula_mask].item()
								
					formula_data = {
						"formula_idx": f_idx.item(), 
						#"formula": formula,
						"formula_prob": f_pred_prob,
						"formula_mass": f_mz,
						"peak_idx": peak_idx,
						"peak_id": peak_labels[peak_idx],
						"frag": []
					}
						
					if joint_batch_idxs is not None:
						bf_pred_formula_node_mask = (b_pred_joint_formula_idxs == f_idx)
						bf_pred_formula_node_logprobs = b_pred_formula_node_logprobs[bf_pred_formula_node_mask]
						bf_pred_formula_node_probs = th.exp(bf_pred_formula_node_logprobs)
						
						for pfn_idx in th.argsort(bf_pred_formula_node_probs, descending = True):
							
							node_idx = b_pred_joint_node_idxs[bf_pred_formula_node_mask][pfn_idx]
							if bf_pred_formula_node_probs[pfn_idx] < min_frag_prob:
								continue
							
							node_idx = node_idx.item()
							#node_cc_bit_mask = cc_bit_mask[node_idx]
							
							#node_atom_indices = node_cc_bit_mask.nonzero(as_tuple=True)[0].tolist()
							# fragment formula node probability, p(n|f)
							nf_pred_prob = bf_pred_formula_node_probs[pfn_idx].item()
							formula_data["frag"].append(
								{
									"node_atom_indices": node_idx,
									"formula_node_prob": nf_pred_prob
								}
							)
					pred_formulas.append(formula_data)

				# assemble row and append
				data_row = [b_spec_id.item(), b_mol_id, b_spec_mzs.tolist(), b_spec_ints.tolist(), b_pred_oos_logprob, pred_peak_positions, pred_formulas]
				preds_l.append(data_row)
   
		# build dataframe with new columns (after processing all batches)
		pred_df = pd.DataFrame(preds_l, columns=["spec_id","mol_id","pred_mzs","pred_ints","pred_oos_logprob","pred_peak_positions","pred_formulas"])
		print(pred_df.head())
		return pred_df
