import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from fragnnet.runner import load_config, init_dataloader
from fragnnet.pl_model import FragGNNPL
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
	parser = argparse.ArgumentParser()
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
	parser.add_argument(
		"--wandb_run_id",
		type=str,
		default="ikohsc16",
	)
	parser.add_argument(
		"--model_save_dp",
		type=str,
		default='saved_ckpts'
	)
	parser.add_argument(
		"--model_ckpt",
		type=str,
		default='saved_ckpts/nist20_inchikey_fraggnn_d3_ma_mi_all_mol_conditon_v2_ikohsc16.ckpt'
	)
	parser.add_argument(
		"--save_dp", 
		type=str, 
		default="output/inference_outputs",
		help="Directory to save inference evaluation results"
	)
	parser.add_argument(
		"--skip_extra_losses", 
		type=booltype, 
		default=False
	)
	parser.add_argument(
		"--disable_preproc",
		type=booltype,
		default=True
	)
	parser.add_argument(
		"--proc_dp",
		type=str,
		default="./data/other/mimedb/proc",
		help="Directory containing mol_df.pkl.gz and spec_df.pkl.gz files"
	)
	parser.add_argument(
		"--frag_dp",
		type=str,
		default="./data/other/mimedb/frag",
	)
	parser.add_argument(
		"--num_predictions",
		type=int,
		default=8,
		help="Number of predictions to output"
	)
	parser.add_argument(
		"--min_intensity",
		type=float,
		default=1.0,
		help="Minimum intensity for predictions n/1000"
	)
	parser.add_argument(
		"--eval_batch_size",
		type=int,
		default=1,
		help="Batch size for evaluation/inference"
	)
	args = parser.parse_args()
	if args.model_ckpt is None:
		os.makedirs(args.model_save_dp, exist_ok=True)	
		model_save_dp = args.model_save_dp
		print(f">> save model to {model_save_dp} from wandb")
		#assert args.wandb_run_id is not None
		print(f">> wandb run id: {args.wandb_run_id}")
		ckpt_fp = get_best_ckpt_from_wandb(model_save_dp, args.wandb_run_id, use_cached = True)
	else:
		ckpt_fp = args.model_ckpt
  
	save_dp = args.save_dp
	# find wandb checkpoints
	template_fp = args.template_fp
	custom_fp = args.custom_fp

	print(f">>> Loading config {custom_fp}")
	config_d = load_config(template_fp, custom_fp)
	# modify config to record relevant metrics
	config_d["auxiliary_scores"] = [] # disable auxiliary scores
	config_d["skip_extra_losses"] = args.skip_extra_losses
	config_d["compile"] = False
	if args.disable_preproc:
		for k in ["spec", "mol", "frag", "magma", "ann"]:
			key = f"{k}_params"
			config_d[key]["preprocess"] = False
			if k in ["frag"]:
				config_d[key]["preload"] = False
	# modify config for formula strings (don't output them)
	config_d["output_formula_str"] = False
	config_d["spec_params"]["prec_type_str"] = False
	config_d["frag_params"]["formula_str"] = False
 	# disable sampler
	config_d["dynamic_batch_sampler"] = False
	config_d["group_sampler"] = False 
	config_d["simple_group_sampler"] = False
	# override batch size
	config_d["train_batch_size"] = 1
	config_d["eval_batch_size"] = args.eval_batch_size

	print("> preload mol_df")
	mol_df = pd.read_pickle(os.path.join(args.proc_dp,'mol_df.pkl.gz'))
	print(f"  num mol: {len(mol_df)}")
	print("> preload spec_df")
	spec_df = pd.read_pickle(os.path.join(args.proc_dp,'spec_df.pkl.gz'))
	print(f"  num spec: {len(spec_df)}")

	print(mol_df.head(5))
	print(spec_df.head(5))

	# Precompute fragment graphs for FragGNN models (this is time-intensive)
	# this is for frag_gnn only
	# Create temporary directory for fragment generation

	if args.frag_dp is not None:
		config_d['frag_dp'] = args.frag_dp
		print(f"> Using provided fragment directory: {config_d['frag_dp']}")
	else:
		temp_frag_dir = tempfile.TemporaryDirectory() if config_d["model_type"] in ["frag_gnn","fragnnet"] else None
		if config_d["model_type"] in ["frag_gnn","fragnnet"]:
			print(f"> Using temporary fragment directory: {temp_frag_dir.name}")
			# Override fragment directory in config to use temporary directory
			config_d['frag_dp'] = temp_frag_dir.name		

			# Prepare inputs for parallel fragment graph generation
			dag_feat_inputs_l = []
			for _, row in tqdm(mol_df.iterrows(), total=len(mol_df)):
				dag_feat_inputs_l.append([
										row['mol'],           # RDKit molecule object
										row['mol_id'],        # Molecule identifier
										config_d['num_depth'] - 1, #args.max_depth,  # Maximum fragmentation depth
										True, # h_prior        # Use hydrogen prior
										config_d['num_hs'], #args.max_h_transfer,  # Maximum hydrogen transfers
										150, #args.max_time,   # Maximum time for fragmentation (seconds)
										False, #args.isotopes, # Include isotopes
										False, #nb_isomorphic, # Number of isomorphic fragments
										-1,                   # Random seed (-1 for no seed)
										config_d['frag_dp'],  # Fragment directory (now temporary)
										False, #args.allow_cached  # Don't use cached fragments
										config_d['frag_params']['compressed'] # Not compressed
										])
			
			# Time the fragment generation process
			start = time.time()
			print(f"> run frag gen")
			# Generate fragment graphs in parallel
			frag_results = data_utils.par_apply(iter(dag_feat_inputs_l),frag_utils.timed_get_dags,True, return_as_generator=True)
			# Consume the generator to complete all fragment computations
			for res in tqdm(frag_results, total = len(dag_feat_inputs_l), desc = "Compute Frags"):
				continue
			end = time.time()
			time_cost = end-start
			print(f" frag gen time cost {time_cost}s")
	
	# filter input data to only those with fragments
	print(">> filtering input without frags")
	all_frags = set([f.split('.')[0] for f in os.listdir(config_d['frag_dp'])])
	print(">> num of frags: ", len(all_frags))
	mol_df = mol_df[mol_df['mol_id'].astype(str).isin(all_frags)]
	print("> num of spec before filtering by dag", len(spec_df))
	spec_df = spec_df[spec_df['mol_id'].isin(mol_df['mol_id'].to_list())]
	print("> num of spec after filtering by dag", len(spec_df))

	config_d['mol_fp'] = mol_df
	config_d['spec_fp'] = spec_df

	print(">>> Loading dataset")
	ds_cls = SpecMolFragDataset
	ds = ds_cls(split="predict_only", **config_d)
	dl = init_dataloader(ds, config_d)
	model_cls = FragGNNPL
	device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")

	ckpt = th.load(ckpt_fp, map_location="cpu", weights_only=False)
	# decompile
	assert not config_d["compile"]
	ckpt = decompile_jit_ckpt(ckpt)
	# init
	model = model_cls(**config_d)
	try:
		model.load_state_dict(ckpt["state_dict"], strict=True)
	except RuntimeError as e:
		print(f"> error when loading from checkpoint, will try with strict=False")
		model.load_state_dict(ckpt["state_dict"], strict=False)
	model.to(device)
	model.eval()
 
	model_save_dp = os.path.join(args.save_dp, args.wandb_run_id)
	os.makedirs(model_save_dp, exist_ok=True)
	pkl_fp = os.path.join(model_save_dp, f"predictions.pkl")

	columns = ['spec_id','mol_id','spec_ce','spec_prec_type','spec_inst_type','pred_mzs','pred_ints','pred_oos']
	with th.inference_mode():
		for _, batch_input in tqdm(enumerate(dl), total=len(dl), desc=" Running inference"):
			spec_ids = batch_input['spec_unique_id']
			mol_ids = batch_input['mol_id']

			batch_input = to_device(batch_input, device)
			batch_result = model.inference_step(batch_input)
			batch_size = batch_input['batch_size'].detach().cpu()

			pred_mzs = batch_result["pred_mzs"].detach().cpu()
			pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()

			has_oos_b = "pred_oos_logprobs" in batch_result
			if has_oos_b:
				pred_oos_logprobs = batch_result["pred_oos_logprobs"].detach().cpu()

			pred_ints = batch_result["pred_ints"].detach().cpu()
			pred_ints = model.ints_untransform_func(pred_ints, pred_batch_idxs)

			rows_to_append = []
			for b_idx in range(batch_size):
				b_peak_mask = pred_batch_idxs == b_idx

				b_pred_oos_logprob = pred_oos_logprobs[b_idx].numpy().item() if has_oos_b else np.nan
				b_spec_id = spec_ids[b_idx]
				b_mol_id = mol_ids[b_idx]
				b_spec_ce = batch_input['spec_ce'][b_idx].item()
				b_spec_prec_type = batch_input['spec_prec_type'][b_idx].item()
				spec_inst_type = batch_input['spec_inst_type'][b_idx].item()
				b_spec_mzs = pred_mzs[b_peak_mask]
				b_spec_ints = pred_ints[b_peak_mask]

				data_row = [b_spec_id.item(), b_mol_id, b_spec_ce, b_spec_prec_type, spec_inst_type, b_spec_mzs.tolist(), b_spec_ints.tolist(), b_pred_oos_logprob]
				rows_to_append.append(data_row)

			# Append to pkl file after each batch
			#batch_df = pd.DataFrame(rows_to_append, columns=columns)
			#if os.path.exists(pkl_fp):
				# Append to existing file
			#	prev_df = pd.read_pickle(pkl_fp)
			#	new_df = pd.concat([prev_df, batch_df], ignore_index=True)
			#	new_df.to_pickle(pkl_fp)
			#else:
			#	batch_df.to_pickle(pkl_fp)

	# clean up temporary fragment directory
	if temp_frag_dir is not None:
		print(f"> Cleaning up temporary fragment directory: {temp_frag_dir.name}")
		temp_frag_dir.cleanup()