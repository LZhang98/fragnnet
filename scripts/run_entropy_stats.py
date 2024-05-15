from frag_gnn.runner import load_config, init_dataset, init_dataloader
from frag_gnn.pl_model import FragGNNPL
from frag_gnn.dataset import SpecMolFragDataset
from frag_gnn.utils.misc_utils import scatter_reduce, scatter_argmax, EPS, booltype, scatter_masked_softmax, scatter_logsumexp
from frag_gnn.utils.script_utils import init_wandb_ckpt, run_inference, select_model_vals, sparse_kl, log_mean, log_std
from frag_gnn.loss import sparse_conditional_entropy_fn

import numpy as np
import torch as th
import pandas as pd
import argparse


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--debug", type=booltype, default=False)
	parser.add_argument("--mode", type=str, choices=["baseline","entropy"], default="entropy")
	parser.add_argument("--tw_thresh", type=float, default=0.025)
	parser.add_argument("--eval_split", type=str, default="test", choices=["train","val","test"])
	parser.add_argument("--batch_cutoff", type=int, default=int(1e6))
	args = parser.parse_args()

	debug = args.debug
	mode = args.mode
	tw_thresh = args.tw_thresh
	eval_split = args.eval_split
	pd.options.display.float_format = '{:.4f}'.format
	batch_cutoff = args.batch_cutoff

	# load configs
	print(">>> Loading configs")
	template_fp = "config/template.yml"
	custom_fp = "config/entropyv2/fraggnn_d3.yml"
	config_d = load_config(template_fp, custom_fp)

	# if debug:
	# 	print(">> debug mode, using small dataset")
	# 	config_d["spec_fp"] = "data/proc/debug_smallv2/spec_df.pkl"
	# 	config_d["mol_fp"] = "data/proc/debug_smallv2/mol_df.pkl"
	# 	config_d["split_dp"] = "data/split/debug_smallv2"
	# 	config_d["ann_fp"] = "data/proc/debug_smallv2/ann_df.pkl"

	if debug:
		if mode == "entropy":
			raise NotImplementedError
			# model_to_run_id = {
			# 	"d3_baseline": "45csl4n4",
			# 	"d3_posfn": "gzb25hmk",
			# 	"d3_negfn": "phg6ta1y"
			# }
		else:
			assert mode == "baseline"
			model_to_run_id = {
				"d3_debug_s0":  "vjrv1nzz",
				"d3_debug_s1": "dkztdob9",
				"d3_debug_s2": "rgw9m0jr"
			}
	else:
		raise NotImplementedError
		# if mode == "entropy":
		# 	model_to_run_id = {
		# 		"d3_posfn_s0": "8rsxiw68",
		# 		"d3_posfn_s1": "x6lrm3ry",
		# 		"d3_posfn_s2": "tu7dqx4x",
		# 		"d3_posfn_s3": "0n3r3c0a",
		# 		"d3_posfn_s4": "z89eby5f",
		# 		"d3_negfn_s5": "sezkh2lu",
		# 		"d3_negfn_s6": "2yba8u45",
		# 		"d3_negfn_s7": "nrq935fe",
		# 		"d3_negfn_s8": "gs6wku8w",
		# 		"d3_negfn_s9": "pbaierew",
		# 		"d3_baseline_s10": "br6jl4du",
		# 		"d3_baseline_s11": "fgxmpnbz",
		# 		"d3_baseline_s12": "r0xbwsyn",
		# 		"d3_baseline_s13": "me1cjx88",
		# 		"d3_baseline_s14": "qqg593r6",
		# 	}
		# else:
		# 	assert mode == "baseline"
		# 	model_to_run_id = {
		# 		"d3_baseline_s0": "u9wy1bez",
		# 		"d3_baseline_s1": "b41o0odi",
		# 		"d3_baseline_s2": "1i6owhp1",
		# 		"d3_baseline_s3": "kduczhxt",
		# 		"d3_baseline_s4": "ppiru3q2",
		# 		"d3_baseline_s5": "5anh3g2x",
		# 		"d3_baseline_s6": "4oaj4vf6",
		# 		"d3_baseline_s7": "p05lunqn",
		# 		"d3_baseline_s8": "kf2fuiya",
		# 		"d3_baseline_s9": "xzvo8kly",
		# 		"d3_baseline_s10": "br6jl4du",
		# 		"d3_baseline_s11": "fgxmpnbz",
		# 		"d3_baseline_s12": "r0xbwsyn",
		# 		"d3_baseline_s13": "me1cjx88",
		# 		"d3_baseline_s14": "qqg593r6",
		# 	}

	num_models = len(model_to_run_id.keys())

	# load datasets
	print(">>> Loading datasets")
	ds = init_dataset(config_d, splits=(eval_split,))[0]
	dl = init_dataloader(ds, config_d)

	model_to_vals = {}
	# model_to_model = {}
	for model_name, run_id in model_to_run_id.items():
		# load model
		print(f">>> Loading model {model_name} - {run_id}")
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
		model = init_wandb_ckpt(run_id,last_ckpt=False,model_cls=FragGNNPL,config_d=config_d)
		model.to(device)
		model.eval()
		# run inference
		vals = run_inference(dl, model, device, eval_split, batch_cutoff, config_d["nb_iso"])
		model_to_vals[model_name] = vals

	# compute ensemble distributions
	
	# P_ens(m)

	select_keys = [
		"pred_mzs",
		"pred_logprobs",
		"pred_batch_idxs",
		"true_mzs",
		"true_logprobs",
		"true_batch_idxs",
		"true_prec_mzs"
	]
	select_d = select_model_vals(model_to_vals,select_keys,stack_dim=0)
	# get target stuff
	true_logprobs = select_d["true_logprobs"][0]
	true_mzs = select_d["true_mzs"][0]
	true_batch_idxs = select_d["true_batch_idxs"][0]
	true_prec_mzs = select_d["true_prec_mzs"][0]

	ind_cos_sim_bin = []
	ind_cos_hun = []

	for i in range(num_models):
		
		print(f">> model {i}")
		
		# get predicted stuff
		pred_logprobs = select_d["pred_logprobs"][i]
		pred_mzs = select_d["pred_mzs"][i]
		pred_batch_idxs = select_d["pred_batch_idxs"][i]
		
		# calculate metrics
		with th.inference_mode():
			metric_input_d = {
				"true_mzs": true_mzs,
				"true_ints": th.exp(true_logprobs),
				"true_batch_idxs": true_batch_idxs,
				"pred_mzs": pred_mzs,
				"pred_ints": th.exp(pred_logprobs),
				"pred_batch_idxs": pred_batch_idxs,
				"true_prec_mzs": true_prec_mzs
			}
			metric_output_d = model.metric_fn(**metric_input_d)
		
		cos_sim_bin = metric_output_d["cos_sim_0.01"]
		cos_hun = metric_output_d["cos_hun"]
		ind_cos_sim_bin.append(cos_sim_bin)
		ind_cos_hun.append(cos_hun)
		
		print(f"> cos_sim_0.01: {cos_sim_bin.mean()}")
		print(f"> cos_hun: {cos_hun.mean()}")
	
	ind_cos_sim_bin = th.stack(ind_cos_sim_bin,dim=0)
	ind_cos_hun = th.stack(ind_cos_hun,dim=0)

	print(">> average individual")
	print(f"> cos_sim_0.01: {ind_cos_sim_bin.mean()}")
	print(f"> cos_hun: {ind_cos_hun.mean()}")

	ens_pred_logprobs = log_mean(select_d["pred_logprobs"],dim=0)
	ens_pred_mzs = select_d["pred_mzs"][0]
	ens_pred_batch_idxs = select_d["pred_batch_idxs"][0]
	
	with th.inference_mode():
		metric_input_d = {
			"true_mzs": true_mzs,
			"true_ints": th.exp(true_logprobs),
			"true_batch_idxs": true_batch_idxs,
			"pred_mzs": ens_pred_mzs,
			"pred_ints": th.exp(ens_pred_logprobs),
			"pred_batch_idxs": ens_pred_batch_idxs,
			"true_prec_mzs": true_prec_mzs
		}
		metric_output_d = model.metric_fn(**metric_input_d)

	ens_cos_sim_bin = metric_output_d["cos_sim_0.01"]
	ens_cos_hun = metric_output_d["cos_hun"]

	print(">> ensemble")
	print(f"> cos_sim_0.01: {ens_cos_sim_bin.mean()}")
	print(f"> cos_hun: {ens_cos_hun.mean()}")

	# P_ens(n|f), P_ens(f)

	select_keys = [
		"pred_joint_logprobs",
		"pred_formula_logprobs",
		"pred_formula_node_logprobs",
		"pred_formula_batch_idxs",
		"pred_joint_formula_idxs",
		"pred_joint_batch_idxs",
		"pred_formula_formula_idxs"
	]
	select_d = select_model_vals(model_to_vals,select_keys,stack_dim=0)

	ind_formula_node_e = []
	ind_formula_node_ne = []

	for i in range(num_models):
		
		print(f">> model {i}")
		
		pred_formula_node_logprobs = select_d["pred_formula_node_logprobs"][i]
		pred_formula_logprobs = select_d["pred_formula_logprobs"][i]
		pred_formula_batch_idxs = select_d["pred_formula_batch_idxs"][i]
		pred_joint_formula_idxs = select_d["pred_joint_formula_idxs"][i]
		pred_joint_batch_idxs = select_d["pred_joint_batch_idxs"][i]

		formula_node_e, formula_node_ne = sparse_conditional_entropy_fn(
			pred_formula_logprobs,
			pred_formula_batch_idxs,
			pred_formula_node_logprobs,
			pred_joint_formula_idxs,
			pred_joint_batch_idxs
		)

		ind_formula_node_e.append(formula_node_e)
		ind_formula_node_ne.append(formula_node_ne)

		print(f"> formula_node_e: {formula_node_e.mean()}")
		print(f"> formula_node_ne: {formula_node_ne.mean()}")

	ind_formula_node_e = th.stack(ind_formula_node_e,dim=0)
	ind_formula_node_ne = th.stack(ind_formula_node_ne,dim=0)

	print(">> average individual")
	print(f"> formula_node_e: {ind_formula_node_e.mean()}")
	print(f"> formula_node_ne: {ind_formula_node_ne.mean()}")

	ens_pred_joint_logprobs = log_mean(select_d["pred_joint_logprobs"],dim=0)
	ens_pred_formula_logprobs = log_mean(select_d["pred_formula_logprobs"],dim=0)
	ens_pred_formula_batch_idxs = select_d["pred_formula_batch_idxs"][0]
	ens_pred_joint_formula_idxs = select_d["pred_joint_formula_idxs"][0]
	ens_pred_joint_batch_idxs = select_d["pred_joint_batch_idxs"][0]
	ens_pred_formula_formula_idxs = select_d["pred_formula_formula_idxs"][0]
	assert th.all(th.arange(ens_pred_formula_formula_idxs.shape[0]) == ens_pred_formula_formula_idxs)
	# ens_pred_formula_logprobs = ens_pred_formula_logprobs[th.argsort(ens_pred_formula_formula_idxs)]
	ens_pred_formula_node_logprobs = ens_pred_joint_logprobs - ens_pred_formula_logprobs[ens_pred_joint_formula_idxs]
	ens_pred_formula_node_logprobs = th.clamp(ens_pred_formula_node_logprobs, max=0.)
	ens_pred_formula_node_lse = scatter_logsumexp(ens_pred_formula_node_logprobs, ens_pred_joint_formula_idxs, dim_size=ens_pred_formula_formula_idxs.shape[0])
	print(ens_pred_formula_node_lse.max(), ens_pred_formula_node_lse.min())

	ens_formula_node_e, ens_formula_node_ne = sparse_conditional_entropy_fn(
		ens_pred_formula_logprobs,
		ens_pred_formula_batch_idxs,
		ens_pred_formula_node_logprobs,
		ens_pred_joint_formula_idxs,
		ens_pred_joint_batch_idxs
	)

	print(">> ensemble")
	print(f"> formula_node_e: {ens_formula_node_e.mean()}")
	print(f"> formula_node_ne: {ens_formula_node_ne.mean()}")

	import pdb; pdb.set_trace()