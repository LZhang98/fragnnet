from fragnnet.runner import load_config, init_dataloader
from fragnnet.pl_model import FragGNNPL
from fragnnet.dataset import SpecMolFragDataset
from fragnnet.utils.script_utils import run_inference
from fragnnet.utils.misc_utils import booltype, get_best_ckpt_from_wandb
from fragnnet.utils.nn_utils import decompile_jit_ckpt
import torch as th
import argparse
import os


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
		"--eval_split", 
		type=str, 
		default="test", 
		choices=["train","val","test","secondary"]
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
		default=False
	)
	parser.add_argument(
		"--auxiliary_scores", 
		type=str, 
		nargs="+", 
		default=[
			"cos_sim",
			"cos_hun",
			"true_oos_prob",
			"true_oos_e"
		]
	)
	parser.add_argument(
		"--skip_extra_losses", 
		type=booltype, 
		default=False
	)
	parser.add_argument(
		"--output_subset", 
		type=str, 
		nargs="+", 
		default=[
			"true_mzs",
			"true_logprobs",
			"true_batch_idxs",
			"pred_mzs",
			"pred_logprobs",
			"pred_batch_idxs",
			"true_oos_prob",
			"oos_prob",
		]
	)
	parser.add_argument(
		"--output_everything",
		type=booltype,
		default=False
	)
	parser.add_argument(
		"--disable_preproc",
		type=booltype,
		default=True
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
  
	eval_split = args.eval_split
	save_dp = args.save_dp
	force_overwrite = args.force_overwrite
	if args.output_everything:
		output_subset = None
	else:
		output_subset = set(args.output_subset)


	# find wandb checkpoints
	template_fp = args.template_fp
	custom_fp = args.custom_fp

	print(f">>> Loading config {custom_fp}")
	config_d = load_config(template_fp, custom_fp)
	# modify config to record relevant metrics
	config_d["auxiliary_scores"] = args.auxiliary_scores
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
	print(">>> Loading dataset")
	ds_cls = SpecMolFragDataset
	ds = ds_cls(split=eval_split, **config_d)
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
 
	# run inference
	batch_cutoff = int(1e6) # this is effectively inf
	nb_iso = False
	vals = run_inference(
		dl, 
		model, 
		device, 
		eval_split, 
		batch_cutoff, 
		nb_iso, 
		output_subset=output_subset, 
		untransform_spec=args.untransform_spec
	)
	model.cpu()
	# save vals
	model_save_dp = os.path.join(args.save_dp,args.wandb_run_id)
	os.makedirs(model_save_dp,exist_ok=True)
	seed_vals_fp = os.path.join(model_save_dp,f"{eval_split}.pkl")
	if os.path.isfile(seed_vals_fp):
		print(f"> Info: {seed_vals_fp} exists already")
		if force_overwrite:
			print(f"> Warning: overwriting {seed_vals_fp}")
			th.save(vals, seed_vals_fp)
	else:
		th.save(vals, seed_vals_fp)

