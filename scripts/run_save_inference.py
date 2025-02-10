from frag_gnn.runner import load_config, init_dataloader
from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.massformer.pl_model import MassFormerPL
from frag_gnn.iceberg.pl_model import IcebergIntenPL
from frag_gnn.graff.pl_model import GrAFFPL
from frag_gnn.dataset import SpecMolDataset, SpecMolFragDataset
from frag_gnn.iceberg.dataset import SpecMolMagmaIntenDataset
from frag_gnn.graff.dataset import SpecMolAnnDataset
from frag_gnn.utils.misc_utils import NestedDefaultDict, booltype
from frag_gnn.utils.script_utils import init_wandb_ckpt, run_inference

import torch as th
import torch._dynamo
import argparse
import wandb
from pprint import pprint
from glob import glob
import os
import gc

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument(
		"--split_types", 
		type=str, 
		nargs="+", 
		default=["inchikey","scaffold"]
	)
	parser.add_argument(
		"--wandb_model_types", 
		type=str, 
		nargs="+", 
		default=[
			"fraggnn_d3",
			"fraggnn_d4",
			"iceberg_inten3_100",
			"iceberg_inten3_all",
			"neims",
			"massformer"
		]
	)
	parser.add_argument(
		"--local_model_types", 
		type=str, 
		nargs="+", 
		default=[]
	)
	parser.add_argument(
		"--eval_split", 
		type=str, 
		default="test", 
		choices=["train","val","test","secondary"]
	)
	parser.add_argument(
		"--num_seeds", 
		type=int, 
		default=5
	)
	parser.add_argument(
		"--local_path", 
		type=str, 
		default="/scratch/hdd001/home/adamo/frag-gnn/local_runs"
	)
	parser.add_argument(
		"--vals_dp", 
		type=str, 
		default="/h/adamo/frag-gnn/inference_outputs"
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
		"--untransform_spec",
		type=booltype,
		default=False
	)
	parser.add_argument(
		"--disable_preproc",
		type=booltype,
		default=False
	)
	args = parser.parse_args()

	eval_split = args.eval_split
	num_seeds = args.num_seeds
	local_path = args.local_path
	vals_dp = args.vals_dp
	force_overwrite = args.force_overwrite
	if args.output_everything:
		output_subset = None
	else:
		output_subset = set(args.output_subset)

	# this is necessary to avoid a pytorch compile issue...
	# TODO: replace this with a proper checkpoint decompilation
	th._dynamo.config.suppress_errors = True

	split_types = args.split_types
	wandb_model_types = args.wandb_model_types
	local_model_types = args.local_model_types
	batch_cutoff = int(1e6)
	
	model_type_to_model_cls = {
		"neims": NeimsPL,
		"massformer": MassFormerPL,
		"fraggnn_d3": FragGNNPL,
		"fraggnn_d4": FragGNNPL,
		"iceberg_inten_100": IcebergIntenPL,
		"iceberg_inten_all": IcebergIntenPL,
		"iceberg_inten2_100": IcebergIntenPL,
		"iceberg_inten2_all": IcebergIntenPL,
		"iceberg_inten3_100": IcebergIntenPL,
		"iceberg_inten3_all": IcebergIntenPL,
		"graff": GrAFFPL
	}

	model_type_to_ds_cls = {
		"neims": SpecMolDataset,
		"massformer": SpecMolDataset,
		"fraggnn_d3": SpecMolFragDataset,
		"fraggnn_d4": SpecMolFragDataset,
		"iceberg_inten_100": SpecMolMagmaIntenDataset,
		"iceberg_inten_all": SpecMolMagmaIntenDataset,
		"iceberg_inten2_100": SpecMolMagmaIntenDataset,
		"iceberg_inten2_all": SpecMolMagmaIntenDataset,
		"iceberg_inten3_100": SpecMolMagmaIntenDataset,
		"iceberg_inten3_all": SpecMolMagmaIntenDataset,
		"graff": SpecMolAnnDataset
	}

	model_type_to_dset = {
		"neims": "nist20v3",
		"massformer": "nist20v3",
		"fraggnn_d3": "nist20v3",
		"fraggnn_d4": "nist20v3",
		"iceberg_inten_100": "nist20v3",
		"iceberg_inten_all": "nist20v3",
		"iceberg_inten2_100": "nist20v3",
		"iceberg_inten2_all": "nist20v3",
		"iceberg_inten3_100": "nist20v3",
		"iceberg_inten3_all": "nist20v3",
		"graff": "nist20v3"
	}

	model_to_run_id = NestedDefaultDict()
	model_to_config = NestedDefaultDict()

	# find wandb checkpoints
	template_fp = "config/template.yml"
	api = wandb.Api()
	for split_type in split_types:
		for model_type in wandb_model_types:
			dset = model_type_to_dset[model_type]
			group_name = f"{dset}_{split_type}_{model_type}"
			runs = api.runs(
				path="frag-gnn/frag-gnn",
				filters={"group":group_name})
			model_to_config[split_type][model_type] = f"config/{dset}_d4_wr00_{split_type}/{model_type}/s0.yml"
			for run in runs:
				seed = int(run.name.split("_")[-1].removeprefix("s"))
				if seed < num_seeds:
					model_to_run_id[split_type][model_type][seed] = run.id
	del api
	pprint(model_to_config)
	pprint(model_to_run_id)

	# find local checkpoints
	model_to_ckpt = NestedDefaultDict()
	for split_type in split_types:
		for model_type in local_model_types:
			dset = model_type_to_dset[model_type]
			model_to_config[split_type][model_type] = f"config/{dset}_d4_wr00_{split_type}/{model_type}/s0.yml"
			for seed in range(num_seeds):
				ckpt = glob(os.path.join(local_path,f"{model_type}/{split_type}/ckpts/s{seed}/*.ckpt"))[0]
				model_to_ckpt[split_type][model_type][seed] = ckpt
	pprint(model_to_config)
	pprint(model_to_ckpt)

	model_types = wandb_model_types + local_model_types

	for split_type in split_types:
		for model_type in model_types:
			print(f">>> Starting model_type {model_type}")
			custom_fp = model_to_config[split_type][model_type]
			print(f">>> Loading config {custom_fp}")
			config_d = load_config(template_fp, custom_fp)
			# modify config to record relevant metrics
			config_d["auxiliary_scores"] = args.auxiliary_scores
			config_d["skip_extra_losses"] = args.skip_extra_losses
			config_d["compile"] = False
			if args.disable_preproc:
				for k in ["spec", "mol", "magma", "ann"]:
					key = f"{k}_params"
					config_d[key]["preprocess"] = False
					if k == "magma":
						config_d[key]["preload"] = False
			print(">>> Loading dataset")
			ds_cls = model_type_to_ds_cls[model_type]
			ds = ds_cls(split=eval_split, **config_d)
			dl = init_dataloader(ds, config_d)
			model_cls = model_type_to_model_cls[model_type]
			device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
			for seed in range(num_seeds):
				print(f">>> Starting seed {seed}")
				if model_type in wandb_model_types:
					run_id = model_to_run_id[split_type][model_type][seed]
					model = init_wandb_ckpt(run_id,last_ckpt=False,model_cls=model_cls,config_d=config_d)
				else:
					assert model_type in local_model_types, model_type
					ckpt_fp = model_to_ckpt[split_type][model_type][seed]
					try:
						model = model_cls.load_from_checkpoint(ckpt_fp,strict=True,**config_d)
					except RuntimeError as e:
						print(f"> error when loading from checkpoint file {ckpt_fp}, will try with strict=False")
						model = model_cls.load_from_checkpoint(ckpt_fp,strict=False,**config_d)
				model.to(device)
				model.eval()
				# define output subset
				if not args.output_everything:
					output_subset = output_subset.union(model.metric_names)
				# run inference
				vals = run_inference(
					dl, 
					model, 
					device, 
					eval_split, 
					batch_cutoff, 
					config_d["nb_iso"], 
					output_subset=output_subset, 
					untransform_spec=args.untransform_spec
				)
				model.cpu()
				# save vals
				seed_vals_dp = os.path.join(args.vals_dp,split_type,model_type,f"s{seed}")
				os.makedirs(seed_vals_dp,exist_ok=True)
				seed_vals_fp = os.path.join(seed_vals_dp,f"{eval_split}.pkl")
				if os.path.isfile(seed_vals_fp):
					print(f"> Info: {seed_vals_fp} exists already")
					if force_overwrite:
						print(f"> Warning: overwriting {seed_vals_fp}")
						th.save(vals, seed_vals_fp)
				else:
					th.save(vals, seed_vals_fp)
				del model
				del vals
				gc.collect()
				if config_d["accelerator"]=="gpu":
					with th.no_grad():
						th.cuda.empty_cache()
