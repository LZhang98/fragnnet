from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.massformer.pl_model import MassFormerPL
from frag_gnn.iceberg.pl_model import IcebergIntenPL
from frag_gnn.dataset import SpecMolDataset, SpecMolFragDataset, init_dataloader
from frag_gnn.iceberg.dataset import SpecMolMagmaIntenDataset
from frag_gnn.utils.misc_utils import NestedDefaultDict, booltype
from frag_gnn.utils.script_utils import init_wandb_ckpt, run_inference

import torch as th
import argparse
import wandb
from pprint import pprint
from glob import glob
import os
import gc

if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--debug", type=booltype, default=False)
	parser.add_argument("--eval_split", type=str, default="test", choices=["train","val","test","secondary"])
	parser.add_argument("--num_seeds", type=int, default=5)
	parser.add_argument("--local_path", type=str, default="/scratch/hdd001/home/adamo/frag-gnn/local_runs")
	parser.add_argument("--vals_dp", type=str, default="/h/adamo/frag-gnn/inference_outputs")
	parser.add_argument("--force_overwrite", type=booltype, default=False)
	args = parser.parse_args()

	debug = args.debug
	eval_split = args.eval_split
	num_seeds = args.num_seeds
	local_path = args.local_path
	vals_dp = args.vals_dp
	force_overwrite = args.force_overwrite

	if not debug:
		split_types = ["inchikey","scaffold"]
		wandb_model_types = ["fraggnn_d3_v2","fraggnn_d4_v2","iceberg_inten_100","iceberg_inten_all"]
		local_model_types = ["neims_v2","massformer_v2"]
		batch_cutoff = int(1e6)
		seed_cutoff = num_seeds
	else:
		split_types = ["inchikey","scaffold"]
		wandb_model_types = []
		local_model_types = []
		batch_cutoff = int(1e6)
		seed_cutoff = num_seeds
	
	model_type_to_model_cls = {
		"neims": NeimsPL,
		"neims_v2": NeimsPL,
		"massformer": MassFormerPL,
		"massformer_v2": MassFormerPL,
		"fraggnn_d3": FragGNNPL,
		"fraggnn_d3_v2": FragGNNPL,
		"fraggnn_d4": FragGNNPL,
		"fraggnn_d4_v2": FragGNNPL,
		"iceberg_inten_100": IcebergIntenPL,
		"iceberg_inten_all": IcebergIntenPL,
	}

	model_type_ds_cls = {
		"neims": SpecMolDataset,
		"neims_v2": SpecMolDataset,
		"massformer": SpecMolDataset,
		"massformer_v2": SpecMolDataset,
		"fraggnn_d3": SpecMolFragDataset,
		"fraggnn_d3_v2": SpecMolFragDataset,
		"fraggnn_d4": SpecMolFragDataset,
		"fraggnn_d4_v2": SpecMolFragDataset,
		"iceberg_inten_100": SpecMolMagmaIntenDataset,
		"iceberg_inten_all": SpecMolMagmaIntenDataset,
	}

	model_to_run_id = NestedDefaultDict()
	model_to_config = NestedDefaultDict()

	# find wandb checkpoints
	template_fp = "config/template.yml"
	api = wandb.Api()
	for split_type in split_types:
		for model_type in wandb_model_types:
			group_name = f"nist20_{split_type}_{model_type}"
			runs = api.runs(
				path="frag-gnn/frag-gnn",
				filters={"group":group_name})
			model_to_config[split_type][model_type] = f"config/nist20_d4_wr00_{split_type}/{model_type}/s0.yml"
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
			model_to_config[split_type][model_type] = f"config/nist20_d4_wr00_{split_type}/{model_type}/s0.yml"
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
			print(">>> Loading dataset")
			ds_cls = model_type_ds_cls[model_type]
			ds = ds_cls(split=eval_split, **config_d)
			dl = init_dataloader(ds, config_d)
			model_cls = model_type_to_model_cls[model_type]
			device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
			for seed in range(seed_cutoff):
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
				output_subset = [
					"true_mzs",
					"true_logprobs",
					"true_batch_idxs",
					"pred_mzs",
					"pred_logprobs",
					"pred_batch_idxs",
					"true_oos_prob",
					"oos_prob",
				]
				output_subset = set(output_subset).union(model.metric_names)
				# print(output_subset)
				# run inference
				vals = run_inference(dl, model, device, eval_split, batch_cutoff, config_d["nb_iso"], output_subset=output_subset)
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
