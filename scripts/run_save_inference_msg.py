from frag_gnn.runner import load_config, init_dataloader
from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.massformer.pl_model import MassFormerPL
from frag_gnn.iceberg.pl_model import IcebergIntenPL
from frag_gnn.dataset import SpecMolDataset, SpecMolFragDataset
from frag_gnn.iceberg.dataset import SpecMolMagmaIntenDataset
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
	parser.add_argument("--split_types", type=str, nargs="+", default=["benchmark"])
	parser.add_argument("--wandb_model_types", type=str, nargs="+", default=["fragnnet_msg"])
	parser.add_argument("--eval_split", type=str, default="test", choices=["train","val","test","secondary"])
	parser.add_argument("--vals_dp", type=str, default="/scratch/hdd001/home/adamo/frag-gnn/inference_outputs/msg")
	parser.add_argument("--force_overwrite", type=booltype, default=False)
	parser.add_argument("--auxiliary_scores", type=str, nargs="+", default=["cos_hun","cos_sim","jss","jss_hun"])
	parser.add_argument("--eval_mz_bin_res", type=float, nargs="+", default=[0.1,0.01])
	parser.add_argument("--eval_bin_sqrt", type=booltype, default=True)
	parser.add_argument("--skip_extra_losses", type=booltype, default=True)
	args = parser.parse_args()

	eval_split = args.eval_split
	vals_dp = args.vals_dp
	force_overwrite = args.force_overwrite

	# this is necessary to avoid a pytorch compile issue...
	# TODO: replace this with a proper checkpoint decompilation
	th._dynamo.config.suppress_errors = True

	split_types = args.split_types
	wandb_model_types = args.wandb_model_types
	batch_cutoff = int(1e6)
	
	model_type_to_model_cls = {
		"fragnnet_msg": FragGNNPL,
	}

	model_type_to_ds_cls = {
		"fragnnet_msg": SpecMolFragDataset,
	}

	# model_type_to_dset = {
	# 	"fraggnn_msg": "nist20v2",
	# }

	model_to_run_id = {
		"benchmark": {
			"fragnnet_msg": "9gnx820u"
		}
	}
	model_to_config = {
		"benchmark": {
			"fragnnet_msg": "config/msg/msg_fragnnet_0,01_v2.yml"
		}
	}

	template_fp = "config/template.yml"

	pprint(model_to_config)
	pprint(model_to_run_id)

	model_types = wandb_model_types

	for split_type in split_types:
		for model_type in model_types:
			print(f">>> Starting model_type {model_type}")
			custom_fp = model_to_config[split_type][model_type]
			print(f">>> Loading config {custom_fp}")
			config_d = load_config(template_fp, custom_fp)
			# modify config to record relevant metrics
			config_d["auxiliary_scores"] = args.auxiliary_scores
			config_d["skip_extra_losses"] = args.skip_extra_losses
			config_d["eval_mz_bin_res"] = args.eval_mz_bin_res
			config_d["eval_bin_sqrt"] = args.eval_bin_sqrt
			# turn off compile
			config_d["compile"] = False
			print(">>> Loading model")
			model_cls = model_type_to_model_cls[model_type]
			device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
			run_id = model_to_run_id[split_type][model_type]
			model = init_wandb_ckpt(run_id,last_ckpt=False,model_cls=model_cls,config_d=config_d)
			model.to(device)
			model.eval()
			print(">>> Loading dataset")
			ds_cls = model_type_to_ds_cls[model_type]
			ds = ds_cls(split=eval_split, **config_d)
			dl = init_dataloader(ds, config_d)
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
			run_vals_dp = os.path.join(args.vals_dp,split_type,model_type)
			os.makedirs(run_vals_dp,exist_ok=True)
			run_vals_fp = os.path.join(run_vals_dp,f"{eval_split}.pkl")
			if os.path.isfile(run_vals_fp):
				print(f"> Info: {run_vals_fp} exists already")
				if force_overwrite:
					print(f"> Warning: overwriting {run_vals_fp}")
					th.save(vals, run_vals_fp)
			else:
				th.save(vals, run_vals_fp)
			del model
			del vals
			gc.collect()
			if config_d["accelerator"]=="gpu":
				with th.no_grad():
					th.cuda.empty_cache()				
