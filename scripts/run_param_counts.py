from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.massformer.pl_model import MassFormerPL
from frag_gnn.iceberg.pl_model import IcebergIntenPL, IcebergGenPL
from frag_gnn.utils.misc_utils import NestedDefaultDict, booltype

import argparse
from pprint import pprint
import pytorch_lightning as pl


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--debug", type=booltype, default=False)
	parser.add_argument("--local_path", type=str, default="/scratch/hdd001/home/adamo/frag-gnn/local_runs")
	args = parser.parse_args()

	debug = args.debug
	local_path = args.local_path

	if not debug:
		wandb_model_types = ["fraggnn_d3_v2","fraggnn_d4_v2","iceberg_gen","iceberg_inten_100","iceberg_inten_all"]
		local_model_types = ["neims_v2","massformer_v2"]
	else:
		wandb_model_types = [] 
		local_model_types = []
	
	model_types = wandb_model_types + local_model_types

	model_type_to_model_cls = {
		"neims": NeimsPL,
		"neims_v2": NeimsPL,
		"massformer": MassFormerPL,
		"massformer_v2": MassFormerPL,
		"fraggnn_d3": FragGNNPL,
		"fraggnn_d3_v2": FragGNNPL,
		"fraggnn_d4": FragGNNPL,
		"fraggnn_d4_v2": FragGNNPL,
		"iceberg_gen": IcebergGenPL,
		"iceberg_inten_100": IcebergIntenPL,
		"iceberg_inten_all": IcebergIntenPL,
	}

	model_to_config = NestedDefaultDict()

	# find wandb checkpoints
	template_fp = "config/template.yml"
	for model_type in model_types:
		if model_type == "iceberg_gen":
			model_to_config[model_type] = f"config/nist20_d4_wr00_inchikey/iceberg_gen.yml"
		else:
			model_to_config[model_type] = f"config/nist20_d4_wr00_inchikey/{model_type}/s0.yml"
	pprint(model_to_config)

	model_to_param_counts = NestedDefaultDict()

	for model_type in model_types:
		print(f">>> Starting model_type {model_type}")
		custom_fp = model_to_config[model_type]
		print(f">>> Loading config {custom_fp}")
		config_d = load_config(template_fp, custom_fp)
		config_d["accelerator"] = "cpu"
		print(f">>> Initializing model")
		model_cls = model_type_to_model_cls[model_type]
		model = model_cls(**config_d)
		if model_type in ["neims","massformer","neims_v2","massformer_v2"]:
			model_param_counts = pl.utilities.model_summary.ModelSummary(model, max_depth=4)
			model_idx = model_param_counts.layer_names.index("model")
			output_idxs = [model_param_counts.layer_names.index(f"model._orig_mod.ffn.{k}") for k in ["forw_out_layer","rev_out_layer","out_gate"]]
			model_to_param_counts[model_type] = model_param_counts.param_nums[model_idx]
			model_to_param_counts[f"{model_type}_output"] = \
				model_param_counts.param_nums[model_idx] - sum([model_param_counts.param_nums[i] for i in output_idxs])
		else:
			model_param_counts = pl.utilities.model_summary.ModelSummary(model, max_depth=1)
			model_idx = model_param_counts.layer_names.index("model")
			model_to_param_counts[model_type] = model_param_counts.param_nums[model_idx]
		del model

	if "iceberg_inten_100" in model_to_param_counts:
		assert "iceberg_gen" in model_to_param_counts
		model_to_param_counts["iceberg_inten_100"] += model_to_param_counts.pop("iceberg_gen")
	model_to_param_counts = {k: f"{v/1e6:.1f}" for k,v in model_to_param_counts.items()}
	pprint(model_to_param_counts)
	import pdb; pdb.set_trace()