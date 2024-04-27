import argparse
import json
import tempfile
import time
import torch as th
from tqdm import tqdm

from frag_gnn.runner import load_config
from frag_gnn.massformer.pl_model import MassFormerPL
from frag_gnn.dataset import SpecMolDataset, SpecMolFragDataset, init_dataloader

from frag_gnn.runner import load_config
from frag_gnn.pl_model import FragGNNPL, NeimsPL
from frag_gnn.massformer.pl_model import MassFormerPL
from frag_gnn.utils.misc_utils import to_device, booltype
from frag_gnn.dataset import SpecMolDataset, init_dataloader
from frag_gnn.utils import frag_utils, data_utils

import os
import multiprocessing
import pandas as pd

from tqdm import tqdm
import numpy as np

def run_speed_test(ckpt_fp,
				 template_fp,
				 custom_fp,
				 task_dp,
				 model_cls,
				 model_ds_cls,
				 save_fp,
				 frag_dp = None,
				 devices = ['cpu','cuda'],
				 batch_sizes = [1,4,16,32,64,128]):

	# load configs
	print("> Loading configs")
	config_d = load_config(template_fp, custom_fp)
	config_d['split_dp'] = task_dp

	#config_d['spec_params']['prec_mz'] = False
	config_d['spec_params']['unique_id'] = False
	config_d['spec_params']["counts"] = False

	# load datasets
	print(f"> Loading datasets {task_dp}")

	print("> preload mol_df")
	mol_df = pd.read_pickle(os.path.join(task_dp,'mol_df.pkl.gz'))

 
	print("> preload spec_fp")
	spec_df = pd.read_pickle(os.path.join(task_dp,'spec_df.pkl.gz'))

	print(f"> Loading models {ckpt_fp}")

	mol_df = mol_df
	spec_df = spec_df
 
	config_d['mol_fp'] = mol_df
	config_d['spec_fp'] = spec_df
	speed_d = {}
 
	# load models
	with tempfile.TemporaryDirectory() as frag_dp:
		frag_compressed = False
		config_d['frag_dp'] = frag_dp
		config_d['frag_params']['compressed'] = frag_compressed
		dag_feat_inputs_l = []
  
		for _, row in tqdm(mol_df.iterrows(), total=len(mol_df)):
			dag_feat_inputs_l.append([
									row['mol'],
									row['mol_id'], 
									config_d['num_depth'] - 1, #args.max_depth,
									True, # h_prior
									config_d['num_hs'], #args.max_h_transfer,
									150, #args.max_time,
									False, #args.isotopes, 
									False, #nb_isomorphic,
									-1,
									frag_dp,
									False, #args.allow_cached
									frag_compressed #compressed
									])
		start = time.time()
		print(f"> run frag gen")
		frag_results = data_utils.par_apply(iter(dag_feat_inputs_l),frag_utils.timed_get_dags,True, return_as_generator=True)
		for res in tqdm(frag_results, total = len(dag_feat_inputs_l), desc = "Compute Frags"):
			continue
		end = time.time()
		time_cost = end-start
		speed_d['frag gen'] = time_cost
		print(f" time cost {time_cost}s")

		for device in devices:
			if device == "cpu":
				th.set_num_threads(multiprocessing.cpu_count())
				th.multiprocessing.set_sharing_strategy('file_system')
				config_d['num_workers']= 0
			else:
				config_d['num_workers']= 24

			print("> load model")
			model = model_cls.load_from_checkpoint(ckpt_fp, map_location=device, strict=False, **config_d)
			model.eval()
			model.freeze()

			for batch_size in batch_sizes:
				start = time.time()
				print(f"> run predict")
				print(f"> device {device}, batch size {batch_size}")
				config_d['eval_batch_size'] = batch_size
				split_type = 'predict'
				ds = model_ds_cls(split=split_type, **config_d)
				dl = init_dataloader(ds, config_d)
				with th.no_grad():
					for _, batch_input in tqdm(enumerate(dl),total=len(dl)):
						to_device(batch_input, device)
						model.predict_step(**batch_input)
				end = time.time()
				time_cost = end-start
				speed_d[f'{device}_{batch_size}'] = time_cost
				print(f" time cost {time_cost}s")
	
	with open(save_fp, "w+") as f:
		for item in speed_d:
			f.write(f"{item},{speed_d[item]}\n")
   
def get_args():
	parser = argparse.ArgumentParser()
	
	parser.add_argument("--template_fp",type=str,default="config/template.yml")

	parser.add_argument("--device", type=str,default="cpu", help="")
	parser.add_argument("--batch_size", type=int,default=1, help="")
	parser.add_argument("--model_type", type=str,default="fragnnet", help="")

	parser.add_argument("--custom_fp",type=str,default="./config/retrieval/inchikey_fraggnn_d3_v2.yml")
	parser.add_argument("--ckpt_fp", type=str,default="./saved_ckpts/icml_inchikey/inchikey_fraggnn_d3_v2/inchikey_fraggnn_d3_v2.ckpt", help="")
	parser.add_argument("--proc_dp", type=str,default="data/benchmark/speed_benchmark/1000", help="")
	parser.add_argument("--save_fp", type=str,default="data/benchmark/speed_benchmark/1000/fraggnn_d3_moreworker.txt", help="")
	args = parser.parse_args()
	return args


if __name__ == "__main__":

	args = get_args()
	#json.dump(vars(args), open(f"{args.save_dp}/args.json", "w"), indent=2)
	
	model_type_to_model_cls = {
		"neims": NeimsPL,
		"massformer": MassFormerPL,
		"fragnnet": FragGNNPL,
		"fraggnn": FragGNNPL,
	}

	model_type_ds_cls = {
		"neims": SpecMolDataset,
		"massformer": SpecMolDataset,
		"fragnnet": SpecMolFragDataset,
		"fraggnn": SpecMolFragDataset,
	}
	if args.model_type not in model_type_ds_cls or args.model_type not in model_type_to_model_cls: 
		print(f"> model_type {args.model_type} is not supported")
		exit()
  
	run_speed_test(ckpt_fp = args.ckpt_fp,   
				 template_fp = args.template_fp,
				 custom_fp = args.custom_fp,
				 task_dp = args.proc_dp,
	 			 model_cls = model_type_to_model_cls[args.model_type],
				 model_ds_cls = model_type_ds_cls[args.model_type],
     			 save_fp=args.save_fp)
				 #device = args.device,
				 #batch_size = args.batch_size)
