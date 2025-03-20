import multiprocessing
import tempfile
import torch as th
import pandas as pd

import os
import argparse
import json

from frag_gnn.runner import load_config
from frag_gnn.iceberg.pl_model import IcebergGenPL, IcebergIntenPL
from frag_gnn.iceberg.dataset import SpecMolMagmaIntenDataset,SpecMolMagmaGenDataset
from frag_gnn.utils.misc_utils import to_device, booltype
from frag_gnn.utils.data_utils import par_apply, seq_apply
from rdkit import RDLogger
from rdkit import Chem
from tqdm import tqdm
import time

def run_iceberg_inten(
				 iceberg_inten_ckpt_fp,   
				 template_fp,
				 custom_fp,
				 magma_dp,
     			 proc_dp,
				 device = None,
				 batch_size = 1):

	print("> run_iceberg_inten")
	config_d = load_config(template_fp, custom_fp)
	config_d['mol_fp'] =  os.path.join(proc_dp,'mol_df.pkl.gz')
	config_d['spec_fp'] =  os.path.join(proc_dp,'spec_df.pkl.gz')
	config_d['split_dp'] = proc_dp
	config_d['magma_dp'] =  magma_dp
 
	config_d['spec_params']['prec_mz'] = True
	config_d['spec_params']['unique_id'] = False
	config_d['spec_params']['counts'] = False
	config_d['eval_batch_size'] = batch_size
	config_d['split'] = "predict"
	config_d['num_workers']= 24
	
	if device is None:
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
		
	if device == "cpu":
		th.set_num_threads(multiprocessing.cpu_count())
		th.multiprocessing.set_sharing_strategy('file_system')
 
	
	ds = SpecMolMagmaIntenDataset(**config_d)
	dl = th.utils.data.DataLoader(
		ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=config_d['num_workers'],
		collate_fn=ds.get_collate_fn(),
	)

	iceberg_inten_model = IcebergIntenPL.load_from_checkpoint(iceberg_inten_ckpt_fp, map_location = device, strict=False, **config_d)
	iceberg_inten_model.eval()

	def wrapper(gen):
		while True:
			try:
				yield next(gen)
			except StopIteration:
				break
			except Exception as e:
				print(e)

	for _, batch_input in tqdm(enumerate(wrapper(iter(dl))),total=len(dl), desc="Running iceberg inten",leave=False):
		to_device(batch_input, device)
		iceberg_inten_model.predict_step(**batch_input)

def run_iceberg_gen(iceberg_gen_inputs,
				 device = "cpu",
				 ):

	def single_predict_mol(iceberg_gen_ckpt_fp, adduct, inchi, smi, spec_id, magma_dp, threshold, max_nodes):
		out_file =  f"{magma_dp}/magma_tree/{spec_id}.json"
	
		iceberg_gen_model = IcebergGenPL.load_from_checkpoint(iceberg_gen_ckpt_fp, map_location = device)
		iceberg_gen_model.eval()

		tree_processor = SpecMolMagmaGenDataset.init_tree_processor(
			pe_embed_k=iceberg_gen_model.hparams.magma_params["pe_embed_k"],
			root_encode=iceberg_gen_model.hparams.magma_params["root_encode"],
			add_hs=iceberg_gen_model.hparams.magma_params["add_hs"]
		)
		try:
			pred = iceberg_gen_model.predict_mol(
				smi,
				adduct=adduct,
				threshold=threshold,
				device=device,
				max_nodes=max_nodes,
				tree_processor=tree_processor
			)
		except Exception as error:
			print("Error",error, adduct, inchi, smi, spec_id)
			output = None
		else:
			output = {
				"root_inchi": inchi,
				"name": spec_id,
				"frags": pred,
				"adduct": adduct
			}

			with open(out_file, "w") as fp:
				json.dump(output, fp,indent=2)

		return output

	if device == 'cpu':
		results = par_apply(iter(iceberg_gen_inputs),single_predict_mol,True, return_as_generator=True)
		for res in tqdm(results, total = len(iceberg_gen_inputs), desc=f"Running iceberg gen on {device}", leave=False):
			continue
	else:
		for iceberg_gen_input in tqdm(iceberg_gen_inputs, desc=f"Running iceberg gen on {device}", leave=False):
			single_predict_mol(*iceberg_gen_input)

def get_args():
    
	parser = argparse.ArgumentParser()

 	#gen params
	parser.add_argument("--iceberg_gen_ckpt_fp", type=str,default="saved_ckpts/icml_inchikey/iceberg_gen/iceberg_gen.ckpt", help="")
	parser.add_argument("--threshold", type=float, default=0.0)
	parser.add_argument("--max_nodes", type=int, default=100)
	parser.add_argument("--iceberg_gen_device", type=str,default="cpu", help="")
	parser.add_argument("--proc_dp", type=str,default="data/benchmark/speed_benchmark/1000", help="")

	#iceberg_inten params
	parser.add_argument("--template_fp",type=str,default="config/template.yml")
	parser.add_argument("--custom_fp",type=str,default="config/nist20_d4_wr00_inchikey/iceberg_inten_100/s0.yml")
	parser.add_argument("--iceberg_inten_ckpt_fp", type=str,default="saved_ckpts/icml_inchikey/iceberg_inten_100_hun/iceberg_inten_100_hun.ckpt", help="")
	parser.add_argument("--iceberg_inten_device", type=str,default="cpu", help="")
	parser.add_argument("--batch_size", type=int, default=32, help="")
	parser.add_argument("--save_fp", type=str, default="data/benchmark/speed_benchmark/1000/iceberg_noworker.txt", help="")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	RDLogger.DisableLog("rdApp.*")
	args = get_args()
	#os.makedirs(args.save_dp, exist_ok=True)
	#json.dump(vars(args), open(f"{args.save_dp}/args.json", "w"), indent=2)
 
	if not th.cuda.is_available():
		args.iceberg_gen_device = "cpu"
		args.iceberg_inten_device = "cpu"

	print(f"> iceberg_inten_device {args.iceberg_inten_device }")
 
	mol_fp = os.path.join(args.proc_dp,'mol_df.pkl.gz')
	spec_fp = os.path.join(args.proc_dp,'spec_df.pkl.gz')

	print(f"> prepare mol df {mol_fp}")
	mol_df = pd.read_pickle(mol_fp)
 
	with tempfile.TemporaryDirectory() as magma_dp:
		os.makedirs(f"{magma_dp}/magma_tree", exist_ok=True)
		mol_df['magma_dp'] = magma_dp
		mol_df['iceberg_gen_ckpt_fp'] = args.iceberg_gen_ckpt_fp
		mol_df['max_nodes'] = args.max_nodes
		mol_df['threshold'] = args.threshold
  		#mol_df['inchi'] = mol_df['mol'].apply(lambda x : Chem.MolToInchi(x))
		mol_df['spec_id'] = mol_df['mol_id']
		mol_df['prec_type'] = "[M+H]+"

		speed_d = {}
  		# run_iceberg_gen 
		iceberg_gen_inputs = mol_df[['iceberg_gen_ckpt_fp','prec_type','inchi','smiles','spec_id', 'magma_dp', 'threshold', 'max_nodes']].values.tolist()
		start = time.time()

		for device in ['cpu']:
			print(f"> run_iceberg_gen")
			run_iceberg_gen(iceberg_gen_inputs, device)
			end = time.time()
			time_cost = end - start
			print(f"> time used {time_cost}s")
			speed_d[f'iceberg_gen_{device}'] = time_cost

		#spec_df_grp.drop(columns = ['mol','smiles','inchi','magma_dp','iceberg_gen_ckpt_fp','threshold','max_nodes'], inplace = True)
		#print(magma_dp)
		#print(f"> run_iceberg_inten")

		for device in ['cuda']:
			for batch_size in [16,32,64,128]:
				start = time.time()
				run_iceberg_inten(iceberg_inten_ckpt_fp = args.iceberg_inten_ckpt_fp,
							template_fp = args.template_fp,
							custom_fp = args.custom_fp,
							proc_dp = args.proc_dp,
							magma_dp = magma_dp,
							device = device,
							batch_size = batch_size)
				end = time.time()
				time_cost = end - start
				print(f"> time used {time_cost}s")
				speed_d[f'iceberg_inten_{device}_{batch_size}'] = time_cost
    
	with open(args.save_fp, "w+") as f:
		for item in speed_d:
			f.write(f"{item},{speed_d[item]}\n")