import time
import torch as th
import pandas as pd

import os
import argparse
import json

from frag_gnn.iceberg.pl_model import IcebergGenPL
from frag_gnn.iceberg.dataset import SpecMolMagmaGenDataset
from frag_gnn.utils.misc_utils import booltype
from frag_gnn.utils.data_utils import par_apply, par_apply_series, seq_apply
from rdkit import Chem
from tqdm import tqdm
from frag_gnn.utils.misc_utils import booltype, get_best_ckpt_from_wandb
from frag_gnn.utils.ms2c_utils import load_model_and_init_config, run_spectra_prediction
from rdkit import RDLogger

def run_iceberg_gen(iceberg_gen_inputs,
				 device = None,
				 ):

	def single_predict_mol(iceberg_gen_ckpt_fp, adduct, inchi, smi, spec_id, magma_dp, threshold, max_nodes):
     
		# disbale  UserWarning: TypedStorage is deprecated.warning
		import warnings
		warnings.filterwarnings("ignore", category=DeprecationWarning) 

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
			print("Error", error, adduct, inchi, smi, spec_id)
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
		for res in tqdm(results, total = len(iceberg_gen_inputs), desc="Running iceberg gen", leave=False):
			continue
	else:
		for iceberg_gen_input in tqdm(iceberg_gen_inputs, desc="Running iceberg gen", leave=False):
			single_predict_mol(*iceberg_gen_input)
			results = seq_apply(iter(iceberg_gen_inputs),single_predict_mol,True)
			for res in tqdm(results, total = len(iceberg_gen_inputs), desc="Running iceberg gen", leave=False):
				continue

def get_args():
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str,default="cpu", help="")
	parser.add_argument("--save_fp", type=str,default="./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50/iceberg_inchikey_s0.pkl", help="")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_inchikey_test_10ppm_MORGAN-R2_50", help="")

	parser.add_argument("--use_cached_ckpt",type=booltype,default=True)
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")

 	#gen params
	parser.add_argument("--iceberg_gen_wandb_run_id",type=str, default= 'i7k9gc3d')
	parser.add_argument("--iceberg_gen_ckpt_fp", type=str,default=None, help="")
	parser.add_argument("--magma_dp", type=str,default="data/ms2c/pubchem/magma/iceberg_gen_inchikey_s0", help="")
	parser.add_argument("--threshold", type=float, default=0.0)
	parser.add_argument("--max_nodes", type=int, default=100)
	parser.add_argument("--iceberg_gen_device", type=str,default="cpu", help="")
	parser.add_argument("--save_magma",type=booltype,default=False)
	parser.add_argument("--use_precomputed_magma",type=booltype,default=False)
 
	#iceberg_intern params
	parser.add_argument("--iceberg_intern_wandb_run_id",type=str, default= 'c8iwu7il')
	parser.add_argument("--iceberg_intern_ckpt_fp", type=str,default=None, help="")
	parser.add_argument("--iceberg_intern_device", type=str,default="cpu", help="")
	parser.add_argument("--iceberg_intern_batch_size", type=int, default=1, help="")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	RDLogger.DisableLog("rdApp.warning")
	args = get_args()

	if not th.cuda.is_available():
		args.iceberg_gen_device = "cpu"
		args.iceberg_intern_device = "cpu"
		print("num parallelizing threads operations",th.get_num_threads())
  
	timestr = time.strftime("%Y%m%d-%H%M%S")
	save_dp = os.path.dirname(args.save_fp)
	os.makedirs(save_dp, exist_ok=True)
	records_dict = vars(args)

	os.makedirs(args.model_save_dp, exist_ok=True)	
	if args.iceberg_gen_ckpt_fp is None:
		model_save_dp = args.model_save_dp
		print(f">> save model to {args.model_save_dp} from wandb with run id: {args.iceberg_gen_wandb_run_id}")
		iceberg_gen_ckpt_fp = get_best_ckpt_from_wandb(args.model_save_dp, args.iceberg_gen_wandb_run_id, use_cached = args.use_cached_ckpt)
	else:
		iceberg_gen_ckpt_fp = args.iceberg_gen_ckpt_fp
	print(f"> iceberg_gen_ckpt_fp: {iceberg_gen_ckpt_fp}")
	records_dict['iceberg_gen_ckpt_fp'] = iceberg_gen_ckpt_fp
 
	if args.iceberg_intern_ckpt_fp is None:
		model_save_dp = args.model_save_dp
		print(f">> save model to {args.model_save_dp} from wandb with run id: {args.iceberg_intern_wandb_run_id}")
		iceberg_intern_ckpt_fp = get_best_ckpt_from_wandb(args.model_save_dp, args.iceberg_intern_wandb_run_id, use_cached = args.use_cached_ckpt)
	else:
		iceberg_intern_ckpt_fp = args.iceberg_intern_ckpt_fp
	print(f"> iceberg_intern_ckpt_fp: {iceberg_intern_ckpt_fp}")
	records_dict['iceberg_intern_ckpt_fp'] = iceberg_intern_ckpt_fp

	json.dump(records_dict, open(f"{args.save_fp}_{timestr}.json", "w"), indent=2)

	print(f"> iceberg_gen_device {args.iceberg_gen_device }")
	print(f"> iceberg_intern_device {args.iceberg_intern_device }")

	mol_fp = os.path.join(args.proc_dp,'mol_df.pkl.gz')
	spec_fp = os.path.join(args.proc_dp,'spec_df.pkl.gz')

	print(f"> prepare mol df {mol_fp}")
	mol_df = pd.read_pickle(mol_fp)

	print(f"> prepare spec df {spec_fp}")
	spec_df = pd.read_pickle(spec_fp)

	#this is for debug
	unique_group_ids = spec_df['group_id'].unique()
	print(unique_group_ids)
	unique_group_ids = unique_group_ids[0:10]
	spec_df = spec_df[spec_df['group_id'].isin(unique_group_ids)]

	print(f"> preapre iceberg gen/ magma")
	os.makedirs(f"{args.magma_dp}/magma_tree", exist_ok=True)
	iceberg_gen_df = spec_df.drop_duplicates(subset='group_id').copy()
	iceberg_gen_df.loc[:,'magma_dp'] = args.magma_dp
	iceberg_gen_df.loc[:,'iceberg_gen_ckpt_fp'] = args.iceberg_gen_ckpt_fp
	iceberg_gen_df.loc[:,'max_nodes'] = args.max_nodes
	iceberg_gen_df.loc[:,'threshold'] = args.threshold
	mol_df.loc[:,"inchi"] = par_apply_series(mol_df["mol"], lambda x : (RDLogger.DisableLog("rdApp.warning"), Chem.MolToInchi(x))[1])
	iceberg_gen_df = iceberg_gen_df.merge(mol_df[['mol_id','mol', 'smiles', "inchi"]], on = 'mol_id', how = 'left')

	# run_iceberg_gen 
	iceberg_gen_inputs = []
	if not args.use_precomputed_magma:
		iceberg_gen_inputs = iceberg_gen_df[['iceberg_gen_ckpt_fp','prec_type','inchi','smiles','group_id','magma_dp', 'threshold', 'max_nodes']].values.tolist()
	else:
		cached_group_ids = [f.split('.')[0] for f in os.listdir(args.magma_dp)]		
		iceberg_gen_inputs = iceberg_gen_df[iceberg_gen_df['group_id'].isin(cached_group_ids)][['iceberg_gen_ckpt_fp','prec_type','inchi','smiles','group_id','magma_dp', 'threshold', 'max_nodes']].values.tolist()
	if len(iceberg_gen_inputs):
		run_iceberg_gen(iceberg_gen_inputs, device = args.iceberg_gen_device)
	
	print(f"> Loading Iceberg_Inten models {iceberg_intern_ckpt_fp}")
	auxiliary_scores = ["cos_sim","cos_hun"]
	eval_mz_bin_res = [0.01]
	model, config_d, device = load_model_and_init_config(ckpt_fp = iceberg_intern_ckpt_fp, 
                                                      device = args.iceberg_intern_device, 
                                                      batch_size = args.iceberg_intern_batch_size, 
                                                      auxiliary_scores = auxiliary_scores,
                                                      eval_mz_bin_res = eval_mz_bin_res)

	df = run_spectra_prediction(model = model,   
				config_d = config_d,
				mol_data_ptr = mol_df,
				spec_data_ptr = spec_df,
				magma_dp = args.magma_dp,
				device = args.device,
    			validate=True)
	df.to_pickle(args.save_fp)