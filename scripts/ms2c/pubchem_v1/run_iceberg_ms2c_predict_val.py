import multiprocessing
import time
import torch as th
import pandas as pd

import os
import argparse
import json

#from frag_gnn.runner import load_config
#from frag_gnn.iceberg.pl_model import IcebergGenPL, IcebergIntenPL
#from frag_gnn.iceberg.pl_model import IcebergGenPL
#from frag_gnn.iceberg.dataset import SpecMolMagmaIntenDataset,SpecMolMagmaGenDataset
from frag_gnn.utils.misc_utils import to_device, booltype
from frag_gnn.utils.data_utils import par_apply, seq_apply
from rdkit import RDLogger
from rdkit import Chem
from tqdm import tqdm
import tarfile
from frag_gnn.utils.misc_utils import booltype, get_ckpt_from_wandb
from frag_gnn.utils.ms2c_utils import load_model_and_init_config, run_prediction

def run_iceberg_intern(
				 iceberg_intern_ckpt_fp,   
				 save_fp,
				 template_fp,
				 custom_fp,
				 task_dp,
				 magma_dp,
     			 mol_df = None,
				 spec_df = None,
				 device = None,
				 batch_size = 1,
			 ):

	#print("> run_iceberg_intern")
	config_d = load_config(template_fp, custom_fp)
	config_d['mol_fp'] =  os.path.join(task_dp,'mol_df.pkl.gz') if mol_df is None else mol_df
	config_d['spec_fp'] =  os.path.join(task_dp,'spec_df.pkl.gz') if spec_df is None else spec_df
	config_d['split_dp'] = task_dp
	config_d['magma_dp'] =  magma_dp
 
	config_d['spec_params']['prec_mz'] = True
	config_d['spec_params']['unique_id'] = False
	config_d['spec_params']['counts'] = False
	config_d['eval_batch_size'] = batch_size
	config_d['split'] = "predict"
	
	if device is None:
		device = th.device("cuda:0") if config_d["accelerator"]=="gpu" else th.device("cpu")
		
	if device == "cpu":
		th.set_num_threads(multiprocessing.cpu_count())
		th.multiprocessing.set_sharing_strategy('file_system')
		config_d['num_workers']= 0    
	
	ds = SpecMolMagmaIntenDataset(**config_d)
	dl = th.utils.data.DataLoader(
		ds,
		batch_size=batch_size,
		shuffle=False,
		num_workers=config_d['num_workers'],
		collate_fn=ds.get_collate_fn(),
	)

	iceberg_intern_model = IcebergIntenPL.load_from_checkpoint(iceberg_intern_ckpt_fp, map_location = device)
	iceberg_intern_model.eval()

	preds_l = []
	def wrapper(gen):
		while True:
			try:
				yield next(gen)
			except StopIteration:
				break
			except Exception as e:
				print(e)

	for _, batch_input in tqdm(enumerate(wrapper(iter(dl))),total=len(dl), desc="Running iceberg intern",leave=False):

		#print(batch_input)
		spec_unique_ids  = batch_input['spec_unique_id']
		mol_smiles = batch_input['mol_smiles']

		batch_result = to_device(batch_input, device)
		#batch_result = iceberg_intern_model.predict_step(**batch_input)
		# use "val" mode to get predict and similar score
		batch_result = iceberg_intern_model._common_step(batch_input,  split="val", no_log=True)

		batch_size = batch_input['batch_size'].detach().cpu()

		pred_mzs = batch_result["pred_mzs"].detach().cpu()
		pred_ints = th.exp(batch_result["pred_logprobs"]).detach().cpu()
		pred_batch_idxs = batch_result["pred_batch_idxs"].detach().cpu()

		cos_sim_bin = batch_result['cos_sim_0.01'].cpu()
		cos_sim_hun = batch_result['cos_hun'].cpu()
		
		for b_idx in range(batch_size):
			b_mask = pred_batch_idxs == b_idx

			b_pred_mzs = pred_mzs[b_mask]
			b_pred_ints = pred_ints[b_mask]

			b_pred_ints = iceberg_intern_model.ints_untransform_func(b_pred_ints)

			b_pred_mzs = b_pred_mzs.numpy()
			b_pred_ints = b_pred_ints.numpy()

			b_cos_sim_bin = cos_sim_bin[b_idx].numpy()
			b_cos_sim_hun = cos_sim_hun[b_idx].numpy()

			#print(b_pred_mzs, b_pred_ints)
			preds_l.append([spec_unique_ids[b_idx].item(), mol_smiles[b_idx], 
                   b_pred_mzs, b_pred_ints, b_cos_sim_bin, b_cos_sim_hun])
			
	df = pd.DataFrame(preds_l, columns=['spec_id','smiles','pred_mzs','pred_ints','cos_sim_bin','cos_sim_hun'] )
	df.to_pickle(save_fp)

def run_iceberg_gen(iceberg_gen_inputs,
				 device = None,
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
		#results = seq_apply(iter(iceberg_gen_inputs),single_predict_mol,True)
		#for res in tqdm(results, total = len(iceberg_gen_inputs), desc="Running iceberg gen", leave=False):
		#	continue

def get_args():
    
	parser = argparse.ArgumentParser()
	parser.add_argument("--device", type=str,default="cuda:0", help="")
	parser.add_argument("--save_fp", type=str,default="./data/ms2c/pubchem/predicted/ms2pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50", help="")
	parser.add_argument("--proc_dp", type=str,default="./data/ms2c/pubchem/proc/pubchem_nist20mona23v3_d4_wr00_scaffold_test_10ppm_MORGAN-R2_50", help="")

	parser.add_argument("--use_cahced_ckpt",type=booltype,default=False)
	parser.add_argument("--model_save_dp",type=str, default="./saved_ckpts")
	#parser.add_argument("--model_ckpt",type=str, default=None)

 	#gen params
	parser.add_argument("--iceberg_gen_wandb_run_id",type=str, default= 'i7k9gc3d')
	parser.add_argument("--iceberg_gen_ckpt_fp", type=str,default=None, help="")
	parser.add_argument("--magma_dp", type=str,default="data/ms2c/pubchem/magma/", help="")
	parser.add_argument("--threshold", type=float, default=0.0)
	parser.add_argument("--max_nodes", type=int, default=100)
	parser.add_argument("--iceberg_gen_device", type=str,default="cpu", help="")
	parser.add_argument("--save_magma",type=booltype,default=True)
	parser.add_argument("--use_precomputed_magma",type=booltype,default=False)
 
	#iceberg_intern params
	parser.add_argument("--iceberg_intern_wandb_run_id",type=str, default= 'c8iwu7il')
	parser.add_argument("--iceberg_intern_ckpt_fp", type=str,default=None, help="")
	parser.add_argument("--iceberg_intern_device", type=str,default="cuda:0", help="")
	parser.add_argument("---iceberg_intern_batch_size", type=int, default=1, help="")

	args = parser.parse_args()
	return args

if __name__ == "__main__":
	
	RDLogger.DisableLog("rdApp.*")
	args = get_args()
	#os.makedirs(args.save_dp, exist_ok=True)
 
	#if args.save_magma:
	#	magma_save_dp = os.path.join(args.save_dp, "magma")
	#	os.makedirs(magma_save_dp, exist_ok=True)
	#	print(f"> save magma data to {magma_save_dp}")
  
	if not th.cuda.is_available():
		args.iceberg_gen_device = "cpu"
		args.iceberg_intern_device = "cpu"
  
	timestr = time.strftime("%Y%m%d-%H%M%S")
	save_dp = os.path.dirname(args.save_fp)
	os.makedirs(save_dp, exist_ok=True)
	records_dict = vars(args)

	os.makedirs(args.model_save_dp, exist_ok=True)	
	if args.iceberg_gen_ckpt_fp is None:
		model_save_dp = args.model_save_dp
		print(f">> save model to {args.model_save_dp} from wandb with run id: {args.iceberg_gen_wandb_run_id}")
		iceberg_gen_ckpt_fp = get_ckpt_from_wandb(args.model_save_dp, args.iceberg_gen_wandb_run_id, use_cached = args.use_cahced_ckpt)
	else:
		iceberg_gen_ckpt_fp = args.iceberg_gen_ckpt_fp
	records_dict['iceberg_gen_ckpt_fp'] = iceberg_gen_ckpt_fp
 
	if args.iceberg_intern_ckpt_fp is None:
		model_save_dp = args.model_save_dp
		print(f">> save model to {args.model_save_dp} from wandb with run id: {args.iceberg_intern_wandb_run_id}")
		iceberg_intern_ckpt_fp = get_ckpt_from_wandb(args.model_save_dp, args.iceberg_intern_wandb_run_id, use_cached = args.use_cahced_ckpt)
	else:
		iceberg_intern_ckpt_fp = args.iceberg_intern_ckpt_fp
	records_dict['iceberg_intern_ckpt_fp'] = iceberg_intern_ckpt_fp
	json.dump(records_dict, open(f"{args.save_fp}_{timestr}.json", "w"), indent=2)

	print(f"> iceberg_gen_device {args.iceberg_gen_device }")
	print(f"> iceberg_intern_device {args.iceberg_intern_device }")
	exit()
	mol_fp = os.path.join(args.task_dp,'mol_df.pkl.gz')
	spec_fp = os.path.join(args.task_dp,'spec_df.pkl.gz')

	print(f"> prepare mol df {mol_fp}")
	mol_df = pd.read_pickle(mol_fp)

	print(f"> prepare spec df {spec_fp}")
	spec_df = pd.read_pickle(spec_fp)

	target_ids = spec_df['target_mol_id'].unique()
	for target_id in (pbar := tqdm(target_ids)):
		pbar.set_description(f"Processing target {target_id}")

		magma_dp = f"{args.magma_dp}/{target_id}"
		os.makedirs(f"{magma_dp}/magma_tree", exist_ok=True)

		spec_df_grp = spec_df[spec_df['target_mol_id'] == target_id].copy()
		spec_df_grp['magma_dp'] = magma_dp
		spec_df_grp['iceberg_gen_ckpt_fp'] = args.iceberg_gen_ckpt_fp
		spec_df_grp['max_nodes'] = args.max_nodes
		spec_df_grp['threshold'] = args.threshold
		spec_df_grp = spec_df_grp.merge(mol_df[['mol_id','mol', 'smiles']], on = 'mol_id', how = 'left')
		spec_df_grp['inchi'] = spec_df_grp['mol'].apply(lambda x : Chem.MolToInchi(x))

		# run_iceberg_gen 
		iceberg_gen_inputs = []
		if not args.use_precomputed_magma:
			iceberg_gen_inputs = spec_df_grp[['iceberg_gen_ckpt_fp','prec_type','inchi','smiles','spec_id', 'magma_dp', 'threshold', 'max_nodes']].values.tolist()
		else:
			cached_spec_ids = [f.split('.')[0] for f in os.listdir(magma_dp)]	
			iceberg_gen_inputs = spec_df_grp[spec_df_grp['spec_id'].isin(cached_spec_ids)][['iceberg_gen_ckpt_fp','prec_type','inchi','smiles','spec_id', 'magma_dp', 'threshold', 'max_nodes']].values.tolist()
		if len(iceberg_gen_inputs):
			run_iceberg_gen(iceberg_gen_inputs, device = args.iceberg_gen_device)
		
		if args.save_magma:
			magma_save_fp = os.path.join(magma_save_dp, f"{target_id}_magma.tar.gz")
			with tarfile.open(magma_save_fp, "w:gz") as tar:
				tar.add(magma_dp, arcname=os.path.basename(magma_dp))

		spec_df_grp.drop(columns = ['mol','smiles','inchi','magma_dp','iceberg_gen_ckpt_fp','threshold','max_nodes'], inplace = True)
		run_iceberg_intern(iceberg_intern_ckpt_fp = args.iceberg_intern_ckpt_fp,
					save_fp = os.path.join(args.save_dp, f"{target_id}.pkl.gz"),
					template_fp = args.template_fp,
					custom_fp = args.custom_fp,
					task_dp = args.task_dp,
					magma_dp = magma_dp,
					mol_df=mol_df,
					spec_df=spec_df_grp,
					device = args.iceberg_intern_device,
					batch_size = args.batch_size)