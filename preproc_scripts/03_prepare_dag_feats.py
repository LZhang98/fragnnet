import pandas as pd
import numpy as np
from tqdm import tqdm
import argparse
import os
from collections import Counter
import wandb
import rdkit.Chem as Chem
from pprint import pformat
import json
import traceback
import sys

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.frag.compute_frags import MAX_NUM_NODES, MAX_NUM_EDGES
import frag_gnn.utils.data_utils as data_utils
from frag_gnn.utils.formula_utils import parse_formula
from frag_gnn.utils.misc_utils import booltype
from frag_gnn.utils.dataset_utils import merge_spec_df

def element_filter(formula,allowed_elements):
	element_counts = parse_formula(formula)
	for k,v in element_counts.items():
		if v > 0 and k not in allowed_elements:
			return False
	return True

def filter_spec_mol(spec_df,mol_df,allowed_elements,num_entries=-1):
	"""
	The purpose of this filtering is to prevent errors
	"""

	masks = []
	## spectrum criteria (TODO remove this)
	# dataset mask
	dset_mask = spec_df["dset"].isin(["nist"])
	masks.append(dset_mask)
	# instrument type
	inst_type_mask = spec_df["inst_type"].isin(["FT"])
	masks.append(inst_type_mask)
	# frag mode
	frag_mode_mask = spec_df["frag_mode"].isin(["HCD"])
	masks.append(frag_mode_mask)
	# ce
	ce_mask = (spec_df["inst_type"]=="FT") & (~spec_df["nce"].isna())
	masks.append(ce_mask)
	# ion mode
	ion_mode_mask = spec_df["ion_mode"] == "P"
	masks.append(ion_mode_mask)
	# precursor type
	prec_type_mask = spec_df["prec_type"].isin(["[M+H]+"])
	masks.append(prec_type_mask)
	# resolution
	res_mask = spec_df["res"].isin([1,2,3,4,5,6,7])
	masks.append(res_mask)
	# spectrum type 
	spec_type_mask = spec_df["spec_type"] == "MS2"
	masks.append(spec_type_mask)
	# precursor mz
	prec_mz_mask = ~spec_df["prec_mz"].isna()
	masks.append(prec_mz_mask)
	## molecule criteria
	# single molecule
	single_mol_ids = mol_df[mol_df["single_mol"]]["mol_id"]
	single_mol_mask = spec_df["mol_id"].isin(single_mol_ids)
	masks.append(single_mol_mask)
	# neutral molecule
	neutral_ids = mol_df[mol_df["charge"]==0]["mol_id"]
	neutral_mask = spec_df["mol_id"].isin(neutral_ids)
	masks.append(neutral_mask)
	# element composition
	element_ids = mol_df[mol_df["formula"].apply(lambda formula: element_filter(formula,allowed_elements))]["mol_id"]
	element_mask = spec_df["mol_id"].isin(element_ids)
	print(f">> element: {element_mask.mean()}")
	masks.append(element_mask)
	# atom count
	atom_count_ids = mol_df[mol_df["num_atoms"]<=MAX_NUM_NODES]["mol_id"]
	atom_count_mask = spec_df["mol_id"].isin(atom_count_ids)
	print(f">> atom count: {atom_count_mask.mean()}")
	masks.append(atom_count_mask)
	# bond count
	bond_count_ids = mol_df[mol_df["num_bonds"]<=MAX_NUM_EDGES]["mol_id"]
	bond_count_mask = spec_df["mol_id"].isin(bond_count_ids)
	print(f">> bond count: {bond_count_mask.mean()}")
	masks.append(bond_count_mask)
	# radical stats
	non_radical_ids = mol_df[mol_df["num_radicals"]==0]["mol_id"]
	non_radical_mask = spec_df["mol_id"].isin(non_radical_ids)
	print(f">> non radical: {non_radical_mask.mean()}")
	masks.append(non_radical_mask)
	# put them together
	all_mask = masks[0]
	for mask in masks:
		all_mask = all_mask & mask
	if np.sum(all_mask) == 0:
		raise ValueError("select removed all items")
	print(f">> everything: {all_mask.mean()}")
	spec_df = spec_df[all_mask].reset_index(drop=True)
	# sample
	if num_entries > 0:
		spec_df = spec_df.sample(n=num_entries,replace=False,random_state=420).reset_index(drop=True)
	# only keep molecules that are in the spectra
	mol_df = mol_df[mol_df["mol_id"].isin(spec_df["mol_id"])].reset_index(drop=True)
	return spec_df, mol_df

def print_and_log(name,series,wandb_flag,stats_d):

	stats = series.describe()
	print(f"> {name}")
	print(stats)
	print()
	for stat in ["mean","std","min","25%","50%","75%","max"]:
		stats_d[f"{name}/{stat}"] = stats[stat]
	if wandb_flag:
		log_d = {"step":0}
		for stat in ["mean","std","min","25%","50%","75%","max"]:
			log_d[f"{name}/{stat}"] = stats[stat]
		wandb.log(log_d)
		
def main(args):

	assert args.max_time < data_utils.JOBLIB_TIMEOUT, (args.max_time,data_utils.JOBLIB_TIMEOUT)

	allowed_elements = args.allowed_elements.split("-")
	# init wandb
	wandb_flag = args.wandb_mode != "off"
	if wandb_flag:
		wandb_config = vars(args)
		wandb_run = wandb.init(
			project="frag-gnn",
			entity="frag-gnn",
			name=args.run_name if args.wandb_run_name is None else args.wandb_run_name,
			mode=args.wandb_mode,
			config=wandb_config,
			dir=args.project_dp,
			group="prepare_dag_feats"
		)

	# read in the molecule data
	print("read in mol")
	mol_df = pd.read_pickle(os.path.join(args.project_dp,args.proc_dp,"mol_df.pkl"))
	print("read in spec")
	spec_df = pd.read_pickle(os.path.join(args.project_dp,args.proc_dp,"spec_df.pkl"))
	print()

	# perform selection
	spec_df, mol_df = filter_spec_mol(spec_df,mol_df,allowed_elements,num_entries=args.num_entries)
	m_spec_df = merge_spec_df(spec_df)

	print(f">> {len(spec_df)} spectra, {len(mol_df)} molecules") #, {len(m_spec_df)} merged spectra")
	print()

	num_atoms = mol_df["num_atoms"]
	print_and_log("num_atoms",num_atoms,wandb_flag,{})

	num_bonds = mol_df["num_bonds"]
	print_and_log("num_bonds",num_bonds,wandb_flag,{})

	os.makedirs(os.path.join(args.frag_dp,args.run_name),exist_ok=True)
	# dag_fp = os.path.join(args.frag_dp,args.run_name,f"dag_df.pkl")
	dag_dp = os.path.join(args.frag_dp,args.run_name,f"dags")
	os.makedirs(dag_dp,exist_ok=True)
	group_stats_fp = os.path.join(args.frag_dp,args.run_name,f"group_stats_df.pkl")
	global_stats_fp = os.path.join(args.frag_dp,args.run_name,f"global_stats.json")

	def timed_get_dags(mol, mol_id, 
			max_depth,h_prior,max_h_transfer, max_time,
			use_isotopes:bool,
			output_dir:str, 
			use_cached = True, 
			compressed = False,
			save_dag = True):
		""" 
		Args:
			mol (_type_): _description_
			mol_id (int): _description_

			output_dir (str): _description_
			compressed (bool, optional): _description_. Defaults to False.

		Raises:
			e: _description_

		Returns:
			_type_: _description_
		"""

		output_file =  f"{output_dir}/{mol_id}.pickle.pbz2" if compressed else f"{output_dir}/{mol_id}.pickle"

		if os.path.exists(output_file) and use_cached:
			dag_d = frag_utils.load_dag_d(output_file, compressed)
		else:
			dag_d = {}
			try:
				mol_d = frag_utils.extract_mol_info(mol)
				# this maybe dangers in multi processing because of scopes
				dag_d = frag_utils.compute_dags(mol_d,max_depth,h_prior,max_h_transfer, max_time, use_isotopes)
			except KeyboardInterrupt as e:
				# let these through
				raise e
			except Exception as e:
				# don't retry, theres a bug
				if type(mol) is not str:
					mol = Chem.MolToSmiles(mol)
				print(f">> Non-timeout error, aborting: {type(e)} {repr(e)} Input {mol}",file=sys.stderr)
				print("> Traceback",file=sys.stderr)
				traceback.print_exc(file=sys.stderr)
				dag_d = {}
			else:
				if save_dag:
					frag_utils.save_dag_d(dag_d, output_file, compressed)
				del dag_d["dag"]
		return dag_d


	def compute_spectra_stats(peaks,formula_mzs,formula_probs,idx_by_h_delta,
			   tolerances:list, max_h_transfer:int):
		"""_summary_

		Args:
			tolerances (list): _description_
			h_transfer (int): _description_

		Returns:
			_type_: _description_
		"""
		reuslts = []
		cols = []
		for tolerance in tolerances:
			for h_transfer in range(1, max_h_transfer + 1):
				keys = [
					f"recall_{tolerance}_h{h_transfer}",
					f"wrecall_{tolerance}_h{h_transfer}",
					f"prec_{tolerance}_h{h_transfer}",
					f"ppt_peak_{tolerance}_h{h_transfer}",
					f"ppt_formula_{tolerance}_h{h_transfer}"
				]

				if 'ppm' in tolerance:
					result = frag_utils.compute_frag_peak_stats(peaks,formula_mzs,formula_probs,idx_by_h_delta,\
						 h_transfer,tolerance=float(tolerance[:-3]),prec_type="[M+H]+", is_ppm = True)
				else:
					result = frag_utils.compute_frag_peak_stats(peaks,formula_mzs,formula_probs,idx_by_h_delta,h_transfer,\
						 tolerance=float(tolerance),prec_type="[M+H]+")
				cols += keys
				reuslts.append(result)

		result_series  = pd.concat(reuslts, axis = 0)
		result_df = result_series.to_frame().T
		result_df.columns = cols
		return result_df
	
	print("> Compute Fragments")
	mol_input_rows = []
	for _, row in tqdm(mol_df.iterrows(), total=mol_df.shape[0], desc= "prepare inputs"):
		mol_input_rows.append([row['mol'],row['mol_id'], args.max_depth,args.h_prior,args.max_h_transfer,args.max_time,args.isotopes, dag_dp, \
			 True, args.compress_dags])
	tqdm_iter = tqdm(mol_input_rows,desc=pformat(timed_get_dags),total=len(mol_input_rows))
	frag_results = data_utils.par_apply(tqdm_iter,timed_get_dags,True)
	print("> Fragments Computed")

	#print(frag_results)
	frag_stats_df = pd.DataFrame({"mol_id":mol_df["mol_id"]})
	frag_stats_df["depth"] = list(map(lambda x: x.pop("max_depth",np.nan),frag_results))
	#frag_stats_df["dag"] = list(map(lambda x: x.pop("dag",np.nan),frag_results))
	frag_stats_df["formula_mzs"] = list(map(lambda x: x.pop("formula_mzs",np.nan),frag_results))
	frag_stats_df["formula_probs"] = list(map(lambda x: x.pop("formula_probs",np.nan),frag_results))
	frag_stats_df["idx_to_formula"] = list(map(lambda x: x.pop("idx_to_formula",np.nan),frag_results))
	frag_stats_df["dag_num_edges"] = list(map(lambda x: x.pop("dag_num_edges",np.nan),frag_results))
	frag_stats_df["dag_num_nodes"] = list(map(lambda x: x.pop("dag_num_nodes",np.nan),frag_results))
	frag_stats_df["dag_sparsity"] = list(map(lambda x: x.pop("dag_sparsity",np.nan),frag_results))
	frag_stats_df["formula_redundancy"] = list(map(lambda x: x.pop("formula_redundancy",np.nan),frag_results))	
	frag_stats_df["idx_by_h_delta"] = list(map(lambda x: x.pop("idx_by_h_delta",np.nan),frag_results))	

	del frag_results

	frag_stats_d = {}
	# count failures, then remove them
	# print(frag_stats_df)
	num_failures = frag_stats_df.isna().any(axis=1).sum()
	frag_stats_d["total_num_failures"] = num_failures
	print(f"> total num failures: {num_failures}")
	print()
	if wandb_flag:
		wandb.log({"total_num_failures": num_failures,"step":0})
	frag_stats_df = frag_stats_df.dropna(axis=0)

	### global properties

	# compute total number of formulae
	unique_formulae = set()
	for idx_to_formula in frag_stats_df["idx_to_formula"].values:
		unique_formulae.update(list(idx_to_formula.values()))
	frag_stats_d["total_num_formulae"] = len(unique_formulae)
	print(f"> total num formulae: {len(unique_formulae)}")
	print()

	# compute total number of depths
	print("> depth:")
	print(frag_stats_df["depth"].value_counts())
	print()
	depth_d = {f"depth/{k}":v for k,v in frag_stats_df["depth"].value_counts().to_dict().items()}
	frag_stats_d.update(depth_d)
	if wandb_flag:
		depth_d["step"] = 0
		wandb.log(depth_d)

	### molecule properties
	frag_stats_d = {}
	print_and_log("dag_num_edges",frag_stats_df["dag_num_edges"],wandb_flag,frag_stats_d)
	print_and_log("dag_num_nodes",frag_stats_df["dag_num_nodes"],wandb_flag,frag_stats_d)
	print_and_log("dag_sparsity",frag_stats_df["dag_sparsity"],wandb_flag,frag_stats_d)
	print_and_log("formula_redundancy",frag_stats_df["formula_redundancy"],wandb_flag,frag_stats_d)

	# count number of formula per molecule
	frag_stats_df.loc[:,"num_formulae"] = frag_stats_df["idx_to_formula"].apply(lambda x: len(x)-1)
	print_and_log("num_formulae",frag_stats_df["num_formulae"],wandb_flag,frag_stats_d)

	assert (frag_stats_df["num_formulae"]>0).all()

	# drop idx_to_formula
	frag_stats_df = frag_stats_df.drop(columns=["idx_to_formula"])

	print("> Compute Spectra Stats")

	stats_cols = ["num_formulae","depth","formula_redundancy","dag_num_edges","dag_num_nodes","dag_sparsity"]
	id_cols = ["mol_id"]
	data_cols = list(set(frag_stats_df.columns)-set(stats_cols)-set(id_cols))

	for merged in [False,True]:

		if not merged:
			spec_key = "spec_id"
			_spec_df = spec_df
			spec_prefix = "spec/"
			mol_prefix = "mol/"
			spec_stats_fp = os.path.join(args.frag_dp,args.run_name,f"spec_stats_df.pkl")
			mol_stats_fp = os.path.join(args.frag_dp,args.run_name,f"mol_stats_df.pkl")
		else:
			spec_key = "group_id"
			_spec_df = m_spec_df
			spec_prefix = "m_spec/"
			mol_prefix = "m_mol/"
			spec_stats_fp = os.path.join(args.frag_dp,args.run_name,f"m_spec_stats_df.pkl")
			mol_stats_fp = os.path.join(args.frag_dp,args.run_name,f"m_mol_stats_df.pkl")

		# compute spectra stats
		peak_spec_df = _spec_df[[spec_key,"mol_id","peaks","prec_mz"]].merge(frag_stats_df[id_cols+data_cols],on="mol_id",how="inner")
		assert peak_spec_df.shape[0] == peak_spec_df.drop_duplicates(subset=[spec_key,"mol_id"]).shape[0]

		spectra_input_rows = []
		tolerances = [tolerance for tolerance in args.tolerances.split("-")]
		for _, row in tqdm(peak_spec_df.iterrows(), total=peak_spec_df.shape[0], desc= "prepare spectra stats inputs"):
			spectra_input_rows.append(
				[
					row['peaks'], 
					row["formula_mzs"], 
					row['formula_probs'], 
					row['idx_by_h_delta'], 
					tolerances, 
					args.max_h_transfer
				]
			)
		tqdm_iter = tqdm(spectra_input_rows,desc=pformat(compute_spectra_stats),total=len(spectra_input_rows))
		# run stats
		stats_results = data_utils.par_apply(tqdm_iter,compute_spectra_stats,True)
		# Loky and job lib should kep ordering
		stats_results_df =  pd.concat(stats_results, axis=0, ignore_index=True)
		metric_keys = list(stats_results_df.columns)

		# add peak stats
		peak_spec_df = pd.concat((peak_spec_df[[spec_key,"mol_id"]],stats_results_df), axis=1)
		# add dag stats
		peak_spec_df = peak_spec_df.merge(frag_stats_df[id_cols+stats_cols],on="mol_id",how="inner")

		# save spectrum-level stats
		peak_spec_df.to_pickle(spec_stats_fp)

		# update the frag stats d
		for key in metric_keys+stats_cols:
			print_and_log(spec_prefix+key,peak_spec_df[key],wandb_flag,frag_stats_d)

		print("> Compute Molecule Stats")	

		# collect across molecules and report summary statistics
		peak_mol_df = peak_spec_df.drop(columns=[spec_key]).groupby("mol_id").agg(np.nanmean).reset_index()
		for key in metric_keys+stats_cols:
			print_and_log(mol_prefix+key,peak_mol_df[key],wandb_flag,frag_stats_d)

		# save molecule-level stats
		peak_mol_df.to_pickle(mol_stats_fp)


	# save the frag stats dict
	with open(global_stats_fp,"w",encoding="utf-8") as f:
		frag_stats_d = {k:str(v) for k,v in frag_stats_d.items()}
		json.dump(frag_stats_d,f,ensure_ascii=False,indent=4)

	if wandb_flag:
		wandb.finish()


if __name__ == "__main__":

	default_elements = "-".join(frag_utils.CANONICAL_ELEMENT_ORDER)

	parser = argparse.ArgumentParser()
	parser.add_argument("--num_entries",type=int,default=-1)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--max_time",type=int,default=150)
	parser.add_argument("--project_dp",type=str,default=os.getcwd())
	parser.add_argument("--frag_dp",type=str,default="data/frag")
	parser.add_argument("--proc_dp",type=str,default="data/proc")
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--allowed_elements",type=str,default=default_elements)
	parser.add_argument("--tolerances",type=str,default="0.01-0.005-0.001-0.0001-10ppm-5ppm")
	parser.add_argument("--isotopes",type=booltype,default=True)
	parser.add_argument("--wandb_mode",type=str,default="online",choices=["online","offline","off"])
	parser.add_argument("--run_name",type=str,default="debug")
	parser.add_argument("--wandb_dp",type=str,default="wandb")
	parser.add_argument("--compress_dags",type=booltype,default=True)
	parser.add_argument("--wandb_run_name",type=str,default=None)
	parser.add_argument("--save_dag",type=booltype,default=True)
	args = parser.parse_args()

	tqdm.pandas()

	main(args)