import pandas as pd
import numpy as np
from tqdm import tqdm
import signal
from contextlib import contextmanager
import argparse
from functools import reduce
import os
from collections import Counter
import wandb
import rdkit
import rdkit.Chem as Chem

import frag_gnn.frag.frag_utils as frag_utils
from frag_gnn.frag.frag_utils import ELEMENT_TO_IDX
from frag_gnn.frag.compute_frags import MAX_NUM_NODES, MAX_NUM_EDGES
import frag_gnn.data_utils as data_utils
from frag_gnn.data_utils import parse_formula
from frag_gnn.misc_utils import booltype, flatten_lol, wandb_symlink, TimeoutException, time_limit
import frag_gnn.formula_utils as formula_utils


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

def prec_in_frag(row):

	prec_formula = row["prec_formula_str"]
	frag_formulae = row["frag_formulae_str"]
	return prec_formula in frag_formulae

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
		print(f"> logging {name}: {log_d.keys()}")
		wandb.log(log_d)

def main(args):

	allowed_elements = args.allowed_elements.split(",")

	# init wandb
	wandb_flag = args.wandb_mode != "off"
	if wandb_flag:
		wandb_config = vars(args)
		wandb_run = wandb.init(
			project="frag-gnn",
			entity="frag-gnn",
			name=args.run_name,
			mode=args.wandb_mode,
			config=wandb_config,
			dir=args.project_dp,
		)

	# read in the molecule data
	print("read in mol")
	mol_df = pd.read_pickle(os.path.join(args.project_dp,args.proc_dp,"mol_df.pkl"))
	print("read in spec")
	spec_df = pd.read_pickle(os.path.join(args.project_dp,args.proc_dp,"spec_df.pkl"))
	print()

	# perform selection
	spec_df, mol_df = filter_spec_mol(spec_df,mol_df,allowed_elements,num_entries=args.num_entries)
	
	print(f">> {len(spec_df)} spectra, {len(mol_df)} molecules")
	print()

	num_atoms = mol_df["num_atoms"]
	print_and_log("num_atoms",num_atoms,wandb_flag,{})

	num_bonds = mol_df["num_bonds"]
	print_and_log("num_bonds",num_bonds,wandb_flag,{})

	dag_fp = "data/frag/debug1.pkl"
	mol_stats_fp = "data/frag/debug2.pkl"
	spec_stats_fp = "data/frag/debug3.pkl"
	# frag_fp = os.path.join(args.project_dp,args.frag_dp,args.frag_stats_df_name)

	def timed_get_dags(mol):
		cur_max_depth = args.max_depth
		dag_d = {}
		while not bool(dag_d): # empty dictionary
			try:
				mol_d = frag_utils.extract_mol_info(mol)
				with time_limit(args.max_time):
					dag_d = frag_utils.compute_dags(mol_d,cur_max_depth,args.h_prior,args.max_h_transfer)
			except TimeoutException as e:
				if cur_max_depth > 1:
					cur_max_depth -= 1
					print(f">> Decrementing depth to {cur_max_depth}")
				else:
					dag_d = {}
			except KeyboardInterrupt as e:
				# let these through
				raise e
			except Exception as e:
				# don't retry, theres a bug
				dag_d = frag_utils.compute_dags(mol_d,cur_max_depth,args.h_prior,args.max_h_transfer)
				print(f">> Non-timeout error, aborting: {e}")
				dag_d = {}
		return dag_d

	# example_mol = mol_df["mol"].iloc[0]
	# mol_d = frag_utils.extract_mol_info(example_mol)
	# dag_d = frag_utils.compute_dags(mol_d,args.max_depth,args.h_prior,args.max_h_transfer)
	# import pdb; pdb.set_trace()

	frag_results = data_utils.seq_apply(mol_df["mol"].tolist(),timed_get_dags)
	frag_depth = list(map(frag_results,lambda x: x.get("max_depth",np.nan)))
	frag_formulae = list(map(frag_results,lambda x: x.get("formulae",np.nan)))
	frag_ccs = list(map(frag_results,lambda x: x.get("ccs",np.nan)))
	frag_cc_gs = list(map(frag_results,lambda x: x.get("cc_gs",np.nan)))
	del frag_results

	import pdb; pdb.set_trace()

	dag_df = pd.DataFrame(
		{
			"mol_id": mol_df["mol_id"],
			"cc_gs": frag_cc_gs,
		}
	)
	dag_df.to_pickle(dag_fp)
	del dag_df

	frag_stats_df = pd.DataFrame(
		{
			"mol_id": mol_df["mol_id"],
			"depth": frag_depth,
			"formulae": frag_formulae,
			"ccs": frag_ccs,
			"dag": frag_cc_gs,
		}
	)

	# count failures, then remove them
	num_failures = frag_stats_df.isna().any(axis=0).sum()
	frag_stats_d["total_num_failures"] = num_failures
	print(f"> num failures: {num_failures}")
	print()
	if wandb_flag:
		wandb.log({"num_failures": num_failures,"step":0})
	frag_stats_df = frag_stats_df.dropna(axis=0)

	frag_stats_d = {}

	### global properties

	# compute total number of formulae
	def formulae_to_array(formulae):
		if len(formulae) == 0:
			return np.ones((0,len(ELEMENT_TO_IDX)),dtype=int)
		else:
			return np.array(list(formulae.keys()),dtype=int).reshape(len(formulae),len(ELEMENT_TO_IDX))
	frag_formulae = frag_stats_df["formulae"].progress_apply(formulae_to_array)
	unique_formulae = np.unique(np.concatenate(frag_formulae.values,axis=0),axis=0).tolist()
	frag_stats_d["total_num_formulae"] = len(unique_formulae)
	print(f"> num unique formulae: {len(unique_formulae)}")
	print()

	# compute total number of depths
	print("> depth:")
	print(frag_stats_df["depth"].value_counts())
	print()
	if wandb_flag:
		depth_d = {f"depth/{k}":v for k,v in frag_stats_df["depth"].value_counts().to_dict().items()}
		frag_stats_d.update(depth_d)
		depth_d["step"] = 0
		wandb.log(depth_d)

	### molecule properties

	# count number of formula per molecule
	frag_stats_df.loc[:,"num_formulae"] = frag_stats_df["formulae"].apply(lambda x: len(x))
	print_and_log("num_formulae",frag_stats_df["num_formulae"],wandb_flag,frag_stats_d)
	assert (frag_stats_df["num_formulae"]>0).all()

	# count number of ccs per molecule
	frag_stats_df.loc[:,"num_ccs"] = frag_stats_df["ccs"].apply(lambda x: len(x))
	print_and_log("num_ccs",frag_stats_df["num_ccs"],wandb_flag,frag_stats_d)
	
	# drop ccs
	frag_stats_df = frag_stats_df.drop(columns=["ccs"])

	# compute formula redundancy
	def get_formula_redundancy(formulae):
		return np.mean(np.array(list(formulae.values())))
	frag_stats_df.loc[:,"formula_redundancy"] = frag_stats_df["formulae"].apply(get_formula_redundancy)
	print_and_log("formula_redundancy",frag_stats_df["formula_redundancy"],wandb_flag,frag_stats_d)

	# compute dag num edges and nodes
	frag_stats_df.loc[:,"dag_num_edges"] = frag_stats_df["dag"].apply(lambda x: x.number_of_edges())
	frag_stats_df.loc[:,"dag_num_nodes"] = frag_stats_df["dag"].apply(lambda x: x.number_of_nodes())
	frag_stats_df.loc[:,"dag_sparsity"] = frag_stats_df["num_edges"]/(frag_stats_df["num_nodes"]*(frag_stats_df["num_nodes"]-1))
	print_and_log("dag_num_edges",frag_stats_df["dag_num_edges"],wandb_flag,frag_stats_d)
	print_and_log("dag_num_nodes",frag_stats_df["dag_num_nodes"],wandb_flag,frag_stats_d)
	print_and_log("dag_sparsity",frag_stats_df["dag_sparsity"],wandb_flag,frag_stats_d)

	# drop dags
	frag_stats_df = frag_stats_df.drop(columns=["dag"])

	# save molecule-level stats
	mol_stats = ["num_formulae","num_ccs","depth","formula_redundancy","dag_num_edges","dag_num_nodes","dag_sparsity"]
	frag_stats_df[["mol_id"]+mol_stats].to_pickle(mol_stats_fp)
	frag_stats_df = frag_stats_df.drop(columns=mol_stats)

	### spectrum properties

	# setup formula information
	def add_prec_to_frag(row):
		prec_formula = row["prec_formula_str"]
		frag_formulae = row["formulae_str"]
		if prec_formula not in frag_formulae:
			formulae = frag_formulae + [prec_formula]
		else:
			formulae = frag_formulae + []
		return formulae
	frag_stats_df = frag_stats_df.merge(mol_df[["mol_id","formula"]],on="mol_id",how="inner").rename(columns={"formula":"prec_formula_str"})
	frag_stats_df.loc[:,"formulae_str"] = frag_stats_df["formulae"].apply(lambda x: list(map(frag_utils.get_formula_string,x)))
	frag_stats_df.loc[:,"formulae_str"] = frag_stats_df.apply(add_prec_to_frag,axis=1)
	
	# drop formulae
	frag_stats_df = frag_stats_df.drop(columns=["formulae"])

	# convert formulae to peak locations
	peak_mzs_counter = lambda x: formula_to_peak_mzs(x,"[M+H]+",isotopes=args.isotopes,counter=True)
	frag_stats_df.loc[:,"peak_mzs_counts"] = data_utils.par_apply_series(frag_stats_df["formulae_str"], lambda x: sum(map(peak_mzs_counter,x),start=Counter()))
	prec_peak_mz = lambda x: formula_to_peak_mzs(x,"[M+H]+",isotopes=False,counter=False)
	frag_stats_df.loc[:,"prec_peak_mz"] = frag_stats_df["prec_formula_str"].progress_apply(lambda x: prec_peak_mz(x)[0])

	# for each spectrum, calculate fraction of peaks explained, fraction of intensity explained by referencing
	peak_spec_df = spec_df[["spec_id","mol_id","peaks","prec_mz"]].merge(frag_stats_df,on="mol_id",how="inner")
	result = peak_spec_df.apply(lambda x: compute_frag_peak_stats(x,tolerance=args.tolerance),axis=1)
	peak_spec_df.loc[:,"frag_recall"] = result[0].values
	peak_spec_df.loc[:,"frag_wrecall"] = result[1].values
	peak_spec_df.loc[:,"frag_prec"] = result[2].values
	peak_spec_df.loc[:,"frag_ppt_peak"] = result[3].values
	peak_spec_df.loc[:,"frag_ppt_formula"] = result[4].values
	del result

	# collect across spectra and report summary statistics
	print_and_log("spec_frag_recall",peak_spec_df["frag_recall"],wandb_flag,frag_stats_d)
	print_and_log("spec_frag_wrecall",peak_spec_df["frag_wrecall"],wandb_flag,frag_stats_d)
	print_and_log("spec_frag_prec",peak_spec_df["frag_prec"],wandb_flag,frag_stats_d)
	print_and_log("spec_frag_ppt_peak",peak_spec_df["frag_ppt_peak"],wandb_flag,frag_stats_d)
	print_and_log("spec_frag_ppt_formula",peak_spec_df["frag_ppt_formula"],wandb_flag,frag_stats_d)

	# collect across molecules and report summary statistics
	agg_dict = {
		"frag_recall":np.mean,
		"frag_wrecall":np.mean,
		"frag_prec":np.mean,
		"frag_ppt_peak":np.mean,
		"frag_ppt_formula":np.mean
	}
	peak_mol_df = peak_spec_df[["mol_id"]+list(agg_dict.keys())].groupby("mol_id").agg(agg_dict).reset_index()
	print_and_log("mol_frag_recall",peak_mol_df["frag_recall"],wandb_flag,frag_stats_d)
	print_and_log("mol_frag_wrecall",peak_mol_df["frag_wrecall"],wandb_flag,frag_stats_d)
	print_and_log("mol_frag_prec",peak_mol_df["frag_prec"],wandb_flag,frag_stats_d)
	print_and_log("mol_frag_ppt_peak",peak_mol_df["frag_ppt_peak"],wandb_flag,frag_stats_d)
	print_and_log("mol_frag_ppt_formula",peak_mol_df["frag_ppt_formula"],wandb_flag,frag_stats_d)

	# if wandb_flag and args.wandb_symlink_dp != "":
	# 	wandb_symlink_dp = os.path.join(args.project_dp,args.wandb_symlink_dp)
	# 	wandb_symlink(wandb_run.dir,wandb_symlink_dp,args.job_id)

	# remove stuff we don't need
	frag_stats_df = frag_stats_df.drop(columns=["peaks","prec_mz","prec_peak_mz","peak_mzs_counts","prec_formula_str","formulae_str"])
	
	# save spectrum-level stats
	frag_stats_df.to_pickle(spec_stats_fp)

	if wandb_flag:
		wandb.finish()


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument("--num_entries",type=int,default=-1)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--max_time",type=int,default=2*60)
	parser.add_argument("--project_dp",type=str,required=True)
	parser.add_argument("--frag_dp",type=str,default="data/frag")
	parser.add_argument("--proc_dp",type=str,default="data/proc")
	parser.add_argument("--frag_stats_df_basename",type=str,default="")
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--allowed_elements",type=str,default="C,H,O,N,F,S,P,Cl")
	parser.add_argument("--tolerance",type=float,default=0.01)
	parser.add_argument("--isotopes",type=booltype,default=True)
	parser.add_argument("--wandb_mode",type=str,default="off",choices=["online","offline","off"])
	parser.add_argument("--run_name",type=str,default="debug")
	parser.add_argument("--wandb_dp",type=str,default="wandb")
	# parser.add_argument("--wandb_symlink_dp",type=str,default="jobs_wandb")
	# parser.add_argument("--job_id",type=str,default="00000")
	args = parser.parse_args()

	tqdm.pandas()

	main(args)