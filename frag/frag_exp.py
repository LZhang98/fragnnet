import pyximport
pyximport.install(language_level=3)
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

import frag_utils
from frag_utils import ELEMENT_TO_IDX
import run_compute_frags
import compute_frags
import data_utils
from misc_utils import booltype, flatten_lol, wandb_symlink
import formula_utils
from plot_utils import plot_histogram

PERIODIC_TABLE = Chem.GetPeriodicTable()

ELEMENT_TO_MASS = {
	"H": PERIODIC_TABLE.GetMostCommonIsotopeMass("H"), #1.007825031898,
	"C": PERIODIC_TABLE.GetMostCommonIsotopeMass("C"), #12.0,
	"N": PERIODIC_TABLE.GetMostCommonIsotopeMass("N"), #14.003074004251,
	"O": PERIODIC_TABLE.GetMostCommonIsotopeMass("O"), #15.994914619257,
	"F": PERIODIC_TABLE.GetMostCommonIsotopeMass("F"), #18.998403162067,
	"Cl": PERIODIC_TABLE.GetMostCommonIsotopeMass("Cl"), #34.96885269,
	"P": PERIODIC_TABLE.GetMostCommonIsotopeMass("P"), #30.9737619986,
	"S": PERIODIC_TABLE.GetMostCommonIsotopeMass("S"), #31.9720711744 
}

ELECTRON_MASS = 0.00054858

PREC_TYPE_TO_MASS_DIFF = {
	"[M+H]+": ELEMENT_TO_MASS["H"] - ELECTRON_MASS,
}

# some utils for function timeout
# adapted from https://stackoverflow.com/questions/366682/how-to-limit-execution-time-of-a-function-call

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
	def signal_handler(signum, frame):
		raise TimeoutException(f"Timed out! ({seconds} seconds)")
	signal.signal(signal.SIGALRM, signal_handler)
	signal.alarm(seconds)
	try:
		yield
	finally:
		signal.alarm(0)

# some utils for filtering

def element_filter(formula,allowed_elements):
	element_counts = data_utils.parse_formula(formula)
	for k,v in element_counts.items():
		if v > 0 and k not in allowed_elements:
			return False
	return True

def atom_count_filter(mol):
	num_atoms = mol.GetNumHeavyAtoms()
	return num_atoms <= compute_frags.MAX_NUM_NODES

def bond_count_filter(mol):
	num_bonds = len(mol.GetBonds())
	return num_bonds <= compute_frags.MAX_NUM_EDGES

def filter_spec_mol(spec_df,mol_df,allowed_elements,num_entries=-1):
	"""
	The purpose of this filtering is to prevent errors
	"""

	masks = []
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
	# single molecule
	multi_mol_ids = mol_df[mol_df["smiles"].str.contains("\.")]["mol_id"]
	single_mol_mask = ~spec_df["mol_id"].isin(multi_mol_ids)
	masks.append(single_mol_mask)
	# neutral molecule
	charges = mol_df["mol"].apply(data_utils.mol_to_charge)
	charged_ids = mol_df[charges!=0]["mol_id"]
	neutral_mask = ~spec_df["mol_id"].isin(charged_ids)
	masks.append(neutral_mask)
	# element composition
	element_ids = mol_df[mol_df["formula"].apply(lambda formula: element_filter(formula,allowed_elements))]["mol_id"]
	element_mask = spec_df["mol_id"].isin(element_ids)
	print(f">> element: {element_mask.mean()}")
	masks.append(element_mask)
	# atom count
	atom_count_ids = mol_df[mol_df["mol"].apply(atom_count_filter)]["mol_id"]
	atom_count_mask = spec_df["mol_id"].isin(atom_count_ids)
	print(f">> atom count: {atom_count_mask.mean()}")
	masks.append(atom_count_mask)
	# bond count
	bond_count_ids = mol_df[mol_df["mol"].apply(bond_count_filter)]["mol_id"]
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

def get_mol_formulae_ccs(smiles,max_depth,h_prior,max_h_transfer):

	mol_d = frag_utils.extract_mol_info(smiles,"[M+H]+")
	num_nodes = mol_d["atom_mask_arr"].shape[0]
	# ccs_to_depths = run_compute_frags.cython_function(mol_d,max_depth,depths=True)
	# formulae = [frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False) for cc_int in ccs_to_depths.keys()]
	# ccs_max_depths = [max(v) for v in ccs_to_depths.values()]
	ccs = run_compute_frags.cython_function(mol_d,max_depth,depths=False)
	formulae = [frag_utils.convert_from_int(num_nodes,cc_int,bitmask=False) for cc_int in ccs]
	formulae = frag_utils.compute_approximate_formulae(formulae,mol_d,h_prior=h_prior,max_h_transfer=max_h_transfer,return_counts=True)
	return formulae, ccs

def formula_to_peak_mzs(formula,prec_type,isotopes=True):

	element_counts = data_utils.parse_formula(formula)
	mass_diff = PREC_TYPE_TO_MASS_DIFF[prec_type]
	if not isotopes:
		peak_mz = 0.
		for k,v in element_counts.items():
			peak_mz += v*ELEMENT_TO_MASS.get(k,0.)
		peak_mz += mass_diff
		peak_mzs = [peak_mz]
	else:
		peak_mzs = formula_utils.get_peaks_for_formula(element_counts)[0]
		peak_mzs = [peak_mz+mass_diff for peak_mz in peak_mzs]
	return peak_mzs

def compute_frag_peak_stats(row,tolerance=0.01):

	true_mzs, true_ints = list(zip(*row["peaks"]))
	pred_mzs = np.array(row["peak_mzs"])
	# compute overlap
	overlap_true_idxs = []
	overlap_true_ints = []
	overlap_pred_idxs = []
	overlap_pred_counts = []
	for true_idx, true_mz in enumerate(true_mzs):
		mz_diffs = np.abs(pred_mzs-true_mz)
		mz_close = mz_diffs < tolerance
		if np.any(mz_close):
			pred_idx = np.nonzero(mz_close)[0]
			overlap_true_idxs.append(true_idx)
			overlap_true_ints.append(true_ints[true_idx])
			overlap_pred_idxs.append(pred_idx)
			overlap_pred_counts.append(len(pred_idx))
	# remove duplicates
	if len(overlap_pred_idxs) > 0:
		overlap_pred_idxs = np.unique(np.concatenate(overlap_pred_idxs,axis=0))
	else:
		overlap_pred_idxs = np.array([])
	recall = len(overlap_true_idxs) / len(true_mzs)
	w_recall = np.sum(overlap_true_ints) / np.sum(true_ints)
	prec = len(overlap_pred_idxs) / len(pred_mzs)
	if len(overlap_pred_counts) > 0:
		avg_count = np.mean(overlap_pred_counts)
	else:
		avg_count = 0.
	return pd.Series([recall,w_recall,prec,avg_count])

def prec_in_frag(row):

	prec_formula = row["prec_formula_str"]
	frag_formulae = row["frag_formulae_str"]
	return prec_formula in frag_formulae

def add_prec_to_frag(row):

	prec_formula = row["prec_formula_str"]
	frag_formulae = row["frag_formulae_str"]
	if prec_formula not in frag_formulae:
		formulae = frag_formulae + [prec_formula]
	else:
		formulae = frag_formulae + []
	return formulae

def print_and_log(name,series,wandb_flag):

	stats = series.describe()
	print(f"> {name}")
	print(stats)
	print()
	if wandb_flag:
		log_d = {"step":0}
		for stat in ["mean","std","50%","min","max"]:
			log_d[f"{name}/{stat}"] = stats[stat]
		hist_data = plot_histogram(series.values,name)
		log_d[f"{name}-hist"] = wandb.Image(hist_data)
		print(f"> logging {name}: {log_d.keys()}")
		wandb.log(log_d)

if __name__ == "__main__":

	# TODO: verify that using different version of software doesn't break compatibility!

	parser = argparse.ArgumentParser()
	parser.add_argument("--num_entries",type=int,default=-1)
	parser.add_argument("--max_depth",type=int,default=4)
	parser.add_argument("--max_time",type=int,default=2*60)
	parser.add_argument("--project_dp",type=str,required=True)
	parser.add_argument("--frag_dp",type=str,default="data/frag")
	parser.add_argument("--proc_dp",type=str,default="data/proc")
	parser.add_argument("--frag_df_name",type=str,default="debug_frag_df.pkl")
	parser.add_argument("--load_frag_df",type=booltype,default=False)
	parser.add_argument("--h_prior",type=booltype,default=True)
	parser.add_argument("--max_h_transfer",type=int,default=4)
	parser.add_argument("--allowed_elements",type=str,default="C,H,O,N,F,S,P,Cl")
	parser.add_argument("--tolerance",type=float,default=0.01)
	parser.add_argument("--isotopes",type=booltype,default=True)
	parser.add_argument("--wandb_mode",type=str,default="off",choices=["online","offline","off"])
	parser.add_argument("--run_name",type=str,default="debug")
	parser.add_argument("--wandb_dp",type=str,default="wandb")
	parser.add_argument("--wandb_symlink_dp",type=str,default="jobs_wandb")
	parser.add_argument("--job_id",type=str,default="00000")
	args = parser.parse_args()

	tqdm.pandas()

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

	# perform selection ([M+H]+ ONLY, no S/P)
	spec_df, mol_df = filter_spec_mol(spec_df,mol_df,allowed_elements,num_entries=args.num_entries)
	
	print(f">> {len(spec_df)} spectra, {len(mol_df)} molecules")
	print()

	num_atoms = mol_df["mol"].apply(lambda x: x.GetNumHeavyAtoms())
	print_and_log("num_atoms",num_atoms,wandb_flag)

	num_bonds = mol_df["mol"].apply(lambda x: len(x.GetBonds()))
	print_and_log("num_bonds",num_bonds,wandb_flag)

	frag_fp = os.path.join(args.project_dp,args.frag_dp,args.frag_df_name)

	if not args.load_frag_df:

		def timed_get_mol_formulae_ccs(smiles):
			cur_max_depth = args.max_depth
			formulae, ccs = None, None
			while formulae is None:
				try:
					with time_limit(args.max_time):
						formulae, ccs = get_mol_formulae_ccs(smiles,cur_max_depth,args.h_prior,args.max_h_transfer)
				except TimeoutException as e:
					if cur_max_depth > 1:
						cur_max_depth -= 1
						print(f">> Decrementing depth to {cur_max_depth}")
					else:
						formulae, ccs = [], []
				except KeyboardInterrupt as e:
					# let these through
					raise e
				except Exception as e:
					# don't retry, theres a bug
					print(f">> Non-timeout error, aborting: {e}")
					formulae, ccs = [], []
			return (cur_max_depth,formulae,ccs)

		# maybe chunk it, to allow for more reductions with each worker?
		frag_results = data_utils.par_apply_series(mol_df["mol"],timed_get_mol_formulae_ccs)
		frag_depth = frag_results.apply(lambda x: x[0])
		frag_formulae = frag_results.apply(lambda x: x[1])
		frag_ccs = frag_results.apply(lambda x: x[2])
		del frag_results

		frag_df = pd.DataFrame(
			{
				"mol_id": mol_df["mol_id"],
				"frag_formulae": frag_formulae,
				"frag_depth": frag_depth,
				"frag_ccs": frag_ccs
			}
		)
		frag_df.to_pickle(frag_fp)

	else:

		print("read in formulae")
		assert os.path.exists(frag_fp), frag_fp
		frag_df = pd.read_pickle(frag_fp)
		print(frag_df.shape)

	# pool formulae across molecules, compute frequencies
	def formulae_to_array(formulae):
		if len(formulae) == 0:
			return np.ones((0,len(ELEMENT_TO_IDX)),dtype=int)
		else:
			return np.array(list(formulae.keys()),dtype=int).reshape(len(formulae),len(ELEMENT_TO_IDX))
	frag_formulae = frag_df["frag_formulae"].progress_apply(formulae_to_array)
	unique_formulae = np.unique(np.concatenate(frag_formulae.values,axis=0),axis=0).tolist()
	print(len(unique_formulae), unique_formulae[:5])
	print()

	# convert to strings
	unique_formulae = list(map(frag_utils.get_formula_string,unique_formulae))
	print(len(unique_formulae), unique_formulae[:5])
	print()

	# count number of formula per molecule (filtering out ones that have 0 formulae)
	frag_df.loc[:,"frag_num_formulae"] = frag_df["frag_formulae"].apply(lambda x: len(x))
	print_and_log("num_formulae",frag_df["frag_num_formulae"],wandb_flag)
	print("> num failures:",(frag_df["frag_num_formulae"]==0).sum())
	print()
	if wandb_flag:
		wandb.log({"num_failures": (frag_df["frag_num_formulae"]==0).sum(),"step":0})
	frag_df = frag_df[frag_df["frag_num_formulae"]>0]
	
	# compute ratio of ccs to formulae
	frag_df.loc[:,"frag_num_ccs"] = frag_df["frag_ccs"].apply(len)
	print_and_log("num_ccs",frag_df["frag_num_ccs"],wandb_flag)
	print_and_log("num_ccs_per_formula",frag_df["frag_num_ccs"]/frag_df["frag_num_formulae"],wandb_flag)

	# compute formula redundancy
	def get_formula_redundancy(formulae):
		return np.mean(np.array(list(formulae.values())))
	frag_df.loc[:,"frag_formula_redundancy"] = frag_df["frag_formulae"].apply(get_formula_redundancy)
	print_and_log("formula_redundancy",frag_df["frag_formula_redundancy"],wandb_flag)

	# check depth
	print("> depth:")
	print(frag_df["frag_depth"].value_counts())
	print()
	# if wandb_flag:
	# 	wandb.log({"depth": wandb.Histogram(frag_df["frag_depth"].values)},commit=True)

	# convert formulae to strings
	frag_df.loc[:,"frag_formulae_str"] = frag_df["frag_formulae"].apply(lambda x: list(map(frag_utils.get_formula_string,x)))
	frag_df = frag_df.merge(mol_df[["mol_id","formula"]],on="mol_id",how="inner").rename(columns={"formula":"prec_formula_str"})
	prec_in_frag_count = frag_df.apply(prec_in_frag,axis=1)
	print(f"> prec in frag: {prec_in_frag_count.mean()}")
	print()
	if wandb_flag:
		wandb.log({"prec_in_frag": prec_in_frag_count.mean(),"step":0})
	frag_df.loc[:,"formulae_str"] = frag_df.apply(add_prec_to_frag,axis=1)

	# convert formulae to peak locations
	mh_formula_to_peak_mzs = lambda x: formula_to_peak_mzs(x,"[M+H]+",isotopes=args.isotopes)
	frag_df.loc[:,"peak_mzs"] = data_utils.par_apply_series(frag_df["formulae_str"], lambda x: flatten_lol(list(map(mh_formula_to_peak_mzs,x))))
	mh_formula_to_peak_mzs = lambda x: formula_to_peak_mzs(x,"[M+H]+",isotopes=False)
	frag_df.loc[:,"prec_peak_mz"] = frag_df["prec_formula_str"].progress_apply(lambda x: mh_formula_to_peak_mzs(x)[0])
	
	# for each spectrum, calculate fraction of peaks explained, fraction of intensity explained by referencing
	peak_spec_df = spec_df[["spec_id","mol_id","peaks","prec_mz"]].merge(frag_df,on="mol_id",how="inner")
	# prec_mz_diff = (peak_spec_df["prec_peak_mz"]-peak_spec_df["prec_mz"]).apply(np.abs)
	# print(f"> prec mz diff")
	# print(prec_mz_diff.describe())
	# print()
	result = peak_spec_df.apply(lambda x: compute_frag_peak_stats(x,tolerance=args.tolerance),axis=1)
	peak_spec_df.loc[:,"frag_recall"] = result[0].values
	peak_spec_df.loc[:,"frag_wrecall"] = result[1].values
	peak_spec_df.loc[:,"frag_prec"] = result[2].values
	peak_spec_df.loc[:,"frag_avg_count"] = result[3].values
	del result

	# collect across spectra and report summary statistics
	print_and_log("spec_frag_recall",peak_spec_df["frag_recall"],wandb_flag)
	print_and_log("spec_frag_wrecall",peak_spec_df["frag_wrecall"],wandb_flag)
	print_and_log("spec_frag_prec",peak_spec_df["frag_prec"],wandb_flag)
	print_and_log("spec_frag_avg_count",peak_spec_df["frag_avg_count"],wandb_flag)

	# collect across molecules and report summary statistics
	agg_dict = {
		"frag_recall":np.mean,
		"frag_wrecall":np.mean,
		"frag_prec":np.mean,
		"frag_avg_count":np.mean
	}
	peak_mol_df = peak_spec_df[["mol_id"]+list(agg_dict.keys())].groupby("mol_id").agg(agg_dict).reset_index()
	print_and_log("mol_frag_recall",peak_mol_df["frag_recall"],wandb_flag)
	print_and_log("mol_frag_wrecall",peak_mol_df["frag_wrecall"],wandb_flag)
	print_and_log("mol_frag_prec",peak_mol_df["frag_prec"],wandb_flag)
	print_and_log("mol_frag_avg_count",peak_mol_df["frag_avg_count"],wandb_flag)

	if wandb_flag and args.wandb_symlink_dp != "":
		wandb_symlink_dp = os.path.join(args.project_dp,args.wandb_symlink_dp)
		wandb_symlink(wandb_run.dir,wandb_symlink_dp,args.job_id)

	if wandb_flag:
		wandb.finish()
