import pandas as pd
import numpy as np

from frag_gnn.utils.spec_utils import merge_sparse_specs
from frag_gnn.utils.formula_utils import parse_formula
from frag_gnn.frag.compute_frags import MAX_NUM_NODES, MAX_NUM_EDGES


def element_filter(formula,allowed_elements):
	try:
		element_counts = parse_formula(formula)
	except ValueError as e:
		return False
	for k,v in element_counts.items():
		if v > 0 and k not in allowed_elements:
			return False
	return True

def filter_spec_mol(spec_df,mol_df,allowed_elements,dsets,num_entries=-1):
	"""
	The purpose of this filtering is to prevent errors
	"""

	masks = []
	## spectrum criteria (TODO remove this)
	# dataset mask
	dset_mask = spec_df["dset"].isin(dsets)
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

def merge_spec_df(spec_df,renormalize=False,sum_ints=True):

	# merge the peaks (they are not normalized here)
	m_func = lambda peakses: merge_sparse_specs(*peakses,renormalize=renormalize,sum_ints=sum_ints)
	m_spec_df = spec_df[["group_id","peaks"]].groupby("group_id").agg({"peaks":m_func}).reset_index()
	# add the metadata back
	meta_spec_df = spec_df.drop(columns=["peaks","spec_id"]).drop_duplicates(subset=["group_id"])
	m_spec_df = m_spec_df.merge(meta_spec_df,on=["group_id"],how="inner")
	return m_spec_df
