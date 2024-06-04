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

def filter_spec_mol(
		spec_df,
		mol_df,
		allowed_elements=None,
		allowed_dsets=None,
		max_peak_mz=None,
		max_prez_mz=None,
		min_prez_mz=None,
		allowed_prec_types:list = None,
		num_entries=-1,
  		allowed_inst_types:list = None,
    	allowed_frag_modes:list = None,
     	allowed_ion_modes:list = None,
      	require_ace:bool = False,
        require_nce:bool = False,):
	"""
	The purpose of this filtering is to prevent errors
	"""
	masks = []
 
	## spectrum criteria (TODO remove this)
	# dataset mask
	if allowed_dsets is not None:
		print(f">> dsets filter: {allowed_dsets}")
		dset_mask = spec_df["dset"].isin(allowed_dsets)
		print(f">> dsets filter mean: {dset_mask.mean()}")
		masks.append(dset_mask)

	# instrument type
	if allowed_inst_types is not None:
		print(f">> inst type: {allowed_inst_types}")
		inst_type_mask = spec_df["inst_type"].isin(allowed_inst_types)
		print(f">> inst type filter mean: {inst_type_mask.mean()}")
		masks.append(inst_type_mask)
  
	# frag mode
	if allowed_frag_modes is not None:
		print(f">> frag mode filter: {allowed_frag_modes}")
		frag_mode_mask = spec_df["frag_mode"].isin(allowed_frag_modes)
		print(f">> frag mode filter mean: {frag_mode_mask.mean()}")
		masks.append(frag_mode_mask)
  
	# ce filter
	if require_ace:
		print(f">> require ace")
		ace_mask = (~spec_df["ace"].isna())
		print(f">> ace filter: {ace_mask.mean()}")
		masks.append(ace_mask)
	if require_nce:
		print(f">> require nce")
		nce_mask = (~spec_df["nce"].isna())
		print(f">> nce filter: {nce_mask.mean()}")
		masks.append(nce_mask)
 
	# ion mode
	if allowed_ion_modes is not None:
		print(f">> ion mode filter: {allowed_ion_modes}")
		ion_mode_mask = spec_df["ion_mode"].isin(allowed_ion_modes)
		print(f">> ion mode filter mean: {ion_mode_mask.mean()}")
		masks.append(ion_mode_mask)
  
	# precursor type
	if allowed_prec_types is not None:
		print(f">> prec type filter: {allowed_prec_types}")
		prec_type_mask = spec_df["prec_type"].isin(allowed_prec_types)
		print(f">> prec type filter mean: {prec_type_mask.mean()}")
		masks.append(prec_type_mask)
  
	# resolution
	# res_mask = spec_df["res"].isin([1,2,3,4,5,6,7])
	# print(f">> res: {res_mask.mean()}")
	# masks.append(res_mask)
 
	# spectrum type 
	spec_type_mask = spec_df["spec_type"] == "MS2"
	print(f">> spec type: {spec_type_mask.mean()}")
	masks.append(spec_type_mask)
	# precursor mz
	prec_mz_mask = ~spec_df["prec_mz"].isna()
	print(f">> prec mz: {prec_mz_mask.mean()}")
	masks.append(prec_mz_mask)
  
	if max_prez_mz is not None:
		print("f >> ")
		max_prez_mz_mask = (spec_df["prec_mz"] <= max_prez_mz)
		print(f">> prec mz: {prec_mz_mask.max_prez_mz_mask()}")
		masks.append(max_prez_mz_mask)

	# max peak mz
	if max_peak_mz is not None:
		def get_max_mz(peaks):
			return max([peak[0] for peak in peaks])
		max_peak_mz_mask = spec_df["peaks"].apply(get_max_mz) <= max_peak_mz
		print(f">> max peak mz: {max_peak_mz_mask.mean()}")
		masks.append(max_peak_mz_mask)

	## molecule criteria
	# single molecule
	single_mol_ids = mol_df[mol_df["single_mol"]]["mol_id"]
	single_mol_mask = spec_df["mol_id"].isin(single_mol_ids)
	print(f">> single mol: {single_mol_mask.mean()}")
	masks.append(single_mol_mask)
	# neutral molecule
	neutral_ids = mol_df[mol_df["charge"]==0]["mol_id"]
	neutral_mask = spec_df["mol_id"].isin(neutral_ids)
	print(f">> neutral: {neutral_mask.mean()}")
	masks.append(neutral_mask)
	# element composition
	if allowed_elements is not None:
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

def merge_spec_df(spec_df,renormalize=False,sum_ints=True,keep_ces=False):

	# merge the peaks (they are not normalized here)
	m_peaks_func = lambda peakses: merge_sparse_specs(*peakses,renormalize=renormalize,sum_ints=sum_ints)
	m_spec_peaks_df = spec_df[["group_id","peaks"]].groupby("group_id").agg({"peaks":m_peaks_func}).reset_index()
	m_spec_meta_df = spec_df.drop(columns=["peaks","spec_id"]).drop_duplicates(subset=["group_id"])
	m_spec_df = m_spec_peaks_df.merge(m_spec_meta_df,on=["group_id"],how="inner")
	if keep_ces:
		m_spec_ce_df = spec_df[["group_id","nce","ace"]].groupby("group_id").agg({"nce":list,"ace":list}).reset_index()
		m_spec_df = m_spec_df.drop(columns=["nce","ace"])
		m_spec_df = m_spec_df.merge(m_spec_ce_df,on=["group_id"],how="inner")
	else:
		m_spec_df.loc[:,"nce"] = np.nan
		m_spec_df.loc[:,"ace"] = np.nan
	return m_spec_df
