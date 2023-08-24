import pandas as pd
import numpy as np


PREC_ADDUCTS = [
	"[M+H]+",
	"[M-H]-",
	"[M+H-H2O]+",
	"[M+Na]+",
	"[2M+H]+",
	"[M+H+2i]+",
	"[2M-H]-",
	"[M+NH4]+",
	"[M+H-2H2O]+",
	"[M+2H]2+",
	"[M+H-NH3]+",
	"[M-H-CO2]-",
	"[M-H-H2O]-",
	"[M+Cl]-",
	"[M+CHO2]-",
	"[M-H+2i]-",
	"[M+H-CH2O2]+",
	"[M+K]+",
	"[M+H-C2H4O2]+",
	"[M+H-C4H8]+",
	"[M+H-3H2O]+"
]


def filter_spec_mol(spec_df,mol_df,num_entries=-1):
	"""
	The purpose of this filtering is to prevent errors
	"""

	masks = []
	## spectrum criteria (TODO remove this)
	# dataset mask
	dset_mask = spec_df["dset"].isin(["nist20_hr","nist20_lr","mona23"])
	masks.append(dset_mask)
	# instrument type
	inst_type_mask = spec_df["inst_type"].isin(["FT","QTOF","IT","QQQ"]) #,"Other"])
	masks.append(inst_type_mask)
	# frag mode
	frag_mode_mask = spec_df["frag_mode"].isin(["HCD","CID"])
	masks.append(frag_mode_mask)
	# ion mode
	ion_mode_mask = spec_df["ion_mode"].isin(["P","N"])
	masks.append(ion_mode_mask)
	# precursor type
	prec_type_mask = spec_df["prec_type"].isin(PREC_ADDUCTS)
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
	# radical stats
	non_radical_ids = mol_df[mol_df["num_radicals"]==0]["mol_id"]
	non_radical_mask = spec_df["mol_id"].isin(non_radical_ids)
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

if __name__ == "__main__":

    spec_df = pd.read_pickle("data/proc_all/spec_df.pkl")
    mol_df = pd.read_pickle("data/proc_all/mol_df.pkl")
    print("num_spec",spec_df.shape[0])
    print("num_mol",mol_df.shape[0])
	print(spec_df["prec_type"].value_counts())
    spec_df, mol_df = filter_spec_mol(spec_df,mol_df)
    print("num_spec",spec_df.shape[0])
    print("num_mol",mol_df.shape[0])
    print(spec_df["prec_type"].value_counts())
    import pdb; pdb.set_trace()

    # prec_type_vcs = spec_df["prec_type"].value_counts()
    # prec_types = prec_type_vcs.index.values[prec_type_vcs.values>1000]
    # print(prec_types)
    # import pdb; pdb.set_trace()
