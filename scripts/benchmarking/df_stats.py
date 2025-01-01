import pandas as pd
import argparse
import os

from frag_gnn.utils.data_utils import par_apply_series, parse_prec_type_str

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

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dp",type=str,default="data/proc_nist20_mona23")
    parser.add_argument("--spec_file",type=str,default="spec_df.pkl")
    parser.add_argument("--mol_file",type=str,default="mol_df.pkl")
    parser.add_argument("--dsets",type=str,nargs="+",default=["nist20_hr","nist20_lr","mona23"])
    args = parser.parse_args()

    assert os.path.isdir(args.proc_dp), args.proc_dp
    spec_fp = os.path.join(args.proc_dp,args.spec_file)
    mol_fp = os.path.join(args.proc_dp,args.mol_file)
    assert os.path.isfile(spec_fp), spec_fp
    assert os.path.isfile(mol_fp), mol_fp

    spec_df = pd.read_pickle(spec_fp)
    mol_df = pd.read_pickle(mol_fp)

    dsets = args.dsets

    # some preliminary selection

    spec_df = spec_df[spec_df["spec_type"]=="MS2"]
    spec_df = spec_df.dropna(subset=["prec_mz","prec_type"],axis=0,how="all")
    # spec_df = spec_df[spec_df["prec_type"].isin(PREC_ADDUCTS)]

    # count number of spectra per dataset

    print(">>> number of spectra per dataset")
    print(spec_df["dset"].value_counts()[:10])
    print("total",spec_df.shape[0])

    # check number of molecules per dataset

    print(">>> number of molecules per dataset")
    dset_mol_ids = {}
    all_mol_ids = set()
    for dset in dsets:
        dset_mol_ids[dset] = set(spec_df[spec_df["dset"]==dset]["mol_id"])
        print(dset,len(dset_mol_ids[dset]))
        all_mol_ids.update(dset_mol_ids[dset])
    print("total",len(all_mol_ids))

    print(">>> number of common molecules between datasets")
    comparisons = set()
    for dset1 in dsets:
        for dset2 in dsets:
            if dset1 != dset2 and (dset1,dset2) not in comparisons:
                print(dset1,dset2,len(dset_mol_ids[dset1].intersection(dset_mol_ids[dset2])))
                comparisons.add((dset1,dset2))
                comparisons.add((dset2,dset1))

    print(">>> number of unique molecules per dataset")
    for dset1 in dsets:
        mol_ids = dset_mol_ids[dset1].copy()
        for dset2 in dsets:
            if dset1 != dset2:
                mol_ids = mol_ids.difference(dset_mol_ids[dset2])
        print(dset1,len(mol_ids))

    # count the number of instrument types per dataset

    print(">>> number of instrument types per dataset")
    for dset in dsets:
        dset_df = spec_df[spec_df["dset"]==dset]
        print(dset)
        print(dset_df["inst_type"].value_counts()[:10])

    # count the number of precursor adducts per dataset

    print(">>> number of precursor adducts per dataset")
    for dset in dsets:
        dset_df = spec_df[spec_df["dset"]==dset]
        print(dset)
        print(dset_df["prec_type"].value_counts()[:10])

    import pdb; pdb.set_trace()
