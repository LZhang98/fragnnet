import pandas as pd
import numpy as np

from frag_gnn.utils.frag_utils import ELEMENTS
from frag_gnn.utils.proc_utils import filter_spec_mol
from frag_gnn.graff.data_utils import fit_ann_library

if __name__ == "__main__":

    spec_df_fp = "data/proc/debug_annot_6/spec_df.pkl"
    mol_df_fp = "data/proc/debug_annot_6/mol_df.pkl"
    ann_df_fp = "data/proc/debug_annot_6/ann_df.pkl"

    spec_df = pd.read_pickle(spec_df_fp)
    mol_df = pd.read_pickle(mol_df_fp)
    ann_df = pd.read_pickle(ann_df_fp)

    k  = 10000
    max_isotope = 3

    spec_df, mol_df = filter_spec_mol(
        spec_df, 
        mol_df,
        dsets=["nist20_hr"],
        prec_types=["[M+H]+"],
        max_peak_mz=1500.,
        allowed_elements=ELEMENTS)

    if "ann_exact_mzs" not in ann_df.columns:
        print("> missing ann_exact_mzs, using ann_peak_mzs instead")
        ann_df["ann_exact_mzs"] = ann_df["ann_peak_mzs"]

    library_df, big_mz_diff_ids = fit_ann_library(spec_df, ann_df, k, max_isotope, debug=True)

    diff_ann_df = ann_df[ann_df["spec_id"].isin(big_mz_diff_ids)]
    spec_id_map = spec_df[spec_df["spec_id"].isin(big_mz_diff_ids)][["spec_id","dset_spec_id"]].drop_duplicates()

    diff_ann_df_counts = diff_ann_df.groupby(by="spec_id").size()
    print(diff_ann_df_counts.value_counts())

    diff_ann_df = diff_ann_df.sample(n=10,replace=False,random_state=420)
    for i in range(10):
        diff_ann_entry = diff_ann_df.iloc[i]
        spec_id = diff_ann_entry["spec_id"]
        dset_spec_id = spec_id_map[spec_id_map["spec_id"]==spec_id]["dset_spec_id"].values[0]
        ann_peak_mzs = np.array(list(map(float,diff_ann_entry["ann_peak_mzs"])))
        ann_exact_mzs = np.array(diff_ann_entry["ann_exact_mzs"])
        ann_diff_mzs = np.abs(ann_peak_mzs-ann_exact_mzs)
        print(f">>> spec_id {spec_id}, dset_spec_id {dset_spec_id}")
        print(ann_peak_mzs)
        print(ann_diff_mzs)
        import pdb; pdb.set_trace()

    import pdb; pdb.set_trace()