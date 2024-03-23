import pandas as pd

from frag_gnn.utils.frag_utils import ELEMENTS
from frag_gnn.utils.proc_utils import filter_spec_mol
from frag_gnn.graff.data_utils import fit_ann_library

if __name__ == "__main__":

    spec_df_fp = "data/proc/debug_annot_2/spec_df.pkl"
    mol_df_fp = "data/proc/debug_annot_2/mol_df.pkl"
    ann_df_fp = "data/proc/debug_annot_2/ann_df.pkl"

    spec_df = pd.read_pickle(spec_df_fp)
    mol_df = pd.read_pickle(mol_df_fp)
    ann_df = pd.read_pickle(ann_df_fp)

    k  = 10000

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

    library_df = fit_ann_library(spec_df, ann_df, k)

    import pdb; pdb.set_trace()