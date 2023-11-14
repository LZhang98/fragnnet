import pandas as pd
import argparse
import os
import numpy as np

from frag_gnn.utils.proc_utils import filter_spec_mol
from frag_gnn.utils.frag_utils import ELEMENTS


def main(args):

    spec_df = pd.read_pickle(os.path.join(args.proc_dp,"spec_df.pkl"))
    mol_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl"))

    f_spec_df, f_mol_df = filter_spec_mol(spec_df,mol_df,ELEMENTS,args.dsets)

    find_min_mz = lambda peaks: min(p[0] for p in peaks)
    find_max_mz = lambda peaks: max(p[0] for p in peaks)
    find_prec_mz = lambda row: any(abs(row["prec_mz"]-p[0]) <= 0.01 for p in row["peaks"])
    # find_prec_mz_wide = lambda row: any(abs(row["prec_mz"]-p[0]) <= 1.0 for p in row["peaks"])
    def find_min_prec_diff(row):
        val = None
        for p in row["peaks"]:
            if p[0] > row["prec_mz"]:
                if val is None:
                    val = p[0]-row["prec_mz"]
                else:
                    val = min(p[0]-row["prec_mz"],val)
        if val is None:
            val = 0.
        return val
    def find_max_prec_diff(row):
        val = None
        for p in row["peaks"]:
            if p[0] > row["prec_mz"]:
                if val is None:
                    val = p[0]-row["prec_mz"]
                else:
                    val = max(p[0]-row["prec_mz"],val)
        if val is None:
            val = 0.
        return val
    def find_ints_over_prec(row):
        ints = 0.
        total_ints = 0.
        for p in row["peaks"]:
            if p[0] > row["prec_mz"]:
                ints += p[1]
            total_ints += p[1]
        return ints/total_ints

    f_spec_df["min_mz"] = f_spec_df["peaks"].apply(find_min_mz)
    f_spec_df["max_mz"] = f_spec_df["peaks"].apply(find_max_mz)
    f_spec_df["min_over_prec"] = f_spec_df["min_mz"] > f_spec_df["prec_mz"]
    f_spec_df["max_over_prec"] = f_spec_df["max_mz"] > f_spec_df["prec_mz"]
    f_spec_df["max_prec_diff"] = f_spec_df.apply(find_max_prec_diff,axis=1)
    f_spec_df["prec_in_peaks"] = f_spec_df.apply(find_prec_mz,axis=1)
    f_spec_df["min_prec_diff"] = f_spec_df.apply(find_min_prec_diff,axis=1)
    f_spec_df["ints_over_prec"] = f_spec_df.apply(find_ints_over_prec,axis=1)

    print("min over prec: ",f_spec_df["min_over_prec"].sum(),f_spec_df["min_over_prec"].mean())
    print("max over prec: ",f_spec_df["max_over_prec"].sum(),f_spec_df["max_over_prec"].mean())
    print("prec in peaks: ",f_spec_df["prec_in_peaks"].sum(),f_spec_df["prec_in_peaks"].mean())

    mf_spec_df = f_spec_df[(~f_spec_df["prec_in_peaks"] & f_spec_df["max_over_prec"])]
    print(mf_spec_df.shape[0])
    print(mf_spec_df["max_prec_diff"].describe())
    print(mf_spec_df["min_prec_diff"].describe())

    print(f_spec_df[f_spec_df["max_over_prec"]]["ints_over_prec"].describe())

    import pdb; pdb.set_trace()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-p",
        "--proc_dp",
        type=str,
        required=True
    )
    parser.add_argument(
        "-d",
        "--dsets",
        type=str,
        nargs="+",
        required=True
    )
    args = parser.parse_args()
    main(args)