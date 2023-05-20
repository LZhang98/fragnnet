import pandas as pd
import os
import argparse

from frag_gnn.spec_utils import merge_sparse_specs

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--proc_dp", type=str, default="data/proc")
    parser.add_argument("--frag_dp", type=str, default="data/frag")
    parser.add_argument("--frag_run", type=str, choices=["d3_h4_re","d3_h4_fe","d4_h4_re","d4_h4_fe"])
    parser.add_argument("--mol_id", type=int, required=True)
    args = parser.parse_args()

    spec_df = pd.read_pickle(os.path.join(args.proc_dp,"spec_df.pkl"))
    mol_df = pd.read_pickle(os.path.join(args.proc_dp,"mol_df.pkl"))

    hcd_df = spec_df[(spec_df["dset"]=="nist")&(spec_df["inst_type"]=="FT")&(spec_df["frag_mode"]=="HCD")&(spec_df["prec_type"]=="[M+H]+")]

    spec_subset = hcd_df[hcd_df["mol_id"]==args.mol_id]
    print(spec_subset)
    peaks = spec_subset["peaks"].tolist()
    merged_peaks = merge_sparse_specs(*peaks)

    print(merged_peaks)
