import argparse
import os

import pandas as pd

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--proc_dp", type=str, required=True)
    parser.add_argument("--split_dp", type=str, required=True)

    args = parser.parse_args()
    print("> reading spec_df from", args.proc_dp)
    spec_df = pd.read_pickle(os.path.join(args.proc_dp, "spec_df.pkl"))
    spec_df = spec_df[["spec_id", "prec_type"]]

    for split in ["train", "val", "test"]:
        if not os.path.exists(os.path.join(args.split_dp, f"{split}_ids.csv")):
            raise FileNotFoundError(f"Split file for {split} not found in {args.split_dp}")
        split_df = pd.read_csv(os.path.join(args.split_dp, f"{split}_ids.csv"))
        split_df = split_df.merge(spec_df, on="spec_id", how="left")

        unique_prec_types = split_df["prec_type"].unique()

        for prec_type in unique_prec_types:
            o_split_dp = f"{args.split_dp}_{prec_type}"
            print(f"Processing split: {split}, prec_type: {prec_type}, output dir: {o_split_dp}")
            os.makedirs(o_split_dp, exist_ok=True)
            split_df[split_df["prec_type"] == prec_type][["spec_id", "mol_id", "group_id"]].to_csv(
                os.path.join(o_split_dp, f"{split}_ids.csv"), index=False
            )
