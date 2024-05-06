import pandas as pd
import argparse
import os

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--old_split_dp", type=str, required=True)
    parser.add_argument("--old_proc_dp", type=str, required=True)
    parser.add_argument("--new_split_dp", type=str, required=True)
    parser.add_argument("--new_proc_dp", type=str, required=True)

    args = parser.parse_args()

    old_proc_df = pd.read_pickle(os.path.join(args.old_proc_dp, "spec_df.pkl"))

    new_proc_df = pd.read_pickle(os.path.join(args.new_proc_dp, "spec_df.pkl"))

    old_proc_df = old_proc_df[["spec_id","mol_id","group_id","dset_spec_id"]]
    new_proc_df = new_proc_df[["spec_id","mol_id","group_id","dset_spec_id"]]

    splits = ["train", "val", "test", "secondary"]

    os.makedirs(args.new_split_dp, exist_ok=True)

    for split in splits:

        print(f"> {split}")

        old_split_fp = os.path.join(args.old_split_dp, f"{split}_ids.csv")
        new_split_fp = os.path.join(args.new_split_dp, f"{split}_ids.csv")
        assert os.path.isfile(old_split_fp), old_split_fp
        assert not os.path.isfile(new_split_fp), new_split_fp

        old_split_df = pd.read_csv(old_split_fp)
        print(old_split_df.shape[0])

        old_split_df = old_split_df.merge(old_proc_df, on="spec_id", how="inner")
        print(old_split_df.shape[0])

        new_split_df = new_proc_df[new_proc_df["dset_spec_id"].isin(old_split_df["dset_spec_id"])]
        print(new_split_df.shape[0])

        new_split_df = new_split_df[["spec_id","mol_id","group_id"]]

        new_split_df.to_csv(new_split_fp, index=False)

        print()
