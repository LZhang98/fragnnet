"""Remove spectra with precursor m/z above a threshold from existing split CSV files.

Iterates over every cv*/train_ids.csv (and val/test if present) under split_dp,
joins against spec_df to get prec_mz, drops rows that exceed max_prec_mz, and
overwrites the CSV files in-place.

Usage:
    python preproc_scripts/filter_split_by_prec_mz.py \
        --proc_dp data/proc/nist23_pcdl \
        --split_dp data/split/ccx_nist23_pcdl \
        --max_prec_mz 1500.0
"""

import argparse
import os

import pandas as pd


def filter_split_dir(split_dir: str, spec_df: pd.DataFrame, max_prec_mz: float) -> None:
    """Filter all split CSVs in a single directory.

    Args:
        split_dir: Path to a directory containing *_ids.csv files.
        spec_df: DataFrame with columns spec_id and prec_mz.
        max_prec_mz: Maximum allowed precursor m/z; rows exceeding this are removed.
    """
    for split in ["train", "val", "test", "secondary"]:
        fp = os.path.join(split_dir, f"{split}_ids.csv")
        if not os.path.exists(fp):
            continue
        df = pd.read_csv(fp)
        before = len(df)
        df = df.merge(spec_df[["spec_id", "prec_mz"]], on="spec_id", how="left")
        mask = df["prec_mz"] <= max_prec_mz
        removed = int((~mask).sum())
        df = df[mask][["spec_id", "mol_id", "group_id"]]
        df.to_csv(fp, index=False)
        if removed:
            print(f"  [{split}] removed {removed}/{before} spectra with prec_mz > {max_prec_mz}")
        else:
            print(f"  [{split}] no spectra removed ({before} rows kept)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Filter split CSVs by precursor m/z threshold")
    parser.add_argument(
        "--proc_dp",
        type=str,
        required=True,
        help="Directory containing spec_df.pkl with prec_mz column",
    )
    parser.add_argument(
        "--split_dp",
        type=str,
        required=True,
        help="Root split directory; scans immediate subdirs (cv*) and the root itself",
    )
    parser.add_argument(
        "--max_prec_mz",
        type=float,
        default=1500.0,
        help="Maximum allowed precursor m/z (default: 1500.0)",
    )
    args = parser.parse_args()

    print(f"> Loading spec_df from {args.proc_dp}")
    spec_df = pd.read_pickle(os.path.join(args.proc_dp, "spec_df.pkl"))
    print(
        f"> {len(spec_df):,} spectra loaded; "
        f"{(spec_df['prec_mz'] > args.max_prec_mz).sum()} have prec_mz > {args.max_prec_mz}"
    )

    # process root-level splits (train_ids.csv directly in split_dp)
    root_csvs = [f for f in os.listdir(args.split_dp) if f.endswith("_ids.csv")]
    if root_csvs:
        print(f"> Filtering root split: {args.split_dp}")
        filter_split_dir(args.split_dp, spec_df, args.max_prec_mz)

    # process cv subdirectories
    for entry in sorted(os.listdir(args.split_dp)):
        subdir = os.path.join(args.split_dp, entry)
        if os.path.isdir(subdir):
            print(f"> Filtering {entry}: {subdir}")
            filter_split_dir(subdir, spec_df, args.max_prec_mz)

    print("> Done.")
