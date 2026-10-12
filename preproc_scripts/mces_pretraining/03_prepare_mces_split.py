import argparse
import json
import os
import random

import pandas as pd

from fragnnet.utils import formula_utils


def make_splits(args):
    ms_mol_df = pd.read_pickle(os.path.join(args.ms_proc_dp, "mol_df.pkl"))
    ms_split_d = {}
    for split_type in ["train", "val", "test"]:
        split_fp = os.path.join(args.ms_split_dp, f"{split_type}_ids.csv")
        split_df = pd.read_csv(split_fp)
        ms_split_d[split_type] = ms_mol_df[ms_mol_df["mol_id"].isin(split_df["mol_id"])][
            "inchikey_s"
        ].unique()

    mol_df = pd.read_pickle(os.path.join(args.proc_dp, "mol_df.pkl"))
    mces_data = pd.read_pickle(os.path.join(args.proc_dp, "mces_pairs.pkl"))

    # Filter MCES pairs by distance range
    print(f"\nTotal MCES pairs before filtering: {len(mces_data)}")
    print(f"Filtering MCES pairs with {args.mce_min} <= mces <= {args.mce_max}")
    mces_data = mces_data[mces_data["mces"] >= args.mce_min]
    if args.mce_max != -1:
        mces_data = mces_data[mces_data["mces"] <= args.mce_max]
    print(f"MCES pairs after filtering: {len(mces_data)}")

    # Create MCES splits based on MS splits
    # Ensure molecules in MS train split are also in MCES train split
    split_d = {}

    # Filter mol_df by MS split assignments based on inchikey_s
    for split_type in ["train", "val", "test"]:
        # Get mol_ids from mol_df that have inchikey_s in the corresponding MS split
        split_mol_ids = mol_df[mol_df["inchikey_s"].isin(ms_split_d[split_type])].index.tolist()
        split_d[split_type] = split_mol_ids
        print(f"{split_type}: {len(split_mol_ids)} molecules")

    total_mols = mol_df["inchikey_s"].nunique()
    assert args.test_frac + args.val_frac < 1.0
    print(f"\nTotal unique molecules in mol_df: {total_mols}")
    expected_val_mol_count = int(total_mols * args.val_frac)
    print(f"Expected val molecules: {expected_val_mol_count}")
    expected_test_mol_count = int(total_mols * args.test_frac)
    print(f"Expected test molecules: {expected_test_mol_count}")
    expected_train_mol_count = total_mols - expected_val_mol_count - expected_test_mol_count
    print(f"Expected train molecules: {expected_train_mol_count}")

    expected_count_d = {
        "train": expected_train_mol_count,
        "val": expected_val_mol_count,
        "test": expected_test_mol_count,
    }

    # Get inchikey_s that are already assigned to splits
    assigned_inchikey_s = set()
    for split_type in ["train", "val", "test"]:
        assigned_inchikey_s.update(
            mol_df[mol_df.index.isin(split_d[split_type])]["inchikey_s"].unique()
        )

    # Available molecules are those not yet assigned to any split
    available_mol_inchikey_s = set(mol_df["inchikey_s"].unique()) - assigned_inchikey_s
    print(f"\nAvailable unassigned molecules: {len(available_mol_inchikey_s)}")

    # Randomly sample from available molecules to fill each split

    random.seed(args.meta_rseed)
    available_list = list(available_mol_inchikey_s)
    random.shuffle(available_list)

    for split_type in ["train", "val", "test"]:
        actual_count = len(split_d[split_type])
        expected_count = expected_count_d[split_type]
        print(f"\n{split_type} molecules: {actual_count} (expected {expected_count})")
        remaining_count = expected_count - actual_count

        if remaining_count > 0 and len(available_list) > 0:
            # Sample min(remaining_count, available) molecules
            n_to_sample = min(remaining_count, len(available_list))
            sampled_inchikey_s = available_list[:n_to_sample]
            available_list = available_list[n_to_sample:]  # Remove sampled from available

            # Get mol_ids corresponding to sampled inchikey_s
            additional_mol_ids = mol_df[
                mol_df["inchikey_s"].isin(sampled_inchikey_s)
            ].index.tolist()
            split_d[split_type].extend(additional_mol_ids)

            print(f"  Added {len(additional_mol_ids)} molecules from unassigned pool")
            print(f"  New total: {len(split_d[split_type])} molecules")
        elif remaining_count > 0:
            print(f"  Warning: Need {remaining_count} more molecules but none available!")
        else:
            print("  Split already has sufficient molecules")

    # Verify no overlap between test and train/val
    train_set = set(split_d["train"])
    val_set = set(split_d["val"])
    test_set = set(split_d["test"])

    print("\nChecking molecule overlaps:")
    print(f"Train ∩ Test: {len(train_set & test_set)} molecules (should be 0)")
    print(f"Val ∩ Test: {len(val_set & test_set)} molecules (should be 0)")
    print(f"Train ∩ Val: {len(train_set & val_set)} molecules")

    # Filter MCES pairs to ensure both molecules in a pair belong to the same split
    # Remove any pairs that cross splits
    mces_split_d = {}

    print("\nFiltering MCES pairs (removing cross-split pairs)...")
    print(f"Total MCES pairs before filtering: {len(mces_data)}")

    for split_type in ["train", "val", "test"]:
        split_set = set(split_d[split_type])

        # Keep only pairs where both mol_id_1 and mol_id_2 are in the same split
        split_pairs = mces_data[
            (mces_data["mol_id_1"].isin(split_set)) & (mces_data["mol_id_2"].isin(split_set))
        ].copy()

        mces_split_d[split_type] = split_pairs
        print(f"{split_type}: {len(split_pairs)} MCES pairs")

    # Verify no pair overlap between splits
    train_pair_indices = set(mces_split_d["train"].index)
    val_pair_indices = set(mces_split_d["val"].index)
    test_pair_indices = set(mces_split_d["test"].index)

    print("\nChecking MCES pair overlaps:")
    print(f"Train ∩ Test pairs: {len(train_pair_indices & test_pair_indices)} (should be 0)")
    print(f"Val ∩ Test pairs: {len(val_pair_indices & test_pair_indices)} (should be 0)")
    print(f"Train ∩ Val pairs: {len(train_pair_indices & val_pair_indices)}")

    total_pairs_after = sum(len(mces_split_d[s]) for s in ["train", "val", "test"])
    removed_pairs = len(mces_data) - total_pairs_after
    print(
        f"Removed {removed_pairs} cross-split pairs ({removed_pairs / len(mces_data) * 100:.2f}%)"
    )

    # Save splits
    os.makedirs(args.split_dp, exist_ok=True)

    # Save MCES pairs (with both molecule IDs)
    for split_type in ["train", "val", "test"]:
        split_fp = os.path.join(args.split_dp, f"{split_type}_ids.csv")
        split_df = mces_split_d[split_type][["mol_id_1", "mol_id_2", "mces"]].copy()
        split_df.reset_index(drop=False, inplace=True)
        split_df.rename(columns={"index": "mces_id"}, inplace=True)
        split_df.to_csv(split_fp, index=False)
        print(f"Saved {split_type} MCES pairs to {split_fp}")

    # Compute MCES statistics for each split
    print(f"\n{'=' * 70}")
    print("MCES Statistics for Each Split")
    print(f"{'=' * 70}")
    mces_stats = {}
    for split_type in ["train", "val", "test"]:
        mces_values = mces_split_d[split_type]["mces"]
        stats = {
            "count": int(len(mces_values)),
            "min": float(mces_values.min()),
            "max": float(mces_values.max()),
            "mean": float(mces_values.mean()),
            "std": float(mces_values.std()),
            "median": float(mces_values.median()),
            "q25": float(mces_values.quantile(0.25)),
            "q75": float(mces_values.quantile(0.75)),
        }
        mces_stats[split_type] = stats

        # Print detailed statistics for this split
        print(f"\n{split_type.upper()} Split:")
        print(f"  Count:    {stats['count']:,} pairs")
        print(f"  Min:      {stats['min']:.1f}")
        print(f"  Q25:      {stats['q25']:.1f}")
        print(f"  Median:   {stats['median']:.1f}")
        print(f"  Mean:     {stats['mean']:.2f}")
        print(f"  Q75:      {stats['q75']:.1f}")
        print(f"  Max:      {stats['max']:.1f}")
        print(f"  Std Dev:  {stats['std']:.2f}")

    print(f"{'=' * 70}\n")

    # Save split metadata
    metadata = {
        "split_type": args.split_type,
        "split_key": args.split_key,
        "ms_split_dp": args.ms_split_dp,
        "mce_min": args.mce_min,
        "mce_max": args.mce_max,
        "total_molecules": len(mol_df),
        "train_mol_size": len(split_d["train"]),
        "val_mol_size": len(split_d["val"]),
        "test_mol_size": len(split_d["test"]),
        "total_mces_pairs": len(mces_data),
        "train_pairs_size": len(mces_split_d["train"]),
        "val_pairs_size": len(mces_split_d["val"]),
        "test_pairs_size": len(mces_split_d["test"]),
        "removed_cross_split_pairs": removed_pairs,
        "cross_split_removal_rate": f"{removed_pairs / len(mces_data) * 100:.2f}%",
        "mces_statistics": mces_stats,
    }

    metadata_fp = os.path.join(args.split_dp, "split_metadata.json")
    with open(metadata_fp, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata to {metadata_fp}")

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split_type", type=str, choices=["random"], default="random")
    parser.add_argument("--split_key", type=str, choices=["inchikey_s"], default="inchikey_s")
    parser.add_argument("--elements", type=str, nargs="+", default=formula_utils.ELEMENTS)

    # non-filtering args
    parser.add_argument("--meta_rseed", type=int, default=42)
    parser.add_argument("--test_frac", type=float, default=0.1)
    parser.add_argument("--val_frac", type=float, default=0.1)
    parser.add_argument("--proc_dp", type=str, default="data/proc/mces")
    parser.add_argument("--split_dp", type=str, default="data/split/mces_nist23_ma_mi_inchikey_m15")
    parser.add_argument("--ms_proc_dp", type=str, default="data/proc/nist23")
    parser.add_argument("--ms_split_dp", type=str, default="data/split/nist23_ma_mi_inchikey")
    parser.add_argument(
        "--mce_min",
        type=int,
        default=0,
        help="minimum value of MCES pairs to process (default: 0s)",
    )
    parser.add_argument(
        "--mce_max",
        type=int,
        default=15,
        help="maximum value of MCES pairs to process (default: 10, -1 to process all pairs)",
    )
    args = parser.parse_args()

    make_splits(args)
