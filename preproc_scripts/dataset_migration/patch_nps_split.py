"""Patch nps_nist23_pcdl_spectraverse proc and splits.

Applies two cleaning steps to the merged NPS dataset:
1. Remove specified source datasets entirely (default: mona23).
2. Deduplicate by (mol_id, prec_type, inst_type, ace), keeping only the
   highest-priority source per group. Same-priority sources are kept together;
   a lower-priority entry is dropped only when a higher-priority duplicate exists.
   Default priority: pcdl > nist23_hr1 = nist23_hr2 > spectraverse.

Outputs:
  <output_proc_dp>/spec_df.pkl  — patched spec_df with reassigned spec_ids
  <output_proc_dp>/mol_df.pkl   — copied unchanged (mol_ids are stable)
  <output_split_dp>/cvN/{train,val,test}_ids.csv  — patched CV split files

Split patching preserves existing molecule-to-fold assignments from old_split_dp:
  a) Old nist spec_ids that survived the patch are translated to new spec_ids.
  b) New pcdl/spectraverse entries are injected into whichever partition their
     molecule already occupies; molecules not in the old splits go to train.

Usage:
    python preproc_scripts/dataset_migration/patch_nps_split.py \\
        --proc_dp data/proc/nps_nist23_pcdl_spectraverse \\
        --old_proc_dp data/proc/nps_nist23 \\
        --old_split_dp data/split/nps_nist23 \\
        --output_proc_dp data/proc/nps_pcdl_spectraverse \\
        --output_split_dp data/split/nps_pcdl_spectraverse
"""

import argparse
import logging
import os
import shutil

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

DEDUP_KEY = ["mol_id", "prec_type", "inst_type", "ace"]


def remove_datasets(spec_df: pd.DataFrame, remove_dsets: list[str]) -> pd.DataFrame:
    """Remove all rows belonging to the specified source datasets.

    Args:
        spec_df: Spectrum dataframe with a 'dset' column.
        remove_dsets: Dataset names to drop (e.g. ['mona23']).

    Returns:
        Filtered dataframe with the listed datasets removed.
    """
    before = len(spec_df)
    spec_df = spec_df[~spec_df["dset"].isin(remove_dsets)].copy()
    removed = before - len(spec_df)
    logger.info("  Removed %d spectra from datasets: %s", removed, remove_dsets)
    for dset in remove_dsets:
        logger.info("    removed dset=%s", dset)
    return spec_df


def dedup_by_priority(
    spec_df: pd.DataFrame, priority: list[str]
) -> pd.DataFrame:
    """Drop lower-priority duplicates sharing the same (mol_id, prec_type, inst_type, ace).

    Within each dedup group, only rows whose source dataset priority equals the
    group minimum are kept.  Same-priority sources (e.g. nist23_hr1 and nist23_hr2)
    are kept together; a lower-priority entry is dropped only when a strictly
    better source exists in the same group.

    Args:
        spec_df: Spectrum dataframe with 'dset' and DEDUP_KEY columns.
        priority: Ordered list of dataset names, best first
            (e.g. ['pcdl', 'nist23_hr1', 'nist23_hr2', 'spectraverse']).

    Returns:
        Deduplicated dataframe.
    """
    priority_map = {dset: rank for rank, dset in enumerate(priority)}
    unknown_rank = len(priority)
    spec_df = spec_df.copy()
    spec_df["_prio"] = spec_df["dset"].map(priority_map).fillna(unknown_rank).astype(int)
    # Compute per-group minimum priority
    min_prio = spec_df.groupby(DEDUP_KEY, dropna=False)["_prio"].transform("min")
    before = len(spec_df)
    spec_df = spec_df[spec_df["_prio"] == min_prio].drop(columns=["_prio"])
    dropped = before - len(spec_df)
    logger.info("  Dedup dropped %d lower-priority duplicate spectra", dropped)
    return spec_df


def load_split_csv(path: str) -> pd.DataFrame:
    """Load a split CSV, dropping the spurious index column if present.

    Args:
        path: Path to the split CSV file.

    Returns:
        DataFrame with columns mol_id, spec_id, group_id.
    """
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df


def write_split_csv(df: pd.DataFrame, path: str) -> None:
    """Write a split dataframe to CSV without an index column.

    Args:
        df: Split dataframe with mol_id, spec_id, group_id columns.
        path: Output file path.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df[["mol_id", "spec_id", "group_id"]].to_csv(path, index=False)


def patch_splits(
    new_spec_df: pd.DataFrame,
    old_spec_df: pd.DataFrame,
    old_split_dp: str,
    output_split_dp: str,
    cv_dirs: list[str],
    extra_dsets: list[str],
) -> None:
    """Patch CV split files to reflect the cleaned spec_df.

    For each CV fold:
      a) Translates surviving old nist spec_ids to new spec_ids via
         (dset, dset_spec_id) cross-reference.
      b) Injects pcdl/spectraverse entries into whichever partition already
         contains the corresponding molecule; truly new molecules go to train.

    Args:
        new_spec_df: Patched spectrum dataframe with reassigned spec_ids.
        old_spec_df: Original nps_nist23 spec_df used for spec_id cross-reference.
        old_split_dp: Root directory of the existing CV splits.
        output_split_dp: Root directory for output splits.
        cv_dirs: Sorted list of CV subdirectory names (e.g. ['cv0', ..., 'cv9']).
        extra_dsets: Dataset names to inject (e.g. ['pcdl', 'spectraverse']).
    """
    # Build lookup: (dset, dset_spec_id) → row in new_spec_df
    # Only for datasets that appear in old splits (nist23_hr1, nist23_hr2)
    shared_in_new = new_spec_df[~new_spec_df["dset"].isin(extra_dsets)]
    new_key_index = shared_in_new.set_index(["dset", "dset_spec_id"])

    # Build lookup: old spec_id → (dset, dset_spec_id)
    old_id_to_key = dict(
        zip(old_spec_df["spec_id"], zip(old_spec_df["dset"], old_spec_df["dset_spec_id"]))
    )

    # Extra entries to inject per mol_id
    extra_df = new_spec_df[new_spec_df["dset"].isin(extra_dsets)]
    mol_to_extra: dict[int, list[dict]] = {}
    for _, row in extra_df.iterrows():
        mol_to_extra.setdefault(int(row["mol_id"]), []).append(
            {"mol_id": int(row["mol_id"]), "spec_id": int(row["spec_id"]), "group_id": int(row["group_id"])}
        )

    for cv in cv_dirs:
        cv_in = os.path.join(old_split_dp, cv)
        cv_out = os.path.join(output_split_dp, cv)

        # Step a: translate old spec_ids to new for all three partitions first
        translated: dict[str, list[dict]] = {}
        mol_to_partition: dict[int, str] = {}

        for partition in ("train", "val", "test"):
            fp = os.path.join(cv_in, f"{partition}_ids.csv")
            if not os.path.exists(fp):
                logger.warning("  Missing %s — skipping", fp)
                translated[partition] = []
                continue

            old_split = load_split_csv(fp)
            rows = []
            for old_sid in old_split["spec_id"]:
                key = old_id_to_key.get(old_sid)
                if key is None:
                    continue
                if key not in new_key_index.index:
                    continue  # removed by mona23 filter or dedup
                new_row = new_key_index.loc[key]
                mol_id = int(new_row["mol_id"])
                rows.append({
                    "mol_id": mol_id,
                    "spec_id": int(new_row["spec_id"]),
                    "group_id": int(new_row["group_id"]),
                })
                mol_to_partition[mol_id] = partition

            translated[partition] = rows

        # Step b: inject extra entries (pcdl/spectraverse)
        new_mol_ids = set(mol_to_extra.keys())
        assigned_mol_ids: set[int] = set()
        for partition in ("train", "val", "test"):
            partition_mols = {r["mol_id"] for r in translated[partition]}
            for mol_id in new_mol_ids & partition_mols:
                translated[partition].extend(mol_to_extra[mol_id])
                assigned_mol_ids.add(mol_id)

        # Molecules new to all partitions → train
        unassigned = new_mol_ids - assigned_mol_ids
        for mol_id in unassigned:
            translated["train"].extend(mol_to_extra[mol_id])

        # Write patched splits
        for partition in ("train", "val", "test"):
            out_path = os.path.join(cv_out, f"{partition}_ids.csv")
            df = pd.DataFrame(translated[partition], columns=["mol_id", "spec_id", "group_id"])
            write_split_csv(df, out_path)

        logger.info(
            "  %s: train=%d  val=%d  test=%d  (%d new-mol → train)",
            cv,
            len(translated["train"]),
            len(translated["val"]),
            len(translated["test"]),
            len(unassigned),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proc_dp",
        default="data/proc/nps_nist23_pcdl_spectraverse",
        help="Source merged proc directory (spec_df.pkl, mol_df.pkl).",
    )
    parser.add_argument(
        "--old_proc_dp",
        default="data/proc/nps_nist23",
        help="Old proc directory used only to cross-reference old spec_ids.",
    )
    parser.add_argument(
        "--old_split_dp",
        default="data/split/nps_nist23",
        help="Root directory of CV splits to patch.",
    )
    parser.add_argument(
        "--output_proc_dp",
        default="data/proc/nps_pcdl_spectraverse",
        help="Output directory for patched proc files.",
    )
    parser.add_argument(
        "--output_split_dp",
        default="data/split/nps_pcdl_spectraverse",
        help="Output directory for patched split files.",
    )
    parser.add_argument(
        "--remove_dsets",
        nargs="+",
        default=["mona23"],
        help="Dataset names to remove entirely.",
    )
    parser.add_argument(
        "--priority",
        nargs="+",
        default=["pcdl", "nist23_hr1", "nist23_hr2", "spectraverse"],
        help="Dedup priority: dataset names ordered best→worst.",
    )
    parser.add_argument(
        "--cv_pattern",
        default="cv",
        help="Prefix for CV fold subdirectory names.",
    )
    args = parser.parse_args()

    # ------------------------------------------------------------------ #
    # Patch spec_df                                                        #
    # ------------------------------------------------------------------ #
    logger.info("Loading spec_df from %s", args.proc_dp)
    spec_df = pd.read_pickle(os.path.join(args.proc_dp, "spec_df.pkl"))
    logger.info("  Loaded %d spectra. dset distribution:", len(spec_df))
    for dset, cnt in spec_df["dset"].value_counts().items():
        logger.info("    %-20s %d", dset, cnt)

    logger.info("Step 1: removing datasets %s", args.remove_dsets)
    spec_df = remove_datasets(spec_df, args.remove_dsets)

    logger.info("Step 2: deduplicating by priority %s", args.priority)
    spec_df = dedup_by_priority(spec_df, args.priority)

    logger.info("After cleaning: %d spectra remain. dset distribution:", len(spec_df))
    for dset, cnt in spec_df["dset"].value_counts().items():
        logger.info("    %-20s %d", dset, cnt)

    logger.info("Step 3: reassigning spec_ids and saving to %s", args.output_proc_dp)
    spec_df = spec_df.reset_index(drop=True)
    spec_df["spec_id"] = np.arange(len(spec_df))
    os.makedirs(args.output_proc_dp, exist_ok=True)
    spec_df.to_pickle(os.path.join(args.output_proc_dp, "spec_df.pkl"))
    shutil.copy(
        os.path.join(args.proc_dp, "mol_df.pkl"),
        os.path.join(args.output_proc_dp, "mol_df.pkl"),
    )
    logger.info("  spec_df saved (%d rows). mol_df copied.", len(spec_df))

    # ------------------------------------------------------------------ #
    # Patch splits                                                         #
    # ------------------------------------------------------------------ #
    logger.info("Loading old spec_df from %s for cross-reference", args.old_proc_dp)
    old_spec_df = pd.read_pickle(os.path.join(args.old_proc_dp, "spec_df.pkl"))

    cv_dirs = sorted(
        d
        for d in os.listdir(args.old_split_dp)
        if d.startswith(args.cv_pattern)
        and os.path.isdir(os.path.join(args.old_split_dp, d))
    )
    if not cv_dirs:
        raise FileNotFoundError(
            f"No CV directories matching '{args.cv_pattern}*' in {args.old_split_dp}"
        )
    logger.info("Found %d CV folds: %s", len(cv_dirs), cv_dirs)

    # Datasets whose entries don't appear in old splits and need to be injected
    old_dsets = set(old_spec_df["dset"].unique())
    extra_dsets = [d for d in spec_df["dset"].unique() if d not in old_dsets]
    logger.info("Datasets to inject into splits: %s", extra_dsets)

    logger.info("Step 4: patching splits → %s", args.output_split_dp)
    patch_splits(
        new_spec_df=spec_df,
        old_spec_df=old_spec_df,
        old_split_dp=args.old_split_dp,
        output_split_dp=args.output_split_dp,
        cv_dirs=cv_dirs,
        extra_dsets=extra_dsets,
    )

    logger.info("Done. Patched proc: %s  Patched splits: %s", args.output_proc_dp, args.output_split_dp)


if __name__ == "__main__":
    main()
