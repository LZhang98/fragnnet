"""Patch CV split files to inject new datasets into existing splits.

The proc directory is unchanged; only the splits are updated.
Existing split rows are preserved as-is. New entries from extra_dsets
are injected into whichever partition already contains the molecule;
molecules not yet in any split go to train.

Outputs:
  <output_split_dp>/cvN/{train,val,test}_ids.csv  — patched CV split files

Usage:
    # Dry run (no files written):
    python preproc_scripts/dataset_migration/patch_nps_split.py \\
        --proc_dp data/proc/nps_pcdl_spectraverse \\
        --old_split_dp data/split/nps_nist23 \\
        --output_split_dp data/split/nps_pcdl_spectraverse \\
        --extra_dsets pcdl spectraverse \\
        --dry-run

    # Full run:
    python preproc_scripts/dataset_migration/patch_nps_split.py \\
        --proc_dp data/proc/nps_pcdl_spectraverse \\
        --old_split_dp data/split/nps_nist23 \\
        --output_split_dp data/split/nps_pcdl_spectraverse \\
        --extra_dsets pcdl spectraverse
"""

import argparse
import logging
import os

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


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
    spec_df: pd.DataFrame,
    old_split_dp: str,
    output_split_dp: str,
    cv_dirs: list[str],
    dsets_order: list[str],
    dry_run: bool = False,
) -> None:
    """Inject new dataset entries into existing CV split files.

    Existing rows are preserved unchanged. Entries from extra_dsets are
    injected into whichever partition already contains the molecule; truly
    new molecules go to train.

    Args:
        new_spec_df: Full spec_df from the (unchanged) proc directory.
        old_split_dp: Root directory of the existing CV splits.
        output_split_dp: Root directory for output splits.
        cv_dirs: Sorted list of CV subdirectory names (e.g. ['cv0', ..., 'cv9']).
        extra_dsets: Dataset names to inject (e.g. ['pcdl', 'spectraverse']).
        dry_run: If True, log intended writes without writing any files.
    """
    # filter by dset first
    logger.info(f"  Dsets: {spec_df['dset'].unique()}")
    spec_df = spec_df[spec_df["dset"].isin(dsets_order)]
    logger.info(f"  After filtering by dset,  {len(spec_df)} spectra remain.")
    logger.info(f"  After filtering, dsets: {spec_df['dset'].unique()}")
    for cv in cv_dirs:
        cv_in = os.path.join(old_split_dp, cv)
        cv_stats = {
            "train": {"mol": 0, "spec": 0, "mol_new": 0, "spec_new": 0},
            "val": {"mol": 0, "spec": 0, "mol_new": 0, "spec_new": 0},
            "test": {"mol": 0, "spec": 0, "mol_new": 0, "spec_new": 0},
        }
        for split in ["train", "val", "test"]:
            # cv_out = os.path.join(output_split_dp, cv)
            cv_split_df = pd.read_csv(os.path.join(cv_in, f"{split}_ids.csv"))
            cv_stats[split]["mol"] = len(cv_split_df["mol_id"].unique())
            cv_stats[split]["spec"] = len(cv_split_df["spec_id"].unique())

            # remove any rows with spec_ids not in the new spec_df (e.g. from old dsets)
            cv_split_df = cv_split_df[cv_split_df["spec_id"].isin(spec_df["spec_id"])]
            cv_stats[split]["mol_new"] = len(cv_split_df["mol_id"].unique())
            cv_stats[split]["spec_new"] = len(cv_split_df["spec_id"].unique())

        logger.info(
            f"  {cv}: train:{cv_stats['train']['mol']} ->{cv_stats['train']['mol_new']} val:{cv_stats['val']['mol']}->{cv_stats['val']['mol_new']}  test:{cv_stats['test']['mol']}->{cv_stats['test']['mol_new']}  ",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--proc_dp",
        default="data/proc/nist23_pcdl_spectraverse",
        help="New proc directory (spec_df.pkl with final spec_ids).",
    )
    parser.add_argument(
        "--old_split_dp",
        default="data/split/ccx_nist23_pcdl_spectraverse",
        help="Root directory of CV splits to patch.",
    )
    parser.add_argument(
        "--output_split_dp",
        default="data/split/ccx_nist23_pcdl_spectraverse",
        help="Output directory for patched split files.",
    )
    parser.add_argument(
        "--dsets_order",
        nargs="+",
        default=["pcdl", "nist23_hr1", "nist23_hr2", "spectraverse"],
        help="Order of datasets to inject into splits (not present in old splits).",
    )
    parser.add_argument(
        "--cv_pattern",
        default="cv",
        help="Prefix for CV fold subdirectory names.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Log what would be written without writing any files.",
    )
    args = parser.parse_args()

    logger.info("Loading spec_df from %s", args.proc_dp)
    spec_df = pd.read_pickle(os.path.join(args.proc_dp, "spec_df.pkl"))
    logger.info("  Loaded %d spectra.", len(spec_df))

    cv_dirs = sorted(
        d
        for d in os.listdir(args.old_split_dp)
        if d.startswith(args.cv_pattern) and os.path.isdir(os.path.join(args.old_split_dp, d))
    )
    if not cv_dirs:
        raise FileNotFoundError(
            f"No CV directories matching '{args.cv_pattern}*' in {args.old_split_dp}"
        )
    logger.info("Found %d CV folds: %s", len(cv_dirs), cv_dirs)
    logger.info("Datasets to inject: %s", args.dsets_order)

    logger.info("Patching splits → %s", args.output_split_dp)
    patch_splits(
        spec_df=spec_df,
        old_split_dp=args.old_split_dp,
        output_split_dp=args.output_split_dp,
        cv_dirs=cv_dirs,
        dsets_order=args.dsets_order,
        dry_run=args.dry_run,
    )

    logger.info("Done. Patched splits: %s", args.output_split_dp)


if __name__ == "__main__":
    main()
