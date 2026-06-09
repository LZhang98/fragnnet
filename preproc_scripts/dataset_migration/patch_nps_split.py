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
    new_spec_df: pd.DataFrame,
    old_split_dp: str,
    output_split_dp: str,
    cv_dirs: list[str],
    extra_dsets: list[str],
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
    extra_df = new_spec_df[new_spec_df["dset"].isin(extra_dsets)]
    mol_to_extra: dict[int, list[dict]] = {}
    for _, row in extra_df.iterrows():
        mol_to_extra.setdefault(int(row["mol_id"]), []).append(
            {
                "mol_id": int(row["mol_id"]),
                "spec_id": int(row["spec_id"]),
                "group_id": int(row["group_id"]),
            }
        )

    for cv in cv_dirs:
        cv_in = os.path.join(old_split_dp, cv)
        cv_out = os.path.join(output_split_dp, cv)

        translated: dict[str, list[dict]] = {}
        mol_to_partition: dict[int, str] = {}

        for partition in ("train", "val", "test"):
            fp = os.path.join(cv_in, f"{partition}_ids.csv")
            if not os.path.exists(fp):
                logger.warning("  Missing %s — skipping", fp)
                translated[partition] = []
                continue

            old_split = load_split_csv(fp)
            rows = old_split[["mol_id", "spec_id", "group_id"]].to_dict("records")
            mol_to_partition.update({int(r["mol_id"]): partition for r in rows})
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
            if dry_run:
                logger.info("  [dry-run] would write %s (%d rows)", out_path, len(df))
            else:
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
        default="data/proc/nps_pcdl_spectraverse",
        help="New proc directory (spec_df.pkl with final spec_ids).",
    )
    parser.add_argument(
        "--old_split_dp",
        default="data/split/nps_nist23",
        help="Root directory of CV splits to patch.",
    )
    parser.add_argument(
        "--output_split_dp",
        default="data/split/nps_pcdl_spectraverse",
        help="Output directory for patched split files.",
    )
    parser.add_argument(
        "--extra_dsets",
        nargs="+",
        default=["pcdl", "spectraverse"],
        help="Dataset names to inject into splits (not present in old splits).",
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
    logger.info("Datasets to inject: %s", args.extra_dsets)

    logger.info("Patching splits → %s", args.output_split_dp)
    patch_splits(
        new_spec_df=spec_df,
        old_split_dp=args.old_split_dp,
        output_split_dp=args.output_split_dp,
        cv_dirs=cv_dirs,
        extra_dsets=args.extra_dsets,
        dry_run=args.dry_run,
    )

    logger.info("Done. Patched splits: %s", args.output_split_dp)


if __name__ == "__main__":
    main()
