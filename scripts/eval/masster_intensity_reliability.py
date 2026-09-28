"""Audit condition-matched NIST20 replicates before estimating intensity noise.

This first stage records source metadata and duplicate peak lists. A group without an
acquisition identifier is explicitly provenance-unknown, even when spectra differ.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SPLITS = ("train", "val", "test", "secondary")
CONDITION = (
    "mol_id", "prec_type", "inst_type", "frag_mode", "dset",
    "col_gas", "res", "ce_type", "ace", "nce",
)


def audit(spec_df: pd.DataFrame, split_frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Return row-level replicate membership and the auditable census summary."""
    if spec_df["spec_id"].duplicated().any():
        raise ValueError("Source spectrum table has duplicate spec_id values")

    membership = pd.concat(
        [frame.assign(split=name) for name, frame in split_frames.items()], ignore_index=True
    )
    if membership["spec_id"].duplicated().any():
        raise ValueError("Saved splits contain repeated spec_id values")
    source = spec_df.merge(
        membership[["spec_id", "mol_id", "split"]],
        on="spec_id",
        suffixes=("", "_split"),
        validate="one_to_one",
    )
    if len(source) != len(membership) or (source["mol_id"] != source["mol_id_split"]).any():
        raise ValueError("Saved split IDs or molecular keys disagree with the source table")
    primary = source[source["split"].isin(("train", "val", "test"))]
    if primary.groupby("mol_id")["split"].nunique().gt(1).any():
        raise ValueError("Molecules overlap across train, validation, and test splits")
    source = source.drop(columns="mol_id_split")

    # Exact recorded CE stays separate from values inferred by the model's CE conversion.
    source["ce_kind"] = "missing"
    source["ce_value"] = float("nan")
    for field in ("ace", "nce"):
        mask = source["ce_kind"].eq("missing") & source[field].notna()
        source.loc[mask, "ce_kind"] = field
        source.loc[mask, "ce_value"] = source.loc[mask, field]

    source["peak_hash"] = source["peaks"].map(
        lambda peaks: hashlib.sha256(repr(peaks).encode("utf-8")).hexdigest()
    )
    group_fields = ["split", *CONDITION]
    source["group_id_audit"] = source.groupby(group_fields, dropna=False, sort=True).ngroup()
    source["repeat_count"] = source.groupby("group_id_audit")["spec_id"].transform("size")
    source["identical_copy"] = source.duplicated(
        ["group_id_audit", "peak_hash"], keep=False
    )
    source["duplicate_record"] = source.duplicated(
        ["dset", "dset_spec_id"], keep=False
    )
    source["provenance_tier"] = "unknown_acquisition"
    source.loc[source["duplicate_record"], "provenance_tier"] = "duplicate_record"
    source.loc[source["identical_copy"], "provenance_tier"] = "identical_peak_list"
    source.loc[source["ce_kind"].eq("missing"), "provenance_tier"] = "missing_recorded_ce"

    groups = source.drop_duplicates("group_id_audit")
    repeated = groups[groups["repeat_count"].ge(2)]
    summary = {
        "schema_version": 1,
        "source_spectra": len(spec_df),
        "split_spectra": len(source),
        "secondary_overlap_molecules": int(
            len(set(source.loc[source["split"].eq("secondary"), "mol_id"])
                & set(primary["mol_id"]))
        ),
        "metadata_columns": spec_df.columns.tolist(),
        "missing_counts": {
            field: int(spec_df[field].isna().sum())
            for field in ("dset_spec_id", "inst_type", "frag_mode", "ace", "nce", "ce_type")
        },
        "replicate_groups": int(len(repeated)),
        "repeat_bands": {
            "2": int(repeated["repeat_count"].eq(2).sum()),
            "3-4": int(repeated["repeat_count"].between(3, 4).sum()),
            "5+": int(repeated["repeat_count"].ge(5).sum()),
        },
        "by_split": {
            name: {
                "spectra": int(len(rows)),
                "molecules": int(rows["mol_id"].nunique()),
                "replicate_groups": int(rows.loc[rows["repeat_count"].ge(2), "group_id_audit"].nunique()),
                "repeat_molecules": int(rows.loc[rows["repeat_count"].ge(2), "mol_id"].nunique()),
            }
            for name, rows in source.groupby("split")
        },
        "identical_copy_rows": int(source["identical_copy"].sum()),
        "duplicate_record_rows": int(source["duplicate_record"].sum()),
        "strict_independent_groups": 0,
        "gate_g1": "stop_observation_noise_head_no_acquisition_identifier",
    }
    return source, summary


def main() -> None:
    """Write the P1 census without modifying the saved source or split tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--spec-fp", type=Path, default=Path("data/proc/nist20/spec_df.pkl"))
    parser.add_argument("--split-dir", type=Path, default=Path("data/split/nist20_inchikey"))
    parser.add_argument(
        "--out-dir", type=Path, default=Path("output/masster_intensity_reliability_j4u25qfd_e050")
    )
    args = parser.parse_args()
    frames = {name: pd.read_csv(args.split_dir / f"{name}_ids.csv") for name in SPLITS}
    rows, summary = audit(pd.read_pickle(args.spec_fp), frames)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows.drop(columns="peaks").to_csv(args.out_dir / "replicate_groups.csv", index=False)
    (args.out_dir / "census_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
