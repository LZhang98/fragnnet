"""Audit imported Enveda RT/CCS labels before enabling the MASSter auxiliary heads."""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def audit_labels(csv_path: str) -> dict:
    """Count usable labels and within-source-molecule/adduct label ranges.

    Only scalar columns are loaded. Repeated-label ranges group source SMILES and
    adduct, not standardized molecules; they describe acquisition variation, not
    a model error floor. No train/validation/test labels fit model parameters.
    """
    frame = pd.read_csv(
        csv_path,
        usecols=[
            "dset_spec_id",
            "smiles",
            "prec_type",
            "rtinseconds",
            "ccs",
            "sample_inlet",
            "column_type",
        ],
    )
    result = {
        "input": str(Path(csv_path).resolve()),
        "rows": len(frame),
        "unique_source_ids": int(frame["dset_spec_id"].nunique()),
        "methods": frame[["sample_inlet", "column_type"]].drop_duplicates().to_dict("records"),
        "targets": {},
    }
    for column in ("rtinseconds", "ccs"):
        values = pd.to_numeric(frame[column], errors="coerce")
        valid = np.isfinite(values) & (values > 0)
        frame[column] = values.where(valid)
        groups = frame.groupby(["smiles", "prec_type"])[column].agg(["count", "min", "max"])
        repeated = groups[groups["count"] > 1]
        ranges = repeated["max"] - repeated["min"]
        result["targets"][column] = {
            "valid": int(valid.sum()),
            "invalid": int((~valid).sum()),
            "valid_quantiles": values[valid].quantile([0, 0.01, 0.5, 0.99, 1]).to_dict(),
            "repeated_source_smiles_adduct_groups": len(repeated),
            "within_group_range_quantiles": ranges.quantile([0.5, 0.9, 0.99, 1]).to_dict(),
        }
    return result


def main() -> None:
    """Write a small label census; refuse to overwrite an existing artifact."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/df/enveda_180_df.csv")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    result = audit_labels(args.csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    logger.info("Audited %d rows; wrote %s", result["rows"], output)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
