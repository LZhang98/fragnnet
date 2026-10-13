"""Read Enveda scalar labels without rebuilding the existing spectrum artifacts."""

import numpy as np
import pandas as pd


def join_rt_ccs_labels(spec_df: pd.DataFrame, labels_fp: str) -> pd.DataFrame:
    """Join the imported Enveda CSV by source spectrum identity, never row position.

    This sidecar is restricted to a single source dataset because its IDs are
    source-local. Every processed row must match; missing/nonpositive target
    values within matched rows remain NaN. Original row order/index are preserved.
    """
    if spec_df["dset"].nunique(dropna=False) != 1:
        raise ValueError("RT/CCS CSV labels require a single source dataset")
    columns = ["dset_spec_id", "rtinseconds", "ccs"]
    labels = pd.read_csv(labels_fp, usecols=columns, dtype={"dset_spec_id": str})
    if labels["dset_spec_id"].isna().any() or labels["dset_spec_id"].duplicated().any():
        raise ValueError("RT/CCS labels must have unique nonmissing dset_spec_id values")
    labels = labels.set_index("dset_spec_id")
    source_ids = spec_df["dset_spec_id"].astype(str)
    if not source_ids.isin(labels.index).all():
        raise ValueError("RT/CCS label CSV does not cover every processed spectrum")
    result = spec_df.copy()
    for column in columns[1:]:
        values = pd.to_numeric(labels[column], errors="coerce")
        values = values.where(np.isfinite(values) & (values > 0))
        result[column] = source_ids.map(values)
    return result


def join_property_labels(spec_df: pd.DataFrame, labels_fp: str, name: str) -> pd.DataFrame:
    """Join one independent RT or CCS label set by source spectrum ID.

    A label set needs only its own target column and may cover a subset of the
    spectra. Unmatched, nonfinite and nonpositive labels become NaN, so one
    property's coverage never filters the other property's training examples.
    """
    column = {"rt": "rtinseconds", "ccs": "ccs"}[name]
    if spec_df["dset"].nunique(dropna=False) != 1:
        raise ValueError("Property CSV labels require a single source dataset")
    labels = pd.read_csv(labels_fp, usecols=["dset_spec_id", column], dtype={"dset_spec_id": str})
    if labels["dset_spec_id"].isna().any() or labels["dset_spec_id"].duplicated().any():
        raise ValueError("Property labels must have unique nonmissing dset_spec_id values")
    values = pd.to_numeric(labels.set_index("dset_spec_id")[column], errors="coerce")
    values = values.where(np.isfinite(values) & (values > 0))
    result = spec_df.copy()
    result[column] = spec_df["dset_spec_id"].astype(str).map(values)
    return result
