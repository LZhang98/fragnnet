"""Export worst predictions from an inference HDF5 file to a TSV.

Selects the worst N spectra by a score column, applies a per-spectrum intensity
filter, and writes a TSV with all metadata and filtered predicted peaks.

Usage:
  python scripts/plotting/export_worst_predictions.py \
      --pred_fp output/inference_eval/4ews8j6f/predictions_test.hdf5 \
      --spec_df_fp data/proc/nist20/spec_df.pkl \
      --n 200 \
      --score_col cos_hun_inst_tol \
      --int_threshold 0.01
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_SCALAR_KEYS = [
    "spec_id",
    "mol_id",
    "cos_sim_0.01",
    "cos_hun_inst_tol",
    "cos_hun_inst_tol_sqrt",
    "jss_hun_inst_tol",
    "jss_hun_inst_tol_sqrt",
    "cos_sim_sqrt_0.01",
    "pred_oos_prob",
    "start_points",
    "end_points",
    "index_rebased",
]

_SPEC_META_COLS = [
    "name",
    "compound_name",
    "mol_name",
    "formula",
    "mol_formula",
    "smiles",
    "mol_smiles",
    "inst_type",
    "prec_type",
    "nce",
    "ace",
    "dset_name",
    "dataset_name",
]


def load_hdf5_meta(pred_fp: str) -> pd.DataFrame:
    """Load per-spectrum scalar metadata from the prediction HDF5.

    Args:
        pred_fp: Path to the predictions HDF5 file.

    Returns:
        DataFrame with one row per spectrum.
    """
    with h5py.File(pred_fp, "r") as f:
        available = [k for k in _SCALAR_KEYS if k in f]
        data = {k: f[k][:] for k in available}
    return pd.DataFrame(data)


def _peaks_to_str(mzs: np.ndarray, ints: np.ndarray) -> str:
    return " ".join(f"{m:.4f}:{i:.2f}" for m, i in zip(mzs.tolist(), ints.tolist()))


def _gt_peaks_to_str(peaks: object) -> str:
    if peaks is None or (isinstance(peaks, float) and np.isnan(peaks)):
        return ""
    if isinstance(peaks, (list, np.ndarray)):
        arr = np.asarray(peaks)
        if arr.ndim == 2 and arr.shape[1] >= 2:
            return " ".join(f"{r[0]:.4f}:{r[1]:.2f}" for r in arr.tolist())
        return ""
    if isinstance(peaks, str):
        return peaks.replace("\n", " ")
    return ""


def export_worst(
    pred_fp: str,
    out_fp: str,
    n: int,
    score_col: str,
    int_threshold: float,
    spec_df: pd.DataFrame | None,
    mol_df: pd.DataFrame | None,
) -> None:
    """Select worst predictions and export to TSV.

    Args:
        pred_fp: Path to predictions HDF5.
        out_fp: Output TSV path.
        n: Number of worst spectra to export.
        score_col: Score column to rank by (ascending = worst first).
        int_threshold: Per-spectrum intensity filter as a fraction of the spectrum
            max intensity. Peaks below ``int_threshold * max(ints)`` are dropped.
            E.g. 0.01 keeps only peaks >= 1% of the spectrum's tallest peak.
        spec_df: Optional ground-truth spec_df for metadata and GT peaks.
        mol_df: Optional mol_df with [mol_id, smiles] for SMILES when not in spec_df.
    """
    meta = load_hdf5_meta(pred_fp)
    logger.info("Loaded %d spectra from %s", len(meta), pred_fp)

    if score_col not in meta.columns:
        raise ValueError(f"Score column '{score_col}' not found. Available: {list(meta.columns)}")

    worst = meta.dropna(subset=[score_col]).nsmallest(n, score_col).reset_index(drop=True)
    logger.info("Selected %d worst spectra by %s", len(worst), score_col)

    starts = worst["start_points"].to_numpy().astype(int)
    ends = worst["end_points"].to_numpy().astype(int)

    with h5py.File(pred_fp, "r") as f:
        pred_mzs_all = f["pred_mzs"][:]
        pred_ints_all = f["pred_ints"][:]

    peaks_list: list[str] = []
    n_peaks_raw: list[int] = []
    n_peaks_filt: list[int] = []

    for s, e in zip(starts, ends):
        pmzs = pred_mzs_all[s:e]
        pints = pred_ints_all[s:e]
        n_peaks_raw.append(len(pmzs))

        if len(pints) > 0 and int_threshold > 0.0:
            thresh_abs = float(np.max(pints)) * int_threshold
            keep = pints >= thresh_abs
            pmzs, pints = pmzs[keep], pints[keep]

        n_peaks_filt.append(len(pmzs))
        peaks_list.append(_peaks_to_str(pmzs, pints))

    worst["n_peaks_raw"] = n_peaks_raw
    worst["n_peaks_filtered"] = n_peaks_filt
    worst["peaks_pred"] = peaks_list

    # Drop internal indexing columns
    worst = worst.drop(columns=["start_points", "end_points"], errors="ignore")

    if spec_df is not None and "spec_id" in spec_df.columns:
        meta_cols = [c for c in _SPEC_META_COLS if c in spec_df.columns]
        worst = worst.merge(
            spec_df[["spec_id"] + meta_cols],
            on="spec_id",
            how="left",
        )

        if "peaks" in spec_df.columns:
            sid_set = set(worst["spec_id"].astype(int).tolist())
            gt_lookup: dict[int, str] = {
                int(row["spec_id"]): _gt_peaks_to_str(row["peaks"])
                for _, row in spec_df[spec_df["spec_id"].isin(sid_set)].iterrows()
            }
            worst["peaks_gt"] = [gt_lookup.get(int(sid), "") for sid in worst["spec_id"]]

    if mol_df is not None and "mol_id" in mol_df.columns:
        mol_cols = [c for c in mol_df.columns if c != "mol_id"]
        worst = worst.merge(
            mol_df[["mol_id"] + mol_cols], on="mol_id", how="left", suffixes=("", "_mol")
        )
        # For conflicts with spec_df columns: keep spec_df value, fill NaN with mol_df value
        for c in mol_cols:
            mol_c = f"{c}_mol"
            if mol_c in worst.columns:
                worst[c] = worst[c].fillna(worst[mol_c])
                worst = worst.drop(columns=[mol_c])

    Path(out_fp).parent.mkdir(parents=True, exist_ok=True)
    worst.to_csv(out_fp, sep="\t", index=False)
    print(f"Saved {len(worst)} rows → {out_fp}")


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export worst predictions to TSV")
    parser.add_argument("--pred_fp", required=True, help="Path to predictions HDF5")
    parser.add_argument("--spec_df_fp", default=None, help="Path to ground-truth spec_df.pkl")
    parser.add_argument(
        "--mol_df_fp", default=None, help="Path to mol_df.pkl with [mol_id, smiles]"
    )
    parser.add_argument(
        "--out_fp",
        default=None,
        help="Output TSV path (default: <pred_fp_dir>/worst_predictions.tsv)",
    )
    parser.add_argument("--n", type=int, default=100, help="Number of worst spectra to export")
    parser.add_argument(
        "--score_col",
        default="cos_hun_inst_tol",
        help="Score column to rank by (default: cos_hun_inst_tol)",
    )
    parser.add_argument(
        "--int_threshold",
        type=float,
        default=0.01,
        help=(
            "Per-spectrum intensity filter: keep peaks >= this fraction of the spectrum's "
            "max intensity (default: 0.01 = 1%%)"
        ),
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()

    out_fp = args.out_fp or str(Path(args.pred_fp).parent / "worst_predictions.tsv")

    spec_df = None
    if args.spec_df_fp is not None:
        logger.info("Loading spec_df from %s", args.spec_df_fp)
        spec_df = pd.read_pickle(args.spec_df_fp)

    mol_df = None
    if args.mol_df_fp is not None:
        logger.info("Loading mol_df from %s", args.mol_df_fp)
        mol_df = pd.read_pickle(args.mol_df_fp)

    export_worst(
        pred_fp=args.pred_fp,
        out_fp=out_fp,
        n=args.n,
        score_col=args.score_col,
        int_threshold=args.int_threshold,
        spec_df=spec_df,
        mol_df=mol_df,
    )


if __name__ == "__main__":
    main()
