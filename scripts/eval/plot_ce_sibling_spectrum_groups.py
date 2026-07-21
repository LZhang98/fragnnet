"""Plot CE-series spectra for groups with suspicious low oracle scores.

Each PDF page is one ``(mol_id, adduct, instrument, frag_mode)`` group. Spectra are sorted by
collision energy and suspicious low-oracle spectra are labeled in red. The intended use is to
inspect spectra where the oracle score is low even though another CE sibling in the same group is
normal, which is likely a bad acquisition/annotation rather than a DAG coverage gap.

Usage:
    python scripts/eval/plot_ce_sibling_spectrum_groups.py \\
        --out_dp data/eval/oracle_cos_nist20_default
"""

from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from oracle_dataset_utils import resolve_dsets_for_spec_fp
from rdkit import Chem
from rdkit.Chem import Draw

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


GROUP_COLS = ["mol_id", "prec_type", "inst_type", "frag_mode"]


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--flags_csv",
        default="data/eval/oracle_cos_nist20_default/oracle_ce_sibling_trend_flags.csv",
    )
    parser.add_argument(
        "--oracle_csv",
        default="data/eval/oracle_cos_nist20_default/oracle_cos_default.csv",
    )
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--out_dp", required=True)
    parser.add_argument("--metric", default="cos_hun_inst_tol_opt")
    parser.add_argument(
        "--max_groups",
        type=int,
        default=None,
        help="Limit number of groups plotted after sorting by worst flagged metric.",
    )
    parser.add_argument(
        "--max_spectra_per_group",
        type=int,
        default=20,
        help="For large CE groups, keep flagged spectra plus representative siblings up to this cap.",
    )
    parser.add_argument("--mz_max", type=float, default=None)
    parser.add_argument("--min_int_frac", type=float, default=0.005)
    parser.add_argument(
        "--max_peaks_per_spectrum",
        type=int,
        default=150,
        help="Plot only the top-N retained peaks by intensity for speed/readability.",
    )
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)
    return args


def add_ce_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add CE value/key columns.

    FT spectra use NCE when available because NCE is the intended comparable scale there.
    Other instruments use ACE first, then fall back to NCE.
    """
    df = df.copy()
    use_nce = df["inst_type"].eq("FT") & df["nce"].notna()
    df["ce_value"] = df["ace"].where(df["ace"].notna(), df["nce"])
    df.loc[use_nce, "ce_value"] = df.loc[use_nce, "nce"]
    df["ce_source"] = np.where(
        use_nce,
        "nce",
        np.where(df["ace"].notna(), "ace", np.where(df["nce"].notna(), "nce", "none")),
    )
    df["ce_key"] = np.where(
        df["ce_value"].notna(),
        df["ce_source"] + "=" + df["ce_value"].round(6).astype(str),
        "ce=NA",
    )
    return df


def peaks_to_arrays(peaks: object) -> tuple[np.ndarray, np.ndarray]:
    """Convert a peaks list into m/z and intensity arrays."""
    if peaks is None or (isinstance(peaks, float) and np.isnan(peaks)):
        return np.array([]), np.array([])
    if isinstance(peaks, str):
        rows = []
        for line in peaks.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2:
                rows.append((float(parts[0]), float(parts[1])))
        peaks = rows
    if len(peaks) == 0:
        return np.array([]), np.array([])
    arr = np.asarray(peaks, dtype=float)
    return arr[:, 0], arr[:, 1]


def choose_group_rows(group: pd.DataFrame, metric: str, max_rows: int) -> pd.DataFrame:
    """Select rows to plot for a large group while preserving flagged context."""
    if len(group) <= max_rows:
        return group
    flagged_ids = set(group.loc[group["is_flagged"], "spec_id"])
    keep_ids = set(flagged_ids)

    # Add the best normal siblings and CE endpoints for context.
    keep_ids.update(group.nlargest(min(4, len(group)), metric)["spec_id"].tolist())
    keep_ids.update(group.nsmallest(min(2, len(group)), "ce_sort")["spec_id"].tolist())
    keep_ids.update(group.nlargest(min(2, len(group)), "ce_sort")["spec_id"].tolist())

    flagged_pos = group.index[group["spec_id"].isin(flagged_ids)].tolist()
    for pos in flagged_pos:
        loc = group.index.get_loc(pos)
        for neigh in [loc - 1, loc + 1]:
            if 0 <= neigh < len(group):
                keep_ids.add(int(group.iloc[neigh]["spec_id"]))

    chosen = group[group["spec_id"].isin(keep_ids)].copy()
    if len(chosen) > max_rows:
        chosen = chosen.sort_values(["is_flagged", metric], ascending=[False, True]).head(max_rows)
    return chosen.sort_values("ce_sort")


def label_for_row(row: pd.Series, metric: str) -> tuple[str, str]:
    """Return subplot title and color for a spectrum row."""
    bits = [
        f"spec {int(row.spec_id)}",
        f"dset={row.dset}",
        str(row.ce_key),
        f"{metric}={row[metric]:.3f}",
    ]
    return " | ".join(bits), metric_color(row[metric])


def metric_color(value: float) -> str:
    """Color spectra by metric in 0.1 bins; good spectra stay blue."""
    if pd.isna(value):
        return "#666666"
    if value < 0.1:
        return "#C44E52"
    if value < 0.2:
        return "#DD8452"
    if value < 0.3:
        return "#E6C84F"
    if value < 0.4:
        return "#55A868"
    if value < 0.5:
        return "#64B5CD"
    return "#4C72B0"


def mol_image_array(smiles: str | None) -> np.ndarray | None:
    """Render a molecule image array from SMILES, returning None on failures."""
    if not smiles or smiles == "nan":
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol, size=(360, 220), kekulize=False)
    return np.asarray(img)


def safe_text(value: object) -> str:
    """Format scalar metadata without leaking pandas missing values into plot text."""
    if value is None or pd.isna(value):
        return ""
    return str(value)


def plot_group_page(
    pdf: PdfPages,
    group: pd.DataFrame,
    metric: str,
    mol_row: pd.Series | None,
    mz_max_arg: float | None,
    min_int_frac: float,
    max_peaks_per_spectrum: int,
    page: int,
) -> None:
    """Plot one CE group page into a PdfPages object."""
    n = len(group)
    ncols = 2 if n > 6 else 1
    nrows = math.ceil(n / ncols)
    fig_h = max(6.2, min(18.0, 1.65 * nrows + 2.4))
    fig = plt.figure(figsize=(13.5, fig_h), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows + 1,
        ncols,
        height_ratios=[1.35] + [1.0] * nrows,
        hspace=0.62,
        wspace=0.18,
    )
    header_ax = fig.add_subplot(grid[0, :])
    header_ax.axis("off")
    axes = np.array(
        [[fig.add_subplot(grid[row + 1, col]) for col in range(ncols)] for row in range(nrows)]
    )
    axes_flat = axes.T.ravel()

    all_mzs = []
    for peaks in group["peaks"]:
        mzs, _ints = peaks_to_arrays(peaks)
        if mzs.size:
            all_mzs.append(mzs)
    if mz_max_arg is not None:
        x_max = mz_max_arg
    elif all_mzs:
        x_max = min(1600.0, max(float(np.nanmax(mzs)) for mzs in all_mzs) * 1.04)
    else:
        x_max = 1000.0

    first = group.iloc[0]
    formula = ""
    smiles = ""
    inchikey = ""
    if mol_row is not None:
        formula = safe_text(mol_row.get("formula", ""))
        smiles = safe_text(mol_row.get("smiles", ""))
        inchikey = safe_text(mol_row.get("inchikey_s", mol_row.get("inchikey", "")))
    flagged_ids = group.loc[group["is_flagged"], "spec_id"].astype(int).tolist()
    dsets = sorted(group["dset"].dropna().astype(str).unique())
    dset_label = ",".join(dsets) if dsets else "NA"
    header_text = (
        f"page {page} | mol {int(first.mol_id)} | {first.prec_type} | {first.inst_type} | "
        f"{first.frag_mode} | dset {dset_label} | formula {formula}\n"
        f"flagged specs: {flagged_ids}"
    )
    if inchikey:
        header_text += f"\n{inchikey}"
    if smiles:
        header_text += f"\n{smiles[:180]}"
    header_ax.text(0.01, 0.96, header_text, transform=header_ax.transAxes, va="top", fontsize=9.5)
    mol_img = mol_image_array(smiles)
    if mol_img is not None:
        mol_ax = header_ax.inset_axes([0.72, 0.02, 0.26, 0.94])
        mol_ax.imshow(mol_img)
        mol_ax.axis("off")
    else:
        header_ax.text(
            0.84,
            0.5,
            "molecule image unavailable",
            transform=header_ax.transAxes,
            ha="center",
            va="center",
            fontsize=8,
            color="#666666",
        )

    for ax, (_idx, row) in zip(axes_flat, group.iterrows()):
        mzs, ints = peaks_to_arrays(row["peaks"])
        title, color = label_for_row(row, metric)
        if mzs.size == 0 or ints.size == 0:
            ax.text(0.5, 0.5, "no peaks", transform=ax.transAxes, ha="center", va="center")
        else:
            max_int = float(np.nanmax(ints))
            if max_int > 0:
                ints = ints / max_int
            keep = ints >= min_int_frac
            mzs = mzs[keep]
            ints = ints[keep]
            if max_peaks_per_spectrum > 0 and len(ints) > max_peaks_per_spectrum:
                top_idx = np.argpartition(ints, -max_peaks_per_spectrum)[-max_peaks_per_spectrum:]
                top_idx = top_idx[np.argsort(mzs[top_idx])]
                mzs = mzs[top_idx]
                ints = ints[top_idx]
            ax.vlines(mzs, 0.0, ints, color=color, linewidth=0.65)
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, x_max)
        ax.set_title(title, fontsize=7.5, loc="left", color=color)
        ax.grid(axis="y", alpha=0.25, linewidth=0.4)
        ax.tick_params(labelsize=7)
        ax.set_ylabel("rel int", fontsize=7)

    for ax in axes_flat[n:]:
        ax.axis("off")
    for ax in axes[-1, :]:
        ax.set_xlabel("m/z", fontsize=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Create a multipage PDF of suspicious CE groups."""
    args = get_args()
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)

    flags = pd.read_csv(args.flags_csv)
    oracle_cols = ["spec_id", args.metric]
    oracle = pd.read_csv(args.oracle_csv, usecols=lambda c: c in set(oracle_cols))
    spec = pd.read_pickle(args.spec_fp)[
        ["spec_id", "mol_id", "prec_type", "inst_type", "frag_mode", "dset", "ace", "nce", "peaks"]
    ]
    if args.dsets:
        spec = spec[spec["dset"].isin(args.dsets)]
    mol = pd.read_pickle(args.mol_fp)
    mol_by_id = mol.set_index("mol_id")

    flag_cols = [
        "spec_id",
        "best_lower_ce",
        "best_higher_ce",
        "best_diff_ce",
        "flag_low_ce_bad_high_ce_ok",
        "flag_high_ce_bad_low_ce_ok",
        "flag_any_diff_ce_ok",
        "flag_trend_drop",
    ]
    flag_cols = [c for c in flag_cols if c in flags.columns]
    flag_by_spec = flags[flag_cols].drop_duplicates("spec_id")

    groups = flags[GROUP_COLS].drop_duplicates()
    plot_df = spec.merge(groups, on=GROUP_COLS, how="inner").merge(oracle, on="spec_id", how="left")
    plot_df = plot_df.merge(flag_by_spec, on="spec_id", how="left")
    plot_df = add_ce_columns(plot_df)
    plot_df["ce_sort"] = plot_df["ce_value"].fillna(np.inf)
    plot_df["is_flagged"] = plot_df["spec_id"].isin(set(flags["spec_id"]))
    for col in [
        "flag_low_ce_bad_high_ce_ok",
        "flag_high_ce_bad_low_ce_ok",
        "flag_any_diff_ce_ok",
        "flag_trend_drop",
    ]:
        if col in plot_df.columns:
            plot_df[col] = pd.Series(
                np.where(plot_df[col].isna(), False, plot_df[col]), index=plot_df.index
            ).astype(bool)

    group_order = (
        flags.groupby(GROUP_COLS, observed=True)[args.metric]
        .min()
        .reset_index(name="worst_metric")
        .sort_values("worst_metric")
    )
    if args.max_groups is not None:
        group_order = group_order.head(args.max_groups)

    pdf_fp = out_dp / "oracle_ce_sibling_spectrum_groups.pdf"
    index_rows = []
    with PdfPages(pdf_fp) as pdf:
        for page, row in enumerate(group_order.itertuples(index=False), start=1):
            key = {
                "mol_id": row.mol_id,
                "prec_type": row.prec_type,
                "inst_type": row.inst_type,
                "frag_mode": row.frag_mode,
            }
            mask = np.ones(len(plot_df), dtype=bool)
            for col, val in key.items():
                mask &= plot_df[col].to_numpy() == val
            group = plot_df[mask].sort_values("ce_sort").copy()
            group = choose_group_rows(group, args.metric, args.max_spectra_per_group)
            mol_row = mol_by_id.loc[row.mol_id] if row.mol_id in mol_by_id.index else None
            if isinstance(mol_row, pd.DataFrame):
                mol_row = mol_row.iloc[0]
            plot_group_page(
                pdf,
                group,
                args.metric,
                mol_row,
                args.mz_max,
                args.min_int_frac,
                args.max_peaks_per_spectrum,
                page,
            )
            flagged = group[group["is_flagged"]]
            index_rows.append(
                {
                    **key,
                    "page": page,
                    "dsets": ";".join(sorted(group["dset"].dropna().astype(str).unique())),
                    "n_plotted_spectra": len(group),
                    "flagged_spec_ids": ";".join(flagged["spec_id"].astype(str)),
                    "worst_metric": flagged[args.metric].min(),
                    "n_low_no_good_reason": int(flagged["flag_any_diff_ce_ok"].sum()),
                    "n_trend_drop": int(flagged["flag_trend_drop"].sum()),
                }
            )
            if page % 100 == 0:
                log.info("Plotted %d/%d groups", page, len(group_order))

    index_fp = out_dp / "oracle_ce_sibling_spectrum_groups_index.csv"
    pd.DataFrame(index_rows).to_csv(index_fp, index=False)
    log.info("Wrote %s", pdf_fp)
    log.info("Wrote %s", index_fp)


if __name__ == "__main__":
    main()
