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
import torch as th
from dag_oracle_scores import (
    _allowed_formula_idxs,
    _dag_ion_mzs,
    _detect_is_compressed,
    _tolerance_for_inst_type,
)
from matplotlib.backends.backend_pdf import PdfPages
from oracle_dataset_utils import resolve_dsets_for_spec_fp
from rdkit import Chem
from rdkit.Chem import Draw

from fragnnet.utils import frag_utils
from fragnnet.utils.formula_utils import PREC_TYPE_TO_MASS_DIFF
from fragnnet.utils.spec_utils import calculate_match_mzs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


GROUP_COLS = ["mol_id", "prec_type", "inst_type", "frag_mode"]
PEAK_TRACK_TOLERANCE_DA = 0.01


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
        "--frag_dp",
        default="data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags",
        help="DAG directory used to color matched/unmatched peaks.",
    )
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
        "--peak_color_mode",
        choices=["match", "metric"],
        default="match",
        help="match: color individual peaks by DAG match status; metric: color whole spectrum.",
    )
    parser.add_argument(
        "--max_h_transfer",
        type=int,
        default=None,
        help="Default: use the DAG's full H-transfer budget for match coloring.",
    )
    parser.add_argument(
        "--no_cmf",
        action="store_true",
        help="For match coloring, use direct-charge-retention ion masses only.",
    )
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


def highest_peak_mz(peaks: object) -> float:
    """Return the m/z of the highest-intensity peak, or NaN if there are no peaks."""
    mzs, ints = peaks_to_arrays(peaks)
    if mzs.size == 0 or ints.size == 0:
        return float("nan")
    return float(mzs[int(np.nanargmax(ints))])


def compute_peak_match_masks(
    plot_df: pd.DataFrame,
    frag_dp: str,
    max_h_transfer: int | None,
    include_cmf: bool,
) -> dict[int, np.ndarray]:
    """Return spec_id -> true peak mask for DAG matches under instrument tolerance."""
    if plot_df.empty:
        return {}
    sample_mol_id = int(plot_df["mol_id"].iloc[0])
    is_compressed = _detect_is_compressed(frag_dp, sample_mol_id)
    masks: dict[int, np.ndarray] = {}

    for mol_id, mol_group in plot_df.groupby("mol_id", observed=True):
        try:
            dag_d = frag_utils.load_frag_d(int(mol_id), frag_dp, is_compressed=is_compressed)
        except FileNotFoundError:
            dag_d = None
        if dag_d is None:
            continue

        theo_mzs_full = dag_d["formula_peak_mzs"]
        theo_probs_full = dag_d["formula_peak_probs"]
        allowed_idx = _allowed_formula_idxs(dag_d["idx_by_h_delta"], max_h_transfer)
        idx_t = th.tensor(allowed_idx, dtype=th.long)
        theo_mzs = theo_mzs_full.index_select(0, idx_t)
        prec_mask = (theo_probs_full.index_select(0, idx_t) > 0.0).float()

        for row in mol_group.itertuples(index=False):
            mzs, _ints = peaks_to_arrays(row.peaks)
            if mzs.size == 0:
                masks[int(row.spec_id)] = np.array([], dtype=bool)
                continue
            inst_tol, inst_tol_min_mz = _tolerance_for_inst_type(row.inst_type)
            dag_ion_mzs = _dag_ion_mzs(theo_mzs, prec_mask, row.prec_type, include_cmf=include_cmf)
            true_mzs = th.tensor(mzs, dtype=th.float32)
            match_mask = calculate_match_mzs(
                true_mzs,
                dag_ion_mzs,
                tolerance=inst_tol,
                relative=True,
                tolerance_min_mz=inst_tol_min_mz,
            ).any(dim=1)
            masks[int(row.spec_id)] = match_mask.cpu().numpy().astype(bool)
    return masks


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
    calc_pmz = row.get("calc_prec_mz", np.nan)
    pmz = row.get("prec_mz", np.nan)
    top_mz = row.get("top_peak_mz", np.nan)
    bits = [
        f"spec {int(row.spec_id)}",
        f"dset={row.dset}",
        str(row.ce_key),
        f"pmz={pmz:.4f}" if pd.notna(pmz) else "pmz=NA",
        f"calc_pmz={calc_pmz:.4f}" if pd.notna(calc_pmz) else "calc_pmz=NA",
        f"top_mz={top_mz:.4f}" if pd.notna(top_mz) else "top_mz=NA",
        f"{metric}={row[metric]:.3f}",
    ]
    return " | ".join(bits), metric_color(row[metric])


def add_peak_label_columns(df: pd.DataFrame, mol_by_id: pd.DataFrame) -> pd.DataFrame:
    """Add computed precursor m/z and highest observed peak m/z for plot labels."""
    out = df.copy()
    exact_mw = out["mol_id"].map(mol_by_id["exact_mw"]) if "exact_mw" in mol_by_id else np.nan
    mass_diff = out["prec_type"].map(PREC_TYPE_TO_MASS_DIFF)
    out["calc_prec_mz"] = exact_mw + mass_diff
    out["top_peak_mz"] = out["peaks"].map(highest_peak_mz)
    return out


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


def format_group_values(values: pd.Series, decimals: int = 4) -> str:
    """Format the distinct finite numeric values represented on a group page."""
    numeric = pd.to_numeric(values, errors="coerce").dropna().unique()
    return ", ".join(f"{value:.{decimals}f}" for value in np.sort(numeric)) or "NA"


def add_peak_tracks(group: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Label cross-CE peaks using a fixed absolute m/z tolerance.

    A track contains at most one peak from each spectrum. Track IDs are ordered by
    their final mean m/z and only tracks present at two or more CE values are retained.
    """
    if group.empty:
        return group.copy(), pd.DataFrame()

    peak_rows = []
    for row in group.itertuples():
        mzs, ints = peaks_to_arrays(row.peaks)
        for peak_idx, (mz, intensity) in enumerate(zip(mzs, ints)):
            peak_rows.append(
                {
                    "row_index": row.Index,
                    "spec_id": int(row.spec_id),
                    "ce_key": row.ce_key,
                    "ce_value": row.ce_value,
                    "peak_idx": peak_idx,
                    "mz": float(mz),
                    "intensity": float(intensity),
                }
            )
    if not peak_rows:
        out = group.copy()
        out["peak_track_ids"] = [np.array([], dtype=int) for _ in range(len(out))]
        return out, pd.DataFrame()

    # 1. Greedily form narrow m/z tracks, preventing two peaks from one spectrum
    # from being assigned to the same track.
    tracks: list[dict[str, object]] = []
    for peak in sorted(peak_rows, key=lambda item: (item["mz"], item["spec_id"])):
        candidates = []
        for track_idx, track in enumerate(tracks):
            if peak["spec_id"] in track["spec_ids"]:
                continue
            center_mz = float(np.mean(track["mzs"]))
            diff = abs(float(peak["mz"]) - center_mz)
            if diff <= PEAK_TRACK_TOLERANCE_DA:
                candidates.append((diff, track_idx))
        if candidates:
            _, track_idx = min(candidates)
            track = tracks[track_idx]
        else:
            track_idx = len(tracks)
            track = {"mzs": [], "spec_ids": set(), "peaks": []}
            tracks.append(track)
        track["mzs"].append(peak["mz"])
        track["spec_ids"].add(peak["spec_id"])
        track["peaks"].append(peak)

    # 2. Give stable m/z-ordered IDs only to tracks observed across multiple CEs.
    cross_ce_tracks = []
    for track in tracks:
        ce_keys = {peak["ce_key"] for peak in track["peaks"]}
        if len(ce_keys) >= 2:
            cross_ce_tracks.append(track)
    cross_ce_tracks.sort(key=lambda track: float(np.mean(track["mzs"])))

    assignments: dict[tuple[object, int], int] = {}
    table_rows = []
    for track_id, track in enumerate(cross_ce_tracks, start=1):
        center_mz = float(np.mean(track["mzs"]))
        n_ce = len({peak["ce_key"] for peak in track["peaks"]})
        for peak in track["peaks"]:
            assignments[(peak["row_index"], peak["peak_idx"])] = track_id
            table_rows.append(
                {
                    "track_id": track_id,
                    "track_mz": center_mz,
                    "n_ce": n_ce,
                    "spec_id": peak["spec_id"],
                    "ce_key": peak["ce_key"],
                    "ce_value": peak["ce_value"],
                    "peak_idx": peak["peak_idx"],
                    "mz": peak["mz"],
                    "intensity": peak["intensity"],
                }
            )

    # 3. Attach arrays aligned one-to-one with each spectrum's original peak array.
    out = group.copy()
    track_arrays = []
    for row_index, row in out.iterrows():
        mzs, _ints = peaks_to_arrays(row["peaks"])
        track_arrays.append(
            np.array(
                [assignments.get((row_index, peak_idx), 0) for peak_idx in range(len(mzs))],
                dtype=int,
            )
        )
    out["peak_track_ids"] = track_arrays
    return out, pd.DataFrame(table_rows)


def keep_tracks_with_unexplainable_peaks(group: pd.DataFrame) -> pd.DataFrame:
    """Keep track labels only when at least one member is unmatched by the DAG."""
    out = group.copy()
    qualifying_ids: set[int] = set()
    for row in out.itertuples():
        track_ids = getattr(row, "peak_track_ids", None)
        match_mask = getattr(row, "match_mask", None)
        if not (
            isinstance(track_ids, np.ndarray)
            and isinstance(match_mask, np.ndarray)
            and len(track_ids) == len(match_mask)
        ):
            continue
        qualifying_ids.update(track_ids[(track_ids > 0) & ~match_mask].astype(int).tolist())

    out["peak_track_ids"] = [
        np.where(np.isin(track_ids, list(qualifying_ids)), track_ids, 0)
        if isinstance(track_ids, np.ndarray)
        else track_ids
        for track_ids in out["peak_track_ids"]
    ]
    return out


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
    exact_mass = safe_text(mol_row.get("exact_mw", "")) if mol_row is not None else ""
    exact_mass_label = f"{float(exact_mass):.4f}" if exact_mass else "NA"
    pmz_label = format_group_values(group["prec_mz"])
    header_text = (
        f"page {page} | mol {int(first.mol_id)} | {first.prec_type} | {first.inst_type} | "
        f"{first.frag_mode} | dset {dset_label} | formula {formula} | "
        f"exact mass {exact_mass_label} | pmz {pmz_label}\n"
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
            track_ids = row.get("peak_track_ids", None)
            if isinstance(track_ids, np.ndarray) and len(track_ids) == len(ints):
                track_ids = track_ids[keep]
            else:
                track_ids = None
            match_mask = row.get("match_mask", None)
            if isinstance(match_mask, np.ndarray) and len(match_mask) == len(ints):
                match_mask = match_mask[keep]
            else:
                match_mask = None
            mzs = mzs[keep]
            ints = ints[keep]
            if max_peaks_per_spectrum > 0 and len(ints) > max_peaks_per_spectrum:
                top_idx = np.argpartition(ints, -max_peaks_per_spectrum)[-max_peaks_per_spectrum:]
                top_idx = top_idx[np.argsort(mzs[top_idx])]
                mzs = mzs[top_idx]
                ints = ints[top_idx]
                if match_mask is not None:
                    match_mask = match_mask[top_idx]
                if track_ids is not None:
                    track_ids = track_ids[top_idx]
            ax.set_ylim(0, 1.05)
            ax.set_xlim(0, x_max)
            if match_mask is None:
                ax.vlines(mzs, 0.0, ints, color=color, linewidth=0.65)
            else:
                ax.vlines(mzs[match_mask], 0.0, ints[match_mask], color="#4C72B0", linewidth=0.7)
                ax.vlines(mzs[~match_mask], 0.0, ints[~match_mask], color="#C44E52", linewidth=0.7)
            if track_ids is not None:
                for mz, intensity, track_id in zip(mzs, ints, track_ids):
                    if track_id > 0:
                        ax.text(
                            mz,
                            min(float(intensity) + 0.025, 1.03),
                            f"T{track_id}",
                            rotation=90,
                            ha="center",
                            va="bottom",
                            fontsize=4.5,
                            color="#222222",
                            clip_on=True,
                        )
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
        [
            "spec_id",
            "mol_id",
            "prec_type",
            "inst_type",
            "frag_mode",
            "dset",
            "ace",
            "nce",
            "prec_mz",
            "peaks",
        ]
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
    plot_df = add_peak_label_columns(plot_df, mol_by_id)
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

    page_groups = []
    selected_frames = []
    track_frames = []
    for row in group_order.itertuples(index=False):
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
        group, track_df = add_peak_tracks(group)
        page_groups.append((key, group))
        selected_frames.append(group)
        if not track_df.empty:
            for col, val in key.items():
                track_df[col] = val
            track_frames.append(track_df)

    if args.peak_color_mode == "match" and selected_frames:
        selected_df = pd.concat(selected_frames, ignore_index=True)
        log.info("Computing DAG match masks for %d plotted spectra", len(selected_df))
        match_masks = compute_peak_match_masks(
            selected_df,
            args.frag_dp,
            args.max_h_transfer,
            include_cmf=not args.no_cmf,
        )
        page_groups = [
            (
                key,
                keep_tracks_with_unexplainable_peaks(
                    group.assign(match_mask=group["spec_id"].astype(int).map(match_masks))
                ),
            )
            for key, group in page_groups
        ]

    pdf_fp = out_dp / "oracle_ce_sibling_spectrum_groups.pdf"
    index_rows = []
    with PdfPages(pdf_fp) as pdf:
        for page, (key, group) in enumerate(page_groups, start=1):
            mol_id = int(key["mol_id"])
            mol_row = mol_by_id.loc[mol_id] if mol_id in mol_by_id.index else None
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
                    "exact_mass": (
                        mol_row.get("exact_mw", np.nan) if mol_row is not None else np.nan
                    ),
                    "pmz": format_group_values(group["prec_mz"]),
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
    tracks_fp = out_dp / "oracle_ce_sibling_peak_tracks.csv"
    track_cols = [
        *GROUP_COLS,
        "track_id",
        "track_mz",
        "n_ce",
        "spec_id",
        "ce_key",
        "ce_value",
        "peak_idx",
        "mz",
        "intensity",
    ]
    tracks_df = (
        pd.concat(track_frames, ignore_index=True)
        if track_frames
        else pd.DataFrame(columns=track_cols)
    )
    tracks_df[track_cols].to_csv(tracks_fp, index=False)
    log.info("Wrote %s", pdf_fp)
    log.info("Wrote %s", index_fp)
    log.info("Wrote %s", tracks_fp)


if __name__ == "__main__":
    main()
