"""Plot predictions from an inference HDF5 file.

Modes:
  1. summary   — cos_sim / cos_hun histograms + CDF.
  2. ce        — violin plots of cos_hun_inst_tol by CE bin, split by instrument type
                 (QTOF: ACE 10 eV bins; FT: NCE 10% bins).  Requires --spec_df_fp.
  3. spectra   — mirror spectra for top/worst/random N predictions. Requires --spec_df_fp.
  4. all       — all of the above.

Usage:
  python scripts/plotting/plot_predictions_hdf5.py \
      --pred_fp output/inference_eval/4ews8j6f/predictions_test.hdf5 \
      --mode summary

  python scripts/plotting/plot_predictions_hdf5.py \
      --pred_fp output/inference_eval/4ews8j6f/predictions_test.hdf5 \
      --spec_df_fp data/proc/nist20/spec_df.pkl \
      --mode ce

  python scripts/plotting/plot_predictions_hdf5.py \
      --pred_fp output/inference_eval/4ews8j6f/predictions_test.hdf5 \
      --spec_df_fp data/proc/nist20/spec_df.pkl \
      --mode all
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Literal

import h5py
import matplotlib.collections as mc
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# HDF5 helpers
# ---------------------------------------------------------------------------


def load_hdf5_meta(pred_fp: str) -> pd.DataFrame:
    """Load per-spectrum metadata from the prediction HDF5 into a DataFrame.

    Args:
        pred_fp: Path to the predictions HDF5 file.

    Returns:
        DataFrame with one row per spectrum and scalar columns from HDF5.
    """
    scalar_keys = [
        "spec_id",
        "mol_id",
        "cos_sim_0.01",
        "cos_hun_inst_tol",
        "cos_sim_sqrt_0.01",
        "pred_oos_prob",
        "start_points",
        "end_points",
        "index_rebased",
    ]
    with h5py.File(pred_fp, "r") as f:
        available = {k for k in scalar_keys if k in f}
        data = {k: f[k][:] for k in available}
    return pd.DataFrame(data)


def get_pred_spectrum(f: h5py.File, row: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    """Extract predicted (mz, intensity) arrays for one spectrum.

    Args:
        f: Open HDF5 file handle.
        row: One row from the metadata DataFrame (must have start_points, end_points).

    Returns:
        Tuple of (pred_mzs, pred_ints) as float32 arrays.
    """
    s, e = int(row["start_points"]), int(row["end_points"])
    return f["pred_mzs"][s:e], f["pred_ints"][s:e]


# ---------------------------------------------------------------------------
# Summary distribution plots
# ---------------------------------------------------------------------------

_METRICS = [
    ("cos_sim_0.01", "Cosine Similarity (tol=0.01 Da)"),
    ("cos_hun_inst_tol", "Hungarian Cosine (instrument tol)"),
]


def plot_summary(meta: pd.DataFrame, save_dp: str) -> None:
    """Plot histogram + CDF for each available score metric.

    Args:
        meta: DataFrame from load_hdf5_meta.
        save_dp: Directory to save figures.
    """
    os.makedirs(save_dp, exist_ok=True)
    available_metrics = [(k, label) for k, label in _METRICS if k in meta.columns]
    if not available_metrics:
        logger.warning("No score metrics found in HDF5.")
        return

    fig, axes = plt.subplots(len(available_metrics), 2, figsize=(12, 4 * len(available_metrics)))
    if len(available_metrics) == 1:
        axes = [axes]

    for row_axes, (key, label) in zip(axes, available_metrics):
        vals = meta[key].dropna().to_numpy()
        ax_hist, ax_cdf = row_axes

        # Histogram
        ax_hist.hist(vals, bins=100, color="steelblue", edgecolor="none", alpha=0.85)
        ax_hist.axvline(
            np.median(vals), color="crimson", lw=1.5, label=f"median={np.median(vals):.3f}"
        )
        ax_hist.set_xlabel(label, fontsize=11)
        ax_hist.set_ylabel("Count", fontsize=11)
        ax_hist.set_title(f"Distribution of {label}", fontsize=12)
        ax_hist.legend(fontsize=10)

        # CDF
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        ax_cdf.plot(sorted_vals, cdf, color="steelblue", lw=1.5)
        for thresh in (0.5, 0.7, 0.9):
            frac = np.mean(vals >= thresh)
            ax_cdf.axvline(thresh, color="grey", lw=0.8, ls="--")
            ax_cdf.text(
                thresh + 0.005, 0.05, f"≥{thresh:.1f}: {frac:.1%}", fontsize=8, color="grey"
            )
        ax_cdf.set_xlabel(label, fontsize=11)
        ax_cdf.set_ylabel("CDF", fontsize=11)
        ax_cdf.set_title(f"CDF of {label}", fontsize=12)
        ax_cdf.set_xlim(0, 1)
        ax_cdf.set_ylim(0, 1)

    fig.tight_layout()
    out_fp = os.path.join(save_dp, "summary_distributions.png")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved summary plot: %s", out_fp)
    print(f"Saved: {out_fp}")


# ---------------------------------------------------------------------------
# Mirror spectrum plot
# ---------------------------------------------------------------------------


def _mirror_ax(
    ax: plt.Axes,
    true_mzs: np.ndarray,
    true_ints: np.ndarray,
    pred_mzs: np.ndarray,
    pred_ints: np.ndarray,
    colors: tuple[str, str] = ("steelblue", "crimson"),
    bar_width: float = 1.0,
) -> None:
    """Draw a mirror (head-to-tail) spectrum comparison on ax.

    Args:
        ax: Matplotlib axes to draw on.
        true_mzs: Ground-truth m/z values.
        true_ints: Ground-truth intensities (0–999.99 scale).
        pred_mzs: Predicted m/z values.
        pred_ints: Predicted intensities.
        colors: (true_color, pred_color).
        bar_width: Line width for peaks.
    """
    if len(true_ints) > 0:
        scale = np.max(true_ints)
    elif len(pred_ints) > 0:
        scale = np.max(pred_ints)
    else:
        return

    all_mzs = np.concatenate([true_mzs, pred_mzs]) if len(true_mzs) else pred_mzs
    x_max = int(np.ceil(np.max(all_mzs) * 1.05))
    y_max = scale * 1.1

    ax.set_xlim(0, x_max)
    ax.set_ylim(-y_max, y_max)

    def _lines(mzs, ints, sign):
        return [((m, 0), (m, sign * i)) for m, i in zip(mzs, ints)]

    if len(true_mzs):
        lc_top = mc.LineCollection(
            _lines(true_mzs, true_ints, +1), colors=colors[0], linewidths=bar_width
        )
        ax.add_collection(lc_top)

    if len(pred_mzs):
        lc_bot = mc.LineCollection(
            _lines(pred_mzs, pred_ints, -1), colors=colors[1], linewidths=bar_width
        )
        ax.add_collection(lc_bot)

    ax.axhline(0, color="black", lw=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_position("zero")
    ax.set_xlabel("m/z", fontsize=10)
    ax.set_ylabel("Intensity", fontsize=10)
    ax.grid(axis="x", color="grey", lw=0.2, alpha=0.4)

    handles = [
        mpatches.Patch(color=colors[0], label="Ground Truth"),
        mpatches.Patch(color=colors[1], label="Predicted"),
    ]
    ax.legend(handles=handles, fontsize=9, loc="upper left")


def plot_spectra(
    meta: pd.DataFrame,
    pred_fp: str,
    spec_df: pd.DataFrame,
    select: Literal["top", "worst", "random"],
    n: int,
    score_col: str,
    save_dp: str,
) -> None:
    """Plot mirror spectra for selected predictions.

    Args:
        meta: Per-spectrum metadata DataFrame.
        pred_fp: Path to HDF5 predictions file.
        spec_df: Ground-truth spectrum DataFrame with columns [spec_id, peaks].
        select: Selection strategy – 'top', 'worst', or 'random'.
        n: Number of spectra to plot.
        score_col: Metric column to rank by.
        save_dp: Directory to save figures.
    """
    if score_col not in meta.columns:
        raise ValueError(f"Score column '{score_col}' not found. Available: {list(meta.columns)}")

    ranked = meta.dropna(subset=[score_col])
    if select == "top":
        subset = ranked.nlargest(n, score_col)
    elif select == "worst":
        subset = ranked.nsmallest(n, score_col)
    else:
        subset = ranked.sample(n=min(n, len(ranked)), random_state=42)

    # Build fast lookup for ground truth peaks
    gt_lookup: dict[int, np.ndarray] = {}
    for _, row in spec_df.iterrows():
        sid = int(row["spec_id"])
        peaks = row["peaks"]
        if peaks is None or (isinstance(peaks, float) and np.isnan(peaks)):
            continue
        if isinstance(peaks, np.ndarray) and peaks.ndim == 2:
            gt_lookup[sid] = peaks  # shape (N, 2): [[mz, int], ...]
        elif isinstance(peaks, str) and peaks.strip():
            rows = []
            for line in peaks.strip().split("\n"):
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        rows.append([float(parts[0]), float(parts[1])])
                    except ValueError:
                        pass
            if rows:
                gt_lookup[sid] = np.array(rows)

    os.makedirs(save_dp, exist_ok=True)

    with h5py.File(pred_fp, "r") as f:
        for _, row in tqdm(subset.iterrows(), total=len(subset), desc="Plotting spectra"):
            sid = int(row["spec_id"])
            pred_mzs, pred_ints = get_pred_spectrum(f, row)

            gt = gt_lookup.get(sid)
            if gt is not None:
                true_mzs, true_ints = gt[:, 0], gt[:, 1]
            else:
                true_mzs, true_ints = np.array([]), np.array([])

            score = row[score_col]
            fig, ax = plt.subplots(figsize=(12, 5))
            _mirror_ax(ax, true_mzs, true_ints, pred_mzs, pred_ints)
            ax.set_title(
                f"spec_id={sid}  mol_id={int(row['mol_id'])}  "
                f"{score_col}={score:.3f}  n_pred={len(pred_mzs)}",
                fontsize=11,
            )
            fig.tight_layout()
            out_fp = os.path.join(save_dp, f"spectrum_{sid}_{select}.png")
            fig.savefig(out_fp, dpi=120, bbox_inches="tight")
            plt.close(fig)

    print(f"Saved {len(subset)} spectra plots to {save_dp}/")


# ---------------------------------------------------------------------------
# CE breakdown violin plots
# ---------------------------------------------------------------------------

_MIN_BIN_N = 30  # drop bins with fewer spectra than this


def _make_ce_bins(
    df: pd.DataFrame,
    ce_col: str,
    bin_size: int,
    ce_max: float,
    score_col: str,
) -> tuple[list[str], list[np.ndarray]]:
    """Bin df by CE column and return (labels, score_arrays) for non-empty bins.

    Args:
        df: DataFrame with ce_col and score_col.
        ce_col: Column name for collision energy values.
        bin_size: Width of each bin.
        ce_max: Upper bound (exclusive) for binning.
        score_col: Score column to collect per bin.

    Returns:
        Tuple of (bin_labels, list_of_score_arrays), only for bins with >= _MIN_BIN_N spectra.
    """
    labels, arrays = [], []
    for lo in range(0, int(ce_max), bin_size):
        hi = lo + bin_size
        mask = (df[ce_col] >= lo) & (df[ce_col] < hi)
        vals = df.loc[mask, score_col].dropna().to_numpy()
        if len(vals) >= _MIN_BIN_N:
            labels.append(f"{lo}–{hi}")
            arrays.append(vals)
    return labels, arrays


def _violin_ax(
    ax: plt.Axes,
    labels: list[str],
    arrays: list[np.ndarray],
    title: str,
    xlabel: str,
    color: str,
    score_col: str,
) -> None:
    """Draw violin + median scatter on ax.

    Args:
        ax: Axes to draw on.
        labels: Bin label strings for x-ticks.
        arrays: One score array per bin.
        title: Axes title.
        xlabel: X-axis label.
        color: Violin fill colour.
        score_col: Score column name (for y-axis label).
    """
    positions = list(range(1, len(labels) + 1))
    parts = ax.violinplot(arrays, positions=positions, showmedians=False, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor(color)
        pc.set_alpha(0.65)

    medians = [np.median(a) for a in arrays]
    counts = [len(a) for a in arrays]
    ax.scatter(positions, medians, color="black", s=25, zorder=3)

    for pos, med, n in zip(positions, medians, counts):
        ax.text(pos, med + 0.018, f"{med:.2f}", ha="center", va="bottom", fontsize=7, color="black")
        ax.text(pos, -0.07, f"n={n}", ha="center", va="top", fontsize=6.5, color="grey",
                transform=ax.get_xaxis_transform())

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel(score_col, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=11)
    ax.axhline(0.5, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.axhline(0.9, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_ce_breakdown(
    meta: pd.DataFrame,
    spec_df: pd.DataFrame,
    score_col: str,
    save_dp: str,
) -> None:
    """Plot cos_hun_inst_tol violin plots broken down by CE bin for QTOF and FT.

    QTOF uses ACE (eV) in 10 eV bins; FT uses NCE (%) in 10% bins.
    Bins with fewer than _MIN_BIN_N spectra are omitted.

    Args:
        meta: Per-spectrum metadata DataFrame from load_hdf5_meta.
        spec_df: Ground-truth spec_df with columns [spec_id, inst_type, ace, nce].
        score_col: Score column to plot (must be in meta).
        save_dp: Directory to save figures.
    """
    if score_col not in meta.columns:
        raise ValueError(f"Score column '{score_col}' not in meta. Available: {list(meta.columns)}")

    os.makedirs(save_dp, exist_ok=True)

    joined = meta[["spec_id", score_col]].merge(
        spec_df[["spec_id", "inst_type", "frag_mode", "ace", "nce"]],
        on="spec_id",
        how="left",
    )

    qtof = joined[(joined["inst_type"] == "QTOF") & joined["ace"].notna() & (joined["ace"] > 0)]
    ft = joined[(joined["inst_type"] == "FT") & joined["nce"].notna()]

    qtof_labels, qtof_arrays = _make_ce_bins(qtof, "ace", 10, qtof["ace"].max() + 10, score_col)
    ft_labels, ft_arrays = _make_ce_bins(ft, "nce", 10, ft["nce"].max() + 10, score_col)

    # Two subplots: QTOF left, FT right
    n_qtof = len(qtof_labels)
    n_ft = len(ft_labels)
    fig_w = max(8, n_qtof * 0.9) + max(8, n_ft * 0.9)
    fig, (ax_qtof, ax_ft) = plt.subplots(1, 2, figsize=(fig_w, 5.5))

    _violin_ax(
        ax_qtof,
        qtof_labels,
        qtof_arrays,
        title=f"QTOF — ACE (eV) breakdown  [n={len(qtof):,}]",
        xlabel="ACE bin (eV)",
        color="#4878CF",
        score_col=score_col,
    )
    _violin_ax(
        ax_ft,
        ft_labels,
        ft_arrays,
        title=f"FT — NCE (%) breakdown  [n={len(ft):,}]",
        xlabel="NCE bin (%)",
        color="#6ACC65",
        score_col=score_col,
    )

    fig.suptitle(f"{score_col} by instrument type and collision energy", fontsize=12, y=1.01)
    fig.tight_layout()
    out_fp = os.path.join(save_dp, "ce_breakdown_violin.png")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved CE breakdown plot: %s", out_fp)
    print(f"Saved: {out_fp}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot predictions from inference HDF5")
    parser.add_argument("--pred_fp", type=str, required=True, help="Path to predictions HDF5")
    parser.add_argument(
        "--spec_df_fp",
        type=str,
        default=None,
        help="Path to ground-truth spec_df.pkl (required for spectra mode)",
    )
    parser.add_argument(
        "--save_dp",
        type=str,
        default=None,
        help="Output directory (default: same dir as pred_fp)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["summary", "ce", "spectra", "all"],
        default="summary",
        help="What to plot (ce requires --spec_df_fp)",
    )
    parser.add_argument(
        "--select",
        type=str,
        choices=["top", "worst", "random"],
        default="top",
        help="Which spectra to plot in spectra mode",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=20,
        help="Number of spectra to plot",
    )
    parser.add_argument(
        "--score_col",
        type=str,
        default="cos_hun_inst_tol",
        help="Score column to rank spectra by",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args = get_args()

    save_dp = args.save_dp or str(Path(args.pred_fp).parent / "plots")

    logger.info("Loading HDF5 metadata from %s", args.pred_fp)
    meta = load_hdf5_meta(args.pred_fp)
    logger.info("Loaded %d spectra", len(meta))

    if args.mode in ("summary", "all"):
        plot_summary(meta, save_dp)

    if args.mode in ("ce", "spectra", "all"):
        if args.spec_df_fp is None:
            raise ValueError("--spec_df_fp is required for 'ce', 'spectra', or 'all' mode")
        logger.info("Loading spec_df from %s", args.spec_df_fp)
        spec_df = pd.read_pickle(args.spec_df_fp)

        if args.mode in ("ce", "all"):
            plot_ce_breakdown(
                meta=meta,
                spec_df=spec_df,
                score_col=args.score_col,
                save_dp=save_dp,
            )

        if args.mode in ("spectra", "all"):
            plot_spectra(
                meta=meta,
                pred_fp=args.pred_fp,
                spec_df=spec_df,
                select=args.select,
                n=args.n,
                score_col=args.score_col,
                save_dp=save_dp,
            )


if __name__ == "__main__":
    main()
