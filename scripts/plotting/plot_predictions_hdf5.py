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
        "cos_hun_inst_tol_sqrt",
        "jss_hun_inst_tol",
        "jss_hun_inst_tol_sqrt",
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
    ("cos_hun_inst_tol_sqrt", "Hungarian Cosine Sqrt (instrument tol)"),
    ("jss_hun_inst_tol", "Hungarian JSS (instrument tol)"),
    ("jss_hun_inst_tol_sqrt", "Hungarian JSS Sqrt (instrument tol)"),
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


_METRIC_COLORS = ["#4878CF", "#e05c1a", "#6ACC65", "#8B6BB1", "#D65F5F"]


def plot_all_metrics(meta: pd.DataFrame, save_dp: str) -> None:
    """Create a single overview figure with all available metrics.

    Layout (4-column grid throughout):
    - CDF rows: one subplot per metric, 4 per row, with threshold annotations.
    - Histogram rows: one subplot per metric, 4 per row, median/mean lines.

    Args:
        meta: DataFrame from load_hdf5_meta.
        save_dp: Directory to save figures.
    """
    os.makedirs(save_dp, exist_ok=True)
    available_metrics = [(k, label) for k, label in _METRICS if k in meta.columns]
    if not available_metrics:
        logger.warning("No score metrics found in HDF5.")
        return

    n = len(available_metrics)
    n_cols = 4
    n_rows_per_section = (n + n_cols - 1) // n_cols
    n_rows_total = 2 * n_rows_per_section  # CDF section + histogram section

    fig, axes = plt.subplots(
        n_rows_total,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows_total),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.55, wspace=0.35)

    # --- CDF subplots (top section) ---
    for idx, ((key, label), color) in enumerate(zip(available_metrics, _METRIC_COLORS)):
        row, col = divmod(idx, n_cols)
        ax = axes[row, col]
        vals = meta[key].dropna().to_numpy()
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
        med = np.median(vals)
        mn = np.mean(vals)
        ax.plot(sorted_vals, cdf, color=color, lw=2.0)
        for thresh in (0.5, 0.7, 0.9):
            frac = np.mean(vals >= thresh)
            ax.axvline(thresh, color="grey", lw=0.8, ls="--", alpha=0.6)
            ax.text(thresh + 0.004, 0.04, f"≥{thresh:.1f}:{frac:.1%}", fontsize=8, color="grey")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Score", fontsize=11)
        ax.set_ylabel("CDF", fontsize=11)
        ax.set_title(f"{label}\nmed={med:.3f}  mean={mn:.3f}", fontsize=12, pad=6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused CDF cells
    for idx in range(n, n_rows_per_section * n_cols):
        row, col = divmod(idx, n_cols)
        axes[row, col].set_visible(False)

    # --- Histogram subplots (bottom section) ---
    for idx, ((key, label), color) in enumerate(zip(available_metrics, _METRIC_COLORS)):
        row = n_rows_per_section + idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        vals = meta[key].dropna().to_numpy()
        med = np.median(vals)
        mn = np.mean(vals)
        ax.hist(vals, bins=80, color=color, edgecolor="none", alpha=0.75)
        ax.axvline(med, color="black", lw=1.5, label=f"med={med:.3f}")
        ax.axvline(mn, color="white", lw=1.5, ls="--", label=f"mean={mn:.3f}")
        ax.set_xlabel("Score", fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.set_title(label, fontsize=12, pad=6)
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # Hide unused histogram cells
    for idx in range(n, n_rows_per_section * n_cols):
        row = n_rows_per_section + idx // n_cols
        col = idx % n_cols
        axes[row, col].set_visible(False)

    fig.suptitle("All Metrics Overview", fontsize=20, y=1.01)
    out_fp = os.path.join(save_dp, "all_metrics_overview.png")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved all-metrics plot: %s", out_fp)
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
    global_labels: list[str],
    arrays_aligned: list[np.ndarray | None],
    title: str,
    xlabel: str,
    color: str,
    score_col: str,
    show_xticklabels: bool = True,
    show_xlabel: bool = True,
) -> None:
    """Draw violin + IQR bar (tol range) + median/mean dots on ax.

    Accepts a globally-aligned arrays list (None for bins with insufficient data)
    so x-axis positions are consistent across all subplots sharing the same axis.

    Args:
        ax: Axes to draw on.
        global_labels: Bin label strings for every position (including empty bins).
        arrays_aligned: Score array per bin, or None for bins with < _MIN_BIN_N spectra.
        title: Axes title.
        xlabel: X-axis label (shown only when show_xlabel=True).
        color: Violin fill colour.
        score_col: Score column name (for y-axis label).
        show_xticklabels: Whether to draw x-tick labels (False for non-bottom rows).
        show_xlabel: Whether to draw the x-axis label (False for non-bottom rows).
    """
    all_positions = list(range(1, len(global_labels) + 1))
    n_total = len(global_labels)

    _MED_COLOR = "#1a1a2e"
    _MEAN_COLOR = "#e05c1a"

    # Collect per-bin counts for x-tick labels
    bin_counts: dict[int, int] = {}

    for pos, arr in zip(all_positions, arrays_aligned):
        if arr is None or len(arr) < _MIN_BIN_N:
            continue

        parts = ax.violinplot([arr], positions=[pos], showmedians=False, showextrema=False)
        for pc in parts["bodies"]:
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor("grey")
            pc.set_linewidth(0.5)

        med = np.median(arr)
        mn = np.mean(arr)
        q1 = np.percentile(arr, 25)
        q3 = np.percentile(arr, 75)
        n = len(arr)
        bin_counts[pos] = n

        ax.vlines([pos], [q1], [q3], color=_MED_COLOR, linewidth=3.5, zorder=4)
        ax.scatter(
            [pos], [med], color="white", s=35, zorder=6, edgecolors=_MED_COLOR, linewidths=1.5
        )
        ax.scatter(
            [pos],
            [mn],
            color=_MEAN_COLOR,
            s=30,
            zorder=6,
            marker="D",
            edgecolors="white",
            linewidths=0.6,
        )

        ax.text(
            pos - 0.3,
            q3 + 0.018,
            f"{med:.2f}",
            ha="center",
            va="bottom",
            fontsize=13,
            color=_MED_COLOR,
            fontweight="bold",
        )
        ax.text(
            pos + 0.3,
            q3 + 0.018,
            f"{mn:.2f}",
            ha="center",
            va="bottom",
            fontsize=13,
            color=_MEAN_COLOR,
            fontweight="bold",
        )
        ax.text(
            pos,
            q1 - 0.01,
            f"±{(q3 - q1) / 2:.2f}",
            ha="center",
            va="top",
            fontsize=11,
            color="#555555",
        )

    # Legend proxy handles
    ax.vlines([], [], [], color=_MED_COLOR, linewidth=3.5, label="IQR (tol range)")
    ax.scatter([], [], color="white", s=35, edgecolors=_MED_COLOR, linewidths=1.5, label="Median")
    ax.scatter(
        [],
        [],
        color=_MEAN_COLOR,
        s=30,
        marker="D",
        edgecolors="white",
        linewidths=0.6,
        label="Mean",
    )

    ax.set_xticks(all_positions)
    if show_xticklabels:
        tick_labels = [
            f"{lbl}\nn={bin_counts[p]:,}" if p in bin_counts else lbl
            for p, lbl in zip(all_positions, global_labels)
        ]
        ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=13)
    else:
        # Show n= as small text inside the plot top for non-bottom rows
        for pos, arr in zip(all_positions, arrays_aligned):
            if arr is None or len(arr) < _MIN_BIN_N:
                continue
            ax.text(
                pos,
                1.02,
                f"n={len(arr):,}",
                ha="center",
                va="bottom",
                fontsize=10,
                color="grey",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xticklabels([])
    ax.set_xlim(0.5, n_total + 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel(score_col, fontsize=14)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=14)
    ax.set_title(title, fontsize=18, pad=12)
    for thresh, ls in [(0.5, "--"), (0.7, ":"), (0.9, "--")]:
        ax.axhline(thresh, color="grey", lw=0.7, ls=ls, alpha=0.55)
        ax.text(n_total + 0.55, thresh, f"{thresh}", va="center", fontsize=11, color="grey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(fontsize=12, loc="lower left", ncol=3)


def _plot_inst_adduct_grid(
    joined: pd.DataFrame,
    inst_type: str,
    ce_col: str,
    ce_label: str,
    color: str,
    score_col: str,
    save_dp: str,
    ce_cap: float | None = None,
) -> None:
    """Produce one figure for a given instrument type, one row per adduct.

    Args:
        joined: Merged DataFrame with spec metadata and scores.
        inst_type: Instrument type string (e.g. 'FT', 'QTOF').
        ce_col: CE column to bin by ('ace' or 'nce').
        ce_label: Human-readable CE axis label.
        color: Violin fill colour.
        score_col: Score column to plot.
        save_dp: Output directory.
        ce_cap: If set, discard spectra with ce_col >= ce_cap before binning.
    """
    inst_df = joined[(joined["inst_type"] == inst_type) & joined[ce_col].notna()]
    if ce_col == "ace":
        inst_df = inst_df[inst_df[ce_col] > 0]
    if ce_cap is not None:
        inst_df = inst_df[inst_df[ce_col] < ce_cap]

    adducts = (
        inst_df.groupby("prec_type")["spec_id"]
        .count()
        .loc[lambda s: s >= _MIN_BIN_N]
        .sort_values(ascending=False)
        .index.tolist()
    )
    if not adducts:
        logger.warning("No adducts with enough data for %s", inst_type)
        return

    ce_max = inst_df[ce_col].max() + 10
    n_rows = len(adducts)

    # Global bin labels from the full inst_df (same for every adduct row)
    global_labels, _ = _make_ce_bins(inst_df, ce_col, 10, ce_max, score_col)
    n_bins = len(global_labels)
    # Build lo edges for global bins so we can align per-adduct data
    global_lo = [int(lbl.split("–")[0]) for lbl in global_labels]

    fig_w = max(32, n_bins * 4.0 + 3)
    fig_h = n_rows * 7.0

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False, sharex=True)

    for row_idx, adduct in enumerate(adducts):
        sub = inst_df[inst_df["prec_type"] == adduct]
        # Align per-adduct arrays to global positions (None where bin is empty)
        arrays_aligned: list[np.ndarray | None] = []
        for lo in global_lo:
            hi = lo + 10
            mask = (sub[ce_col] >= lo) & (sub[ce_col] < hi)
            vals = sub.loc[mask, score_col].dropna().to_numpy()
            arrays_aligned.append(vals if len(vals) >= _MIN_BIN_N else None)

        ax = axes[row_idx, 0]
        is_bottom = row_idx == n_rows - 1
        _violin_ax(
            ax,
            global_labels,
            arrays_aligned,
            title=f"{inst_type} · {adduct}  [n={len(sub):,}]",
            xlabel=ce_label,
            color=color,
            score_col=score_col,
            show_xticklabels=is_bottom,
            show_xlabel=is_bottom,
        )

    fig.suptitle(
        f"{score_col} — {inst_type} by adduct + {ce_label}",
        fontsize=18,
        y=1.005,
    )
    fig.subplots_adjust(hspace=0.35)
    fname = f"ce_breakdown_{inst_type.lower()}_violin.png"
    out_fp = os.path.join(save_dp, fname)
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_fp)
    print(f"Saved: {out_fp}")


def plot_ce_breakdown(
    meta: pd.DataFrame,
    spec_df: pd.DataFrame,
    score_col: str,
    save_dp: str,
) -> None:
    """Plot score violin plots broken down by CE bin, adduct and instrument type.

    Produces two figures: one for QTOF (ACE bins) and one for FT (NCE bins).
    Each figure has one row per adduct type. The IQR bar on each violin shows
    the tolerance spread (tol range) of predictions for that CE bin.

    Args:
        meta: Per-spectrum metadata DataFrame from load_hdf5_meta.
        spec_df: Ground-truth spec_df with columns [spec_id, inst_type, prec_type, ace, nce].
        score_col: Score column to plot (must be in meta).
        save_dp: Directory to save figures.
    """
    if score_col not in meta.columns:
        raise ValueError(f"Score column '{score_col}' not in meta. Available: {list(meta.columns)}")

    os.makedirs(save_dp, exist_ok=True)

    joined = meta[["spec_id", score_col]].merge(
        spec_df[["spec_id", "inst_type", "prec_type", "ace", "nce"]],
        on="spec_id",
        how="left",
    )

    _plot_inst_adduct_grid(
        joined, "QTOF", "ace", "ACE (eV)", "#4878CF", score_col, save_dp, ce_cap=50
    )
    _plot_inst_adduct_grid(joined, "FT", "nce", "NCE (%)", "#6ACC65", score_col, save_dp, ce_cap=70)


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
        choices=["summary", "ce", "spectra", "all", "plot_all"],
        default="summary",
        help="What to plot (ce/all requires --spec_df_fp; plot_all = summary + 4-col CDF/hist overview)",
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

    if args.mode == "plot_all":
        plot_summary(meta, save_dp)
        plot_all_metrics(meta, save_dp)

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
