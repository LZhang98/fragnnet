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
from scipy import stats
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
    ("cos_hun_inst_tol", "Cosine (instrument tol)"),
    ("cos_hun_inst_tol_sqrt", "Cosine Sqrt (instrument tol)"),
    ("jss_hun_inst_tol", "JSS (instrument tol)"),
    ("jss_hun_inst_tol_sqrt", "JSS Sqrt (instrument tol)"),
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

    Layout: 2 metric pairs per row. Each metric shows histogram + CDF side by side.
    For N metrics: ceil(N/2) rows × 4 cols.

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
    n_cols = 4  # 2 metrics per row: [hist0] [cdf0] [hist1] [cdf1]
    n_rows = (n + 1) // 2  # ceil(n/2)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(6 * n_cols, 5 * n_rows),
        squeeze=False,
    )
    fig.subplots_adjust(hspace=0.40, wspace=0.35)

    for idx, ((key, label), color) in enumerate(zip(available_metrics, _METRIC_COLORS)):
        row = idx // 2
        col_pair = 2 * (idx % 2)  # 0 or 2
        col_hist = col_pair
        col_cdf = col_pair + 1

        vals = meta[key].dropna().to_numpy()
        med = np.median(vals)
        mn = np.mean(vals)
        sorted_vals = np.sort(vals)
        cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)

        # Histogram
        ax_hist = axes[row, col_hist]
        ax_hist.hist(vals, bins=80, color=color, edgecolor="none", alpha=0.75)
        ax_hist.axvline(med, color="black", lw=1.5, label=f"med={med:.3f}")
        ax_hist.axvline(mn, color="#FF6B35", lw=1.5, ls="--", label=f"mean={mn:.3f}")
        ax_hist.set_xlabel("Score", fontsize=11)
        ax_hist.set_ylabel("Count", fontsize=11)
        ax_hist.set_title(f"{label} — histogram", fontsize=12, pad=6)
        ax_hist.legend(fontsize=9)
        ax_hist.spines["top"].set_visible(False)
        ax_hist.spines["right"].set_visible(False)

        # CDF
        ax_cdf = axes[row, col_cdf]
        ax_cdf.plot(sorted_vals, cdf, color=color, lw=2.0)
        for thresh in (0.5, 0.7, 0.9):
            frac = np.mean(vals >= thresh)
            ax_cdf.axvline(thresh, color="grey", lw=0.8, ls="--", alpha=0.6)
            ax_cdf.text(thresh + 0.004, 0.04, f"≥{thresh:.1f}:{frac:.1%}", fontsize=8, color="grey")
        ax_cdf.set_xlim(0, 1)
        ax_cdf.set_ylim(0, 1)
        ax_cdf.set_xlabel("Score", fontsize=11)
        ax_cdf.set_ylabel("CDF", fontsize=11)
        ax_cdf.set_title(f"{label} — CDF", fontsize=12, pad=6)
        ax_cdf.spines["top"].set_visible(False)
        ax_cdf.spines["right"].set_visible(False)

    # Hide unused cells
    for idx in range(n, n_rows * 2):
        row = idx // 2
        col_pair = 2 * (idx % 2)
        axes[row, col_pair].set_visible(False)
        axes[row, col_pair + 1].set_visible(False)

    fig.suptitle("All Metrics Overview (Histogram + CDF)", fontsize=20, y=1.01)
    out_fp = os.path.join(save_dp, "all_metrics_overview.png")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved all-metrics plot: %s", out_fp)
    print(f"Saved: {out_fp}")


# ---------------------------------------------------------------------------
# Peak count & intensity stats
# ---------------------------------------------------------------------------


def plot_metrics_ci(meta: pd.DataFrame, save_dp: str) -> None:
    """Plot bar chart with IQR for each metric.

    Shows median (bar) with IQR (25th-75th percentile) as error bars.

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
    fig, ax = plt.subplots(figsize=(max(8, n * 1.5), 6))

    x_pos = np.arange(n)
    medians = []
    lower_iqr = []
    upper_iqr = []
    means = []
    labels = []

    for (key, label), color in zip(available_metrics, _METRIC_COLORS):
        vals = meta[key].dropna().to_numpy()
        med = np.median(vals)
        q25 = np.percentile(vals, 25)
        q75 = np.percentile(vals, 75)
        mn = np.mean(vals)
        medians.append(med)
        lower_iqr.append(med - q25)
        upper_iqr.append(q75 - med)
        means.append(mn)
        labels.append(label)

    errors = [lower_iqr, upper_iqr]
    bars = ax.bar(
        x_pos, medians, color=_METRIC_COLORS, alpha=0.75, edgecolor="black", linewidth=0.8
    )
    ax.errorbar(
        x_pos,
        medians,
        yerr=errors,
        fmt="none",
        ecolor="black",
        capsize=5,
        capthick=1.5,
        linewidth=1.5,
    )
    ax.scatter(
        x_pos,
        means,
        color="white",
        s=80,
        zorder=5,
        edgecolors="black",
        linewidths=1.5,
        marker="o",
        label="Mean",
    )

    ax.set_xlabel("Metric", fontsize=13)
    ax.set_ylabel("Score", fontsize=13)
    ax.set_title("Metrics: Median ± IQR (with Mean)", fontsize=16, pad=15)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.legend(fontsize=11, loc="upper right")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.tight_layout()
    out_fp = os.path.join(save_dp, "metrics_ci_bar.png")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved metrics CI bar plot: %s", out_fp)
    print(f"Saved: {out_fp}")


def _hist_ax(
    ax: plt.Axes,
    vals: np.ndarray,
    title: str,
    xlabel: str,
    color: str,
    bins: int = 80,
    log_scale: bool = False,
) -> None:
    """Draw annotated histogram with median/mean lines."""
    med = np.median(vals)
    mn = np.mean(vals)
    ax.hist(vals, bins=bins, color=color, edgecolor="none", alpha=0.75, log=log_scale)
    ax.axvline(med, color="black", lw=1.5, label=f"med={med:.2f}")
    ax.axvline(mn, color="#FF6B35", lw=1.5, ls="--", label=f"mean={mn:.2f}")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13, pad=6)
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _cdf_ax(
    ax: plt.Axes,
    vals: np.ndarray,
    title: str,
    xlabel: str,
    color: str,
) -> None:
    """Draw CDF with percentile annotations."""
    sorted_vals = np.sort(vals)
    cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals)
    ax.plot(sorted_vals, cdf, color=color, lw=2.0)
    p50, p90, p99 = np.percentile(vals, [50, 90, 99])
    for pct, val, ls in [(50, p50, "--"), (90, p90, ":"), (99, p99, "-.")]:
        ax.axvline(val, color="grey", lw=0.8, ls=ls)
        ax.text(val, 0.05, f"p{pct}={val:.1f}", fontsize=8, color="grey", rotation=90, va="bottom")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel("CDF", fontsize=11)
    ax.set_title(title, fontsize=13, pad=6)
    ax.set_ylim(0, 1)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_peak_stats(
    meta: pd.DataFrame,
    pred_fp: str,
    save_dp: str,
    filter_col: str = "index_rebased",
    spec_df: pd.DataFrame | None = None,
) -> None:
    """Plot per-spectrum peak count and per-peak intensity distributions.

    Shows overall histograms and CDFs. If filter_col exists in meta, adds
    split panels for filter=True and filter=False groups.
    If spec_df is provided, also shows predicted vs ground-truth peak count comparison.

    Args:
        meta: DataFrame from load_hdf5_meta (must have start_points, end_points, spec_id).
        pred_fp: Path to predictions HDF5 (for pred_ints).
        save_dp: Output directory.
        filter_col: Optional boolean column in meta to split plots by.
        spec_df: Optional ground-truth spec_df (must have spec_id, "mz", "intensity" columns).
    """
    if "start_points" not in meta.columns or "end_points" not in meta.columns:
        logger.warning("start_points/end_points not in meta — skipping peak stats.")
        return

    os.makedirs(save_dp, exist_ok=True)

    n_peaks_pred = (meta["end_points"] - meta["start_points"]).to_numpy()

    with h5py.File(pred_fp, "r") as f:
        all_ints = f["pred_ints"][:]

    has_filter = filter_col in meta.columns
    has_gt = spec_df is not None

    if has_filter:
        mask_true = meta[filter_col].to_numpy().astype(bool)
        mask_false = ~mask_true

        # Per-spectrum peak counts for each group
        n_peaks_pred_true = n_peaks_pred[mask_true]
        n_peaks_pred_false = n_peaks_pred[mask_false]

        # Per-peak intensities mapped back through spectrum index
        starts = meta["start_points"].to_numpy()
        ends = meta["end_points"].to_numpy()
        peak_filter = np.zeros(len(all_ints), dtype=bool)
        for i in np.where(mask_true)[0]:
            peak_filter[starts[i] : ends[i]] = True
        ints_true = all_ints[peak_filter]
        ints_false = all_ints[~peak_filter]

    # Get ground truth peak counts if spec_df provided
    n_peaks_gt = None
    peak_delta = None
    over_pred = None
    under_pred = None
    exact_match = None

    if has_gt:
        if "spec_id" not in meta.columns:
            logger.warning("spec_id not in meta — cannot compare with ground truth")
        else:
            # Build a dict for fast lookup instead of iterating + pandas lookup
            spec_peaks_dict = {}
            for _, row in spec_df.iterrows():
                spec_id = int(row["spec_id"])
                peaks = row["peaks"]
                if peaks is not None and not (isinstance(peaks, float) and np.isnan(peaks)):
                    if isinstance(peaks, (list, np.ndarray)):
                        spec_peaks_dict[spec_id] = len(peaks)
                    elif isinstance(peaks, str) and peaks.strip():
                        spec_peaks_dict[spec_id] = len(peaks.strip().split("\n"))

            n_peaks_gt = np.zeros(len(meta), dtype=int)
            for idx, spec_id in meta["spec_id"].items():
                n_peaks_gt[idx] = spec_peaks_dict.get(int(spec_id), 0)

            peak_delta = n_peaks_pred - n_peaks_gt
            over_pred = (peak_delta > 0).sum()
            under_pred = (peak_delta < 0).sum()
            exact_match = (peak_delta == 0).sum()

    n_cols = 4 if has_gt else (3 if has_filter else 2)
    colors_all = ["#4878CF", "#4878CF"]
    colors_true = ["#6ACC65", "#6ACC65"]
    colors_false = ["#D65F5F", "#D65F5F"]
    colors_gt = ["#E8A83C", "#E8A83C"]

    fig, axes = plt.subplots(2, n_cols, figsize=(6 * n_cols, 10), squeeze=False)
    fig.subplots_adjust(hspace=0.45, wspace=0.35)

    # --- Row 0: peak count ---
    _hist_ax(axes[0, 0], n_peaks_pred, "Peak count — Predicted", "# peaks", colors_all[0])
    _cdf_ax(axes[0, 1], n_peaks_pred, "Peak count CDF — Predicted", "# peaks", colors_all[0])

    if has_gt:
        # Ground truth peak count
        _hist_ax(axes[0, 2], n_peaks_gt, "Peak count — Ground Truth", "# peaks", colors_gt[0])
        # Over/Under prediction delta
        ax = axes[0, 3]
        ax.hist(peak_delta, bins=80, color="#7030A0", edgecolor="none", alpha=0.7)
        ax.axvline(0, color="red", linestyle="--", linewidth=2, label="Perfect prediction")
        ax.set_xlabel("Predicted - Ground Truth (# peaks)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Frequency", fontsize=11, fontweight="bold")
        stats_text = (
            f"Over-predicted: {over_pred} ({over_pred / len(meta) * 100:.1f}%)\n"
            f"Under-predicted: {under_pred} ({under_pred / len(meta) * 100:.1f}%)\n"
            f"Exact match: {exact_match} ({exact_match / len(meta) * 100:.1f}%)\n"
            f"Mean delta: {peak_delta.mean():.2f}"
        )
        ax.text(
            0.98,
            0.97,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
        )
        ax.set_title("Prediction Error Distribution", fontsize=12, fontweight="bold")
        ax.legend(loc="upper left")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.2)

    elif has_filter:
        _hist_ax(
            axes[0, 2],
            n_peaks_pred_true if mask_true.sum() > 0 else np.array([0]),
            f"Peak count  {filter_col}=T ({mask_true.sum():,})\nvs =F ({mask_false.sum():,})",
            "# peaks",
            colors_true[0],
        )
        axes[0, 2].hist(
            n_peaks_pred_false, bins=80, color=colors_false[0], edgecolor="none", alpha=0.55
        )

    # --- Row 1: intensity ---
    _hist_ax(
        axes[1, 0], all_ints, "Intensity — all peaks", "Intensity", colors_all[1], log_scale=True
    )
    _cdf_ax(axes[1, 1], all_ints, "Intensity CDF — all peaks", "Intensity", colors_all[1])

    if has_filter and not has_gt:
        _hist_ax(
            axes[1, 2],
            ints_true if len(ints_true) > 0 else np.array([0]),
            f"Intensity  {filter_col}=T vs =F\n(log scale)",
            "Intensity",
            colors_true[1],
            log_scale=True,
        )
        axes[1, 2].hist(
            ints_false, bins=80, color=colors_false[1], edgecolor="none", alpha=0.55, log=True
        )
    elif has_gt:
        # Show predicted vs GT peak count comparison scatter
        ax = axes[1, 2]
        ax.scatter(n_peaks_gt, n_peaks_pred, alpha=0.5, s=20, color="#1f77b4")
        # Add perfect prediction line
        max_peaks = max(n_peaks_gt.max(), n_peaks_pred.max())
        ax.plot([0, max_peaks], [0, max_peaks], "r--", linewidth=2, label="Perfect prediction")
        ax.set_xlabel("Ground Truth Peak Count", fontsize=11, fontweight="bold")
        ax.set_ylabel("Predicted Peak Count", fontsize=11, fontweight="bold")
        ax.set_title("Predicted vs Ground Truth", fontsize=12, fontweight="bold")
        ax.legend()
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(alpha=0.2)

        # Show intensity distribution for second row
        _hist_ax(
            axes[1, 3],
            all_ints,
            "Intensity — Predicted peaks\n(log scale)",
            "Intensity",
            colors_all[1],
            log_scale=True,
        )

    filter_note = f" (split by {filter_col})" if has_filter else ""
    gt_note = " + Ground Truth Comparison" if has_gt else ""
    fig.suptitle(f"Peak Count & Intensity Stats{filter_note}{gt_note}", fontsize=16, y=1.01)
    out_fp = os.path.join(save_dp, "peak_stats.png")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved peak stats plot: %s", out_fp)
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

        ax.vlines([pos], [q1], [q3], color=_MED_COLOR, linewidth=2, zorder=4)
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

    # Legend proxy handles
    ax.vlines([], [], [], color=_MED_COLOR, linewidth=2, label="IQR (tol range)")
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
        ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=9)
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
                fontsize=7,
                color="grey",
                transform=ax.get_xaxis_transform(),
            )
        ax.set_xticklabels([])
    ax.set_xlim(0.5, n_total + 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel(score_col, fontsize=10)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=10)
    ax.set_title(title, fontsize=12, pad=12)
    for thresh, ls in [(0.5, "--"), (0.7, ":"), (0.9, "--")]:
        ax.axhline(thresh, color="grey", lw=0.7, ls=ls, alpha=0.55)
        ax.text(n_total + 0.55, thresh, f"{thresh}", va="center", fontsize=8, color="grey")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(0.8)
    ax.spines["bottom"].set_linewidth(0.8)
    ax.grid(axis="y", alpha=0.2, linestyle="-", linewidth=0.5)
    ax.legend(fontsize=8, loc="lower left", ncol=3, framealpha=0.95, edgecolor="grey")


def _plot_ce_breakdown_grid(
    joined: pd.DataFrame,
    score_col: str,
    save_dp: str,
) -> None:
    """Plot CE breakdown bars as grid: QTOF (ACE) in row 0, FT (NCE) in row 1.

    Creates a single figure with 2 rows (QTOF and FT) and columns for each adduct.
    Both rows show x-axis with their respective units.

    Args:
        joined: DataFrame with score_col, inst_type, prec_type, ace, nce.
        score_col: Score column to plot.
        save_dp: Output directory.
    """
    df = joined[[score_col, "inst_type", "prec_type", "ace", "nce"]].copy()

    # Get adducts across all inst types
    adducts_all = (
        df.groupby("prec_type")[score_col]
        .count()
        .loc[lambda s: s >= _MIN_BIN_N]
        .sort_values(ascending=False)
        .index.tolist()
    )
    if not adducts_all:
        logger.warning("No adducts with enough data")
        return

    n_adducts = len(adducts_all)
    n_rows = 2  # QTOF and FT

    fig, axes = plt.subplots(n_rows, n_adducts, figsize=(6 * n_adducts, 5 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")

    # Row 0: QTOF (ACE), Row 1: FT (NCE)
    for row_idx, (inst_type, ce_col, ce_cap, color) in enumerate(
        [
            ("QTOF", "ace", 50, "#5B9BD5"),  # Blue
            ("FT", "nce", 70, "#70AD47"),  # Green
        ]
    ):
        for col_idx, adduct in enumerate(adducts_all):
            ax = axes[row_idx, col_idx]

            sub = df[(df["prec_type"] == adduct) & (df["inst_type"] == inst_type)].copy()
            if ce_col == "ace":
                sub = sub[sub[ce_col] > 0]
            sub = sub[sub[ce_col] < ce_cap]

            if len(sub) < _MIN_BIN_N:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, 160)
                ax.set_ylim(0, 1.05)
                ax.set_title(f"{adduct}", fontsize=12)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                continue

            ce_max = sub[ce_col].max() + 10
            bin_edges = list(range(0, int(ce_max), 10))

            means = []
            lows = []
            highs = []
            bin_centers = []

            for lo in bin_edges[:-1]:
                hi = lo + 10
                mask = (sub[ce_col] >= lo) & (sub[ce_col] < hi)
                vals = sub.loc[mask, score_col].dropna().to_numpy()
                if len(vals) >= _MIN_BIN_N:
                    mean = np.mean(vals)
                    lo_ci, hi_ci = _ci_95(vals)
                    means.append(mean)
                    lows.append(lo_ci)
                    highs.append(hi_ci)
                    bin_centers.append((lo + hi) / 2)

            if len(means) == 0:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0, 160)
                ax.set_ylim(0, 1.05)
                ax.set_title(f"{adduct}", fontsize=12)
                continue

            # Ensure errors are non-negative
            errors = [
                np.maximum(np.array(means) - np.array(lows), 0),
                np.maximum(np.array(highs) - np.array(means), 0),
            ]

            # Plot bars and error bars
            bin_width = 8
            ax.bar(
                bin_centers,
                means,
                width=bin_width,
                color=color,
                alpha=0.7,
                edgecolor="black",
                linewidth=0.8,
            )
            ax.errorbar(
                bin_centers,
                means,
                yerr=errors,
                fmt="none",
                ecolor="black",
                capsize=4,
                linewidth=1.5,
            )
            ax.scatter(
                bin_centers, means, color="red", s=50, zorder=5, edgecolors="darkred", linewidths=1
            )

            ax.set_xlim(-5, 165)
            ax.set_ylim(0, 1.05)
            ax.set_ylabel("Score", fontsize=10)
            ax.set_title(f"{adduct}", fontsize=12)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            ax.grid(axis="y", alpha=0.2, linestyle="-")

            # Show x-axis on both rows
            ce_unit = "eV" if inst_type == "QTOF" else "NCE"
            ax.set_xlabel(f"Collision Energy ({ce_unit})", fontsize=10)

    # Add row labels on the left
    fig.text(0.02, 0.75, "QTOF", fontsize=11, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.25, "FT", fontsize=11, fontweight="bold", rotation=90, va="center")

    fig.suptitle(
        f"{score_col} — Collision Energy Breakdown (95% CI)", fontsize=13, fontweight="bold"
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.3, left=0.08)

    out_fp = os.path.join(save_dp, "ce_breakdown_bar_all.png")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved CE breakdown grid: {out_fp}")
    print(f"Saved: {out_fp}")


def _plot_ce_breakdown_violin_inst(
    df: pd.DataFrame,
    inst_type: str,
    ce_col: str,
    ce_cap: float,
    color: str,
    adducts_all: list[str],
    score_col: str,
    save_dp: str,
) -> None:
    """Plot CE breakdown violins for one instrument type (QTOF or FT) with adducts as columns.

    Args:
        df: DataFrame with score_col, inst_type, prec_type, ce_col.
        inst_type: Instrument type ('QTOF' or 'FT').
        ce_col: CE column to bin by ('ace' or 'nce').
        ce_cap: Upper limit for CE values.
        color: Violin fill colour.
        adducts_all: List of adduct types to plot as columns.
        score_col: Score column to plot.
        save_dp: Output directory.
    """
    # Compute global CE bins for this instrument type
    inst_full = df[(df["inst_type"] == inst_type) & df[ce_col].notna()]
    if ce_col == "ace":
        inst_full = inst_full[inst_full[ce_col] > 0]
    inst_full = inst_full[inst_full[ce_col] < ce_cap]

    if len(inst_full) == 0:
        logger.warning(f"No data for {inst_type}")
        return

    ce_max = inst_full[ce_col].max() + 10
    global_labels, _ = _make_ce_bins(inst_full, ce_col, 10, ce_max, score_col)
    global_lo = [int(lbl.split("–")[0]) for lbl in global_labels]
    n_bins = len(global_labels)

    n_cols = len(adducts_all)
    fig_w = max(36, n_bins * 4.5 + 4)
    fig_h = 7

    fig, axes = plt.subplots(1, n_cols, figsize=(fig_w / n_cols * n_cols, fig_h), squeeze=False)
    fig.patch.set_facecolor("white")

    for col_idx, adduct in enumerate(adducts_all):
        ax = axes[0, col_idx]

        sub = df[(df["prec_type"] == adduct) & (df["inst_type"] == inst_type)].copy()
        if ce_col == "ace":
            sub = sub[sub[ce_col] > 0]
        sub = sub[sub[ce_col] < ce_cap]

        if len(sub) < _MIN_BIN_N:
            ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
            ax.set_xlim(0.5, len(global_labels) + 0.5)
            ax.set_ylim(0, 1.05)
            ax.set_title(f"{adduct}", fontsize=12, fontweight="bold")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            continue

        # Align per-adduct arrays to global positions
        arrays_aligned: list[np.ndarray | None] = []
        for lo in global_lo:
            hi = lo + 10
            mask = (sub[ce_col] >= lo) & (sub[ce_col] < hi)
            vals = sub.loc[mask, score_col].dropna().to_numpy()
            arrays_aligned.append(vals if len(vals) >= _MIN_BIN_N else None)

        ce_unit = "eV" if inst_type == "QTOF" else "NCE"
        _violin_ax(
            ax,
            global_labels,
            arrays_aligned,
            title=f"{adduct}  [n={len(sub):,}]",
            xlabel=f"Collision Energy ({ce_unit})",
            color=color,
            score_col=score_col,
            show_xticklabels=True,
            show_xlabel=True,
        )

    ce_unit = "eV" if inst_type == "QTOF" else "NCE"
    fig.suptitle(
        f"{score_col} — {inst_type} Collision Energy Breakdown (Violin, {ce_unit})",
        fontsize=16,
        fontweight="bold",
        y=1.005,
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.3)

    fname = f"ce_breakdown_violin_{inst_type.lower()}.png"
    out_fp = os.path.join(save_dp, fname)
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved CE breakdown violin plot for {inst_type}: {out_fp}")
    print(f"Saved: {out_fp}")


def _plot_ce_breakdown_grid_violin(
    joined: pd.DataFrame,
    score_col: str,
    save_dp: str,
) -> None:
    """Plot CE breakdown violins as grid: QTOF (ACE) in row 0, FT (NCE) in row 1.

    Creates a single figure with 2 rows (QTOF and FT) and columns for each adduct.
    Both rows show x-axis with their respective units.

    Args:
        joined: DataFrame with score_col, inst_type, prec_type, ace, nce.
        score_col: Score column to plot.
        save_dp: Output directory.
    """
    df = joined[[score_col, "inst_type", "prec_type", "ace", "nce"]].copy()

    # Get adducts across all inst types
    adducts_all = (
        df.groupby("prec_type")[score_col]
        .count()
        .loc[lambda s: s >= _MIN_BIN_N]
        .sort_values(ascending=False)
        .index.tolist()
    )
    if not adducts_all:
        logger.warning("No adducts with enough data")
        return

    n_adducts = len(adducts_all)
    n_rows = 2  # QTOF and FT

    # Compute max bins to determine figure width
    inst_full_qtof = df[(df["inst_type"] == "QTOF") & df["ace"].notna()]
    inst_full_qtof = inst_full_qtof[inst_full_qtof["ace"] > 0]
    inst_full_qtof = inst_full_qtof[inst_full_qtof["ace"] < 50]
    qtof_ce_max = inst_full_qtof["ace"].max() + 10 if len(inst_full_qtof) > 0 else 50
    qtof_labels, _ = (
        _make_ce_bins(inst_full_qtof, "ace", 10, qtof_ce_max, score_col)
        if len(inst_full_qtof) > 0
        else ([], [])
    )
    qtof_n_bins = len(qtof_labels)

    inst_full_ft = df[(df["inst_type"] == "FT") & df["nce"].notna()]
    inst_full_ft = inst_full_ft[inst_full_ft["nce"] < 70]
    ft_ce_max = inst_full_ft["nce"].max() + 10 if len(inst_full_ft) > 0 else 70
    ft_labels, _ = (
        _make_ce_bins(inst_full_ft, "nce", 10, ft_ce_max, score_col)
        if len(inst_full_ft) > 0
        else ([], [])
    )
    ft_n_bins = len(ft_labels)

    n_bins = max(qtof_n_bins, ft_n_bins)
    fig_w = max(36, n_bins * 4.5 + 4)
    fig_h = 7 * n_rows

    fig, axes = plt.subplots(n_rows, n_adducts, figsize=(6 * n_adducts, 5 * n_rows), squeeze=False)
    fig.patch.set_facecolor("white")

    # Row 0: QTOF (ACE), Row 1: FT (NCE)
    for row_idx, (inst_type, ce_col, ce_cap, color) in enumerate(
        [
            ("QTOF", "ace", 50, "#5B9BD5"),  # Blue
            ("FT", "nce", 70, "#70AD47"),  # Green
        ]
    ):
        # Compute global CE bins for this instrument type
        inst_full = df[(df["inst_type"] == inst_type) & df[ce_col].notna()]
        if ce_col == "ace":
            inst_full = inst_full[inst_full[ce_col] > 0]
        inst_full = inst_full[inst_full[ce_col] < ce_cap]

        if len(inst_full) == 0:
            for col_idx in range(n_adducts):
                ax = axes[row_idx, col_idx]
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.axis("off")
            continue

        ce_max = inst_full[ce_col].max() + 10
        global_labels, _ = _make_ce_bins(inst_full, ce_col, 10, ce_max, score_col)
        global_lo = [int(lbl.split("–")[0]) for lbl in global_labels]

        for col_idx, adduct in enumerate(adducts_all):
            ax = axes[row_idx, col_idx]

            sub = df[(df["prec_type"] == adduct) & (df["inst_type"] == inst_type)].copy()
            if ce_col == "ace":
                sub = sub[sub[ce_col] > 0]
            sub = sub[sub[ce_col] < ce_cap]

            if len(sub) < _MIN_BIN_N:
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes)
                ax.set_xlim(0.5, len(global_labels) + 0.5)
                ax.set_ylim(0, 1.05)
                ax.set_title(f"{adduct}", fontsize=12, fontweight="bold")
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                continue

            # Align per-adduct arrays to global positions
            arrays_aligned: list[np.ndarray | None] = []
            for lo in global_lo:
                hi = lo + 10
                mask = (sub[ce_col] >= lo) & (sub[ce_col] < hi)
                vals = sub.loc[mask, score_col].dropna().to_numpy()
                arrays_aligned.append(vals if len(vals) >= _MIN_BIN_N else None)

            ce_unit = "eV" if inst_type == "QTOF" else "NCE"
            _violin_ax(
                ax,
                global_labels,
                arrays_aligned,
                title=f"{adduct}",
                xlabel=f"Collision Energy ({ce_unit})",
                color=color,
                score_col=score_col,
                show_xticklabels=True,
                show_xlabel=True,
            )

    # Add row labels on the left
    fig.text(0.02, 0.75, "QTOF", fontsize=11, fontweight="bold", rotation=90, va="center")
    fig.text(0.02, 0.25, "FT", fontsize=11, fontweight="bold", rotation=90, va="center")

    fig.suptitle(
        f"{score_col} — Collision Energy Breakdown (Violin)", fontsize=13, fontweight="bold"
    )
    fig.subplots_adjust(hspace=0.35, wspace=0.3, left=0.08)

    out_fp = os.path.join(save_dp, "ce_breakdown_violin_all.png")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved CE breakdown violin grid: {out_fp}")
    print(f"Saved: {out_fp}")


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

    fig_w = max(36, n_bins * 4.5 + 4)
    fig_h = max(n_rows * 7.5, 8)

    fig, axes = plt.subplots(n_rows, 1, figsize=(fig_w, fig_h), squeeze=False, sharex=True)
    fig.patch.set_facecolor("white")

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
        f"{score_col} — {inst_type} by Adduct & {ce_label}",
        fontsize=20,
        fontweight="bold",
        y=1.005,
    )
    fig.subplots_adjust(hspace=0.45)
    fname = f"ce_breakdown_{inst_type.lower()}_violin.png"
    out_fp = os.path.join(save_dp, fname)
    fig.savefig(out_fp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_fp)
    print(f"Saved: {out_fp}")


def plot_ce_breakdown(
    meta: pd.DataFrame,
    spec_df: pd.DataFrame,
    score_col: str,
    save_dp: str,
) -> None:
    """Plot CE breakdown bar charts with 95% CI, merging QTOF (ACE) and FT (NCE).

    Creates a single figure showing bars for each CE bin with 95% confidence interval
    error bars and mean points, organized in a grid by adduct type.
    QTOF bars are blue, FT bars are green.

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

    _plot_ce_breakdown_grid(joined, score_col, save_dp)
    _plot_ce_breakdown_grid_violin(joined, score_col, save_dp)


# ---------------------------------------------------------------------------
# Bar plots with 95% confidence intervals
# ---------------------------------------------------------------------------


def _ci_95(arr: np.ndarray) -> tuple[float, float]:
    """Compute 95% confidence interval for the mean.

    Args:
        arr: Array of values.

    Returns:
        Tuple of (lower_bound, upper_bound) for the 95% CI.
    """
    if len(arr) < 2:
        return 0.0, 0.0
    mean = np.mean(arr)
    se = stats.sem(arr)
    margin = se * stats.t.ppf((1 + 0.95) / 2, len(arr) - 1)
    return mean - margin, mean + margin


def _bar_ax(
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
    """Draw bar plot with 95% CI error bars.

    Args:
        ax: Axes to draw on.
        global_labels: Bin label strings for every position (including empty bins).
        arrays_aligned: Score array per bin, or None for bins with < _MIN_BIN_N spectra.
        title: Axes title.
        xlabel: X-axis label (shown only when show_xlabel=True).
        color: Bar fill colour.
        score_col: Score column name (for y-axis label).
        show_xticklabels: Whether to draw x-tick labels (False for non-bottom rows).
        show_xlabel: Whether to draw the x-axis label (False for non-bottom rows).
    """
    all_positions = list(range(1, len(global_labels) + 1))
    n_total = len(global_labels)

    means = []
    lower_cis = []
    upper_cis = []
    bin_counts: dict[int, int] = {}

    for pos, arr in zip(all_positions, arrays_aligned):
        if arr is None or len(arr) < _MIN_BIN_N:
            means.append(0)
            lower_cis.append(0)
            upper_cis.append(0)
            continue

        mean = np.mean(arr)
        lo, hi = _ci_95(arr)
        error_lower = mean - lo
        error_upper = hi - mean

        means.append(mean)
        lower_cis.append(error_lower)
        upper_cis.append(error_upper)
        bin_counts[pos] = len(arr)

    # Plot bars only for non-empty bins
    valid_positions = [
        pos
        for pos, arr in zip(all_positions, arrays_aligned)
        if arr is not None and len(arr) >= _MIN_BIN_N
    ]
    valid_means = [means[pos - 1] for pos in valid_positions]
    valid_lower = [lower_cis[pos - 1] for pos in valid_positions]
    valid_upper = [upper_cis[pos - 1] for pos in valid_positions]

    if valid_positions:
        ax.bar(
            valid_positions,
            valid_means,
            yerr=[valid_lower, valid_upper],
            capsize=5,
            color=color,
            alpha=0.75,
            edgecolor="black",
            linewidth=1.2,
            error_kw={"elinewidth": 2, "capthick": 2, "ecolor": "black"},
        )

        # Add value labels on bars
        for pos, mean in zip(valid_positions, valid_means):
            ax.text(
                pos,
                mean + 0.03,
                f"{mean:.2f}",
                ha="center",
                va="bottom",
                fontsize=11,
                fontweight="bold",
            )

    ax.set_xticks(all_positions)
    if show_xticklabels:
        tick_labels = [
            f"{lbl}\nn={bin_counts[p]:,}" if p in bin_counts else lbl
            for p, lbl in zip(all_positions, global_labels)
        ]
        ax.set_xticklabels(tick_labels, rotation=40, ha="right", fontsize=12)
    else:
        ax.set_xticklabels([])
        # Show n= as small text inside the plot top for non-bottom rows
        for pos in valid_positions:
            if pos in bin_counts:
                ax.text(
                    pos,
                    1.02,
                    f"n={bin_counts[pos]:,}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="grey",
                    transform=ax.get_xaxis_transform(),
                )

    ax.set_xlim(0.5, n_total + 0.5)
    ax.set_ylim(-0.05, 1.12)
    ax.set_ylabel(score_col, fontsize=13)
    if show_xlabel:
        ax.set_xlabel(xlabel, fontsize=13)
    ax.set_title(title, fontsize=17, pad=12)

    # Add threshold reference lines
    for thresh, ls in [(0.5, "--"), (0.7, ":"), (0.9, "--")]:
        ax.axhline(thresh, color="grey", lw=0.7, ls=ls, alpha=0.55)
        ax.text(n_total + 0.55, thresh, f"{thresh}", va="center", fontsize=10, color="grey")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    ax.legend(
        [mpatches.Patch(facecolor=color, edgecolor="black", label="Mean ± 95% CI")],
        ["Mean ± 95% CI"],
        fontsize=11,
        loc="upper left",
    )


def _plot_inst_adduct_grid_bar(
    joined: pd.DataFrame,
    inst_type: str,
    ce_col: str,
    ce_label: str,
    color: str,
    score_col: str,
    save_dp: str,
    ce_cap: float | None = None,
) -> None:
    """Produce one bar plot figure for a given instrument type, one row per adduct.

    Args:
        joined: Merged DataFrame with spec metadata and scores.
        inst_type: Instrument type string (e.g. 'FT', 'QTOF').
        ce_col: CE column to bin by ('ace' or 'nce').
        ce_label: Human-readable CE axis label.
        color: Bar fill colour.
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
    fig_h = n_rows * 6.5

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
        _bar_ax(
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
        f"{score_col} — {inst_type} by adduct + {ce_label} (Mean ± 95% CI)",
        fontsize=18,
        y=1.005,
    )
    fig.subplots_adjust(hspace=0.45)
    fname = f"ce_breakdown_{inst_type.lower()}_bar_ci.png"
    out_fp = os.path.join(save_dp, fname)
    fig.savefig(out_fp, dpi=200, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved: %s", out_fp)
    print(f"Saved: {out_fp}")


def plot_ce_breakdown_bar(
    meta: pd.DataFrame,
    spec_df: pd.DataFrame,
    score_col: str,
    save_dp: str,
) -> None:
    """Plot bar graphs with 95% CI for metrics, binned by CE, grouped by adduct + inst type.

    Produces two figures: one for QTOF (ACE 10 eV bins) and one for FT (NCE 10% bins).
    Each figure has one row per adduct type. Error bars show 95% confidence intervals
    of the mean.

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

    _plot_inst_adduct_grid_bar(
        joined, "QTOF", "ace", "ACE (eV)", "#4878CF", score_col, save_dp, ce_cap=50
    )
    _plot_inst_adduct_grid_bar(
        joined, "FT", "nce", "NCE (%)", "#6ACC65", score_col, save_dp, ce_cap=70
    )


def plot_all_metrics_ce_summary_bar(
    meta: pd.DataFrame,
    spec_df: pd.DataFrame,
    save_dp: str,
) -> None:
    """Plot summary bar charts with all metrics grouped by CE bins and instrument type.

    Creates one figure with two subplots (QTOF and FT), each showing bars for all
    available metrics grouped by CE bins. Error bars show 95% CI around the mean.

    Args:
        meta: Per-spectrum metadata DataFrame from load_hdf5_meta.
        spec_df: Ground-truth spec_df with columns [spec_id, inst_type, prec_type, ace, nce].
        save_dp: Directory to save figures.
    """
    available_metrics = [k for k, _ in _METRICS if k in meta.columns]
    metric_labels = [label for k, label in _METRICS if k in meta.columns]
    if not available_metrics:
        logger.warning("No score metrics found in HDF5.")
        return

    os.makedirs(save_dp, exist_ok=True)

    joined = meta[["spec_id"] + available_metrics].merge(
        spec_df[["spec_id", "inst_type", "ace", "nce"]],
        on="spec_id",
        how="left",
    )

    fig, axes = plt.subplots(2, 1, figsize=(max(12, len(available_metrics) * 2), 12), squeeze=False)

    for row_idx, (inst_type, ce_col, ce_label, ce_max, ce_cap) in enumerate(
        [
            ("QTOF", "ace", "ACE (eV)", 50, 50),
            ("FT", "nce", "NCE (%)", 70, 70),
        ]
    ):
        ax = axes[row_idx, 0]
        inst_df = joined[(joined["inst_type"] == inst_type) & joined[ce_col].notna()].copy()
        if ce_col == "ace":
            inst_df = inst_df[inst_df[ce_col] > 0]
        inst_df = inst_df[inst_df[ce_col] < ce_cap]

        if len(inst_df) == 0:
            logger.warning(f"No data for {inst_type}")
            ax.text(
                0.5,
                0.5,
                f"No data for {inst_type}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            continue

        # Define CE bins
        bin_edges = list(range(0, int(ce_max) + 10, 10))
        bin_labels = [f"{lo}-{hi}" for lo, hi in zip(bin_edges[:-1], bin_edges[1:])]
        inst_df["ce_bin"] = pd.cut(inst_df[ce_col], bins=bin_edges, labels=bin_labels, right=False)

        # Prepare data: for each metric, for each CE bin, compute mean and CI
        n_metrics = len(available_metrics)
        n_bins = len(bin_labels)
        bar_width = 0.8 / n_bins

        colors_ce = plt.cm.Blues(np.linspace(0.4, 0.9, n_bins))

        x_pos = np.arange(n_metrics)
        for bin_idx, (bin_label, ce_color) in enumerate(zip(bin_labels, colors_ce)):
            bin_data = inst_df[inst_df["ce_bin"] == bin_label]
            if len(bin_data) == 0:
                continue

            means = []
            medians = []
            lower_cis = []
            upper_cis = []
            for metric_col in available_metrics:
                vals = bin_data[metric_col].dropna().to_numpy()
                if len(vals) < 2:
                    means.append(0)
                    medians.append(0)
                    lower_cis.append(0)
                    upper_cis.append(0)
                else:
                    mean = np.mean(vals)
                    med = np.median(vals)
                    lo, hi = _ci_95(vals)
                    means.append(mean)
                    medians.append(med)
                    lower_cis.append(mean - lo)
                    upper_cis.append(hi - mean)

            errors = [lower_cis, upper_cis]
            offset = (bin_idx - n_bins / 2 + 0.5) * bar_width
            ax.bar(
                x_pos + offset,
                means,
                bar_width,
                label=bin_label,
                color=[ce_color] * n_metrics,
                alpha=0.75,
                edgecolor=_METRIC_COLORS,
                linewidth=2.0,
            )
            ax.errorbar(
                x_pos + offset,
                means,
                yerr=errors,
                fmt="none",
                ecolor="black",
                capsize=2,
                linewidth=0.8,
                alpha=0.7,
            )
            # Overlay median points
            ax.scatter(
                x_pos + offset,
                medians,
                color="white",
                s=30,
                zorder=5,
                edgecolors="black",
                linewidths=1.0,
                marker="D",
            )

        ax.set_xlabel("Metric", fontsize=12)
        ax.set_ylabel("Score", fontsize=12)
        ax.set_title(f"{inst_type} by {ce_label}", fontsize=14, pad=10)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(metric_labels, rotation=45, ha="right", fontsize=10)
        ax.set_ylim(0, 1.1)
        ax.legend(fontsize=9, loc="upper right", ncol=2, title="CE Bin")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("All Metrics by Collision Energy", fontsize=16, y=1.00)
    fig.tight_layout()
    out_fp = os.path.join(save_dp, "all_metrics_ce_summary.png")
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved CE summary bar plot: {out_fp}")
    print(f"Saved: {out_fp}")


def plot_all_metrics_ce_breakdown_bar(
    meta: pd.DataFrame,
    spec_df: pd.DataFrame,
    save_dp: str,
) -> None:
    """Plot CE breakdown bar charts with median ± IQR for all available metrics.

    For each metric, generates bar charts binned by CE (QTOF/FT).

    Args:
        meta: Per-spectrum metadata DataFrame from load_hdf5_meta.
        spec_df: Ground-truth spec_df with columns [spec_id, inst_type, prec_type, ace, nce].
        save_dp: Directory to save figures.
    """
    available_metrics = [k for k, _ in _METRICS if k in meta.columns]
    if not available_metrics:
        logger.warning("No score metrics found in HDF5.")
        return

    for score_col in available_metrics:
        logger.info(f"Plotting CE breakdown for {score_col}...")
        plot_ce_breakdown_bar(meta, spec_df, score_col, save_dp)


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
        choices=["summary", "ce", "ce_bar", "spectra", "all", "plot_all", "peak_stats"],
        default="summary",
        help="What to plot (ce/ce_bar/spectra/all require --spec_df_fp; ce=violin, ce_bar=bar+95CI; plot_all=all stats; peak_stats=peak count+intensity)",
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
    parser.add_argument(
        "--filter_col",
        type=str,
        default="index_rebased",
        help="Boolean column in HDF5 to split peak stats plots by (used in plot_all / peak_stats modes)",
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

    if args.mode == "peak_stats":
        plot_peak_stats(meta, args.pred_fp, save_dp, filter_col=args.filter_col)

    if args.mode == "plot_all":
        # Load spec_df for peak stats and CE breakdown
        spec_df_for_peaks = None
        if args.spec_df_fp is not None:
            logger.info("Loading spec_df from %s", args.spec_df_fp)
            spec_df_for_peaks = pd.read_pickle(args.spec_df_fp)

        # Plot peak stats with ground truth comparison
        plot_peak_stats(meta, args.pred_fp, save_dp, filter_col=args.filter_col, spec_df=spec_df_for_peaks)

        # Plot CE breakdown if spec_df available
        if spec_df_for_peaks is not None:
            plot_ce_breakdown(
                meta=meta,
                spec_df=spec_df_for_peaks,
                score_col=args.score_col,
                save_dp=save_dp,
            )


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

    if args.mode == "ce_bar":
        if args.spec_df_fp is None:
            raise ValueError("--spec_df_fp is required for 'ce_bar' mode")
        logger.info("Loading spec_df from %s", args.spec_df_fp)
        spec_df = pd.read_pickle(args.spec_df_fp)
        plot_ce_breakdown_bar(
            meta=meta,
            spec_df=spec_df,
            score_col=args.score_col,
            save_dp=save_dp,
        )


if __name__ == "__main__":
    main()
