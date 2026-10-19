"""Recompute QTOF cosine-Hungarian similarity at 20 ppm and compare to stored 10 ppm results.

Loads all 9 CV folds of NIST23 predictions, filters to QTOF spectra, then recomputes
cos_hun at {10, 15, 20, 30} ppm using the stored pred_mzs/pred_ints against the true
peaks from spec_df.  Reports median/mean comparison and per-ACE-bin breakdown.

Usage:
    python scripts/debug/oracle/qtof_20ppm_reeval.py
"""

from __future__ import annotations

import ast
import logging
import pickle
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

DATA_ROOT = Path("data")
TOLERANCE_MIN_MZ = 200.0  # Da floor for relative tolerance
PPM_TOLERANCES = [10, 15, 20, 30]  # ppm values to evaluate


# ---------------------------------------------------------------------------
# Peak parsing helpers
# ---------------------------------------------------------------------------

def parse_peaks(peaks) -> tuple[np.ndarray, np.ndarray]:
    """Parse stored peaks into (mzs, intensities) float64 arrays."""
    if peaks is None:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if isinstance(peaks, str):
        try:
            peaks = ast.literal_eval(peaks)
        except Exception:
            return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    if not peaks:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    arr = np.array(peaks, dtype=np.float64)
    return arr[:, 0], arr[:, 1]


# ---------------------------------------------------------------------------
# Cosine-Hungarian at arbitrary ppm tolerance
# ---------------------------------------------------------------------------

def cos_hun_ppm(
    true_mzs: np.ndarray,
    true_ints: np.ndarray,
    pred_mzs: np.ndarray,
    pred_ints: np.ndarray,
    ppm: float,
) -> float:
    """Compute cosine-Hungarian similarity using a relative ppm tolerance.

    Args:
        true_mzs: Measured peak m/z values of shape (N,).
        true_ints: Measured peak intensities of shape (N,).
        pred_mzs: Predicted peak m/z values of shape (M,).
        pred_ints: Predicted peak intensities of shape (M,).
        ppm: Relative tolerance in ppm.

    Returns:
        Cosine-Hungarian score in [0, 1].
    """
    if len(true_mzs) == 0 or len(pred_mzs) == 0:
        return 0.0

    tol_ratio = ppm * 1e-6
    # Relative tolerance with floor at TOLERANCE_MIN_MZ
    divisors = np.maximum(true_mzs, TOLERANCE_MIN_MZ)
    diff = np.abs(true_mzs[:, None] - pred_mzs[None, :])  # (N, M)
    match_mask = diff / divisors[:, None] < tol_ratio  # (N, M)

    true_match = match_mask.any(axis=1)
    pred_match = match_mask.any(axis=0)
    if not true_match.any():
        return 0.0

    # L2-normalise intensities
    t_norm = np.linalg.norm(true_ints)
    p_norm = np.linalg.norm(pred_ints)
    if t_norm == 0 or p_norm == 0:
        return 0.0
    t = true_ints / t_norm
    p = pred_ints / p_norm

    # Score matrix restricted to matchable peaks
    t_sub = t[true_match]
    p_sub = p[pred_match]
    sub_mask = match_mask[np.ix_(true_match, pred_match)]
    score = sub_mask * np.outer(t_sub, p_sub)

    row_idx, col_idx = linear_sum_assignment(-score)
    return float(score[row_idx, col_idx].sum())


def cos_sim_binned(
    true_mzs: np.ndarray,
    true_ints: np.ndarray,
    pred_mzs: np.ndarray,
    pred_ints: np.ndarray,
    bin_size: float = 0.01,
) -> float:
    """Binned cosine similarity (reproduces stored cos_sim_0.01 metric).

    Each m/z is rounded to the nearest bin of width bin_size.  Intensities
    falling in the same bin are summed, then the standard cosine is computed
    over the union of occupied bins.

    Args:
        true_mzs: Measured peak m/z values.
        true_ints: Measured peak intensities.
        pred_mzs: Predicted peak m/z values.
        pred_ints: Predicted peak intensities.
        bin_size: Bin width in Da. Defaults to 0.01.

    Returns:
        Binned cosine similarity in [0, 1].
    """
    if len(true_mzs) == 0 or len(pred_mzs) == 0:
        return 0.0
    true_bins = np.round(true_mzs / bin_size).astype(np.int64)
    pred_bins = np.round(pred_mzs / bin_size).astype(np.int64)
    all_bins = np.union1d(true_bins, pred_bins)
    n = len(all_bins)
    bin_idx = {b: i for i, b in enumerate(all_bins)}
    t_vec = np.zeros(n)
    p_vec = np.zeros(n)
    for b, intensity in zip(true_bins, true_ints):
        t_vec[bin_idx[b]] += intensity
    for b, intensity in zip(pred_bins, pred_ints):
        p_vec[bin_idx[b]] += intensity
    t_norm = np.linalg.norm(t_vec)
    p_norm = np.linalg.norm(p_vec)
    if t_norm == 0 or p_norm == 0:
        return 0.0
    return float(np.dot(t_vec / t_norm, p_vec / p_norm))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    import gc

    # ── Step 1: load spec_df, extract QTOF metadata+peaks, then free it ────
    log.info("Loading spec_df (NIST23) — will keep only QTOF rows...")
    with open(DATA_ROOT / "proc/nist23/spec_df.pkl", "rb") as f:
        spec_df = pickle.load(f)

    qtof_spec = spec_df[spec_df["inst_type"] == "QTOF"][
        ["spec_id", "mol_id", "peaks", "ace", "prec_type", "dset"]
    ].copy()
    del spec_df
    gc.collect()
    log.info("QTOF spectra in spec_df: %d", len(qtof_spec))

    # Remove MoNA
    qtof_spec = qtof_spec[qtof_spec["dset"] != "mona23"].copy()

    # Build lookup dict: spec_id → (peaks, ace, prec_type)
    qtof_ids = set(qtof_spec["spec_id"].values)
    spec_meta = {
        int(row.spec_id): (row.peaks, row.ace, row.prec_type)
        for row in qtof_spec.itertuples(index=False)
    }
    del qtof_spec
    gc.collect()
    log.info("QTOF spec_ids (no MoNA): %d", len(qtof_ids))

    # ── Step 2: process one CV fold at a time ───────────────────────────────
    results: list[dict] = []

    for cv_idx in range(9):
        zip_path = DATA_ROOT / f"predicted/nps_inchikey_nist23_nist23_cv{cv_idx}.zip"
        log.info("Processing fold %d ...", cv_idx)
        with zipfile.ZipFile(zip_path) as zf:
            pkl_name = next(n for n in zf.namelist() if n.endswith("predictions_test.pkl"))
            with zf.open(pkl_name) as f:
                cv_df = pickle.load(f)

        # Keep only QTOF test spectra
        cv_df = cv_df[cv_df["spec_id"].isin(qtof_ids)].copy()
        log.info("  QTOF rows in fold %d: %d", cv_idx, len(cv_df))

        for row in tqdm(cv_df.itertuples(index=False), total=len(cv_df), desc=f"fold {cv_idx}"):
            sid = int(row.spec_id)
            if sid not in spec_meta:
                continue
            peaks_raw, ace, prec_type = spec_meta[sid]
            true_mzs, true_ints = parse_peaks(peaks_raw)
            pred_mzs = np.array(row.pred_mzs, dtype=np.float64)
            pred_ints = np.array(row.pred_ints, dtype=np.float64)

            entry: dict = {
                "spec_id": sid,
                "mol_id": row.mol_id,
                "ace": ace,
                "prec_type": prec_type,
                "stored_cos_hun": row.cos_hun,
                "recomp_cos_sim_0.01da": cos_sim_binned(
                    true_mzs, true_ints, pred_mzs, pred_ints, 0.01
                ),
            }
            for ppm in PPM_TOLERANCES:
                entry[f"cos_hun_{ppm}ppm"] = cos_hun_ppm(
                    true_mzs, true_ints, pred_mzs, pred_ints, ppm
                )
            results.append(entry)

        del cv_df
        gc.collect()

    res_df = pd.DataFrame(results)
    log.info("Total QTOF results: %d", len(res_df))

    # ── Summary statistics ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("QTOF Performance: Stored vs Recomputed at multiple ppm tolerances")
    print("=" * 70)

    cols_report = (
        ["stored_cos_hun", "recomp_cos_sim_0.01da"]
        + [f"cos_hun_{p}ppm" for p in PPM_TOLERANCES]
    )
    summary = res_df[cols_report].agg(["mean", "median"]).round(4)
    print(summary.to_string())

    # ── Per-ACE-bin breakdown ───────────────────────────────────────────────
    ace_edges = [0, 10, 20, 30, 50, 100, 200]
    ace_labels = ["0-10", "10-20", "20-30", "30-50", "50-100", "100-200"]
    valid_ace = res_df[
        res_df["ace"].notna() & (res_df["ace"] > 0) & (res_df["ace"] <= ace_edges[-1])
    ].copy()
    valid_ace["ace_bin"] = pd.cut(valid_ace["ace"], bins=ace_edges, labels=ace_labels, right=True)

    print("\n" + "=" * 70)
    print("Median cos_hun by ACE bin — stored (10 ppm) vs 20 ppm vs 30 ppm")
    print("=" * 70)
    ace_stats = (
        valid_ace.groupby("ace_bin", observed=True)[
            ["stored_cos_hun", "cos_hun_10ppm", "cos_hun_20ppm", "cos_hun_30ppm"]
        ]
        .agg(["median", "count"])
        .round(4)
    )
    print(ace_stats.to_string())

    # ── Delta summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Δ median cos_hun: 20 ppm vs stored (10 ppm)")
    print("=" * 70)
    delta = (
        res_df["cos_hun_20ppm"].median() - res_df["stored_cos_hun"].median()
    )
    print(f"  Global: {delta:+.4f}  (stored 10ppm median={res_df['stored_cos_hun'].median():.4f}, "
          f"20ppm median={res_df['cos_hun_20ppm'].median():.4f})")

    # ── Save results ────────────────────────────────────────────────────────
    out_path = DATA_ROOT / "predicted/qtof_20ppm_reeval.pkl"
    res_df.to_pickle(out_path)
    log.info("Saved results to %s", out_path)

    plot_results(res_df)


def plot_results(res_df: pd.DataFrame) -> None:
    """Generate and save three-panel QTOF tolerance comparison figure.

    Args:
        res_df: DataFrame with columns stored_cos_hun, cos_hun_{10,15,20,30}ppm,
            recomp_cos_sim_0.01da, ace, prec_type.
    """
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3, "font.size": 11})

    tol_cols = ["stored_cos_hun", "cos_hun_10ppm", "cos_hun_15ppm", "cos_hun_20ppm", "cos_hun_30ppm"]
    tol_labels = ["10 ppm\n(stored)", "10 ppm\n(recomp)", "15 ppm", "20 ppm", "30 ppm"]
    colors = ["#d62728", "#ff7f0e", "#bcbd22", "#2ca02c", "#1f77b4"]

    ace_edges = [0, 10, 20, 30, 50]
    ace_labels_plot = ["0–10 eV", "10–20 eV", "20–30 eV", "30–50 eV"]
    valid_ace = res_df[
        res_df["ace"].notna() & (res_df["ace"] > 0) & (res_df["ace"] <= ace_edges[-1])
    ].copy()
    valid_ace["ace_bin"] = pd.cut(valid_ace["ace"], bins=ace_edges, labels=ace_labels_plot, right=True)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # ── Panel 1: violin distribution per tolerance ──────────────────────────
    ax = axes[0]
    groups = [res_df[c].dropna().values for c in tol_cols]
    vp = ax.violinplot(groups, positions=range(len(tol_cols)), showmedians=True, showextrema=False)
    for body, color in zip(vp["bodies"], colors):
        body.set_facecolor(color)
        body.set_alpha(0.7)
    vp["cmedians"].set_color("black")
    vp["cmedians"].set_linewidth(2)
    medians = [np.median(g) for g in groups]
    for i, (med, color) in enumerate(zip(medians, colors)):
        ax.text(i, med + 0.02, f"{med:.3f}", ha="center", va="bottom", fontsize=9,
                color="black", fontweight="bold")
    ax.set_xticks(range(len(tol_cols)))
    ax.set_xticklabels(tol_labels, fontsize=10)
    ax.set_ylabel("cos_hun")
    ax.set_ylim(0, 1.05)
    ax.set_title("QTOF: cos_hun distribution\nby matching tolerance")

    # ── Panel 2: median per ACE bin grouped bar chart ───────────────────────
    ax = axes[1]
    ace_cols = ["stored_cos_hun", "cos_hun_10ppm", "cos_hun_20ppm", "cos_hun_30ppm"]
    ace_col_labels = ["10 ppm (stored)", "10 ppm (recomp)", "20 ppm", "30 ppm"]
    ace_bar_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#1f77b4"]

    ace_med = (
        valid_ace.groupby("ace_bin", observed=True)[ace_cols]
        .median()
        .reindex(ace_labels_plot)
    )
    n_bins = len(ace_labels_plot)
    n_tols = len(ace_cols)
    width = 0.18
    x = np.arange(n_bins)
    for j, (col, label, color) in enumerate(zip(ace_cols, ace_col_labels, ace_bar_colors)):
        offset = (j - (n_tols - 1) / 2) * width
        vals = ace_med[col].values
        bars = ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, val + 0.01, f"{val:.2f}",
                        ha="center", va="bottom", fontsize=7.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(ace_labels_plot)
    ax.set_ylabel("Median cos_hun")
    ax.set_ylim(0, 1.1)
    ax.set_title("QTOF: median cos_hun by ACE bin\nand matching tolerance")
    ax.legend(fontsize=9, loc="upper right")

    # ── Panel 3: CDF of 20 ppm vs stored ────────────────────────────────────
    ax = axes[2]
    cdf_pairs = [
        ("stored_cos_hun", "10 ppm (stored)", "#d62728"),
        ("cos_hun_20ppm",  "20 ppm",          "#2ca02c"),
        ("cos_hun_30ppm",  "30 ppm",           "#1f77b4"),
        ("recomp_cos_sim_0.01da", "0.01 Da binned", "#9467bd"),
    ]
    for col, label, color in cdf_pairs:
        vals = np.sort(res_df[col].dropna().values)
        cdf = np.arange(1, len(vals) + 1) / len(vals)
        ax.plot(vals, cdf, label=label, color=color, linewidth=2)
    for thresh in [0.5, 0.7, 0.9]:
        ax.axvline(thresh, color="grey", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("cos_hun")
    ax.set_ylabel("Cumulative fraction")
    ax.set_title("QTOF: CDF of cos_hun\nby matching tolerance")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig_path = DATA_ROOT / "predicted/qtof_20ppm_reeval.png"
    fig.savefig(fig_path, bbox_inches="tight")
    log.info("Saved figure to %s", fig_path)
    plt.show()


if __name__ == "__main__":
    main()
