"""Plot NIST20 spectra with isotope peaks labelled from ann_df annotations.

Each spectrum is saved as a separate PNG in scripts/debug/figs/ann_iso_peaks/.
Each plot shows:
  - Annotated fragment peaks (blue sticks, labelled with NIST product formula)
  - NIST-annotated isotope peaks (tomato sticks, labelled M+k + product formula)
  - Unannotated peaks (grey sticks)
  - Dashed vertical line at prec_mz
  - Precursor formula annotated in top-left

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/debug_ann_df_iso_peaks.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROC_DP = Path("data/proc")
FIG_DIR = Path("figs/ann_iso_peaks")

N_PLOTS = 50
MZ_TOL_MATCH = 0.015  # 15 mDa to match observed peak to annotated m/z


def match_ann_to_obs(
    obs_mzs: np.ndarray,
    ann_mzs: list[float],
    ann_iso_ks: list[int],
    ann_products: list[str],
    mz_tol: float = MZ_TOL_MATCH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Match annotated peaks to observed spectrum peaks by m/z proximity.

    Args:
        obs_mzs: Observed peak m/z values of shape (n_peaks,).
        ann_mzs: Annotated peak m/z values from ann_df.
        ann_iso_ks: Isotope order k for each annotated peak (0 = monoisotopic).
        ann_products: NIST product formula string for each annotated peak.
        mz_tol: m/z tolerance in Da for matching.

    Returns:
        Tuple of:
          - iso_k_arr: int array of shape (n_peaks,), -1 = unannotated,
            0 = annotated monoisotopic, k>0 = annotated M+k isotope.
          - is_ann: bool array of shape (n_peaks,), True if peak was annotated.
          - label_arr: list of str, one per observed peak; empty string if unannotated.
    """
    iso_k_arr = np.full(len(obs_mzs), -1, dtype=int)
    is_ann = np.zeros(len(obs_mzs), dtype=bool)
    label_arr: list[str] = [""] * len(obs_mzs)

    for ann_mz, k, product in zip(ann_mzs, ann_iso_ks, ann_products):
        d = np.abs(obs_mzs - ann_mz)
        best = int(np.argmin(d))
        if d[best] <= mz_tol:
            # If multiple annotations match the same peak, prefer highest k
            if not is_ann[best] or k > iso_k_arr[best]:
                iso_k_arr[best] = k
                is_ann[best] = True
                label_arr[best] = str(product)

    return iso_k_arr, is_ann, label_arr


def plot_one(h: dict, out_path: Path, idx: int, total: int) -> None:
    """Save one spectrum plot with ann_df isotope labels to out_path.

    Args:
        h: Hit dict with keys: spec_id, formula, prec_type, inst_type, prec_mz,
           mzs, ints, iso_k_arr, is_ann, label_arr.
        out_path: File path to save PNG.
        idx: 0-based hit index (for title numbering).
        total: Total number of hits (for title numbering).
    """
    mzs = h["mzs"]
    ints = h["ints"]
    prec_mz = h["prec_mz"]
    iso_k_arr = h["iso_k_arr"]   # -1 = unannotated, 0 = monoisotopic, k>0 = isotope
    is_ann = h["is_ann"]
    label_arr = h["label_arr"]   # NIST product formula string per peak

    ints_norm = ints / ints.max() * 100 if ints.max() > 0 else ints.copy()

    # Zoom window: prec_mz - 5 Da to max mz + 1 Da
    zoom_lo = prec_mz - 5.0
    zoom_hi = mzs.max() + 1.0

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="steelblue", lw=1.5, label="Annotated fragment (M+0)"),
        Line2D([0], [0], color="tomato", lw=1.5, label="Annotated isotope (M+k)"),
        Line2D([0], [0], color="grey", lw=1.0, label="Unannotated peak", alpha=0.5),
    ]

    def _draw_peaks(ax: plt.Axes) -> None:
        for i, (mz, rel_int) in enumerate(zip(mzs, ints_norm)):
            k = int(iso_k_arr[i])
            if not is_ann[i]:
                color, lw, zo, alpha = "grey", 0.8, 1, 0.45
            elif k == 0:
                color, lw, zo, alpha = "steelblue", 1.2, 2, 0.9
            else:
                color, lw, zo, alpha = "tomato", 2.0, 3, 0.95
            ax.vlines(mz, 0, rel_int, colors=color, lw=lw, alpha=alpha, zorder=zo)

        if prec_mz > 0:
            ax.axvline(prec_mz, color="black", lw=1.0, ls="--", alpha=0.4, zorder=0)
        ax.set_ylim(0, 130)
        ax.set_xlabel("m/z", fontsize=9)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 5), width_ratios=[2, 1])

    # ── Left: full spectrum ───────────────────────────────────────────────────
    _draw_peaks(ax_full)
    ax_full.set_ylabel("Relative intensity (%)", fontsize=9)
    ax_full.set_title("Full spectrum", fontsize=9)
    ax_full.text(
        0.01,
        0.97,
        f"Formula: {h['formula']}",
        transform=ax_full.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "lightyellow", "alpha": 0.8},
    )
    ax_full.legend(handles=legend_handles, fontsize=7, loc="upper right")
    ax_full.axvspan(zoom_lo, zoom_hi, color="gold", alpha=0.12, zorder=0)
    if prec_mz > 0:
        ax_full.text(
            prec_mz, 107, f"prec\n{prec_mz:.3f}",
            fontsize=6, color="black", ha="center", va="bottom", alpha=0.6,
        )

    # ── Right: zoomed into prec region ───────────────────────────────────────
    _draw_peaks(ax_zoom)
    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    ax_zoom.set_title(
        f"Zoom: prec±5 Da → max mz  ({zoom_lo:.1f}–{zoom_hi:.1f})", fontsize=9
    )
    if prec_mz > 0:
        ax_zoom.text(
            prec_mz, 107, f"prec\n{prec_mz:.4f}",
            fontsize=7, color="black", ha="center", va="bottom", alpha=0.7,
        )

    # Label all annotated peaks on the zoom panel with NIST product formula
    for i, (mz, rel_int) in enumerate(zip(mzs, ints_norm)):
        if not is_ann[i]:
            continue
        if not (zoom_lo <= mz <= zoom_hi):
            continue
        k = int(iso_k_arr[i])
        lbl = label_arr[i]
        if k > 0:
            text = f"M+{k}  {lbl}\n{mz:.4f}"
            color = "tomato"
        else:
            text = f"{lbl}\n{mz:.4f}"
            color = "steelblue"
        ax_zoom.annotate(
            text,
            xy=(mz, rel_int),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=7,
            color=color,
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Label annotated isotope peaks on the full spectrum panel
    for i, (mz, rel_int) in enumerate(zip(mzs, ints_norm)):
        k = int(iso_k_arr[i])
        if not is_ann[i] or k == 0:
            continue
        lbl = label_arr[i]
        ax_full.annotate(
            f"M+{k}  {lbl}\n{mz:.3f}",
            xy=(mz, rel_int),
            xytext=(0, 5),
            textcoords="offset points",
            fontsize=6,
            color="tomato",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Count annotated isotope peaks for the title
    n_iso_peaks = int((iso_k_arr > 0).sum())
    iso_orders = sorted(set(int(k) for k in iso_k_arr[iso_k_arr > 0]))
    ks_str = ", ".join(f"M+{k}" for k in iso_orders) if iso_orders else "none"

    fig.suptitle(
        f"[{idx + 1}/{total}]  spec_id={h['spec_id']}  |  {h['prec_type']}  |  "
        f"inst={h['inst_type']}  |  prec_mz={prec_mz:.4f}  |  "
        f"ann isotope peaks: {n_iso_peaks} ({ks_str})",
        fontsize=9,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {FIG_DIR}")

    log.info("Loading ann_df…")
    ann_df = pd.read_pickle(PROC_DP / "nist20/ann_df.pkl")[
        ["spec_id", "ann_peak_mzs", "ann_isotopes", "ann_products"]
    ].copy()
    log.info(f"  ann_df: {len(ann_df):,} annotated spectra")

    # Keep spectra that have at least one isotope peak (k > 0)
    ann_df["n_iso"] = ann_df["ann_isotopes"].apply(
        lambda x: sum(1 for v in x if float(v) != 0)
    )
    ann_df = ann_df[ann_df["n_iso"] > 0].reset_index(drop=True)
    log.info(f"  Spectra with ≥1 annotated isotope peak: {len(ann_df):,}")

    ann_spec_ids = set(ann_df["spec_id"].unique())

    log.info("Loading spec_df + mol_df…")
    spec_df = pd.read_pickle(PROC_DP / "nist20/spec_df.pkl")[
        ["spec_id", "peaks", "prec_mz", "inst_type", "mol_id", "prec_type"]
    ]
    mol_df = pd.read_pickle(PROC_DP / "nist20/mol_df.pkl")[["mol_id", "formula"]]

    spec_df = spec_df[spec_df["spec_id"].isin(ann_spec_ids)].copy()
    spec_df = spec_df.merge(mol_df, on="mol_id", how="left")
    log.info(f"  Matched spec_df rows: {len(spec_df):,}")
    del mol_df

    # Join ann_df onto spec_df
    merged = spec_df.merge(
        ann_df[["spec_id", "ann_peak_mzs", "ann_isotopes", "ann_products"]],
        on="spec_id",
        how="inner",
    ).reset_index(drop=True)
    log.info(f"  Merged rows: {len(merged):,}")

    log.info(f"Building plots for first {N_PLOTS} spectra…")
    hits: list[dict] = []
    for _, row in merged.iterrows():
        if len(hits) >= N_PLOTS:
            break

        peaks = row["peaks"]
        if not peaks:
            continue
        ann_mzs_raw = row["ann_peak_mzs"]
        ann_iso_ks_raw = row["ann_isotopes"]
        ann_products_raw = row["ann_products"]
        if not ann_mzs_raw:
            continue

        mzs = np.array([p[0] for p in peaks], dtype=np.float64)
        ints = np.array([p[1] for p in peaks], dtype=np.float64)

        try:
            ann_mzs_f = [float(v) for v in ann_mzs_raw]
            ann_iso_ks_i = [int(float(v)) for v in ann_iso_ks_raw]
            ann_products_s = [str(v) for v in ann_products_raw]
        except (TypeError, ValueError):
            continue

        iso_k_arr, is_ann, label_arr = match_ann_to_obs(
            mzs, ann_mzs_f, ann_iso_ks_i, ann_products_s
        )

        hits.append(
            {
                "spec_id": row["spec_id"],
                "formula": str(row.get("formula", "")),
                "prec_type": str(row.get("prec_type", "")),
                "inst_type": str(row.get("inst_type", "")),
                "prec_mz": float(row["prec_mz"]) if pd.notna(row["prec_mz"]) else 0.0,
                "mzs": mzs,
                "ints": ints,
                "iso_k_arr": iso_k_arr,
                "is_ann": is_ann,
                "label_arr": label_arr,
            }
        )

    log.info(f"Saving {len(hits)} plots…")
    for i, h in enumerate(hits):
        out_path = FIG_DIR / f"{i + 1:03d}_spec{h['spec_id']}.png"
        plot_one(h, out_path, i, len(hits))
        if (i + 1) % 10 == 0:
            log.info(f"  {i + 1}/{len(hits)} saved")

    log.info(f"Done. All plots in {FIG_DIR}")


if __name__ == "__main__":
    main()
