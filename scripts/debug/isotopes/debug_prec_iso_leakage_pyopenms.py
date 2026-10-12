"""Plot the first 50 NIST20 spectra that have pyopenms-confirmed isotope peaks
above the precursor m/z (precursor isotope leakage).

Each spectrum is saved as a separate PNG in scripts/debug/figs/prec_iso_leakage/.
Each plot shows:
  - Fragment peaks (blue sticks)
  - Fragment isotope peaks flagged by pyopenms (tomato sticks)
  - Above-prec leakage peaks (crimson sticks, labelled M+k)
  - M, M+1, M+2 precursor peaks: m/z + raw intensity labelled
  - Precursor formula annotated in top-left
  - Dashed vertical line at prec_mz

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/isotopes/debug_prec_iso_leakage_pyopenms.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fragnnet.dataset.isotope_cleanup import (
    detect_isotope_peaks_for_training,
    detect_isotope_peaks_formula_aware,
    isotope_offsets_from_formula,
)
from fragnnet.utils.formula_utils import parse_formula

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROC_DP = Path("data/proc")
FIG_DIR = Path("figs/prec_iso_leakage")

MAX_K = 3
N_PLOTS = 50

# m/z tolerance by instrument — ratio window is formula-based and instrument-agnostic
MZ_TOL_BY_INST: dict[str, float] = {
    "FT": 0.004,  # Orbitrap sub-ppm; excludes H-gain (4.47 mDa from 13C)
    "QTOF": 0.010,  # ~5 ppm at m/z 500
    "IT": 0.020,  # unit resolution
    "QQQ": 0.020,
}
MZ_TOL_DEFAULT = 0.020  # fallback for unknown inst_type


def find_prec_leakage(
    mzs: np.ndarray,
    ints: np.ndarray,
    prec_mz: float,
    element_counts: dict[str, int],
    mz_tol: float = MZ_TOL_DEFAULT,
    max_k: int = MAX_K,
) -> list[tuple[int, int, float]]:
    """Return (peak_idx, k, delta_mz) for above-prec peaks confirmed as isotope leakage.

    A candidate is confirmed if m/z matches prec_mz + pyopenms_offset(k) within mz_tol.
    No intensity ratio check is applied — the ratio I(M+k)/I(M+0) for leakage peaks
    is determined by the MS1 isolation window, not by natural abundance, and is
    therefore unpredictable from the molecular formula alone.

    Args:
        mzs: All peak m/z values.
        ints: All peak intensities.
        prec_mz: Precursor m/z.
        element_counts: Molecular formula element counts from parse_formula.
        mz_tol: m/z tolerance in Da.
        max_k: Maximum isotope order to check.

    Returns:
        List of (peak_index, k, delta_mz) for confirmed leakage peaks.
    """
    offsets = isotope_offsets_from_formula(element_counts, max_isotope=max_k)

    # Check each k in order; stop as soon as one is missing — M+k requires M+1…M+(k-1)
    # all confirmed (envelope coherence: no M+2 without M+1).
    confirmed_ks: set[int] = set()
    leakage = []
    for k, (offset, _exp_ratio) in enumerate(offsets, start=1):
        # Envelope coherence: all lower orders must be confirmed first
        if k > 1 and (k - 1) not in confirmed_ks:
            break

        target = prec_mz + offset
        d = np.abs(mzs - target)
        best_idx = int(np.argmin(d))
        if d[best_idx] > mz_tol or mzs[best_idx] <= prec_mz + mz_tol:
            break  # M+k missing — no point checking M+(k+1)

        confirmed_ks.add(k)
        leakage.append((int(best_idx), k, float(d[best_idx])))

    return leakage


def find_prec_mk_peaks(
    mzs: np.ndarray,
    ints: np.ndarray,
    prec_mz: float,
    element_counts: dict[str, int],
    mz_tol: float = MZ_TOL_DEFAULT,
    max_k: int = MAX_K,
) -> tuple[dict[int, tuple[float, float]], dict[int, float]]:
    """Find M+0..M+max_k peaks near prec_mz and their expected ratios from pyopenms.

    Args:
        mzs: Peak m/z values.
        ints: Peak intensities.
        prec_mz: Precursor m/z.
        element_counts: Molecular formula element counts.
        mz_tol: m/z tolerance in Da.
        max_k: Maximum isotope order.

    Returns:
        Tuple of:
          - obs: dict k -> (observed_mz, raw_intensity) for k=0,1,...
          - exp_ratios: dict k -> expected M+k/M+0 intensity ratio (from pyopenms)
    """
    offsets = isotope_offsets_from_formula(element_counts, max_isotope=max_k)
    obs: dict[int, tuple[float, float]] = {}
    exp_ratios: dict[int, float] = {}

    # k=0: M+0 precursor peak — use a wide fixed tolerance regardless of inst_type,
    # since we are locating the precursor peak in the spectrum, not distinguishing isotopes.
    M0_TOL = 0.05
    d = np.abs(mzs - prec_mz)
    idx = int(np.argmin(d))
    if d[idx] <= M0_TOL:
        obs[0] = (float(mzs[idx]), float(ints[idx]))

    # k=1,2,...: leakage peaks above prec
    for k, (offset, exp_ratio) in enumerate(offsets, start=1):
        exp_ratios[k] = float(exp_ratio)
        target = prec_mz + offset
        d = np.abs(mzs - target)
        idx = int(np.argmin(d))
        if d[idx] <= mz_tol and mzs[idx] > prec_mz + mz_tol:
            obs[k] = (float(mzs[idx]), float(ints[idx]))

    return obs, exp_ratios


def plot_one(
    h: dict,
    out_path: Path,
    idx: int,
) -> None:
    """Save one spectrum plot to out_path.

    Args:
        h: Hit dict with keys: spec_id, formula, prec_type, inst_type, prec_mz,
           mzs, ints, leakage, iso_mask, mk_peaks, exp_ratios.
        out_path: File path to save PNG.
        idx: 0-based hit index (for title numbering).
    """
    mzs = h["mzs"]
    ints = h["ints"]
    prec_mz = h["prec_mz"]
    leakage_idxs = {peak_idx for peak_idx, _, _ in h["leakage"]}
    iso_mask = h["iso_mask"]
    mk_peaks = h["mk_peaks"]  # dict k -> (mz, raw_int)
    exp_ratios = h["exp_ratios"]  # dict k -> expected M+k/M+0 ratio

    ints_norm = ints / ints.max() * 100 if ints.max() > 0 else ints.copy()
    m0_raw_int = mk_peaks[0][1] if 0 in mk_peaks else None

    # Zoom window: prec_mz - 5 Da to max mz + 1 Da
    zoom_lo = prec_mz - 5.0
    zoom_hi = mzs.max() + 1.0

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color="steelblue", lw=1.5, label="Fragment"),
        Line2D([0], [0], color="tomato", lw=1.5, label="Fragment isotope (pyopenms)"),
        Line2D([0], [0], color="crimson", lw=2.0, label="Prec leakage M+k"),
        Line2D([0], [0], color="darkgreen", lw=2.0, label="M+0 precursor peak"),
    ]

    def _draw_peaks(ax: plt.Axes) -> None:
        for i, (mz, rel_int) in enumerate(zip(mzs, ints_norm)):
            if i in leakage_idxs:
                color, lw, zorder = "crimson", 2.0, 3
            elif iso_mask[i]:
                color, lw, zorder = "tomato", 1.4, 2
            else:
                color, lw, zorder = "steelblue", 1.0, 1
            ax.vlines(mz, 0, rel_int, colors=color, lw=lw, alpha=0.9, zorder=zorder)
        if 0 in mk_peaks:
            m0_mz, _ = mk_peaks[0]
            m0_norm = ints_norm[int(np.argmin(np.abs(mzs - m0_mz)))]
            ax.vlines(m0_mz, 0, m0_norm, colors="darkgreen", lw=2.0, alpha=0.9, zorder=4)
        ax.axvline(prec_mz, color="black", lw=1.0, ls="--", alpha=0.4, zorder=0)
        ax.set_ylim(0, 120)
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
    # shade the zoom region
    ax_full.axvspan(zoom_lo, zoom_hi, color="gold", alpha=0.12, zorder=0)
    ax_full.axvline(prec_mz, color="black", lw=1.0, ls="--", alpha=0.4)
    ax_full.text(
        prec_mz,
        104,
        f"prec\n{prec_mz:.3f}",
        fontsize=6,
        color="black",
        ha="center",
        va="bottom",
        alpha=0.6,
    )

    # ── Right: zoomed into prec region ───────────────────────────────────────
    _draw_peaks(ax_zoom)
    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    ax_zoom.set_title(f"Zoom: prec±5 Da → max mz  ({zoom_lo:.1f}–{zoom_hi:.1f})", fontsize=9)
    ax_zoom.axvline(prec_mz, color="black", lw=1.0, ls="--", alpha=0.4)
    ax_zoom.text(
        prec_mz,
        104,
        f"prec\n{prec_mz:.4f}",
        fontsize=7,
        color="black",
        ha="center",
        va="bottom",
        alpha=0.7,
    )

    # Label M+0 on zoom panel
    if 0 in mk_peaks:
        m0_mz, m0_raw = mk_peaks[0]
        m0_norm = ints_norm[int(np.argmin(np.abs(mzs - m0_mz)))]
        ax_zoom.annotate(
            f"M\n{m0_mz:.4f}\n{m0_raw:.0f}",
            xy=(m0_mz, m0_norm),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=8,
            color="darkgreen",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    # Label leakage peaks on zoom panel
    for peak_idx, k, _delta in h["leakage"]:
        mz = mzs[peak_idx]
        rel_int = ints_norm[peak_idx]
        raw_int = ints[peak_idx]
        exp_r = exp_ratios.get(k, float("nan"))
        obs_r = (raw_int / m0_raw_int) if m0_raw_int else float("nan")
        ratio_line = (
            f"exp:{exp_r:.3f}  obs:{obs_r:.3f}" if not np.isnan(obs_r) else f"exp:{exp_r:.3f}"
        )
        ax_zoom.annotate(
            f"M+{k}\n{mz:.4f}\n{raw_int:.0f}\n{ratio_line}",
            xy=(mz, rel_int),
            xytext=(0, 6),
            textcoords="offset points",
            fontsize=8,
            color="crimson",
            ha="center",
            va="bottom",
            fontweight="bold",
        )

    ks_str = ", ".join(f"M+{k}" for _, k, _ in sorted(h["leakage"], key=lambda x: x[1]))
    fig.suptitle(
        f"[{idx + 1}/{N_PLOTS}]  spec_id={h['spec_id']}  |  {h['prec_type']}  |  "
        f"inst={h['inst_type']}  mz_tol={h['mz_tol'] * 1000:.0f} mDa  |  "
        f"prec_mz={prec_mz:.4f}  |  leakage: {ks_str}",
        fontsize=9,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _load_df() -> pd.DataFrame:
    log.info("Loading NIST20 spec_df + mol_df…")
    spec_df = pd.read_pickle(PROC_DP / "nist20/spec_df.pkl")[
        ["spec_id", "peaks", "prec_mz", "inst_type", "mol_id", "prec_type"]
    ]
    mol_df = pd.read_pickle(PROC_DP / "nist20/mol_df.pkl")[["mol_id", "formula"]]
    df = spec_df.merge(mol_df, on="mol_id", how="left")
    del spec_df, mol_df
    df = df[df["formula"].notna() & df["prec_mz"].notna() & (df["prec_mz"] > 0)].reset_index(
        drop=True
    )
    log.info(f"  {len(df):,} spectra with formula + prec_mz")
    return df


def _scan_row(row: pd.Series) -> dict | None:
    """Return a hit dict if row has confirmed precursor isotope leakage, else None."""
    peaks = row["peaks"]
    if not peaks:
        return None
    prec_mz = float(row["prec_mz"])
    formula = str(row["formula"])
    inst_type = str(row.get("inst_type", ""))
    mz_tol = MZ_TOL_BY_INST.get(inst_type, MZ_TOL_DEFAULT)

    mzs = np.array([p[0] for p in peaks], dtype=np.float64)
    ints = np.array([p[1] for p in peaks], dtype=np.float64)

    if not np.any(mzs > prec_mz + mz_tol):
        return None
    try:
        ec = parse_formula(formula)
    except Exception:
        return None

    leakage = find_prec_leakage(mzs, ints, prec_mz, ec, mz_tol=mz_tol)
    if not leakage:
        return None

    return {
        "spec_id": row["spec_id"],
        "formula": formula,
        "prec_type": row.get("prec_type", ""),
        "inst_type": inst_type,
        "prec_mz": prec_mz,
        "mz_tol": mz_tol,
        "mzs": mzs,
        "ints": ints,
        "leakage": leakage,
    }


def count_mode(df: pd.DataFrame) -> None:
    """Scan all spectra and print leakage counts broken down by inst_type and k."""
    import collections

    total = len(df)
    n_candidates = 0  # has peaks above prec
    n_hit = 0
    by_inst: dict[str, int] = collections.Counter()
    by_k: dict[int, int] = collections.Counter()  # highest k confirmed per spectrum

    log.info(f"Scanning all {total:,} spectra…")
    for i, (_, row) in enumerate(df.iterrows()):
        if (i + 1) % 50_000 == 0:
            log.info(f"  {i + 1:,}/{total:,}  hits so far: {n_hit:,}")

        peaks = row["peaks"]
        if not peaks:
            continue
        prec_mz = float(row["prec_mz"])
        inst_type = str(row.get("inst_type", ""))
        mz_tol = MZ_TOL_BY_INST.get(inst_type, MZ_TOL_DEFAULT)

        mzs = np.array([p[0] for p in peaks], dtype=np.float64)
        ints = np.array([p[1] for p in peaks], dtype=np.float64)

        if not np.any(mzs > prec_mz + mz_tol):
            continue
        n_candidates += 1

        formula = str(row["formula"])
        try:
            ec = parse_formula(formula)
        except Exception:
            continue

        leakage = find_prec_leakage(mzs, ints, prec_mz, ec, mz_tol=mz_tol)
        if not leakage:
            continue

        n_hit += 1
        by_inst[inst_type] += 1
        max_k = max(k for _, k, _ in leakage)
        by_k[max_k] += 1

    log.info("\n── Precursor isotope leakage — full NIST20 count ──────────────────")
    log.info(f"  Total spectra (formula+prec_mz): {total:,}")
    log.info(
        f"  Have any peak > prec_mz:         {n_candidates:,}  ({n_candidates / total * 100:.1f}%)"
    )
    log.info(f"  Confirmed leakage (pyopenms):     {n_hit:,}  ({n_hit / total * 100:.1f}%)")
    log.info(f"  Confirmed / candidates:           {n_hit / n_candidates * 100:.1f}%")
    log.info("\n  By inst_type:")
    for inst, cnt in sorted(by_inst.items(), key=lambda x: -x[1]):
        log.info(f"    {inst or '(unknown)':8s}  {cnt:,}")
    log.info("\n  By highest k confirmed:")
    for k, cnt in sorted(by_k.items()):
        log.info(f"    M+{k}  {cnt:,}")


def plot_mode(df: pd.DataFrame) -> None:
    """Find first N_PLOTS hits, build full hit dicts, and save PNGs."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {FIG_DIR}")
    log.info(f"Scanning for pyopenms-confirmed above-prec isotope leakage (first {N_PLOTS})…")

    hits: list[dict] = []
    for _, row in df.iterrows():
        if len(hits) >= N_PLOTS:
            break
        h = _scan_row(row)
        if h is None:
            continue

        # Augment with fields needed for plotting
        ec = parse_formula(h["formula"])
        mz_tol = h["mz_tol"]
        iso_mask = detect_isotope_peaks_formula_aware(
            h["mzs"], h["ints"], ec, mz_tol=mz_tol, max_isotope=MAX_K
        )
        mk_peaks, exp_ratios = find_prec_mk_peaks(
            h["mzs"], h["ints"], h["prec_mz"], ec, mz_tol=mz_tol
        )
        h["iso_mask"] = iso_mask
        h["mk_peaks"] = mk_peaks
        h["exp_ratios"] = exp_ratios
        hits.append(h)

    log.info(f"Found {len(hits)} spectra — saving individual plots…")
    for i, h in enumerate(hits):
        out_path = FIG_DIR / f"{i + 1:03d}_spec{h['spec_id']}.png"
        plot_one(h, out_path, i)
        if (i + 1) % 10 == 0:
            log.info(f"  {i + 1}/{len(hits)} saved")

    log.info(f"Done. All plots in {FIG_DIR}")


# co-isolation fraction threshold (f = R_k/R_exp_prec) used in eval_mode
MIN_COISO: float = 0.05

# How many top-TP / top-FP / top-FN spectra to save as eval plots
N_EVAL_PLOTS: int = 10


def plot_eval_one(
    h: dict,
    out_path: Path,
    idx: int,
    total: int,
    category: str,
) -> None:
    """Save one eval-mode spectrum plot showing TP/FP/FN peak coloring.

    Args:
        h: Dict with keys: spec_id, formula, prec_type, prec_mz, mzs, ints,
           peak_labels (dict[int, str]: "tp" | "fp" | "fn" | "other"),
           n_tp, n_fp, n_fn, fired.
        out_path: File path to save PNG.
        idx: 0-based index within this category.
        total: Total plots in this category.
        category: One of "TP", "FP", "FN_fired", "FN_nofire".
    """
    from matplotlib.lines import Line2D

    mzs = h["mzs"]
    ints = h["ints"]
    prec_mz = h["prec_mz"]
    peak_labels: dict[int, str] = h["peak_labels"]

    ints_norm = ints / ints.max() * 100 if ints.max() > 0 else ints.copy()
    zoom_lo = prec_mz - 5.0
    zoom_hi = float(mzs.max()) + 1.0

    _COLOR = {"tp": "green", "fp": "crimson", "fn": "darkorange", "other": "steelblue"}
    _LW = {"tp": 1.8, "fp": 1.8, "fn": 1.8, "other": 0.8}
    _ZO = {"tp": 3, "fp": 4, "fn": 2, "other": 1}

    legend_handles = [
        Line2D([0], [0], color="green", lw=1.8, label=f"TP (n={h['n_tp']})"),
        Line2D([0], [0], color="crimson", lw=1.8, label=f"FP (n={h['n_fp']})"),
        Line2D([0], [0], color="darkorange", lw=1.8, label=f"FN missed (n={h['n_fn']})"),
        Line2D([0], [0], color="steelblue", lw=0.8, label="Fragment (other)"),
    ]

    def _draw_peaks(ax: plt.Axes) -> None:
        for i, (mz, rel_int) in enumerate(zip(mzs, ints_norm)):
            lbl = peak_labels.get(i, "other")
            ax.vlines(
                mz, 0, rel_int,
                colors=_COLOR[lbl], lw=_LW[lbl], alpha=0.9, zorder=_ZO[lbl],
            )
        ax.axvline(prec_mz, color="black", lw=1.0, ls="--", alpha=0.4)
        ax.set_ylim(0, 120)
        ax.set_xlabel("m/z", fontsize=9)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(16, 5), width_ratios=[2, 1])

    _draw_peaks(ax_full)
    ax_full.set_ylabel("Relative intensity (%)", fontsize=9)
    ax_full.set_title("Full spectrum", fontsize=9)
    ax_full.text(
        0.01,
        0.97,
        f"Formula: {h['formula']}  |  {h['prec_type']}",
        transform=ax_full.transAxes,
        fontsize=8,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "lightyellow", "alpha": 0.8},
    )
    ax_full.legend(handles=legend_handles, fontsize=7, loc="upper right")
    ax_full.axvspan(zoom_lo, zoom_hi, color="gold", alpha=0.12, zorder=0)
    ax_full.text(
        prec_mz, 104, f"prec\n{prec_mz:.3f}", fontsize=6, color="black",
        ha="center", va="bottom", alpha=0.6,
    )

    _draw_peaks(ax_zoom)
    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    ax_zoom.set_title(f"Zoom: prec−5 Da → max mz  ({zoom_lo:.1f}–{zoom_hi:.1f})", fontsize=9)
    ax_zoom.text(
        prec_mz, 104, f"prec\n{prec_mz:.4f}", fontsize=7, color="black",
        ha="center", va="bottom", alpha=0.7,
    )

    fired_str = "fired" if h["fired"] else "no-fire"
    fig.suptitle(
        f"[{idx + 1}/{total}]  {category} ({fired_str})  spec_id={h['spec_id']}  "
        f"TP={h['n_tp']}  FP={h['n_fp']}  FN={h['n_fn']}  "
        f"prec_mz={prec_mz:.4f}",
        fontsize=9,
        y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _build_eval_plot_data(
    spec_id: int,
    counts: list[int],
    ft_df: pd.DataFrame,
    ann_by_spec: dict,
    match_tol: float,
) -> dict | None:
    """Build plot data dict for one spectrum by re-running the detector.

    Args:
        spec_id: Spectrum ID to look up.
        counts: [tp, fp, fn, fired] from the main eval pass.
        ft_df: FT spectra DataFrame indexed by spec_id.
        ann_by_spec: Mapping spec_id → annotation dict (formula, prec_type, mzs, isos).
        match_tol: m/z tolerance for annotation matching.

    Returns:
        Dict for plot_eval_one, or None if data unavailable.
    """
    if spec_id not in ft_df.index or spec_id not in ann_by_spec:
        return None
    spec_row = ft_df.loc[spec_id]
    peaks = spec_row["peaks"]
    if not peaks:
        return None

    prec_mz = float(spec_row["prec_mz"])
    ann = ann_by_spec[spec_id]
    formula = str(ann["formula"]) if pd.notna(ann["formula"]) else None
    prec_type = str(ann["prec_type"]) if pd.notna(ann["prec_type"]) else None

    mzs = np.array([p[0] for p in peaks], dtype=np.float64)
    ints = np.array([p[1] for p in peaks], dtype=np.float64)

    mask = detect_isotope_peaks_for_training(
        mzs,
        ints,
        prec_mz=prec_mz,
        inst_type="FT",
        formula=formula,
        prec_type=prec_type,
        min_coiso_fraction=MIN_COISO,
    )

    ann_mzs = ann["ann_peak_mzs"]
    ann_iso = ann["ann_isotopes"]

    peak_iso_label: dict[int, int] = {}
    for ann_mz, is_iso in zip(ann_mzs, ann_iso):
        d = np.abs(mzs - ann_mz)
        best = int(np.argmin(d))
        if d[best] > match_tol * 10:
            continue
        if best not in peak_iso_label or int(is_iso) > peak_iso_label[best]:
            peak_iso_label[best] = int(is_iso)

    peak_labels: dict[int, str] = {}
    for peak_idx, iso_label in peak_iso_label.items():
        flagged = bool(mask[peak_idx])
        if iso_label > 0:
            peak_labels[peak_idx] = "tp" if flagged else "fn"
        elif flagged:
            peak_labels[peak_idx] = "fp"

    return {
        "spec_id": spec_id,
        "formula": formula or "",
        "prec_type": prec_type or "",
        "prec_mz": prec_mz,
        "mzs": mzs,
        "ints": ints,
        "peak_labels": peak_labels,
        "n_tp": counts[0],
        "n_fp": counts[1],
        "n_fn": counts[2],
        "fired": bool(counts[3]),
    }


def _save_eval_plots(
    spec_counts: dict,
    ft_df: pd.DataFrame,
    ann_by_spec: dict,
    match_tol: float,
) -> None:
    """Save top-N_EVAL_PLOTS spectra for each eval category (TP / FP / FN-fired / FN-nofire).

    Categories:
        TP        — spectra with most TP peaks (sorted by tp desc)
        FP        — spectra with most FP peaks (sorted by fp desc)
        FN_fired  — spectra where detector fired but still has FN peaks (fn desc)
        FN_nofire — spectra where detector never fired but had GT isotope peaks (fn desc)

    Args:
        spec_counts: Mapping spec_id → [tp, fp, fn, fired].
        ft_df: FT spectra indexed by spec_id.
        ann_by_spec: Annotation data keyed by spec_id.
        match_tol: m/z tolerance used during scoring.
    """
    eval_fig_root = FIG_DIR.parent / "eval_plots"

    categories: list[tuple[str, str, callable, callable]] = [
        ("TP", "top_tp", lambda v: v[0] > 0, lambda v: -v[0]),
        ("FP", "top_fp", lambda v: v[1] > 0, lambda v: -v[1]),
        ("FN_fired", "top_fn_fired", lambda v: v[2] > 0 and v[3] == 1, lambda v: -v[2]),
        ("FN_nofire", "top_fn_nofire", lambda v: v[2] > 0 and v[3] == 0, lambda v: -v[2]),
    ]

    for cat_name, subdir, filter_fn, sort_key in categories:
        candidates = [(sid, v) for sid, v in spec_counts.items() if filter_fn(v)]
        candidates.sort(key=lambda x: sort_key(x[1]))
        top = candidates[:N_EVAL_PLOTS]

        out_dir = eval_fig_root / subdir
        out_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"\nSaving {len(top)} {cat_name} plots → {out_dir}")

        for idx, (spec_id, counts) in enumerate(top):
            h = _build_eval_plot_data(spec_id, counts, ft_df, ann_by_spec, match_tol)
            if h is None:
                log.info(f"  [{idx + 1}] spec_id={spec_id} — data unavailable, skipping")
                continue
            out_path = out_dir / f"{idx + 1:03d}_spec{spec_id}.png"
            plot_eval_one(h, out_path, idx, len(top), cat_name)
            log.info(
                f"  [{idx + 1}/{len(top)}] spec_id={spec_id} "
                f"TP={counts[0]} FP={counts[1]} FN={counts[2]} fired={bool(counts[3])}"
            )


def eval_mode(df: pd.DataFrame) -> None:
    """Evaluate detect_isotope_peaks_for_training (formula mode, min_coiso=MIN_COISO)
    against NIST ann_df.

    Scoring is per raw peak (not per annotation): multiple NIST annotations that map
    to the same observed peak are deduplicated — the peak is labelled isotope if ANY
    of its annotations has ann_isotope > 0.

    True positive  = detector flags a peak whose best NIST annotation is isotope (k > 0)
    False positive = detector flags a peak whose best NIST annotation is a fragment (k == 0)
    False negative = detector misses a peak whose best NIST annotation is isotope

    Only annotated peaks are scored; unannotated peaks are not penalised.
    Saves top-N_EVAL_PLOTS spectra per category (TP / FP / FN-fired / FN-nofire).
    """
    import collections

    log.info("Loading ann_df…")
    ann_df = pd.read_pickle(PROC_DP / "nist20/ann_df.pkl")[
        ["spec_id", "formula", "prec_type", "ann_peak_mzs", "ann_isotopes"]
    ]

    # Keep only FT spectra (the only ones our detector handles)
    ft_spec_ids = set(df[df["inst_type"] == "FT"]["spec_id"].tolist())
    ann_df = ann_df[ann_df["spec_id"].isin(ft_spec_ids)].reset_index(drop=True)
    log.info(f"  FT annotated spectra: {len(ann_df):,}")

    ft_df = df[df["inst_type"] == "FT"].set_index("spec_id")
    match_tol = MZ_TOL_BY_INST["FT"]  # 4 mDa

    # Build ann_by_spec for second-pass plot building (first row per spec_id wins)
    ann_by_spec: dict[int, dict] = {}
    for _, arow in ann_df.iterrows():
        sid = int(arow["spec_id"])
        if sid not in ann_by_spec:
            ann_by_spec[sid] = {
                "formula": arow["formula"],
                "prec_type": arow["prec_type"],
                "ann_peak_mzs": np.array(arow["ann_peak_mzs"], dtype=np.float64),
                "ann_isotopes": np.array(arow["ann_isotopes"], dtype=np.int32),
            }

    # Global accumulators
    total_tp = 0
    total_fp = 0
    total_fn = 0
    n_fired = 0

    # Per-spectrum tracking: spec_id -> [tp, fp, fn, fired (0/1)]
    spec_counts: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])

    # FP concentration
    fp_per_spec: dict[int, int] = collections.Counter()

    # FN root-cause counters
    fn_no_fire = 0
    fn_fired_missed = 0
    fn_no_m0 = 0
    fn_monotone = 0
    fn_ratio = 0
    fn_other = 0
    fn_ratio_oor_vals: list[float] = []  # obs/R_expected for ratio-window rejections

    # No-fire diagnostic
    nofire_spec_seen: set[int] = set()
    nofire_no_formula = 0
    nofire_no_pairs = 0
    nofire_f_vals: list[float] = []
    nofire_f_median_vals: list[float] = []

    # Import helpers needed for FN sub-classification
    from fragnnet.dataset.isotope_cleanup import (
        _expected_mk_ratio,
        _parse_adduct_counts,
    )
    from fragnnet.utils.formula_utils import C13_SPACING as _C13
    from fragnnet.utils.formula_utils import parse_formula as _parse_formula

    _ratio_lo = 0.05  # must match detect_isotope_peaks_for_training default
    _ratio_hi = 10.0
    _mz_tol = MZ_TOL_BY_INST["FT"]

    def _classify_fn_peak(
        frag_idx: int,
        smzs: np.ndarray,
        sints: np.ndarray,
        sort_idx: np.ndarray,
        mol_counts: dict | None,
        adduct_counts: dict,
        adduct_delta_mass: float,
        prec_neutral_mass: float,
        r_exp_prec: dict[int, float],
        leak_ratios: dict[int, float],
        r_leak: dict[int, float],
        max_isotope: int,
    ) -> tuple[str, float | None]:
        """Return (reason, obs_over_expected); reason in 'no_m0'|'monotone'|'ratio'|'other'."""
        si = np.where(sort_idx == frag_idx)[0]
        if len(si) == 0:
            return "other", None
        i = si[0]
        frag_mz = smzs[i]
        frag_int = sints[i]

        for k in range(1, max_isotope + 1):
            if k not in leak_ratios:
                continue
            k_tol = _mz_tol if k == 1 else _mz_tol * 2
            target = frag_mz - k * _C13
            lo = np.searchsorted(smzs, target - k_tol, side="left")
            hi = np.searchsorted(smzs, target + k_tol, side="right")
            if lo >= hi:
                continue
            best_j = lo + int(np.argmax(sints[lo:hi]))
            m0_int = sints[best_j]
            m0_mz = smzs[best_j]
            if m0_int <= frag_int:
                return "monotone", None
            if m0_int <= 0:
                continue
            obs = frag_int / m0_int
            if mol_counts is not None and k in r_leak and prec_neutral_mass > 0:
                neutral_frag = m0_mz - adduct_delta_mass
                mol_scale = max(0.0, min(1.0, neutral_frag / prec_neutral_mass))
                r_exp_frag = _expected_mk_ratio(mol_counts, mol_scale, k, adduct_counts)
                r_expected = r_exp_frag * r_leak[k]
            else:
                r_expected = leak_ratios[k]
            if not (_ratio_lo * r_expected <= obs <= _ratio_hi * r_expected):
                oor = obs / r_expected if r_expected > 0 else float("nan")
                return "ratio", oor
            return "other", None
        return "no_m0", None

    for i, row in ann_df.iterrows():
        if (i + 1) % 20_000 == 0:
            log.info(f"  {i + 1:,}/{len(ann_df):,}")

        spec_id = row["spec_id"]
        if spec_id not in ft_df.index:
            continue
        spec_row = ft_df.loc[spec_id]
        peaks = spec_row["peaks"]
        if not peaks:
            continue

        prec_mz = float(spec_row["prec_mz"])
        formula = str(row["formula"]) if pd.notna(row["formula"]) else None
        prec_type = str(row["prec_type"]) if pd.notna(row["prec_type"]) else None
        ann_mzs = np.array(row["ann_peak_mzs"], dtype=np.float64)
        ann_iso = np.array(row["ann_isotopes"], dtype=np.int32)

        mzs = np.array([p[0] for p in peaks], dtype=np.float64)
        ints = np.array([p[1] for p in peaks], dtype=np.float64)

        mask = detect_isotope_peaks_for_training(
            mzs,
            ints,
            prec_mz=prec_mz,
            inst_type="FT",
            formula=formula,
            prec_type=prec_type,
            min_coiso_fraction=MIN_COISO,
        )

        fired = bool(mask.any())
        if fired:
            n_fired += 1
            spec_counts[spec_id][3] = 1

        # Precompute sorted arrays + formula data for FN sub-classification
        sort_idx_arr = np.argsort(mzs)
        smzs = mzs[sort_idx_arr]
        sints = ints[sort_idx_arr]
        _mol_counts: dict | None = None
        _adduct_counts: dict = {}
        _adduct_delta: float = 0.0
        _prec_neutral: float = 0.0
        _r_exp_prec: dict[int, float] = {}
        if formula is not None:
            try:
                _mol_counts = _parse_formula(formula)
                _adduct_counts, _adduct_delta = _parse_adduct_counts(prec_type)
                _prec_neutral = prec_mz - _adduct_delta
                for _k in range(1, MAX_K + 1):
                    _r_exp_prec[_k] = _expected_mk_ratio(_mol_counts, 1.0, _k, _adduct_counts)
            except Exception:
                _mol_counts = None

        # Deduplicate annotations by raw peak index
        peak_iso_label: dict[int, int] = {}
        for ann_mz, is_iso in zip(ann_mzs, ann_iso):
            d = np.abs(mzs - ann_mz)
            idx = int(np.argmin(d))
            if d[idx] > match_tol * 10:
                continue
            if idx not in peak_iso_label or int(is_iso) > peak_iso_label[idx]:
                peak_iso_label[idx] = int(is_iso)

        # Score each unique observed peak
        for peak_idx, iso_label in peak_iso_label.items():
            flagged = bool(mask[peak_idx])
            if iso_label > 0:
                if flagged:
                    total_tp += 1
                    spec_counts[spec_id][0] += 1
                else:
                    total_fn += 1
                    spec_counts[spec_id][2] += 1
            else:
                if flagged:
                    total_fp += 1
                    spec_counts[spec_id][1] += 1
                    fp_per_spec[spec_id] += 1

            # FN root-cause
            if iso_label > 0 and not flagged:
                if not fired:
                    fn_no_fire += 1
                    if spec_id not in nofire_spec_seen:
                        nofire_spec_seen.add(spec_id)
                        if _mol_counts is None:
                            nofire_no_formula += 1
                        else:
                            best_f: float | None = None
                            all_f_vals: list[float] = []
                            for pi in range(len(smzs)):
                                m0_int = sints[pi]
                                if m0_int <= 0:
                                    continue
                                target = smzs[pi] + _C13
                                lo = np.searchsorted(smzs, target - _mz_tol, side="left")
                                hi = np.searchsorted(smzs, target + _mz_tol, side="right")
                                if lo >= hi:
                                    continue
                                mk_int = sints[lo + int(np.argmax(sints[lo:hi]))]
                                if mk_int <= 0 or mk_int >= m0_int:
                                    continue
                                r_raw = mk_int / m0_int
                                neutral_frag = smzs[pi] - _adduct_delta
                                ms = (
                                    max(0.0, min(1.0, neutral_frag / _prec_neutral))
                                    if _prec_neutral > 0
                                    else 1.0
                                )
                                r_exp = _expected_mk_ratio(_mol_counts, ms, 1, _adduct_counts)
                                f_est = r_raw / r_exp if r_exp > 0 else 0.0
                                all_f_vals.append(f_est)
                                if best_f is None or f_est > best_f:
                                    best_f = f_est
                            if best_f is None:
                                nofire_no_pairs += 1
                            else:
                                nofire_f_vals.append(best_f)
                                nofire_f_median_vals.append(float(np.median(all_f_vals)))
                else:
                    fn_fired_missed += 1
                    flagged_idxs = np.where(mask)[0]
                    _leak_ratios: dict[int, float] = {}
                    _r_leak: dict[int, float] = {}
                    if len(flagged_idxs) > 0 and _mol_counts is not None:
                        ratios_k1 = []
                        for fi in flagged_idxs:
                            fi_mz = mzs[fi]
                            fi_int = ints[fi]
                            target = fi_mz - _C13
                            lo = np.searchsorted(smzs, target - _mz_tol, side="left")
                            hi = np.searchsorted(smzs, target + _mz_tol, side="right")
                            if lo >= hi:
                                continue
                            m0_int = sints[lo + int(np.argmax(sints[lo:hi]))]
                            if m0_int > 0 and m0_int > fi_int:
                                ratios_k1.append(fi_int / m0_int)
                        if ratios_k1:
                            r_k1 = float(np.median(ratios_k1))
                            _leak_ratios[1] = r_k1
                            if _r_exp_prec.get(1, 0) > 0:
                                _r_leak[1] = r_k1 / _r_exp_prec[1]
                    reason, oor = _classify_fn_peak(
                        peak_idx,
                        smzs,
                        sints,
                        sort_idx_arr,
                        _mol_counts,
                        _adduct_counts,
                        _adduct_delta,
                        _prec_neutral,
                        _r_exp_prec,
                        _leak_ratios,
                        _r_leak,
                        max_isotope=MAX_K,
                    )
                    if reason == "no_m0":
                        fn_no_m0 += 1
                    elif reason == "monotone":
                        fn_monotone += 1
                    elif reason == "ratio":
                        fn_ratio += 1
                        if oor is not None and not np.isnan(oor):
                            fn_ratio_oor_vals.append(oor)
                    else:
                        fn_other += 1

    # ── Summary stats ────────────────────────────────────────────────────────
    tp, fp, fn = total_tp, total_fp, total_fn
    pv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    rv = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = 2 * pv * rv / (pv + rv) if (pv + rv) > 0 else float("nan")
    log.info(
        f"\n── Evaluation vs NIST ann_isotopes (FT, formula mode, min_coiso={MIN_COISO}) ──"
    )
    log.info(
        f"  fired={n_fired:,}  TP={tp:,}  FP={fp:,}  FN={fn:,}  "
        f"prec={pv:.3f}  rec={rv:.3f}  F1={f1:.3f}"
    )

    # FP concentration
    fp_counts = sorted(fp_per_spec.values())
    if fp_counts:
        from statistics import median

        log.info(
            f"\n── FP concentration (min_coiso={MIN_COISO}, "
            f"{len(fp_counts):,} spectra with ≥1 FP) ──"
        )
        log.info(f"  Median FPs per spec: {median(fp_counts):.1f}  Max: {max(fp_counts)}")
        for threshold in (5, 10, 20):
            heavy = sum(1 for c in fp_counts if c >= threshold)
            heavy_fp = sum(c for c in fp_counts if c >= threshold)
            log.info(
                f"  Spectra with ≥{threshold} FPs: {heavy:,}  "
                f"({heavy_fp / sum(fp_counts) * 100:.1f}% of all FPs)"
            )

    top_fp_list = sorted(spec_counts.items(), key=lambda x: -x[1][1])[:10]
    log.info(f"\n  Top-10 spectra by FP (min_coiso={MIN_COISO}):")
    log.info(f"  {'spec_id':>10}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    for sid, v in top_fp_list:
        log.info(f"  {sid:>10}  {v[0]:>5}  {v[1]:>5}  {v[2]:>5}")

    # FN root-cause breakdown
    total_fn_count = fn_no_fire + fn_fired_missed
    log.info(f"\n── FN root-cause breakdown (min_coiso={MIN_COISO}) ─────────────────")
    log.info(f"  Total FN:                      {total_fn_count:,}")
    if total_fn_count > 0:
        log.info(
            f"  Detector never fired:          {fn_no_fire:,}  "
            f"({fn_no_fire / total_fn_count * 100:.1f}%)"
        )
        log.info(
            f"  Detector fired, peak missed:   {fn_fired_missed:,}  "
            f"({fn_fired_missed / total_fn_count * 100:.1f}%)"
        )
    if fn_fired_missed > 0:
        log.info(
            f"    ├ no M+0 partner found:      {fn_no_m0:,}  "
            f"({fn_no_m0 / fn_fired_missed * 100:.1f}%)"
        )
        log.info(
            f"    ├ monotonicity failed:        {fn_monotone:,}  "
            f"({fn_monotone / fn_fired_missed * 100:.1f}%)"
        )
        log.info(
            f"    ├ ratio window rejected:      {fn_ratio:,}  "
            f"({fn_ratio / fn_fired_missed * 100:.1f}%)"
        )
        log.info(
            f"    └ other (envelope/unknown):   {fn_other:,}  "
            f"({fn_other / fn_fired_missed * 100:.1f}%)"
        )

    if fn_ratio_oor_vals:
        oor = np.array(fn_ratio_oor_vals)
        log.info(f"\n── Ratio-window FN: obs/R_expected distribution ({len(oor):,} peaks) ──")
        log.info(
            f"  Below window (obs < {_ratio_lo}×): {(oor < _ratio_lo).sum():,}  "
            f"({(oor < _ratio_lo).mean() * 100:.1f}%)"
        )
        log.info(
            f"  Above window (obs > {_ratio_hi}×): {(oor > _ratio_hi).sum():,}  "
            f"({(oor > _ratio_hi).mean() * 100:.1f}%)"
        )
        log.info(
            f"  Percentiles: p5={np.percentile(oor, 5):.3f}  p10={np.percentile(oor, 10):.3f}  "
            f"p25={np.percentile(oor, 25):.3f}  median={np.median(oor):.3f}  "
            f"p75={np.percentile(oor, 75):.3f}  p90={np.percentile(oor, 90):.3f}  "
            f"p95={np.percentile(oor, 95):.3f}"
        )
        log.info("  Recoverable by widening ratio_lo/ratio_hi:")
        for lo_thr, hi_thr in [(0.3, 3.0), (0.2, 5.0), (0.1, 5.0), (0.05, 10.0)]:
            n_rec = int(((oor >= lo_thr) & (oor <= hi_thr)).sum())
            log.info(
                f"    [{lo_thr}, {hi_thr}]: recovers {n_rec:,}  ({n_rec / len(oor) * 100:.1f}%)"
            )

    # No-fire spectra: f-value distribution
    n_nofire_specs = len(nofire_spec_seen)
    n_has_pairs = len(nofire_f_vals)
    log.info(f"\n── No-fire spectrum analysis (min_coiso={MIN_COISO}) ────────────────")
    log.info(f"  Unique no-fire spectra with ≥1 FN peak: {n_nofire_specs:,}")
    if n_nofire_specs:
        log.info(
            f"  No formula available:                   {nofire_no_formula:,}  "
            f"({nofire_no_formula / n_nofire_specs * 100:.1f}%)"
        )
        log.info(
            f"  Formula but no M+k pair found:          {nofire_no_pairs:,}  "
            f"({nofire_no_pairs / n_nofire_specs * 100:.1f}%)"
        )
        log.info(
            f"  Has ≥1 pair (f estimable):              {n_has_pairs:,}  "
            f"({n_has_pairs / n_nofire_specs * 100:.1f}%)"
        )
    if nofire_f_vals:
        f_arr = np.array(nofire_f_vals)
        fm_arr = np.array(nofire_f_median_vals)
        thresholds = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30]
        log.info("  f distribution — BEST pair per spectrum (max f across all pairs):")
        log.info(
            f"    p10={np.percentile(f_arr, 10):.3f}  p25={np.percentile(f_arr, 25):.3f}  "
            f"median={np.median(f_arr):.3f}  p75={np.percentile(f_arr, 75):.3f}  "
            f"p90={np.percentile(f_arr, 90):.3f}  mean={f_arr.mean():.3f}"
        )
        log.info("  f distribution — MEDIAN across all pairs per spectrum:")
        log.info(
            f"    p10={np.percentile(fm_arr, 10):.3f}  p25={np.percentile(fm_arr, 25):.3f}  "
            f"median={np.median(fm_arr):.3f}  p75={np.percentile(fm_arr, 75):.3f}  "
            f"p90={np.percentile(fm_arr, 90):.3f}  mean={fm_arr.mean():.3f}"
        )
        log.info("  Recoverable by strategy (f ≥ threshold):")
        log.info(f"    {'threshold':>10}  {'by best-pair':>14}  {'by median':>12}")
        for thr in thresholds:
            n_best = int((f_arr >= thr).sum())
            n_med = int((fm_arr >= thr).sum())
            log.info(
                f"    f ≥ {thr:.2f}:    {n_best:>7,} ({n_best / n_has_pairs * 100:4.1f}%)   "
                f"{n_med:>7,} ({n_med / n_has_pairs * 100:4.1f}%)"
            )

    # ── Save top-N plots per category ────────────────────────────────────────
    _save_eval_plots(spec_counts, ft_df, ann_by_spec, match_tol)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Precursor isotope leakage analysis")
    parser.add_argument(
        "--count",
        action="store_true",
        help="Count leakage across all NIST20 (no plots saved)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="Evaluate detect_isotope_peaks_for_training vs NIST annotations",
    )
    args = parser.parse_args()

    df = _load_df()
    if args.count:
        count_mode(df)
    elif args.eval:
        eval_mode(df)
    else:
        plot_mode(df)


if __name__ == "__main__":
    main()
