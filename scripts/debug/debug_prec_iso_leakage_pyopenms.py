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
    conda run -n FRAGNNET-GPU python scripts/debug/debug_prec_iso_leakage_pyopenms.py
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fragnnet.utils.formula_utils import isotope_offsets_from_formula, parse_formula
from fragnnet.utils.spec_utils import (
    detect_isotope_peaks_for_training,
    detect_isotope_peaks_formula_aware,
)

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


# min_coiso_fraction values to sweep in eval_mode (co-isolation fraction f = R_k/R_exp_prec)
MIN_COISO_SWEEP: list[float] = [0.05, 0.10, 0.20]


def eval_mode(df: pd.DataFrame) -> None:
    """Evaluate detect_isotope_peaks_for_training (formula mode) against NIST ann_df.

    Always uses formula + prec_type (always available in practice).

    Sweeps min_coiso_fraction values (MIN_COISO_SWEEP): the minimum co-isolation
    fraction f = R_k/R_exp_prec required to trigger flagging.  Lower values detect
    weaker co-isolation but may increase false positives.

    Also reports per-spectrum FP breakdown for the first sweep value to diagnose
    whether FPs are concentrated in a few spectra.

    Scoring is per raw peak (not per annotation): multiple NIST annotations that map
    to the same observed peak are deduplicated — the peak is labelled isotope if ANY
    of its annotations has ann_isotope > 0.  This prevents a single flagged peak with
    N NIST annotations from inflating FP/TP counts by N×.

    True positive  = detector flags a peak whose best NIST annotation is isotope (k > 0)
    False positive = detector flags a peak whose best NIST annotation is a fragment (k == 0)
    False negative = detector misses a peak whose best NIST annotation is isotope

    Only annotated peaks are scored; unannotated peaks are not penalised.
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

    # Per-sweep accumulators
    sweep_stats: dict[float, dict[str, int]] = {
        mcf: {"tp": 0, "fp": 0, "fn": 0} for mcf in MIN_COISO_SWEEP
    }
    sweep_fired: dict[float, int] = dict.fromkeys(MIN_COISO_SWEEP, 0)

    # FP concentration for the first sweep value only
    primary_mlf = MIN_COISO_SWEEP[0]
    spec_fp_counts: dict[int, list[int]] = collections.defaultdict(lambda: [0, 0, 0])

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

        masks: dict[float, np.ndarray] = {}
        for mcf in MIN_COISO_SWEEP:
            masks[mcf] = detect_isotope_peaks_for_training(
                mzs,
                ints,
                prec_mz=prec_mz,
                inst_type="FT",
                formula=formula,
                prec_type=prec_type,
                min_coiso_fraction=mcf,
            )

        # Track fired count and FP concentration for primary sweep value
        primary_fired = bool(masks[primary_mlf].any())
        if primary_fired:
            sweep_fired[primary_mlf] += 1

        # Deduplicate annotations by raw peak index: one entry per peak, taking
        # max(is_iso) so that a peak annotated as both fragment and isotope is
        # treated as isotope (conservative — avoids penalising the detector).
        peak_iso_label: dict[int, int] = {}
        for ann_mz, is_iso in zip(ann_mzs, ann_iso):
            d = np.abs(mzs - ann_mz)
            idx = int(np.argmin(d))
            if d[idx] > match_tol * 10:
                continue
            if idx not in peak_iso_label or int(is_iso) > peak_iso_label[idx]:
                peak_iso_label[idx] = int(is_iso)

        # Score each unique observed peak across all sweep values
        for peak_idx, iso_label in peak_iso_label.items():
            for mcf, mask in masks.items():
                flagged = bool(mask[peak_idx])
                if iso_label > 0:
                    if flagged:
                        sweep_stats[mcf]["tp"] += 1
                    else:
                        sweep_stats[mcf]["fn"] += 1
                else:
                    if flagged:
                        sweep_stats[mcf]["fp"] += 1
                        if mcf == primary_mlf and primary_fired:
                            spec_fp_counts[spec_id][1] += 1

            if primary_fired and iso_label > 0:
                # tp/fn within fired spectra (for FP concentration table)
                flagged = bool(masks[primary_mlf][peak_idx])
                if flagged:
                    spec_fp_counts[spec_id][0] += 1
                else:
                    spec_fp_counts[spec_id][2] += 1

        # Track fired for non-primary sweep values
        for mcf in MIN_COISO_SWEEP[1:]:
            if masks[mcf].any():
                sweep_fired[mcf] += 1

    log.info("\n── Evaluation vs NIST ann_isotopes (FT, formula mode — min_coiso_fraction sweep) ──")
    log.info(
        f"  {'min_coiso':>10}  {'fired':>7}  {'TP':>7}  {'FP':>7}  {'FN':>8}  "
        f"{'prec':>7}  {'rec':>7}  {'F1':>7}"
    )
    for mcf in MIN_COISO_SWEEP:
        s = sweep_stats[mcf]
        tp, fp, fn = s["tp"], s["fp"], s["fn"]
        pv = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
        rv = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
        f1 = 2 * pv * rv / (pv + rv) if (pv + rv) > 0 else float("nan")
        log.info(
            f"  {mcf:>10.2f}  {sweep_fired[mcf]:>7,}  {tp:>7,}  {fp:>7,}  {fn:>8,}  "
            f"{pv:>7.3f}  {rv:>7.3f}  {f1:>7.3f}"
        )

    # FP concentration for primary sweep value
    fp_counts = sorted(v[1] for v in spec_fp_counts.values() if v[1] > 0)
    if fp_counts:
        from statistics import median

        log.info(
            f"\n── FP concentration for min_coiso={primary_mlf} "
            f"({len(fp_counts):,} spectra with ≥1 FP) ──"
        )
        log.info(f"  Median FPs per spec: {median(fp_counts):.1f}  Max: {max(fp_counts)}")
        for threshold in (5, 10, 20):
            heavy = sum(1 for c in fp_counts if c >= threshold)
            heavy_fp = sum(c for c in fp_counts if c >= threshold)
            log.info(
                f"  Spectra with ≥{threshold} FPs: {heavy:,}  "
                f"({heavy_fp / sum(fp_counts) * 100:.1f}% of all FPs)"
            )

    top_fp = sorted(spec_fp_counts.items(), key=lambda x: -x[1][1])[:10]
    log.info(f"\n  Top-10 spectra by FP (max_leak={primary_mlf}):")
    log.info(f"  {'spec_id':>10}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    for sid, (tp, fp, fn) in top_fp:
        log.info(f"  {sid:>10}  {tp:>5}  {fp:>5}  {fn:>5}")


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
