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
from fragnnet.utils.spec_utils import detect_isotope_peaks_formula_aware

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


# Single universal ratio window — the isotope ratio I(M+k)/I(M+0) is a physical
# property of the molecule, not the instrument. mz_tol handles instrument-specific
# mass accuracy; the ratio gate is formula-based and applies equally to all instruments.
# [0.5, 2.0] allows ±2x around the expected ratio — covers real measurement noise
# without being so loose it accepts coincidental peaks.
RATIO_LO = 0.7
RATIO_HI = 1.5


def find_prec_leakage(
    mzs: np.ndarray,
    ints: np.ndarray,
    prec_mz: float,
    element_counts: dict[str, int],
    mz_tol: float = MZ_TOL_DEFAULT,
    max_k: int = MAX_K,
    lo: float = RATIO_LO,
    hi: float = RATIO_HI,
) -> list[tuple[int, int, float]]:
    """Return (peak_idx, k, delta_mz) for above-prec peaks confirmed as isotope leakage.

    A candidate is confirmed only if:
      1. m/z matches prec_mz + pyopenms_offset(k) within mz_tol, AND
      2. obs_ratio = I(candidate) / I(M+0) is within [lo * exp_ratio, hi * exp_ratio].
         If no M+0 peak is found, the ratio check is skipped (m/z only).

    Args:
        mzs: All peak m/z values.
        ints: All peak intensities.
        prec_mz: Precursor m/z.
        element_counts: Molecular formula element counts from parse_formula.
        mz_tol: m/z tolerance in Da.
        max_k: Maximum isotope order to check.
        lo: Lower multiplier on expected ratio.
        hi: Upper multiplier on expected ratio.

    Returns:
        List of (peak_index, k, delta_mz) for confirmed leakage peaks.
    """
    offsets = isotope_offsets_from_formula(element_counts, max_isotope=max_k)

    # Find M+0 precursor peak intensity for ratio check — wide tolerance,
    # locating the peak not discriminating isotopes.
    M0_TOL = 0.05
    d0 = np.abs(mzs - prec_mz)
    m0_idx = int(np.argmin(d0))
    m0_int = float(ints[m0_idx]) if d0[m0_idx] <= M0_TOL else None

    # Check each k in order; stop as soon as one fails — M+k requires M+1…M+(k-1)
    # all confirmed (envelope coherence: no M+2 without M+1).
    confirmed_ks: set[int] = set()
    leakage = []
    for k, (offset, exp_ratio) in enumerate(offsets, start=1):
        # Envelope coherence: all lower orders must be confirmed first
        if k > 1 and (k - 1) not in confirmed_ks:
            break

        target = prec_mz + offset
        d = np.abs(mzs - target)
        best_idx = int(np.argmin(d))
        if d[best_idx] > mz_tol or mzs[best_idx] <= prec_mz + mz_tol:
            break  # M+k missing — no point checking M+(k+1)

        # Ratio check against M+0
        if m0_int is not None and m0_int > 0:
            obs_ratio = float(ints[best_idx]) / m0_int
            if not (lo * exp_ratio <= obs_ratio <= hi * exp_ratio):
                break  # ratio wrong — stop envelope here

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
        Line2D([0], [0], color="tomato",    lw=1.5, label="Fragment isotope (pyopenms)"),
        Line2D([0], [0], color="crimson",   lw=2.0, label="Prec leakage M+k"),
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
        0.01, 0.97, f"Formula: {h['formula']}",
        transform=ax_full.transAxes, fontsize=8, va="top", ha="left",
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "lightyellow", "alpha": 0.8},
    )
    ax_full.legend(handles=legend_handles, fontsize=7, loc="upper right")
    # shade the zoom region
    ax_full.axvspan(zoom_lo, zoom_hi, color="gold", alpha=0.12, zorder=0)
    ax_full.axvline(prec_mz, color="black", lw=1.0, ls="--", alpha=0.4)
    ax_full.text(prec_mz, 104, f"prec\n{prec_mz:.3f}", fontsize=6,
                 color="black", ha="center", va="bottom", alpha=0.6)

    # ── Right: zoomed into prec region ───────────────────────────────────────
    _draw_peaks(ax_zoom)
    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    ax_zoom.set_title(f"Zoom: prec±5 Da → max mz  ({zoom_lo:.1f}–{zoom_hi:.1f})", fontsize=9)
    ax_zoom.axvline(prec_mz, color="black", lw=1.0, ls="--", alpha=0.4)
    ax_zoom.text(prec_mz, 104, f"prec\n{prec_mz:.4f}", fontsize=7,
                 color="black", ha="center", va="bottom", alpha=0.7)

    # Label M+0 on zoom panel
    if 0 in mk_peaks:
        m0_mz, m0_raw = mk_peaks[0]
        m0_norm = ints_norm[int(np.argmin(np.abs(mzs - m0_mz)))]
        ax_zoom.annotate(
            f"M\n{m0_mz:.4f}\n{m0_raw:.0f}",
            xy=(m0_mz, m0_norm), xytext=(0, 6), textcoords="offset points",
            fontsize=8, color="darkgreen", ha="center", va="bottom", fontweight="bold",
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
            xy=(mz, rel_int), xytext=(0, 6), textcoords="offset points",
            fontsize=8, color="crimson", ha="center", va="bottom", fontweight="bold",
        )

    ks_str = ", ".join(f"M+{k}" for _, k, _ in sorted(h["leakage"], key=lambda x: x[1]))
    fig.suptitle(
        f"[{idx + 1}/{N_PLOTS}]  spec_id={h['spec_id']}  |  {h['prec_type']}  |  "
        f"inst={h['inst_type']}  mz_tol={h['mz_tol'] * 1000:.0f} mDa  "
        f"ratio=[{RATIO_LO:.2f}×, {RATIO_HI:.2f}×]  |  "
        f"prec_mz={prec_mz:.4f}  |  leakage: {ks_str}",
        fontsize=9, y=1.01,
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
        "ratio_window": (RATIO_LO, RATIO_HI),
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
    log.info(f"  Have any peak > prec_mz:         {n_candidates:,}  ({n_candidates/total*100:.1f}%)")
    log.info(f"  Confirmed leakage (pyopenms):     {n_hit:,}  ({n_hit/total*100:.1f}%)")
    log.info(f"  Confirmed / candidates:           {n_hit/n_candidates*100:.1f}%")
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


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Precursor isotope leakage analysis")
    parser.add_argument(
        "--count",
        action="store_true",
        help="Count leakage across all NIST20 (no plots saved)",
    )
    args = parser.parse_args()

    df = _load_df()
    if args.count:
        count_mode(df)
    else:
        plot_mode(df)


if __name__ == "__main__":
    main()
