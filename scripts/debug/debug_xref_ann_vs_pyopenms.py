"""Cross-reference NIST ann_df isotope labels with pyopenms precursor leakage detection.

For each spectrum that has above-prec peaks AND ann_df annotations, compare the two
detection sources peak-by-peak in the precursor region (prec_mz - 5 Da → max_mz).

Agreement categories (per above-prec peak):
  - AGREE-ISO   (both): ann_df k>0 AND pyopenms leakage  → tomato
  - POM-ONLY    : pyopenms detects leakage, ann_df k=0 or unannotated → orange
  - ANN-ONLY    : ann_df k>0, pyopenms missed it         → purple
  - AGREE-FRAG  : ann_df k=0 (annotated fragment)        → steelblue
  - UNANNOTATED : no ann_df label                        → grey
  - M+0 prec peak                                        → darkgreen

Each PNG shows:
  - Left: full spectrum with colour-coded peaks and leakage labels
  - Right: zoom into prec region with per-peak labels (M+k, formula, exp/obs ratio)

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/debug_xref_ann_vs_pyopenms.py
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
FIG_DIR = Path("figs/xref_ann_vs_pyopenms")

MAX_K = 3
N_PLOTS = 50
MZ_TOL_MATCH = 0.015   # ann_df peak → obs peak match tolerance (Da)
RATIO_LO = 0.7
RATIO_HI = 1.5

MZ_TOL_BY_INST: dict[str, float] = {
    "FT": 0.004,
    "QTOF": 0.010,
    "IT": 0.020,
    "QQQ": 0.020,
}
MZ_TOL_DEFAULT = 0.020


# ── Agreement category constants ──────────────────────────────────────────────
AGREE_ISO = "agree_iso"    # both say isotope
POM_ONLY = "pom_only"      # pyopenms only
ANN_ONLY = "ann_only"      # ann_df only
AGREE_FRAG = "agree_frag"  # ann_df says fragment (k=0), pyopenms agrees (no leakage)
UNANNOTATED = "unannotated"

_COLOR: dict[str, str] = {
    AGREE_ISO: "tomato",
    POM_ONLY: "darkorange",
    ANN_ONLY: "mediumpurple",
    AGREE_FRAG: "steelblue",
    UNANNOTATED: "grey",
}
_LW: dict[str, float] = {
    AGREE_ISO: 2.2,
    POM_ONLY: 2.0,
    ANN_ONLY: 2.0,
    AGREE_FRAG: 1.2,
    UNANNOTATED: 0.8,
}
_ALPHA: dict[str, float] = {
    AGREE_ISO: 0.95,
    POM_ONLY: 0.90,
    ANN_ONLY: 0.90,
    AGREE_FRAG: 0.85,
    UNANNOTATED: 0.40,
}
_ZO: dict[str, int] = {
    AGREE_ISO: 5,
    POM_ONLY: 4,
    ANN_ONLY: 4,
    AGREE_FRAG: 2,
    UNANNOTATED: 1,
}


# ── Helpers from debug_prec_iso_leakage_pyopenms ──────────────────────────────

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

    Args:
        mzs: All peak m/z values.
        ints: All peak intensities.
        prec_mz: Precursor m/z.
        element_counts: Molecular formula element counts from parse_formula.
        mz_tol: m/z tolerance in Da.
        max_k: Maximum isotope order.
        lo: Lower multiplier on expected ratio.
        hi: Upper multiplier on expected ratio.

    Returns:
        List of (peak_index, k, delta_mz) for confirmed leakage peaks.
    """
    offsets = isotope_offsets_from_formula(element_counts, max_isotope=max_k)
    M0_TOL = 0.05
    d0 = np.abs(mzs - prec_mz)
    m0_idx = int(np.argmin(d0))
    m0_int = float(ints[m0_idx]) if d0[m0_idx] <= M0_TOL else None

    confirmed_ks: set[int] = set()
    leakage = []
    for k, (offset, exp_ratio) in enumerate(offsets, start=1):
        if k > 1 and (k - 1) not in confirmed_ks:
            break
        target = prec_mz + offset
        d = np.abs(mzs - target)
        best_idx = int(np.argmin(d))
        if d[best_idx] > mz_tol or mzs[best_idx] <= prec_mz + mz_tol:
            break
        if m0_int is not None and m0_int > 0:
            obs_ratio = float(ints[best_idx]) / m0_int
            if not (lo * exp_ratio <= obs_ratio <= hi * exp_ratio):
                break
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
    """Find M+0..M+max_k peaks near prec_mz and expected ratios from pyopenms.

    Args:
        mzs: Peak m/z values.
        ints: Peak intensities.
        prec_mz: Precursor m/z.
        element_counts: Molecular formula element counts.
        mz_tol: m/z tolerance in Da.
        max_k: Maximum isotope order.

    Returns:
        Tuple of:
          - obs: dict k -> (observed_mz, raw_intensity)
          - exp_ratios: dict k -> expected M+k/M+0 intensity ratio
    """
    offsets = isotope_offsets_from_formula(element_counts, max_isotope=max_k)
    obs: dict[int, tuple[float, float]] = {}
    exp_ratios: dict[int, float] = {}

    M0_TOL = 0.05
    d = np.abs(mzs - prec_mz)
    idx = int(np.argmin(d))
    if d[idx] <= M0_TOL:
        obs[0] = (float(mzs[idx]), float(ints[idx]))

    for k, (offset, exp_ratio) in enumerate(offsets, start=1):
        exp_ratios[k] = float(exp_ratio)
        target = prec_mz + offset
        d = np.abs(mzs - target)
        idx = int(np.argmin(d))
        if d[idx] <= mz_tol and mzs[idx] > prec_mz + mz_tol:
            obs[k] = (float(mzs[idx]), float(ints[idx]))

    return obs, exp_ratios


# ── Helper from debug_ann_df_iso_peaks ────────────────────────────────────────

def match_ann_to_obs(
    obs_mzs: np.ndarray,
    ann_mzs: list[float],
    ann_iso_ks: list[int],
    ann_products: list[str],
    mz_tol: float = MZ_TOL_MATCH,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Match ann_df annotations to observed peaks by m/z proximity.

    Args:
        obs_mzs: Observed peak m/z values of shape (n_peaks,).
        ann_mzs: Annotated peak m/z values from ann_df.
        ann_iso_ks: Isotope order k per annotated peak (0 = monoisotopic).
        ann_products: NIST product formula string per annotated peak.
        mz_tol: m/z tolerance in Da.

    Returns:
        Tuple of:
          - iso_k_arr: int array (n_peaks,), -1=unannotated, 0=monoisotopic, k>0=isotope.
          - is_ann: bool array (n_peaks,).
          - label_arr: list of str, NIST product formula per peak (empty if unannotated).
    """
    iso_k_arr = np.full(len(obs_mzs), -1, dtype=int)
    is_ann = np.zeros(len(obs_mzs), dtype=bool)
    label_arr: list[str] = [""] * len(obs_mzs)

    for ann_mz, k, product in zip(ann_mzs, ann_iso_ks, ann_products):
        d = np.abs(obs_mzs - ann_mz)
        best = int(np.argmin(d))
        if d[best] <= mz_tol:
            if not is_ann[best] or k > iso_k_arr[best]:
                iso_k_arr[best] = k
                is_ann[best] = True
                label_arr[best] = str(product)

    return iso_k_arr, is_ann, label_arr


# ── Category assignment ───────────────────────────────────────────────────────

def assign_categories(
    mzs: np.ndarray,
    iso_k_arr: np.ndarray,
    is_ann: np.ndarray,
    pom_leakage_idxs: set[int],
    pom_leakage_k: dict[int, int],
) -> list[str]:
    """Assign an agreement category to each observed peak.

    Args:
        mzs: Observed peak m/z values.
        iso_k_arr: Per-peak isotope k from ann_df (-1 = unannotated).
        is_ann: True where ann_df matched a peak.
        pom_leakage_idxs: Set of peak indices flagged as leakage by pyopenms.
        pom_leakage_k: dict peak_idx -> k for pyopenms leakage peaks.

    Returns:
        List of category strings, one per peak.
    """
    cats: list[str] = []
    for i in range(len(mzs)):
        ann_k = int(iso_k_arr[i])
        ann_iso = is_ann[i] and ann_k > 0
        pom_iso = i in pom_leakage_idxs

        if ann_iso and pom_iso:
            cats.append(AGREE_ISO)
        elif pom_iso and not ann_iso:
            cats.append(POM_ONLY)
        elif ann_iso and not pom_iso:
            cats.append(ANN_ONLY)
        elif is_ann[i] and ann_k == 0:
            cats.append(AGREE_FRAG)
        else:
            cats.append(UNANNOTATED)
    return cats


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_one(h: dict, out_path: Path, idx: int, total: int) -> None:
    """Save one cross-reference spectrum plot to out_path.

    Args:
        h: Hit dict with keys: spec_id, formula, prec_type, inst_type, prec_mz,
           mz_tol, mzs, ints, cats, label_arr, iso_k_arr, pom_leakage,
           mk_peaks, exp_ratios.
        out_path: File path to save PNG.
        idx: 0-based hit index.
        total: Total hits.
    """
    mzs = h["mzs"]
    ints = h["ints"]
    prec_mz = h["prec_mz"]
    cats = h["cats"]
    label_arr = h["label_arr"]
    iso_k_arr = h["iso_k_arr"]
    pom_leakage = h["pom_leakage"]          # list of (peak_idx, k, delta_mz)
    mk_peaks = h["mk_peaks"]               # dict k -> (mz, raw_int)
    exp_ratios = h["exp_ratios"]           # dict k -> expected ratio

    ints_norm = ints / ints.max() * 100 if ints.max() > 0 else ints.copy()
    m0_raw_int = mk_peaks[0][1] if 0 in mk_peaks else None

    zoom_lo = prec_mz - 5.0
    zoom_hi = mzs.max() + 1.0

    from matplotlib.lines import Line2D

    legend_handles = [
        Line2D([0], [0], color=_COLOR[AGREE_ISO],   lw=2.0, label="Agree: isotope (both)"),
        Line2D([0], [0], color=_COLOR[POM_ONLY],    lw=2.0, label="pyopenms only"),
        Line2D([0], [0], color=_COLOR[ANN_ONLY],    lw=2.0, label="ann_df only"),
        Line2D([0], [0], color=_COLOR[AGREE_FRAG],  lw=1.2, label="Agree: fragment (ann_df k=0)"),
        Line2D([0], [0], color=_COLOR[UNANNOTATED], lw=0.8, label="Unannotated", alpha=0.5),
        Line2D([0], [0], color="darkgreen",         lw=2.0, label="M+0 precursor"),
    ]

    def _draw_peaks(ax: plt.Axes) -> None:
        for i, (mz, rel_int) in enumerate(zip(mzs, ints_norm)):
            cat = cats[i]
            ax.vlines(
                mz, 0, rel_int,
                colors=_COLOR[cat], lw=_LW[cat], alpha=_ALPHA[cat], zorder=_ZO[cat],
            )
        # M+0 precursor overlay
        if 0 in mk_peaks:
            m0_mz, _ = mk_peaks[0]
            m0_norm = ints_norm[int(np.argmin(np.abs(mzs - m0_mz)))]
            ax.vlines(m0_mz, 0, m0_norm, colors="darkgreen", lw=2.2, alpha=0.95, zorder=6)
        if prec_mz > 0:
            ax.axvline(prec_mz, color="black", lw=1.0, ls="--", alpha=0.4, zorder=0)
        ax.set_ylim(0, 130)
        ax.set_xlabel("m/z", fontsize=9)

    fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(18, 5), width_ratios=[2, 1])

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
    ax_full.axvspan(zoom_lo, zoom_hi, color="gold", alpha=0.10, zorder=0)
    if prec_mz > 0:
        ax_full.text(prec_mz, 108, f"prec\n{prec_mz:.3f}",
                     fontsize=6, color="black", ha="center", va="bottom", alpha=0.6)

    # Label disagree peaks on full panel (pom-only, ann-only) so they stand out
    for i, (mz, rel_int) in enumerate(zip(mzs, ints_norm)):
        cat = cats[i]
        if cat not in (POM_ONLY, ANN_ONLY, AGREE_ISO):
            continue
        lbl = label_arr[i]
        ann_k = int(iso_k_arr[i])
        pom_k = next((k for pidx, k, _ in pom_leakage if pidx == i), None)
        parts = []
        if ann_k > 0:
            parts.append(f"ann M+{ann_k}")
        if pom_k is not None:
            parts.append(f"pom M+{pom_k}")
        if lbl:
            parts.append(lbl)
        ax_full.annotate(
            "\n".join(parts) + f"\n{mz:.3f}",
            xy=(mz, rel_int), xytext=(0, 5), textcoords="offset points",
            fontsize=5.5, color=_COLOR[cat], ha="center", va="bottom", fontweight="bold",
        )

    # ── Right: zoom into prec region ─────────────────────────────────────────
    _draw_peaks(ax_zoom)
    ax_zoom.set_xlim(zoom_lo, zoom_hi)
    ax_zoom.set_title(f"Zoom: prec±5 Da → max mz  ({zoom_lo:.1f}–{zoom_hi:.1f})", fontsize=9)
    if prec_mz > 0:
        ax_zoom.text(prec_mz, 108, f"prec\n{prec_mz:.4f}",
                     fontsize=7, color="black", ha="center", va="bottom", alpha=0.7)

    # Label M+0 on zoom
    if 0 in mk_peaks:
        m0_mz, m0_raw = mk_peaks[0]
        m0_norm = ints_norm[int(np.argmin(np.abs(mzs - m0_mz)))]
        ax_zoom.annotate(
            f"M\n{m0_mz:.4f}\n{m0_raw:.0f}",
            xy=(m0_mz, m0_norm), xytext=(0, 6), textcoords="offset points",
            fontsize=8, color="darkgreen", ha="center", va="bottom", fontweight="bold",
        )

    # Label all annotated + leakage peaks on zoom panel
    pom_k_by_idx = {pidx: k for pidx, k, _ in pom_leakage}
    pom_delta_by_idx = {pidx: delta for pidx, _, delta in pom_leakage}
    for i, (mz, rel_int) in enumerate(zip(mzs, ints_norm)):
        if not (zoom_lo <= mz <= zoom_hi):
            continue
        cat = cats[i]
        if cat == UNANNOTATED:
            continue

        ann_k = int(iso_k_arr[i])
        pom_k = pom_k_by_idx.get(i)
        lbl = label_arr[i]

        lines: list[str] = []
        # isotope order tags
        if ann_k > 0 and pom_k is not None:
            lines.append(f"ann M+{ann_k} / pom M+{pom_k}")
        elif ann_k > 0:
            lines.append(f"ann M+{ann_k}")
        elif pom_k is not None:
            lines.append(f"pom M+{pom_k}")
        # formula
        if lbl:
            lines.append(lbl)
        # m/z
        lines.append(f"{mz:.4f}")
        # ratio info for pyopenms leakage peaks
        if pom_k is not None:
            exp_r = exp_ratios.get(pom_k, float("nan"))
            delta = pom_delta_by_idx.get(i, float("nan"))
            obs_r = (ints[i] / m0_raw_int) if m0_raw_int else float("nan")
            lines.append(f"Δmz={delta * 1000:.1f}mDa")
            if not np.isnan(obs_r):
                lines.append(f"exp:{exp_r:.3f}  obs:{obs_r:.3f}")

        ax_zoom.annotate(
            "\n".join(lines),
            xy=(mz, rel_int), xytext=(0, 6), textcoords="offset points",
            fontsize=7, color=_COLOR[cat], ha="center", va="bottom", fontweight="bold",
        )

    # ── Counts for title ─────────────────────────────────────────────────────
    n_agree = sum(1 for c in cats if c == AGREE_ISO)
    n_pom_only = sum(1 for c in cats if c == POM_ONLY)
    n_ann_only = sum(1 for c in cats if c == ANN_ONLY)

    fig.suptitle(
        f"[{idx + 1}/{total}]  spec_id={h['spec_id']}  |  {h['prec_type']}  |  "
        f"inst={h['inst_type']}  mz_tol={h['mz_tol'] * 1000:.0f} mDa  |  "
        f"prec_mz={prec_mz:.4f}  |  "
        f"agree={n_agree}  pom-only={n_pom_only}  ann-only={n_ann_only}",
        fontsize=9, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    log.info(f"Output directory: {FIG_DIR}")

    log.info("Loading ann_df…")
    ann_df = pd.read_pickle(PROC_DP / "nist20/ann_df.pkl")[
        ["spec_id", "ann_peak_mzs", "ann_isotopes", "ann_products"]
    ].copy()
    ann_spec_ids = set(ann_df["spec_id"].unique())
    log.info(f"  ann_df: {len(ann_df):,} annotated spectra")

    log.info("Loading spec_df + mol_df…")
    spec_df = pd.read_pickle(PROC_DP / "nist20/spec_df.pkl")[
        ["spec_id", "peaks", "prec_mz", "inst_type", "mol_id", "prec_type"]
    ]
    mol_df = pd.read_pickle(PROC_DP / "nist20/mol_df.pkl")[["mol_id", "formula"]]
    df = (
        spec_df[spec_df["spec_id"].isin(ann_spec_ids)]
        .merge(mol_df, on="mol_id", how="left")
        .merge(ann_df, on="spec_id", how="inner")
        .reset_index(drop=True)
    )
    del spec_df, mol_df, ann_df
    log.info(f"  Merged: {len(df):,} spectra")

    # Keep only spectra with valid formula + prec_mz
    df = df[df["formula"].notna() & df["prec_mz"].notna() & (df["prec_mz"] > 0)].reset_index(
        drop=True
    )
    log.info(f"  With formula + prec_mz: {len(df):,}")
    log.info(
        f"Scanning for cross-reference candidates "
        f"(above-prec peaks OR ann isotope, first {N_PLOTS})…"
    )

    hits: list[dict] = []
    for _, row in df.iterrows():
        if len(hits) >= N_PLOTS:
            break

        peaks = row["peaks"]
        if not peaks:
            continue
        prec_mz = float(row["prec_mz"])
        formula = str(row["formula"])
        inst_type = str(row.get("inst_type", ""))
        mz_tol = MZ_TOL_BY_INST.get(inst_type, MZ_TOL_DEFAULT)

        mzs = np.array([p[0] for p in peaks], dtype=np.float64)
        ints = np.array([p[1] for p in peaks], dtype=np.float64)

        # Parse ann_df annotations
        try:
            ann_mzs_f = [float(v) for v in row["ann_peak_mzs"]]
            ann_iso_ks_i = [int(float(v)) for v in row["ann_isotopes"]]
            ann_products_s = [str(v) for v in row["ann_products"]]
        except (TypeError, ValueError):
            continue

        # Parse formula for pyopenms
        try:
            ec = parse_formula(formula)
        except Exception:
            continue

        # pyopenms leakage detection
        pom_leakage = find_prec_leakage(mzs, ints, prec_mz, ec, mz_tol=mz_tol)
        pom_leakage_idxs = {pidx for pidx, _, _ in pom_leakage}
        pom_leakage_k = {pidx: k for pidx, k, _ in pom_leakage}

        # ann_df annotation matching
        iso_k_arr, is_ann, label_arr = match_ann_to_obs(
            mzs, ann_mzs_f, ann_iso_ks_i, ann_products_s
        )

        # Require at least one interesting above-prec event from either source
        has_pom = len(pom_leakage) > 0
        has_ann_above_prec = any(
            mzs[i] > prec_mz + mz_tol and iso_k_arr[i] > 0
            for i in range(len(mzs))
        )
        if not has_pom and not has_ann_above_prec:
            continue

        cats = assign_categories(mzs, iso_k_arr, is_ann, pom_leakage_idxs, pom_leakage_k)
        mk_peaks, exp_ratios = find_prec_mk_peaks(mzs, ints, prec_mz, ec, mz_tol=mz_tol)

        hits.append(
            {
                "spec_id": row["spec_id"],
                "formula": formula,
                "prec_type": str(row.get("prec_type", "")),
                "inst_type": inst_type,
                "prec_mz": prec_mz,
                "mz_tol": mz_tol,
                "mzs": mzs,
                "ints": ints,
                "cats": cats,
                "label_arr": label_arr,
                "iso_k_arr": iso_k_arr,
                "pom_leakage": pom_leakage,
                "mk_peaks": mk_peaks,
                "exp_ratios": exp_ratios,
            }
        )

    log.info(f"Found {len(hits)} spectra — saving plots…")
    agree_count = sum(
        1 for h in hits if any(c == AGREE_ISO for c in h["cats"])
    )
    pom_only_count = sum(
        1 for h in hits if any(c == POM_ONLY for c in h["cats"])
    )
    ann_only_count = sum(
        1 for h in hits if any(c == ANN_ONLY for c in h["cats"])
    )
    log.info(
        f"  Spectra with agree-iso peaks: {agree_count}  "
        f"pom-only: {pom_only_count}  ann-only: {ann_only_count}"
    )

    for i, h in enumerate(hits):
        out_path = FIG_DIR / f"{i + 1:03d}_spec{h['spec_id']}.png"
        plot_one(h, out_path, i, len(hits))
        if (i + 1) % 10 == 0:
            log.info(f"  {i + 1}/{len(hits)} saved")

    log.info(f"Done. All plots in {FIG_DIR}")


if __name__ == "__main__":
    main()
