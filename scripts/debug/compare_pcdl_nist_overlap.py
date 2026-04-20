"""Compare overlapping PCDL and NIST spectra and save mirror plots.

The default comparison is intentionally strict:

  - same molecule by ``inchikey_s``
  - same precursor/adduct
  - same instrument type
  - same fragmentation mode
  - same absolute collision energy (ACE)

For each matchable PCDL spectrum, the script finds the best NIST spectrum in the
same metadata group by m/z-tolerant cosine similarity.  It writes a CSV of best
matches and saves mirror plots for selected pairs.

Run from repo root:

    conda run --no-capture-output -n FRAGNNET-GPU \
      python scripts/debug/compare_pcdl_nist_overlap.py \
      --nist-dset nist20 \
      --n-plots 50
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fragnnet.utils.plot_utils import get_mol_im

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


DEFAULT_PROC_DP = Path("data/proc")
DEFAULT_OUT_DIR = Path("figs/pcdl_nist_overlap")
DEFAULT_ADDUCTS = ("[M+H]+", "[M-H]-")
C13_SPACING = 1.003355


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find exact PCDL/NIST metadata overlaps and save mirror plots.",
    )
    parser.add_argument("--proc-dp", type=Path, default=DEFAULT_PROC_DP)
    parser.add_argument("--pcdl-dset", default="pcdl")
    parser.add_argument("--nist-dset", default="nist20", choices=["nist20", "nist23"])
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--mz-tol", type=float, default=0.05)
    parser.add_argument("--adducts", nargs="+", default=list(DEFAULT_ADDUCTS))
    parser.add_argument("--inst-type", default="QTOF")
    parser.add_argument("--frag-mode", default="CID")
    parser.add_argument(
        "--ce-match",
        choices=["ace", "rounded_ace"],
        default="ace",
        help="Collision-energy key used in matching. rounded_ace uses --ace-round decimals.",
    )
    parser.add_argument("--ace-round", type=int, default=0)
    parser.add_argument(
        "--n-plots",
        type=int,
        default=50,
        help="Number of mirror plots to save. Use 0 to skip plots.",
    )
    parser.add_argument(
        "--plot-selection",
        choices=["top", "bottom", "mixed"],
        default="mixed",
        help="Which matched pairs to plot by cosine rank.",
    )
    parser.add_argument(
        "--bad-thresholds",
        nargs="+",
        type=float,
        default=[0.5, 0.7, 0.85],
        help="Cosine thresholds for bad-match CSV exports.",
    )
    parser.add_argument(
        "--max-pcdl",
        type=int,
        default=0,
        help="Optional limit on matchable PCDL spectra after filtering; 0 means all.",
    )
    return parser.parse_args()


def _load_joined(proc_dp: Path, dset: str) -> pd.DataFrame:
    spec = pd.read_pickle(proc_dp / dset / "spec_df.pkl")
    mol = pd.read_pickle(proc_dp / dset / "mol_df.pkl")[["mol_id", "inchikey_s", "formula", "smiles"]]
    return spec.merge(mol, on="mol_id", how="left")


def _filter_metadata(
    df: pd.DataFrame,
    adducts: set[str],
    inst_type: str,
    frag_mode: str,
) -> pd.DataFrame:
    out = df[
        df["prec_type"].astype(str).isin(adducts)
        & df["inst_type"].astype(str).eq(inst_type)
        & df["frag_mode"].astype(str).eq(frag_mode)
        & df["ace"].notna()
        & df["inchikey_s"].notna()
    ].copy()
    return out


def _add_match_key(df: pd.DataFrame, ce_match: str, ace_round: int) -> pd.DataFrame:
    out = df.copy()
    if ce_match == "rounded_ace":
        out["match_ce"] = out["ace"].astype(float).round(ace_round)
    else:
        out["match_ce"] = out["ace"].astype(float)
    out["match_key"] = list(
        zip(
            out["inchikey_s"].astype(str),
            out["prec_type"].astype(str),
            out["inst_type"].astype(str),
            out["frag_mode"].astype(str),
            out["match_ce"].astype(float),
        )
    )
    return out


def _peak_arrays(peaks: list | tuple | np.ndarray, normalize: bool = True) -> tuple[np.ndarray, np.ndarray]:
    if peaks is None or len(peaks) == 0:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    arr = np.asarray(peaks, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return np.empty(0, dtype=np.float64), np.empty(0, dtype=np.float64)
    mzs = arr[:, 0]
    ints = arr[:, 1]
    ok = np.isfinite(mzs) & np.isfinite(ints) & (ints > 0)
    mzs = mzs[ok]
    ints = ints[ok]
    if len(mzs) == 0:
        return mzs, ints
    order = np.argsort(mzs)
    mzs = mzs[order]
    ints = ints[order]
    if normalize:
        norm = float(np.linalg.norm(ints))
        if norm > 0:
            ints = ints / norm
    return mzs, ints


def _match_pairs(
    pcdl_mzs: np.ndarray,
    nist_mzs: np.ndarray,
    mz_tol: float,
) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    i = 0
    j = 0
    while i < len(pcdl_mzs) and j < len(nist_mzs):
        delta = pcdl_mzs[i] - nist_mzs[j]
        if abs(delta) <= mz_tol:
            best_j = j
            best_abs = abs(delta)
            jj = j + 1
            while jj < len(nist_mzs) and nist_mzs[jj] <= pcdl_mzs[i] + mz_tol:
                abs_delta = abs(pcdl_mzs[i] - nist_mzs[jj])
                if abs_delta < best_abs:
                    best_abs = abs_delta
                    best_j = jj
                jj += 1
            pairs.append((i, best_j))
            i += 1
            j = best_j + 1
        elif delta < 0:
            i += 1
        else:
            j += 1
    return pairs


def _cosine_tol(
    pcdl_arr: tuple[np.ndarray, np.ndarray],
    nist_arr: tuple[np.ndarray, np.ndarray],
    mz_tol: float,
) -> float:
    p_mzs, p_ints = pcdl_arr
    n_mzs, n_ints = nist_arr
    return float(sum(p_ints[i] * n_ints[j] for i, j in _match_pairs(p_mzs, n_mzs, mz_tol)))


def _isotope_like_indices(mzs: np.ndarray, ints: np.ndarray, mz_tol: float) -> set[int]:
    """Return indices with a stronger lower-m/z M+1 or M+2 partner."""
    out: set[int] = set()
    for i, mz in enumerate(mzs):
        for k in (1, 2):
            target = mz - k * C13_SPACING
            lo = np.searchsorted(mzs, target - mz_tol, side="left")
            hi = np.searchsorted(mzs, target + mz_tol, side="right")
            if lo < hi and float(np.max(ints[lo:hi])) > float(ints[i]):
                out.add(i)
                break
    return out


def _residual_features(
    pcdl_arr: tuple[np.ndarray, np.ndarray],
    nist_arr: tuple[np.ndarray, np.ndarray],
    mz_tol: float,
) -> dict[str, float | int | str]:
    """Compute diagnostics that separate unmatched peaks from intensity mismatch."""
    p_mzs, p_ints = pcdl_arr
    n_mzs, n_ints = nist_arr
    pairs = _match_pairs(p_mzs, n_mzs, mz_tol)
    p_used = {i for i, _ in pairs}
    n_used = {j for _, j in pairs}
    p_un = set(range(len(p_mzs))) - p_used
    n_un = set(range(len(n_mzs))) - n_used
    p_iso = _isotope_like_indices(p_mzs, p_ints, mz_tol)
    n_iso = _isotope_like_indices(n_mzs, n_ints, mz_tol)

    matched_sqdiff = float(sum((p_ints[i] - n_ints[j]) ** 2 for i, j in pairs))
    components = {
        "pcdl_unmatched_l2sq": float(sum(p_ints[i] ** 2 for i in p_un)),
        "nist_unmatched_noniso_l2sq": float(sum(n_ints[i] ** 2 for i in n_un - n_iso)),
        "nist_unmatched_isotope_like_l2sq": float(sum(n_ints[i] ** 2 for i in n_un & n_iso)),
        "matched_intensity_sqdiff": matched_sqdiff,
    }
    dominant = max(components, key=components.get)
    return {
        "n_pairs": len(pairs),
        "pcdl_n_peaks": len(p_mzs),
        "nist_n_peaks": len(n_mzs),
        "pcdl_unmatched_l2sq": components["pcdl_unmatched_l2sq"],
        "pcdl_unmatched_isotope_like_l2sq": float(sum(p_ints[i] ** 2 for i in p_un & p_iso)),
        "nist_unmatched_l2sq": float(sum(n_ints[i] ** 2 for i in n_un)),
        "nist_unmatched_isotope_like_l2sq": components["nist_unmatched_isotope_like_l2sq"],
        "nist_unmatched_noniso_l2sq": components["nist_unmatched_noniso_l2sq"],
        "matched_intensity_sqdiff": matched_sqdiff,
        "dominant_residual": dominant,
    }


def _rel_intensity_arrays(peaks: list | tuple | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mzs, ints = _peak_arrays(peaks, normalize=False)
    if len(ints) == 0:
        return mzs, ints
    max_int = float(ints.max())
    if max_int > 0:
        ints = ints / max_int * 100.0
    return mzs, ints


def _smiles_value(row: pd.Series) -> str | None:
    smiles = row.get("smiles")
    if smiles is None or pd.isna(smiles):
        return None
    smiles = str(smiles).strip()
    return smiles or None


def _add_mol_axis(fig: plt.Figure, gs_cell, smiles: str | None) -> None:
    mol_ax = fig.add_subplot(gs_cell)
    mol_ax.axis("off")
    if smiles is None:
        return
    try:
        mol_im = get_mol_im(smiles, width=500, height=500)
    except Exception as exc:
        log.warning("Could not render molecule for SMILES %r: %s", smiles, exc)
        return
    mol_ax.imshow(np.asarray(mol_im))


def _plot_mirror(
    pcdl_row: pd.Series,
    nist_row: pd.Series,
    cosine: float,
    mz_tol: float,
    out_path: Path,
) -> None:
    p_mzs, p_ints = _rel_intensity_arrays(pcdl_row["peaks"])
    n_mzs, n_ints = _rel_intensity_arrays(nist_row["peaks"])

    smiles = _smiles_value(pcdl_row) or _smiles_value(nist_row)
    if smiles is None:
        fig, ax = plt.subplots(figsize=(12, 6))
    else:
        fig = plt.figure(figsize=(15, 6), constrained_layout=True)
        gs = fig.add_gridspec(1, 2, width_ratios=[4.5, 1.2], wspace=0.12)
        ax = fig.add_subplot(gs[0])
        _add_mol_axis(fig, gs[1], smiles)

    ax.vlines(p_mzs, 0, p_ints, color="#1f77b4", lw=0.9, alpha=0.85, label="PCDL")
    ax.vlines(n_mzs, 0, -n_ints, color="#d62728", lw=0.9, alpha=0.85, label="NIST")
    ax.axhline(0, color="black", lw=0.8)

    if pcdl_row.get("prec_mz", 0) and np.isfinite(float(pcdl_row["prec_mz"])):
        ax.axvline(float(pcdl_row["prec_mz"]), color="#1f77b4", lw=0.8, ls="--", alpha=0.35)
    if nist_row.get("prec_mz", 0) and np.isfinite(float(nist_row["prec_mz"])):
        ax.axvline(float(nist_row["prec_mz"]), color="#d62728", lw=0.8, ls="--", alpha=0.35)

    lo_candidates = []
    hi_candidates = []
    if len(p_mzs):
        lo_candidates.append(float(p_mzs.min()))
        hi_candidates.append(float(p_mzs.max()))
    if len(n_mzs):
        lo_candidates.append(float(n_mzs.min()))
        hi_candidates.append(float(n_mzs.max()))
    if lo_candidates:
        pad = max(10.0, (max(hi_candidates) - min(lo_candidates)) * 0.02)
        ax.set_xlim(min(lo_candidates) - pad, max(hi_candidates) + pad)

    ax.set_ylim(-110, 110)
    ax.set_xlabel("m/z")
    ax.set_ylabel("Relative intensity (%)")
    ax.legend(loc="upper right")

    title = (
        f"{pcdl_row['inchikey_s']} | {pcdl_row['prec_type']} | "
        f"{pcdl_row['inst_type']}/{pcdl_row['frag_mode']} | ACE={float(pcdl_row['ace']):g} | "
        f"cos={cosine:.3f} @ +/- {mz_tol:g} Da"
    )
    subtitle = (
        f"PCDL {pcdl_row['dset']}:{pcdl_row['dset_spec_id']} "
        f"(spec_id={int(pcdl_row['spec_id'])}, prec_mz={float(pcdl_row['prec_mz']):.4f}) | "
        f"NIST {nist_row['dset']}:{nist_row['dset_spec_id']} "
        f"(spec_id={int(nist_row['spec_id'])}, prec_mz={float(nist_row['prec_mz']):.4f})"
    )
    ax.set_title(f"{title}\n{subtitle}", fontsize=10)
    if smiles is None:
        fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _select_plot_rows(scores: pd.DataFrame, n_plots: int, mode: str) -> pd.DataFrame:
    if n_plots <= 0 or scores.empty:
        return scores.iloc[0:0]
    if mode == "top":
        return scores.sort_values("best_cos", ascending=False).head(n_plots)
    if mode == "bottom":
        return scores.sort_values("best_cos", ascending=True).head(n_plots)

    n_low = n_plots // 3
    n_high = n_plots // 3
    n_mid = n_plots - n_low - n_high
    low = scores.sort_values("best_cos", ascending=True).head(n_low)
    high = scores.sort_values("best_cos", ascending=False).head(n_high)
    mid_sorted = scores.assign(mid_dist=(scores["best_cos"] - scores["best_cos"].median()).abs())
    mid = mid_sorted.sort_values("mid_dist", ascending=True).head(n_mid)
    return pd.concat([high, mid, low], ignore_index=True).drop_duplicates("pcdl_spec_id")


def _print_summary(scores: pd.DataFrame, label: str, mz_tol: float) -> None:
    log.info("")
    log.info("=" * 80)
    log.info("%s", label)
    log.info("matched PCDL spectra: %s", f"{len(scores):,}")
    if scores.empty:
        return
    quantiles = scores["best_cos"].quantile([0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
    log.info("best cosine quantiles @ +/- %.4g Da:\n%s", mz_tol, quantiles.to_string())
    for thresh in (0.3, 0.5, 0.7, 0.85):
        n = int((scores["best_cos"] >= thresh).sum())
        log.info("cos >= %.2f: %s (%.1f%%)", thresh, f"{n:,}", n / len(scores) * 100)
    log.info("by adduct:\n%s", scores.groupby("prec_type")["best_cos"].agg(["count", "median", "mean"]))
    log.info("by ACE:\n%s", scores.groupby("ace")["best_cos"].agg(["count", "median", "mean"]))
    log.info("=" * 80)


def _write_bad_tables(scores: pd.DataFrame, out_dir: Path, thresholds: list[float]) -> None:
    bad_dir = out_dir / "bad_matches"
    bad_dir.mkdir(parents=True, exist_ok=True)
    for threshold in thresholds:
        bad = scores[scores["best_cos"] < threshold].sort_values("best_cos")
        safe_threshold = str(threshold).replace(".", "p")
        out_csv = bad_dir / f"bad_cos_lt_{safe_threshold}.csv"
        bad.to_csv(out_csv, index=False)
        log.info("Wrote %s bad matches with cos < %.3g to %s", len(bad), threshold, out_csv)

    worst = scores.sort_values("best_cos").head(100)
    worst.to_csv(bad_dir / "worst_100.csv", index=False)
    if not worst.empty and "dominant_residual" in worst:
        log.info(
            "Dominant residuals for worst 100:\n%s",
            worst["dominant_residual"].value_counts().to_string(),
        )


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.out_dir / "mirror_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading PCDL and %s processed data...", args.nist_dset)
    pcdl = _load_joined(args.proc_dp, args.pcdl_dset)
    nist = _load_joined(args.proc_dp, args.nist_dset)

    adducts = set(args.adducts)
    pcdl = _filter_metadata(pcdl, adducts, args.inst_type, args.frag_mode)
    nist = _filter_metadata(nist, adducts, args.inst_type, args.frag_mode)

    pcdl = _add_match_key(pcdl, args.ce_match, args.ace_round)
    nist = _add_match_key(nist, args.ce_match, args.ace_round)
    nist = nist[nist["match_ce"].isin(set(pcdl["match_ce"]))].copy()

    nist_keys = set(nist["match_key"])
    pcdl_keys = set(pcdl["match_key"])
    pcdl_matchable = pcdl[pcdl["match_key"].isin(nist_keys)].copy()
    nist_matchable = nist[nist["match_key"].isin(pcdl_keys)].copy()
    if args.max_pcdl > 0:
        pcdl_matchable = pcdl_matchable.head(args.max_pcdl).copy()

    log.info("PCDL filtered rows: %s", f"{len(pcdl):,}")
    log.info("%s filtered rows: %s", args.nist_dset, f"{len(nist):,}")
    log.info("overlap metadata keys: %s", f"{len(set(pcdl['match_key']) & set(nist['match_key'])):,}")
    log.info(
        "overlap molecules: %s",
        f"{len(set(pcdl_matchable['inchikey_s']) & set(nist_matchable['inchikey_s'])):,}",
    )
    log.info(
        "PCDL spectra with at least one exact metadata NIST candidate: %s (%.1f%%)",
        f"{len(pcdl_matchable):,}",
        len(pcdl_matchable) / len(pcdl) * 100 if len(pcdl) else 0.0,
    )

    nist_rows_by_spec_id = nist_matchable.set_index("spec_id", drop=False)
    nist_by_key: dict[tuple, list[tuple[int, tuple[np.ndarray, np.ndarray]]]] = {}
    log.info("Precomputing normalized NIST peak arrays...")
    for key, sub in nist_matchable.groupby("match_key", sort=False):
        entries = []
        for row in sub.itertuples(index=False):
            entries.append((int(row.spec_id), _peak_arrays(row.peaks)))
        nist_by_key[key] = entries

    log.info("Finding best NIST spectrum for each matchable PCDL spectrum...")
    rows = []
    for row in pcdl_matchable.itertuples(index=False):
        pcdl_arr = _peak_arrays(row.peaks)
        best_cos = -1.0
        best_spec_id = -1
        best_arr = (np.empty(0), np.empty(0))
        candidates = nist_by_key.get(row.match_key, [])
        for nist_spec_id, nist_arr in candidates:
            cos = _cosine_tol(pcdl_arr, nist_arr, args.mz_tol)
            if cos > best_cos:
                best_cos = cos
                best_spec_id = nist_spec_id
                best_arr = nist_arr
        out_row = {
            "pcdl_spec_id": int(row.spec_id),
            "nist_spec_id": int(best_spec_id),
            "pcdl_dset": row.dset,
            "pcdl_dset_spec_id": row.dset_spec_id,
            "nist_dset": args.nist_dset,
            "nist_dset_spec_id": (
                nist_rows_by_spec_id.loc[int(best_spec_id), "dset_spec_id"]
                if best_spec_id in nist_rows_by_spec_id.index
                else ""
            ),
            "inchikey_s": row.inchikey_s,
            "formula": row.formula,
            "smiles": row.smiles,
            "prec_type": row.prec_type,
            "inst_type": row.inst_type,
            "frag_mode": row.frag_mode,
            "ace": float(row.ace),
            "match_ce": float(row.match_ce),
            "pcdl_prec_mz": float(row.prec_mz),
            "best_cos": best_cos,
            "n_candidates": len(candidates),
        }
        out_row.update(_residual_features(pcdl_arr, best_arr, args.mz_tol))
        rows.append(out_row)

    scores = pd.DataFrame(rows)
    out_csv = args.out_dir / f"pcdl_vs_{args.nist_dset}_best_matches.csv"
    scores.to_csv(out_csv, index=False)
    log.info("Wrote %s", out_csv)
    _print_summary(scores, f"PCDL vs {args.nist_dset}", args.mz_tol)
    _write_bad_tables(scores, args.out_dir, args.bad_thresholds)

    plot_rows = _select_plot_rows(scores, args.n_plots, args.plot_selection)
    if len(plot_rows) > 0:
        log.info("Saving %s mirror plots to %s", len(plot_rows), plot_dir)
    pcdl_rows_by_spec_id = pcdl_matchable.set_index("spec_id", drop=False)
    for i, row in enumerate(plot_rows.itertuples(index=False), start=1):
        pcdl_row = pcdl_rows_by_spec_id.loc[int(row.pcdl_spec_id)]
        nist_row = nist_rows_by_spec_id.loc[int(row.nist_spec_id)]
        out_path = plot_dir / (
            f"{i:03d}_cos{float(row.best_cos):.3f}_"
            f"pcdl{int(row.pcdl_spec_id)}_{args.nist_dset}{int(row.nist_spec_id)}.png"
        )
        _plot_mirror(pcdl_row, nist_row, float(row.best_cos), args.mz_tol, out_path)
        if i % 10 == 0:
            log.info("  saved %s/%s", i, len(plot_rows))

    log.info("Done.")


if __name__ == "__main__":
    main()
