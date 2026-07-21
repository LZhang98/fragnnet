"""Split the zero-cosine oracle tail into recoverable vs. broken at a valid loose tolerance.

The zero-cosine spectra (oracle ceiling == 0 at the per-instrument tolerance) match no DAG
fragment at all. To decide whether that is a borderline mass miscalibration (peaks just outside
instrument tolerance) or a gross failure (wrong annotation / >0.05 Da error / noise), we re-score
those spectra at a fixed 0.05 Da tolerance -- chosen because it is looser than BOTH instrument
tolerances (FT ~0.01 Da at m/z 1000, QTOF ~0.015 Da at low m/z). A fixed 0.01 Da probe is invalid
for QTOF because it is *tighter* than QTOF's own 20 ppm window and can never recover a QTOF miss.

This re-scores only the affected molecules through ``dag_oracle_scores.score_dag`` (reusing the
exact matching / ion-mass / filter-tier logic), reproducing the ``1_above_prec`` no-UWPR tier that
produced the source CSV, then classifies each zero-cos spectrum and plots the corrected split.

Usage:
    python scripts/eval/zero_cos_recoverability.py \\
        --oracle_csv data/eval/oracle_cos_nist20_default/oracle_cos_default.csv \\
        --frag_dp data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags \\
        --out_dp data/eval/oracle_cos_nist20_default \\
        --inst_types FT QTOF \\
        --prec_types "[M+H]+" "[M-H]-" "[M+Na]+" "[M+Cl]-"
"""

from __future__ import annotations

import argparse
import logging
import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dag_oracle_scores import score_dag  # noqa: E402  (needs sys.path tweak first)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]
TIER = "1_above_prec"  # source CSV tier: above-precursor peaks dropped, no UWPR denoise
COS_COL = "cos_hun_inst_tol_opt"


def tier_col(df: pd.DataFrame, base: str, tier: str = TIER) -> str:
    """Return the tiered metric column, with fallback for legacy unsuffixed CSVs."""
    suffixed = f"{base}__{tier}"
    if suffixed in df.columns:
        return suffixed
    if base in df.columns:
        return base
    raise KeyError(f"Missing column {base!r} or {suffixed!r}")


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--oracle_csv",
        default="data/eval/oracle_cos_nist20_default/oracle_cos_default.csv",
    )
    parser.add_argument("--frag_dp", default="data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags")
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument("--classyfire_fp", default="data/classyfire/nist20_mol_classyfire.pkl")
    parser.add_argument("--out_dp", required=True)
    parser.add_argument("--inst_types", nargs="+", default=["FT", "QTOF"])
    parser.add_argument(
        "--prec_types", nargs="+", default=["[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-"]
    )
    parser.add_argument("--top_classes", type=int, default=15)
    return parser.parse_args()


def main() -> None:
    """Re-score the zero-cos tail at 0.05 Da, classify recoverable/broken, and plot."""
    args = get_args()
    out_dp = args.out_dp
    os.makedirs(out_dp, exist_ok=True)

    orc = pd.read_csv(args.oracle_csv)
    orc = orc[orc["prec_type"].isin(args.prec_types) & orc["inst_type"].isin(args.inst_types)]
    cos_col = tier_col(orc, COS_COL)
    ach_col = tier_col(orc, "n_achievable_peaks_inst_tol")
    zero_mask = orc[cos_col] == 0.0
    zero_ids = set(orc.loc[zero_mask, "spec_id"])
    zero_mol_ids = set(orc.loc[zero_mask, "mol_id"])
    log.info("Zero-cos: %d spectra across %d molecules", len(zero_ids), len(zero_mol_ids))
    if not zero_ids:
        log.info("No zero-cos spectra in scope; nothing to re-score.")
        return

    # Re-score every spectrum of the affected molecules (spectra of the same mols come for free and
    # are used to validate that we reproduced the source tier).
    spec_df = pd.read_pickle(args.spec_fp)
    spec_df = spec_df[
        spec_df["mol_id"].isin(zero_mol_ids)
        & spec_df["prec_type"].isin(args.prec_types)
        & spec_df["inst_type"].isin(args.inst_types)
    ][["spec_id", "mol_id", "peaks", "prec_mz", "prec_type", "inst_type", "frag_mode"]].copy()
    spec_df = spec_df[spec_df["peaks"].apply(len) > 0]
    spec_df["contam_mzs"] = [[] for _ in range(len(spec_df))]  # no UWPR: tiers 2/3 == 1_above_prec
    log.info("Re-scoring %d spectra across %d molecules", len(spec_df), spec_df["mol_id"].nunique())

    res = score_dag(spec_df, args.frag_dp)

    # --- validate we reproduced the source CSV's tier (cos_hun_inst_tol_opt on 1_above_prec) ---
    chk = res.merge(orc[["spec_id", cos_col, ach_col]], on="spec_id", how="inner")
    cos_diff = (chk[f"{COS_COL}__{TIER}"] - chk[cos_col]).abs().max()
    ach_diff = (
        chk[f"n_achievable_peaks_inst_tol__{TIER}"] - chk[ach_col]
    ).abs().max()
    log.info("Tier reproduction check: max |cos diff|=%.2e, max |achievable diff|=%d", cos_diff, ach_diff)
    if cos_diff > 1e-4 or ach_diff > 0:
        raise ValueError(
            f"Re-score does not reproduce source tier '{TIER}' "
            f"(cos diff {cos_diff:.2e}, achievable diff {ach_diff}); tier/DAG mismatch."
        )

    # --- classify the zero-cos spectra ---
    z = res[res["spec_id"].isin(zero_ids)].copy()
    z["n_ach_inst_tol"] = z[f"n_achievable_peaks_inst_tol__{TIER}"]
    z["n_ach_0p01da"] = z[f"n_achievable_peaks_0p01da__{TIER}"]
    z["n_ach_0p05da"] = z[f"n_achievable_peaks_0p05da__{TIER}"]
    z["n_true_peaks"] = z[f"n_true_peaks__{TIER}"]
    if (z["n_ach_inst_tol"] != 0).any():
        raise ValueError("Some zero-cos spectra have >0 achievable at inst tol after re-score.")
    z["recoverable_0p05da"] = z["n_ach_0p05da"] > 0

    mol = pd.read_pickle(args.mol_fp)[["mol_id", "smiles", "formula", "num_atoms"]]
    cf = pd.read_pickle(args.classyfire_fp)[["mol_id", "klass"]]
    z = z.merge(mol, on="mol_id", how="left").merge(cf, on="mol_id", how="left")
    z["klass"] = z["klass"].fillna("(unclassified)")

    out_cols = [
        "spec_id", "mol_id", "prec_type", "inst_type", "frag_mode", "n_true_peaks",
        "n_ach_inst_tol", "n_ach_0p01da", "n_ach_0p05da", "recoverable_0p05da",
        "num_atoms", "klass", "formula", "smiles",
    ]
    z[out_cols].sort_values(["recoverable_0p05da", "mol_id"]).to_csv(
        os.path.join(out_dp, "zero_cos_recoverability.csv"), index=False
    )

    # --- report ---
    log.info("=== Zero-cos recoverability at 0.05 Da (valid for FT and QTOF) ===")
    for inst in args.inst_types + ["ALL"]:
        sub = z if inst == "ALL" else z[z["inst_type"] == inst]
        if len(sub) == 0:
            continue
        rec = int(sub["recoverable_0p05da"].sum())
        log.info(
            "  %-5s: %4d zero-cos  ->  %4d recoverable (%.0f%%) / %4d broken (%.0f%%)",
            inst, len(sub), rec, 100 * rec / len(sub),
            len(sub) - rec, 100 * (len(sub) - rec) / len(sub),
        )
    # For comparison, the old (invalid-for-QTOF) 0.01 Da probe:
    old_rec = int((z["n_ach_0p01da"] > 0).sum())
    log.info("  (old 0.01 Da probe would call %d recoverable -- undercounts QTOF)", old_rec)

    plot_recoverability(z, args.top_classes, os.path.join(out_dp, "zero_cos_recoverability.png"))


def plot_recoverability(z: pd.DataFrame, top_classes: int, out_fp: str) -> None:
    """Plot the corrected recoverable/broken split (0.05 Da) by instrument and class.

    Args:
        z: Classified zero-cos DataFrame with ``recoverable_0p05da`` and ``inst_type``.
        top_classes: Number of ClassyFire classes to show.
        out_fp: Path to write the PNG.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (A) recoverable vs broken, stacked by instrument
    ax = axes[0]
    ct = (
        z.assign(status=z["recoverable_0p05da"].map({True: "recoverable", False: "broken"}))
        .pivot_table(index="status", columns="inst_type", values="spec_id", aggfunc="count", fill_value=0)
        .reindex(["recoverable", "broken"])
    )
    ct.plot(kind="bar", stacked=True, ax=ax, color=COLORS[: ct.shape[1]], edgecolor="black", linewidth=0.4)
    for i, status in enumerate(ct.index):
        tot = int(ct.loc[status].sum())
        ax.text(i, tot, f"{tot}\n({100 * tot / len(z):.0f}%)", ha="center", va="bottom")
    ax.set_ylabel("# zero-cos spectra")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.set_title(f"A. Zero-cos split @0.05 Da (n={len(z)})")
    ax.legend(title="instrument")

    # (B) sparsity
    ax = axes[1]
    ax.hist(z["n_true_peaks"].clip(upper=30), bins=range(0, 32), color=COLORS[0],
            edgecolor="white", linewidth=0.3)
    ax.set_xlabel("# true peaks (clipped at 30)")
    ax.set_ylabel("# zero-cos spectra")
    ax.set_title(f"B. Zero-cos sparsity (median {int(z['n_true_peaks'].median())} peaks)")

    # (C) broken-only by class (the genuine data-quality residue)
    ax = axes[2]
    broken = z[~z["recoverable_0p05da"]]
    kc = broken["klass"].value_counts().head(top_classes)[::-1]
    ax.barh(kc.index, kc.values, color=COLORS[3], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("# broken zero-cos spectra")
    ax.set_title("C. Broken (@0.05 Da) by ClassyFire class")

    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_fp)


if __name__ == "__main__":
    main()
