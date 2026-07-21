"""Characterize and plot the DAG-unexplainable spectra in NIST from oracle-ceiling scores.

Consumes the per-spectrum CSV from ``scripts/eval/dag_oracle_scores.py``. A spectrum is
"unexplainable" when its oracle cosine ceiling (the best cosine any model could reach given
only the DAG's fragment coverage) falls below a threshold. This script restricts to a clean
instrument/adduct scope, then:

  1. summarizes the oracle ceiling by adduct x instrument and by ClassyFire class,
  2. isolates the zero-cosine tail (oracle matches NO peak) and splits it into
     "recoverable" (>=1 peak matches at a loose 0.01 Da tolerance -> borderline miscalibration)
     vs. "broken" (no peak matches even at 0.01 Da -> wrong annotation / gross mass error / noise),
  3. writes per-spectrum, per-molecule, and per-class CSVs, and
  4. renders two figures: an unexplainability overview and a zero-cosine diagnostic.

Usage:
    python scripts/eval/unexplainable_analysis.py \\
        --oracle_csv data/eval/oracle_cos_nist20_default/oracle_cos_default.csv \\
        --out_dp data/eval/oracle_cos_nist20_default \\
        --inst_types FT QTOF \\
        --prec_types "[M+H]+" "[M-H]-" "[M+Na]+" "[M+Cl]-" \\
        --cos_col cos_hun_inst_tol_opt \\
        --thresh 0.5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COLORS = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3", "#937860"]


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--oracle_csv",
        default="data/eval/oracle_cos_nist20_default/oracle_cos_default.csv",
        help="Per-spectrum CSV from dag_oracle_scores.py.",
    )
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument("--classyfire_fp", default="data/classyfire/nist20_mol_classyfire.pkl")
    parser.add_argument("--out_dp", required=True, help="Output directory for CSVs + figures.")
    parser.add_argument("--inst_types", nargs="+", default=["FT", "QTOF"])
    parser.add_argument(
        "--prec_types", nargs="+", default=["[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-"]
    )
    parser.add_argument(
        "--cos_col",
        default="cos_hun_inst_tol_opt",
        help="Oracle-ceiling column to threshold on.",
    )
    parser.add_argument("--thresh", type=float, default=0.5, help="Unexplainable if cos < thresh.")
    parser.add_argument("--min_class_count", type=int, default=500)
    parser.add_argument("--top_classes", type=int, default=15)
    return parser.parse_args()


def load_scoped(args: argparse.Namespace) -> pd.DataFrame:
    """Load the oracle CSV, filter to the instrument/adduct scope, and attach structure + class.

    Args:
        args: Parsed CLI arguments.

    Returns:
        Scoped per-spectrum DataFrame with ``smiles``, ``formula``, ``num_atoms``,
        ``superklass``, ``klass``, and a boolean ``unexpl`` column added.

    Raises:
        ValueError: If the scope filter removes every row.
    """
    df = pd.read_csv(args.oracle_csv)
    df = df[df["prec_type"].isin(args.prec_types) & df["inst_type"].isin(args.inst_types)].copy()
    if df.empty:
        raise ValueError("Scope filter (inst_types / prec_types) matched no rows.")

    mol = pd.read_pickle(args.mol_fp)[["mol_id", "smiles", "formula", "num_atoms"]]
    cf = pd.read_pickle(args.classyfire_fp)[["mol_id", "superklass", "klass"]]
    df = df.merge(mol, on="mol_id", how="left").merge(cf, on="mol_id", how="left")
    df["klass"] = df["klass"].fillna("(unclassified)")
    df["superklass"] = df["superklass"].fillna("(unclassified)")
    df["unexpl"] = df[args.cos_col] < args.thresh

    log.info(
        "Scope: %d spectra, %d molecules; overall unexplainable %.2f%% (%s < %.2f)",
        len(df),
        df["mol_id"].nunique(),
        100 * df["unexpl"].mean(),
        args.cos_col,
        args.thresh,
    )
    return df


def summarize_by_class(df: pd.DataFrame, cos_col: str, level: str = "klass") -> pd.DataFrame:
    """Aggregate unexplainability by a ClassyFire level.

    Args:
        df: Scoped per-spectrum DataFrame from :func:`load_scoped`.
        cos_col: Oracle-ceiling column name.
        level: ClassyFire column to group by (``klass`` or ``superklass``).

    Returns:
        Per-class DataFrame with spectra/molecule counts, unexplainable count,
        percent unexplainable, and mean ceiling, indexed by class.
    """
    g = df.groupby(level).agg(
        n_spectra=(cos_col, "size"),
        n_mols=("mol_id", "nunique"),
        n_unexpl=("unexpl", "sum"),
        mean_ceiling=(cos_col, "mean"),
    )
    g["pct_unexpl"] = 100 * g["n_unexpl"] / g["n_spectra"]
    g["n_unexpl"] = g["n_unexpl"].astype(int)
    return g


def dump_tables(df: pd.DataFrame, args: argparse.Namespace, out_dp: Path) -> pd.DataFrame:
    """Write per-spectrum, per-molecule, and per-class unexplainable CSVs.

    Args:
        df: Scoped per-spectrum DataFrame.
        args: Parsed CLI arguments.
        out_dp: Output directory.

    Returns:
        The zero-cosine per-spectrum DataFrame (for the diagnostic figure).
    """
    cos_col = args.cos_col
    unexpl = df[df["unexpl"]].copy()

    spec_cols = [
        "spec_id", "mol_id", "prec_type", "inst_type", "frag_mode", cos_col,
        "n_true_peaks", "n_achievable_peaks_inst_tol", "n_achievable_peaks_0p01da",
        "dag_num_nodes", "num_atoms", "klass", "formula", "smiles",
    ]
    spec_cols = [c for c in spec_cols if c in unexpl.columns]
    unexpl[spec_cols].sort_values(cos_col).to_csv(out_dp / "unexpl_spectra.csv", index=False)

    mols = (
        unexpl.groupby("mol_id")
        .agg(
            n_unexpl_spectra=("spec_id", "size"),
            mean_ceiling=(cos_col, "mean"),
            adducts=("prec_type", lambda s: ",".join(sorted(s.unique()))),
            inst=("inst_type", lambda s: ",".join(sorted(s.unique()))),
            num_atoms=("num_atoms", "first"),
            klass=("klass", "first"),
            formula=("formula", "first"),
            smiles=("smiles", "first"),
        )
        .sort_values(["mean_ceiling", "n_unexpl_spectra"], ascending=[True, False])
    )
    mols.to_csv(out_dp / "unexpl_mols.csv")

    summarize_by_class(df, cos_col, "klass").sort_values(
        "n_unexpl", ascending=False
    ).to_csv(out_dp / "unexpl_by_classyfire_klass.csv")

    # Zero-cosine tail (oracle matches nothing). The recoverable-vs-broken split needs a tolerance
    # looser than every per-instrument tol and is done by scripts/eval/zero_cos_recoverability.py
    # (a 0.05 Da re-score); a fixed 0.01 Da probe is invalid for QTOF (tighter than its 20 ppm).
    zero = df[df[cos_col] == 0.0].copy()
    zero.sort_values("mol_id").to_csv(out_dp / "zero_cos_spectra.csv", index=False)

    log.info(
        "Unexplainable: %d spectra / %d molecules. Zero-cos: %d spectra / %d molecules "
        "(run zero_cos_recoverability.py for the recoverable/broken split).",
        len(unexpl),
        unexpl["mol_id"].nunique(),
        len(zero),
        zero["mol_id"].nunique(),
    )
    return zero


def plot_overview(df: pd.DataFrame, args: argparse.Namespace, out_fp: Path) -> None:
    """Render the 2x2 unexplainability overview figure.

    Panels: (A) oracle-ceiling histogram, (B) percent unexplainable by adduct x instrument,
    (C) top classes by percent unexplainable, (D) top classes by raw unexplainable count.

    Args:
        df: Scoped per-spectrum DataFrame.
        args: Parsed CLI arguments.
        out_fp: Path to write the PNG.
    """
    cos_col = args.cos_col
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))

    # (A) ceiling distribution
    ax = axes[0, 0]
    ax.hist(df[cos_col], bins=60, color=COLORS[0], edgecolor="white", linewidth=0.3)
    ax.axvline(args.thresh, color=COLORS[3], ls="--", lw=1.5, label=f"thresh={args.thresh}")
    ax.set_yscale("log")
    ax.set_xlabel(f"oracle ceiling ({cos_col})")
    ax.set_ylabel("spectra (log)")
    ax.set_title("A. Oracle cosine ceiling distribution")
    ax.legend()

    # (B) pct unexplainable by adduct x inst
    ax = axes[0, 1]
    pt = (
        df.groupby(["prec_type", "inst_type"])["unexpl"].mean().mul(100).unstack("inst_type")
        .reindex(args.prec_types)
    )
    pt.plot(kind="bar", ax=ax, color=COLORS[: pt.shape[1]], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("adduct")
    ax.set_ylabel("% unexplainable")
    ax.set_title("B. % unexplainable by adduct x instrument")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(title="instrument")

    kl = summarize_by_class(df, cos_col, "klass")
    kl_big = kl[kl["n_spectra"] >= args.min_class_count]

    # (C) top classes by pct unexplainable
    ax = axes[1, 0]
    top_pct = kl_big.sort_values("pct_unexpl", ascending=False).head(args.top_classes)[::-1]
    ax.barh(top_pct.index, top_pct["pct_unexpl"], color=COLORS[1], edgecolor="black", linewidth=0.4)
    ax.axvline(100 * df["unexpl"].mean(), color="gray", ls=":", label="scope overall")
    ax.set_xlabel("% unexplainable")
    ax.set_title(f"C. Top classes by % unexplainable (>={args.min_class_count} spectra)")
    ax.legend()

    # (D) top classes by raw unexplainable count
    ax = axes[1, 1]
    top_cnt = kl.sort_values("n_unexpl", ascending=False).head(args.top_classes)[::-1]
    ax.barh(top_cnt.index, top_cnt["n_unexpl"], color=COLORS[2], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("# unexplainable spectra")
    ax.set_title("D. Top classes by unexplainable count")

    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_fp)


def plot_zero_cos(zero: pd.DataFrame, top_classes: int, out_fp: Path) -> None:
    """Render the 1x3 zero-cosine diagnostic figure.

    Panels: (A) zero-cos count by adduct x instrument, (B) true-peak sparsity histogram,
    (C) zero-cos spectra by ClassyFire class. The recoverable/broken split is computed
    separately by ``scripts/eval/zero_cos_recoverability.py`` (needs a 0.05 Da re-score).

    Args:
        zero: Zero-cosine per-spectrum DataFrame from :func:`dump_tables`.
        top_classes: Number of classes to show in panel C.
        out_fp: Path to write the PNG.
    """
    if zero.empty:
        log.warning("No zero-cosine spectra in scope; skipping zero-cos figure.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (A) zero-cos count by adduct x instrument
    ax = axes[0]
    ct = zero.pivot_table(
        index="prec_type", columns="inst_type", values="spec_id", aggfunc="count", fill_value=0
    )
    ct.plot(kind="bar", ax=ax, color=COLORS[: ct.shape[1]], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("adduct")
    ax.set_ylabel("# zero-cos spectra")
    ax.tick_params(axis="x", rotation=0)
    ax.set_title(f"A. Zero-cos by adduct x instrument (n={len(zero)})")
    ax.legend(title="instrument")

    # (B) sparsity
    ax = axes[1]
    tp = zero["n_true_peaks"].clip(upper=30)
    ax.hist(tp, bins=range(0, 32), color=COLORS[0], edgecolor="white", linewidth=0.3)
    ax.set_xlabel("# true peaks (clipped at 30)")
    ax.set_ylabel("# zero-cos spectra")
    ax.set_title(f"B. Zero-cos sparsity (median {int(zero['n_true_peaks'].median())} peaks)")

    # (C) by class
    ax = axes[2]
    kc = zero["klass"].value_counts().head(top_classes)[::-1]
    ax.barh(kc.index, kc.values, color=COLORS[4], edgecolor="black", linewidth=0.4)
    ax.set_xlabel("# zero-cos spectra")
    ax.set_title("C. Zero-cos spectra by ClassyFire class")

    fig.tight_layout()
    fig.savefig(out_fp, dpi=150, bbox_inches="tight")
    plt.close(fig)
    log.info("Wrote %s", out_fp)


def main() -> None:
    """Run the unexplainability analysis and write CSVs + figures."""
    args = get_args()
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)

    df = load_scoped(args)
    zero = dump_tables(df, args, out_dp)
    plot_overview(df, args, out_dp / "unexplainable_overview.png")
    plot_zero_cos(zero, args.top_classes, out_dp / "zero_cos_diagnostic.png")


if __name__ == "__main__":
    main()
