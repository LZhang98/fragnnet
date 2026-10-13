"""Export the zero-cosine oracle spectra to MGF + metadata CSV for external formula decomposition.

The zero-cosine tail (oracle ceiling == 0 at instrument tolerance) matches no DAG fragment.
This dumps those spectra in a form MSBuddy (or any formula-decomposition tool) can consume, so
each peak m/z can be tested against the precursor's element budget:
  - if a peak decomposes to NO valid sub-formula -> the m/z is wrong (miscalibration / contaminant),
  - if it decomposes to a valid formula absent from the DAG -> a genuine fragmentation gap.

Two triage signals are attached per spectrum so the MSBuddy result can be interpreted in context:
  - cross-CE control: within the same (mol_id, adduct, instrument), is there a sibling spectrum at
    another collision energy that IS well explained? (best_sibling_cos, n_siblings, all_ce_zero).
    The DAG is CE-independent, so a good sibling proves coverage is fine -> the zero is a per-spectrum
    acquisition artifact.
  - 0.05 Da recoverability from zero_cos_recoverability.py (recoverable_0p05da), if present.

Outputs:
  - zero_cos.mgf: one BEGIN IONS block per spectrum (TITLE=spec_id, PEPMASS, CHARGE, ADDUCT, peaks).
  - zero_cos_export_meta.csv: per-spectrum metadata keyed by spec_id, for joining MSBuddy output back.

Usage:
    python scripts/eval/export_zero_cos.py \\
        --oracle_csv data/eval/oracle_cos_nist20_default/oracle_cos_default.csv \\
        --out_dp data/eval/oracle_cos_nist20_default \\
        --inst_types FT QTOF \\
        --prec_types "[M+H]+" "[M-H]-" "[M+Na]+" "[M+Cl]-"
"""

from __future__ import annotations

import argparse
import logging
import os

import pandas as pd

from oracle_dataset_utils import resolve_dsets_for_spec_fp

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)

COS_COL = "cos_hun_inst_tol_opt"
DEFAULT_TIER = "1_above_prec"
# Adduct -> integer charge (sign only; all these are singly charged).
ADDUCT_CHARGE = {"[M+H]+": 1, "[M+Na]+": 1, "[M+Cl]-": -1, "[M-H]-": -1}


def resolve_metric_col(df: pd.DataFrame, base: str, tier: str = DEFAULT_TIER) -> str:
    """Resolve a metric column from current tier-suffixed or legacy unsuffixed oracle CSVs."""
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
        "--oracle_csv", default="data/eval/oracle_cos_nist20_default/oracle_cos_default.csv"
    )
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df/oracle rows to these dataset labels."
    )
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument("--classyfire_fp", default="data/classyfire/nist20_mol_classyfire.pkl")
    parser.add_argument(
        "--recoverability_csv",
        default="data/eval/oracle_cos_nist20_default/zero_cos_recoverability.csv",
        help="Optional output of zero_cos_recoverability.py; merged if present.",
    )
    parser.add_argument("--out_dp", required=True)
    parser.add_argument("--inst_types", nargs="+", default=["ORB", "ORB_F", "QTOF"])
    parser.add_argument(
        "--prec_types", nargs="+", default=["[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-"]
    )
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)
    return args


def cross_ce_stats(scoped: pd.DataFrame) -> pd.DataFrame:
    """Compute per-spectrum cross-CE sibling statistics.

    Siblings are other spectra of the same (mol_id, prec_type, inst_type) -- the DAG is shared, so
    they differ mainly by collision energy. A high best-sibling ceiling proves the DAG covers this
    molecule+adduct, so a cos-0 at one CE is an acquisition artifact rather than a coverage gap.

    Args:
        scoped: Focused per-spectrum oracle DataFrame (must include COS_COL and the grouping keys).

    Returns:
        DataFrame indexed by spec_id with columns n_siblings, best_sibling_cos, all_ce_zero.
    """
    cos_col = resolve_metric_col(scoped, COS_COL)
    grp = ["mol_id", "prec_type", "inst_type"]
    g = scoped.groupby(grp)[cos_col]
    out = pd.DataFrame({"spec_id": scoped["spec_id"].to_numpy()})
    out["grp_size"] = g.transform("size").to_numpy()
    out["grp_max"] = g.transform("max").to_numpy()
    # best sibling excluding self: if self is the unique max, fall back to the 2nd best via group max
    # of the others. For a zero-cos row self==0, so grp_max already equals the best sibling.
    out["n_siblings"] = out["grp_size"] - 1
    out["best_sibling_cos"] = out["grp_max"]  # valid for zero-cos rows (self contributes 0)
    out["all_ce_zero"] = (out["grp_max"] == 0.0) & (out["grp_size"] > 1)
    return out.set_index("spec_id")[["n_siblings", "best_sibling_cos", "all_ce_zero"]]


def write_mgf(meta: pd.DataFrame, peaks_by_id: dict, out_fp: str) -> None:
    """Write one MGF BEGIN IONS block per spectrum.

    Args:
        meta: Per-spectrum metadata indexed by spec_id (needs prec_mz, prec_type, charge).
        peaks_by_id: Mapping spec_id -> list of (mz, intensity) peak tuples.
        out_fp: Path to write the MGF.
    """
    with open(out_fp, "w") as fh:
        for spec_id, row in meta.iterrows():
            # MGF CHARGE is written as magnitude + sign (e.g. "1+" / "1-"); all adducts here are z=1.
            charge = row["charge"]
            sign = "+" if charge > 0 else "-"
            fh.write("BEGIN IONS\n")
            fh.write(f"TITLE={spec_id}\n")
            fh.write(f"SCANS={spec_id}\n")
            fh.write(f"PEPMASS={row['prec_mz']:.6f}\n")
            fh.write(f"CHARGE={abs(charge)}{sign}\n")
            fh.write(f"ADDUCT={row['prec_type']}\n")
            for mz, inten in peaks_by_id[spec_id]:
                fh.write(f"{mz:.6f} {inten:.6g}\n")
            fh.write("END IONS\n\n")
    log.info("Wrote %s (%d spectra)", out_fp, len(meta))


def main() -> None:
    """Build the zero-cos MGF + metadata export."""
    args = get_args()
    os.makedirs(args.out_dp, exist_ok=True)

    orc = pd.read_csv(args.oracle_csv)
    if args.dsets:
        spec_ids = pd.read_pickle(args.spec_fp)[["spec_id", "dset"]]
        spec_ids = spec_ids[spec_ids["dset"].isin(args.dsets)]
        orc = orc.merge(spec_ids, on="spec_id", how="inner")
    scoped = orc[
        orc["prec_type"].isin(args.prec_types) & orc["inst_type"].isin(args.inst_types)
    ].copy()
    cos_col = resolve_metric_col(scoped, COS_COL)
    n_true_col = resolve_metric_col(scoped, "n_true_peaks")
    if n_true_col != "n_true_peaks":
        scoped["n_true_peaks"] = scoped[n_true_col]
    # Sibling stats must see the FULL scope (all CEs), not just the zero-cos rows -- the control is
    # precisely the non-zero siblings, so compute this before subsetting to the zero-cos tail.
    sib = cross_ce_stats(scoped)

    zero = scoped[scoped[cos_col] == 0.0].copy().set_index("spec_id")
    zero = zero.join(sib, how="left")
    log.info("Zero-cos spectra: %d across %d molecules", len(zero), zero["mol_id"].nunique())

    # structure + class
    mol = pd.read_pickle(args.mol_fp)[["mol_id", "smiles", "formula", "num_atoms", "exact_mw"]]
    cf = pd.read_pickle(args.classyfire_fp)[["mol_id", "superklass", "klass"]]
    zero = zero.reset_index().merge(mol, on="mol_id", how="left").merge(cf, on="mol_id", how="left")

    # peaks + precursor m/z from spec_df (the oracle CSV carries neither; peaks are the raw,
    # unfiltered measured m/z -- exactly what a formula-decomposition tool needs to see).
    spec_cols = ["spec_id", "dset", "prec_mz", "peaks"]
    spec = pd.read_pickle(args.spec_fp)[spec_cols]
    if args.dsets:
        spec = spec[spec["dset"].isin(args.dsets)]
    spec = spec.drop(columns=["dset"])
    zero = zero.merge(spec, on="spec_id", how="left")
    peaks_by_id = dict(zip(zero["spec_id"], zero["peaks"]))
    zero["n_peaks"] = zero["peaks"].apply(len)
    # charge sign drives the MGF CHARGE line; fail loud if an adduct slips through without a mapping.
    zero["charge"] = zero["prec_type"].map(ADDUCT_CHARGE)
    if zero["charge"].isna().any():
        missing = sorted(zero.loc[zero["charge"].isna(), "prec_type"].unique())
        raise ValueError(f"No charge mapping for adduct(s): {missing}")

    # optional 0.05 Da recoverability merge
    if os.path.isfile(args.recoverability_csv):
        rec = pd.read_csv(args.recoverability_csv)[
            ["spec_id", "recoverable_0p05da", "n_ach_0p05da"]
        ]
        zero = zero.merge(rec, on="spec_id", how="left")
        log.info("Merged 0.05 Da recoverability from %s", args.recoverability_csv)
    else:
        log.warning("No recoverability CSV at %s; skipping those columns.", args.recoverability_csv)

    meta_cols = [
        "spec_id", "mol_id", "prec_type", "charge", "inst_type", "frag_mode", "prec_mz",
        "n_peaks", "n_true_peaks", "n_siblings", "best_sibling_cos", "all_ce_zero",
        "formula", "exact_mw", "num_atoms", "superklass", "klass", "smiles",
    ]
    # Splice the optional recoverability columns in right after the cross-CE flags, so all triage
    # signals sit together in the CSV. Only added when zero_cos_recoverability.py has been run.
    if "recoverable_0p05da" in zero.columns:
        meta_cols[meta_cols.index("all_ce_zero") + 1 : meta_cols.index("all_ce_zero") + 1] = [
            "recoverable_0p05da", "n_ach_0p05da"
        ]
    # Sort the systematic residue to the top: all-CE-zero first, then worst cross-CE control -- the
    # rows a formula-decomposition pass most needs to look at (siblings can't already vindicate them).
    meta = zero[meta_cols].sort_values(["all_ce_zero", "best_sibling_cos"], ascending=[False, True])
    meta_fp = os.path.join(args.out_dp, "zero_cos_export_meta.csv")
    meta.to_csv(meta_fp, index=False)
    log.info("Wrote %s", meta_fp)

    write_mgf(meta.set_index("spec_id"), peaks_by_id, os.path.join(args.out_dp, "zero_cos.mgf"))

    # quick triage summary
    log.info("Triage: %d all-CE-zero (systematic), %d have a sibling cos>=0.9 (artifact), %d solo",
             int(meta["all_ce_zero"].sum()),
             int((meta["best_sibling_cos"] >= 0.9).sum()),
             int((meta["n_siblings"] == 0).sum()))


if __name__ == "__main__":
    main()
