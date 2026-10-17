"""Report all low-oracle spectra and groups, including ones missed by CE-sibling flags.

This complements ``oracle_ce_sibling_trend_audit.py``. The CE-sibling audit only flags low spectra
when another collision energy in the same group looks normal. This script also accounts for low
singletons, all-low groups, and groups with no normal CE sibling.

Usage:
    python scripts/debug/oracle/oracle_bad_group_report.py \\
        --out_dp data/eval/oracle_cos_nist20_default
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

from oracle_ce_sibling_trend_audit import add_sibling_features
from oracle_dataset_utils import resolve_dsets_for_spec_fp
from plot_ce_sibling_spectrum_groups import (
    GROUP_COLS,
    add_ce_columns,
    add_peak_label_columns,
    choose_group_rows,
    compute_peak_match_masks,
    plot_group_page,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
log = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--oracle_csv",
        default="data/eval/oracle_cos_nist20_default/oracle_cos_default.csv",
        help="Per-spectrum oracle CSV.",
    )
    parser.add_argument(
        "--flags_csv",
        default="data/eval/oracle_cos_nist20_default/oracle_ce_sibling_trend_flags.csv",
        help="CE-sibling trend flag CSV.",
    )
    parser.add_argument("--spec_fp", default="data/proc/nist20/spec_df.pkl")
    parser.add_argument("--mol_fp", default="data/proc/nist20/mol_df.pkl")
    parser.add_argument(
        "--frag_dp",
        default="data/frag/nist20_ma_mi_d3_h4_isoFalse_smpFalse/dags",
        help="DAG directory used to color matched/unmatched peaks.",
    )
    parser.add_argument(
        "--dsets", "--dset", nargs="+", default=None, help="Filter spec_df to these dataset labels."
    )
    parser.add_argument("--out_dp", required=True)
    parser.add_argument("--metric", default="cos_hun_inst_tol_opt")
    parser.add_argument("--low_thresh", type=float, default=0.30)
    parser.add_argument("--normal_thresh", type=float, default=0.90)
    parser.add_argument("--inst_types", nargs="+", default=["ORB", "ORB_F", "QTOF"])
    parser.add_argument(
        "--prec_types", nargs="+", default=["[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-"]
    )
    parser.add_argument(
        "--max_spectra_per_group",
        type=int,
        default=20,
        help="For large groups, keep low/flagged spectra plus representative siblings.",
    )
    parser.add_argument("--mz_max", type=float, default=None)
    parser.add_argument("--min_int_frac", type=float, default=0.005)
    parser.add_argument("--max_peaks_per_spectrum", type=int, default=150)
    parser.add_argument(
        "--peak_color_mode",
        choices=["match", "metric"],
        default="match",
        help="match: color individual peaks by DAG match status; metric: color whole spectrum.",
    )
    parser.add_argument(
        "--max_h_transfer",
        type=int,
        default=None,
        help="Default: use the DAG's full H-transfer budget for match coloring.",
    )
    parser.add_argument(
        "--no_cmf",
        action="store_true",
        help="For match coloring, use direct-charge-retention ion masses only.",
    )
    parser.add_argument(
        "--max_plot_groups",
        type=int,
        default=None,
        help="Optionally cap plotted missing groups after sorting by worst metric.",
    )
    args = parser.parse_args()
    args.dsets = resolve_dsets_for_spec_fp(args.spec_fp, args.dsets)
    return args


def classify_low_spectra(scored: pd.DataFrame, flag_ids: set[int], metric: str) -> pd.DataFrame:
    """Classify low spectra by why they are or are not in the CE-sibling flag set."""
    low = scored[scored["flag_low_metric"]].copy()
    low["in_ce_sibling_flags"] = low["spec_id"].astype(int).isin(flag_ids)
    low["bad_category"] = "ce_sibling_flagged"
    missing = ~low["in_ce_sibling_flags"]
    low.loc[missing & low["group_n"].eq(1), "bad_category"] = "singleton_group"
    low.loc[
        missing & low["group_n"].gt(1) & low["group_low_n"].eq(low["group_n"]),
        "bad_category",
    ] = "multi_group_all_low"
    low.loc[
        missing
        & low["group_n"].gt(1)
        & low["group_low_n"].lt(low["group_n"])
        & low["group_normal_n"].eq(0),
        "bad_category",
    ] = "multi_group_no_normal_sibling"
    low.loc[
        missing & low["bad_category"].eq("ce_sibling_flagged"),
        "bad_category",
    ] = "unflagged_other"
    low["metric"] = low[metric]
    return low


def summarize_groups(
    low: pd.DataFrame, scored: pd.DataFrame, metric: str, normal_thresh: float
) -> pd.DataFrame:
    """Build one row per group that contains at least one low spectrum."""
    rows = []
    group_meta = scored.groupby(GROUP_COLS, observed=True)
    for key, low_g in low.groupby(GROUP_COLS, observed=True):
        key_d = dict(zip(GROUP_COLS, key, strict=True))
        full_g = group_meta.get_group(key)
        categories = sorted(low_g["bad_category"].unique())
        group_category = "ce_sibling_flagged_group" if bool(low_g["in_ce_sibling_flags"].any()) else categories[0]
        rows.append(
            {
                **key_d,
                "group_category": group_category,
                "bad_categories": ";".join(categories),
                "dsets": ";".join(sorted(full_g["dset"].dropna().astype(str).unique())),
                "n_spectra": len(full_g),
                "n_bad_spectra": len(low_g),
                "n_ce_sibling_flagged": int(low_g["in_ce_sibling_flags"].sum()),
                "n_normal_spectra": int((full_g[metric] >= normal_thresh).sum()),
                "worst_metric": float(low_g[metric].min()),
                "median_bad_metric": float(low_g[metric].median()),
                "best_group_metric": float(full_g[metric].max()),
                "ce_keys": ";".join(full_g.sort_values("ce_sort")["ce_key"].astype(str).unique()),
                "bad_spec_ids": ";".join(low_g.sort_values("ce_sort")["spec_id"].astype(int).astype(str)),
            }
        )
    return pd.DataFrame(rows).sort_values(["group_category", "worst_metric", "mol_id"])


def plot_category_summary(group_df: pd.DataFrame, out_fp: Path) -> None:
    """Plot bad group counts by instrument/adduct and category."""
    if group_df.empty:
        log.warning("No bad groups to plot; skipping %s", out_fp)
        return
    counts = (
        group_df.groupby(["inst_type", "prec_type", "group_category"], observed=True)
        .size()
        .reset_index(name="n_groups")
    )
    labels = (
        counts[["inst_type", "prec_type"]]
        .drop_duplicates()
        .sort_values(["inst_type", "prec_type"])
        .assign(label=lambda x: x["inst_type"] + " " + x["prec_type"])
    )
    cats = [
        "ce_sibling_flagged_group",
        "singleton_group",
        "multi_group_all_low",
        "multi_group_no_normal_sibling",
        "unflagged_other",
    ]
    colors = {
        "ce_sibling_flagged_group": "#C44E52",
        "singleton_group": "#4C72B0",
        "multi_group_all_low": "#55A868",
        "multi_group_no_normal_sibling": "#DD8452",
        "unflagged_other": "#8172B3",
    }
    pivot = counts.pivot_table(
        index=["inst_type", "prec_type"], columns="group_category", values="n_groups", fill_value=0
    ).reindex(labels.set_index(["inst_type", "prec_type"]).index, fill_value=0)

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(pivot))
    bottom = np.zeros(len(pivot))
    for cat in cats:
        vals = pivot[cat].to_numpy() if cat in pivot.columns else np.zeros(len(pivot))
        if vals.sum() == 0:
            continue
        ax.bar(x, vals, bottom=bottom, label=cat, color=colors[cat], edgecolor="black", linewidth=0.4)
        bottom += vals
    ax.set_xticks(x)
    ax.set_xticklabels(labels["label"], rotation=35, ha="right")
    ax.set_ylabel("# low-oracle groups")
    ax.set_title("Low-oracle groups by instrument/adduct")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fp, dpi=160, bbox_inches="tight")
    plt.close(fig)


def df_to_md_table(df: pd.DataFrame) -> str:
    """Render a small DataFrame as a GitHub-flavored markdown table without tabulate."""
    if df.empty:
        return "_none_"
    text_df = df.copy()
    for col in text_df.columns:
        text_df[col] = text_df[col].map(lambda x: "" if pd.isna(x) else str(x))
    header = "| " + " | ".join(text_df.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(text_df.columns)) + " |"
    rows = ["| " + " | ".join(row) + " |" for row in text_df.to_numpy(dtype=str)]
    return "\n".join([header, sep, *rows])


def write_findings_doc(
    out_fp: Path,
    spec_df: pd.DataFrame,
    low: pd.DataFrame,
    group_df: pd.DataFrame,
    missing_groups: pd.DataFrame,
    args: argparse.Namespace,
) -> None:
    """Write a short markdown summary of the findings."""
    cat_counts = (
        low.groupby("bad_category", observed=True)
        .agg(n_spectra=("spec_id", "size"), n_mols=("mol_id", "nunique"))
        .reset_index()
    )
    group_cat_counts = (
        group_df.groupby("group_category", observed=True)
        .agg(n_groups=("mol_id", "size"), n_bad_spectra=("n_bad_spectra", "sum"))
        .reset_index()
    )
    by_inst_adduct = (
        low[~low["in_ce_sibling_flags"]]
        .groupby(["inst_type", "prec_type"], observed=True)
        .size()
        .reset_index(name="n_missing_bad_spectra")
        .sort_values("n_missing_bad_spectra", ascending=False)
    )
    text = [
        "# Oracle Bad Group Findings",
        "",
        f"Scope: `{', '.join(args.dsets)}` from `{args.spec_fp}`.",
        f"Metric: `{args.metric} < {args.low_thresh}` is treated as low/bad.",
        f"Normal sibling threshold: `{args.normal_thresh}`.",
        "",
        "## Counts",
        "",
        f"- Scoped spectra: `{len(spec_df)}`",
        f"- Low/bad spectra: `{len(low)}`",
        f"- Low/bad groups: `{len(group_df)}`",
        f"- CE-sibling flagged spectra already in existing PDF: `{int(low['in_ce_sibling_flags'].sum())}`",
        f"- Low/bad spectra not in CE-sibling flags/PDF: `{int((~low['in_ce_sibling_flags']).sum())}`",
        f"- Low/bad groups not in CE-sibling PDF: `{len(missing_groups)}`",
        "",
        "## Spectrum Categories",
        "",
        df_to_md_table(cat_counts),
        "",
        "## Group Categories",
        "",
        df_to_md_table(group_cat_counts),
        "",
        "## Missing Bad Spectra By Instrument/Adduct",
        "",
        df_to_md_table(by_inst_adduct),
        "",
        "## Outputs",
        "",
        "- `oracle_bad_spectra_all.csv`: every low/bad spectrum in scope.",
        "- `oracle_bad_groups_all.csv`: every group with at least one low/bad spectrum.",
        "- `oracle_bad_groups_by_inst_adduct.png`: category count plot by instrument/adduct.",
        "- `oracle_bad_missing_spectrum_groups.pdf`: spectrum pages for low/bad groups not covered by the CE-sibling PDF.",
        "",
    ]
    out_fp.write_text("\n".join(text), encoding="utf-8")


def main() -> None:
    """Create CSVs, figures, and markdown report for all low-oracle groups."""
    args = get_args()
    out_dp = Path(args.out_dp)
    out_dp.mkdir(parents=True, exist_ok=True)

    flags = pd.read_csv(args.flags_csv)
    flag_ids = set(flags["spec_id"].astype(int))
    oracle = pd.read_csv(args.oracle_csv)
    spec = pd.read_pickle(args.spec_fp)[
        [
            "spec_id",
            "mol_id",
            "prec_type",
            "inst_type",
            "frag_mode",
            "dset",
            "ace",
            "nce",
            "prec_mz",
            "peaks",
        ]
    ]
    if args.dsets:
        spec = spec[spec["dset"].isin(args.dsets)]

    scope = oracle.merge(
        spec.drop(columns=["peaks"]),
        on=["spec_id", "mol_id", "prec_type", "inst_type", "frag_mode"],
        how="inner",
    )
    scope = scope[
        scope["prec_type"].isin(args.prec_types) & scope["inst_type"].isin(args.inst_types)
    ].copy()
    if scope.empty:
        raise ValueError("Scope filter matched no spectra.")
    scope = add_ce_columns(scope)
    scope["ce_sort"] = scope["ce_value"].fillna(np.inf)
    scope["flag_low_metric"] = scope[args.metric] < args.low_thresh
    scope["group_n"] = scope.groupby(GROUP_COLS, observed=True)["spec_id"].transform("size")
    scope["group_low_n"] = scope.groupby(GROUP_COLS, observed=True)[args.metric].transform(
        lambda s: (s < args.low_thresh).sum()
    )
    scope["group_normal_n"] = scope.groupby(GROUP_COLS, observed=True)[args.metric].transform(
        lambda s: (s >= args.normal_thresh).sum()
    )
    scope = add_sibling_features(scope, GROUP_COLS, args.metric)

    low = classify_low_spectra(scope, flag_ids, args.metric)
    group_df = summarize_groups(low, scope, args.metric, args.normal_thresh)
    missing_groups = group_df[group_df["n_ce_sibling_flagged"].eq(0)].copy()

    spectra_fp = out_dp / "oracle_bad_spectra_all.csv"
    group_fp = out_dp / "oracle_bad_groups_all.csv"
    summary_plot_fp = out_dp / "oracle_bad_groups_by_inst_adduct.png"
    missing_pdf_fp = out_dp / "oracle_bad_missing_spectrum_groups.pdf"
    doc_fp = out_dp / "oracle_bad_group_findings.md"

    spectra_cols = [
        "spec_id",
        *GROUP_COLS,
        "dset",
        "bad_category",
        "in_ce_sibling_flags",
        "ce_key",
        "ace",
        "nce",
        args.metric,
        "best_lower_ce",
        "best_higher_ce",
        "best_diff_ce",
        "group_n",
        "group_low_n",
        "group_normal_n",
    ]
    spectra_cols = [c for c in spectra_cols if c in low.columns]
    low[spectra_cols].sort_values(["bad_category", args.metric, "mol_id"]).to_csv(
        spectra_fp, index=False
    )
    group_df.to_csv(group_fp, index=False)
    plot_category_summary(group_df, summary_plot_fp)

    mol = pd.read_pickle(args.mol_fp)
    mol_by_id = mol.set_index("mol_id")
    plot_df = spec.merge(
        missing_groups[GROUP_COLS].drop_duplicates(), on=GROUP_COLS, how="inner"
    ).merge(oracle[["spec_id", args.metric]], on="spec_id", how="left")
    plot_df = add_ce_columns(plot_df)
    plot_df = add_peak_label_columns(plot_df, mol_by_id)
    plot_df["ce_sort"] = plot_df["ce_value"].fillna(np.inf)
    plot_df["is_flagged"] = plot_df[args.metric] < args.low_thresh

    missing_order = missing_groups.sort_values(["worst_metric", "mol_id"])
    if args.max_plot_groups is not None:
        missing_order = missing_order.head(args.max_plot_groups)

    page_groups = []
    selected_frames = []
    for row in missing_order.itertuples(index=False):
        key = {col: getattr(row, col) for col in GROUP_COLS}
        mask = np.ones(len(plot_df), dtype=bool)
        for col, val in key.items():
            mask &= plot_df[col].to_numpy() == val
        group = plot_df[mask].sort_values("ce_sort").copy()
        group = choose_group_rows(group, args.metric, args.max_spectra_per_group)
        page_groups.append((key, group))
        selected_frames.append(group)

    if args.peak_color_mode == "match" and selected_frames:
        selected_df = pd.concat(selected_frames, ignore_index=True)
        log.info("Computing DAG match masks for %d plotted missing-group spectra", len(selected_df))
        match_masks = compute_peak_match_masks(
            selected_df,
            args.frag_dp,
            args.max_h_transfer,
            include_cmf=not args.no_cmf,
        )
        page_groups = [
            (key, group.assign(match_mask=group["spec_id"].astype(int).map(match_masks)))
            for key, group in page_groups
        ]

    with PdfPages(missing_pdf_fp) as pdf:
        for page, (key, group) in enumerate(page_groups, start=1):
            mol_id = int(key["mol_id"])
            mol_row = mol_by_id.loc[mol_id] if mol_id in mol_by_id.index else None
            if isinstance(mol_row, pd.DataFrame):
                mol_row = mol_row.iloc[0]
            plot_group_page(
                pdf,
                group,
                args.metric,
                mol_row,
                args.mz_max,
                args.min_int_frac,
                args.max_peaks_per_spectrum,
                page,
            )
            if page % 100 == 0:
                log.info("Plotted %d/%d missing groups", page, len(page_groups))

    write_findings_doc(doc_fp, scope, low, group_df, missing_groups, args)
    log.info("Wrote %s", spectra_fp)
    log.info("Wrote %s", group_fp)
    log.info("Wrote %s", summary_plot_fp)
    log.info("Wrote %s", missing_pdf_fp)
    log.info("Wrote %s", doc_fp)


if __name__ == "__main__":
    main()
