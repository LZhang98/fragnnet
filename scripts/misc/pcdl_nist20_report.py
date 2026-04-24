"""Generate a PDF report for the PCDL vs NIST20 spectral overlap analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages

# ── paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("/home/feiw/gitRepos/frag-gnn/figs/pcdl_nist20_overlap_mztol0015_with_mol")
OUT_PDF = DATA_DIR / "pcdl_nist20_report.pdf"

MAIN_CSV = DATA_DIR / "pcdl_vs_nist20_best_matches.csv"
BAD_05 = DATA_DIR / "bad_matches" / "bad_cos_lt_0p5.csv"
BAD_07 = DATA_DIR / "bad_matches" / "bad_cos_lt_0p7.csv"
BAD_085 = DATA_DIR / "bad_matches" / "bad_cos_lt_0p85.csv"
WORST = DATA_DIR / "bad_matches" / "worst_100.csv"

# ── palette ──────────────────────────────────────────────────────────────────
BLUE = "#2c7bb6"
RED = "#d7191c"
GREEN = "#1a9641"
ORANGE = "#fdae61"
PURPLE = "#756bb1"
GREY = "#969696"
CE_COLORS = {10.0: BLUE, 20.0: GREEN, 40.0: ORANGE, 4.0: PURPLE, 5.0: GREY}
RESIDUAL_COLORS = {
    "matched_intensity_sqdiff": BLUE,
    "pcdl_unmatched_l2sq": RED,
    "nist_unmatched_noniso_l2sq": ORANGE,
    "nist_unmatched_isotope_like_l2sq": GREEN,
}
RESIDUAL_LABELS = {
    "matched_intensity_sqdiff": "Matched Δintensity",
    "pcdl_unmatched_l2sq": "PCDL unmatched (non-iso)",
    "nist_unmatched_noniso_l2sq": "NIST unmatched (non-iso)",
    "nist_unmatched_isotope_like_l2sq": "NIST unmatched (iso-like)",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "figure.dpi": 150,
    }
)

COS_BINS = [0, 0.5, 0.7, 0.85, 0.9, 0.95, 1.001]
COS_LABELS = ["<0.50", "0.50–0.70", "0.70–0.85", "0.85–0.90", "0.90–0.95", "≥0.95"]
PCDL_DSET = "pcdl"
NIST_DSET = "nist20_hr"
MONA_DSET = "mona23"


def bin_cos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["cos_bin"] = pd.cut(df["best_cos"], bins=COS_BINS, labels=COS_LABELS, right=False)
    return df


def _normalize_sources(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize source labels to explicit values used by the report logic.

    The report expects three source labels:
    - pcdl (PCDL side)
    - nist20_hr (NIST20-HR side)
    - mona23 (MoNA side)
    """
    out = df.copy()

    pcdl_source = out["pcdl_dset"].astype(str).str.strip().str.lower()
    if not pcdl_source.eq(PCDL_DSET).all():
        bad = sorted(set(pcdl_source[~pcdl_source.eq(PCDL_DSET)]))
        raise ValueError(
            f"Unexpected PCDL source(s) in pcdl_dset: {bad}. Expected only '{PCDL_DSET}'."
        )
    out["pcdl_source"] = pcdl_source

    nist_dset = out["nist_dset"].astype(str).str.strip().str.lower()
    nist_spec = out["nist_dset_spec_id"].astype(str).str.strip().str.lower()

    nist_source = pd.Series("", index=out.index, dtype="object")
    nist_source = nist_source.mask(nist_dset.eq(NIST_DSET), NIST_DSET)
    nist_source = nist_source.mask(nist_dset.eq(MONA_DSET), MONA_DSET)

    # Backward compatibility for legacy labels and ID-prefixed records.
    nist_source = nist_source.mask(nist_source.eq("") & nist_dset.eq("nist20"), NIST_DSET)
    nist_source = nist_source.mask(
        nist_source.eq("") & nist_spec.str.startswith(NIST_DSET), NIST_DSET
    )
    nist_source = nist_source.mask(
        nist_source.eq("") & nist_spec.str.startswith(MONA_DSET), MONA_DSET
    )

    unknown = sorted(set(nist_dset[nist_source.eq("")]))
    if unknown:
        raise ValueError(
            "Could not map nist source labels to expected values "
            f"('{NIST_DSET}' or '{MONA_DSET}'). Unknown labels: {unknown}."
        )

    out["nist_source"] = nist_source
    return out


# ── load data ────────────────────────────────────────────────────────────────
df_all = pd.read_csv(MAIN_CSV)  # unfiltered – used only for mirror plot metadata
df_all = _normalize_sources(df_all)

mona_mask_all = df_all["nist_source"].eq(MONA_DSET) & df_all["pcdl_source"].eq(PCDL_DSET)
nist_mask_all = df_all["nist_source"].eq(NIST_DSET) & df_all["pcdl_source"].eq(PCDL_DSET)
mona_pcdl_ids = set(df_all.loc[mona_mask_all, "pcdl_spec_id"])

df = df_all.loc[nist_mask_all].reset_index(drop=True)
df = bin_cos(df)

# Load or create worst_100
if WORST.exists():
    worst = pd.read_csv(WORST)
    worst = _normalize_sources(worst)
else:
    # Create worst_100 from NIST vs PCDL data (non-MoNA)
    worst = df_all.loc[nist_mask_all].nsmallest(100, "best_cos")

worst = worst.loc[
    worst["nist_source"].eq(NIST_DSET) & worst["pcdl_source"].eq(PCDL_DSET)
].reset_index(drop=True)
n_total = len(df)


# ── helper functions ──────────────────────────────────────────────────────────
def _pct(n: int, total: int) -> str:
    return f"{n} ({100 * n / total:.1f}%)"


def add_value_labels(ax: plt.Axes, bars, fmt: str = "{:.0f}", fontsize: int = 7) -> None:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                h + ax.get_ylim()[1] * 0.01,
                fmt.format(h),
                ha="center",
                va="bottom",
                fontsize=fontsize,
            )


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 0 – Cover table
# ═══════════════════════════════════════════════════════════════════════════════
def make_cover(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.patch.set_facecolor("white")

    # header
    header_ax = fig.add_axes([0, 0.91, 1, 0.09])
    header_ax.set_facecolor("#2c3e50")
    header_ax.axis("off")
    header_ax.text(
        0.5,
        0.62,
        "PCDL vs NIST20 Spectral Overlap Report",
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
        color="white",
    )
    header_ax.text(
        0.5,
        0.18,
        "m/z tolerance = 0.015 Da  |  QTOF / CID",
        ha="center",
        va="center",
        fontsize=10,
        color="#ecf0f1",
    )

    # build table data
    main_ces = [10.0, 20.0, 40.0]
    adducts = ["[M+H]+", "[M-H]-"]

    # overall stats block
    cos_bins_count = df["cos_bin"].value_counts(sort=False)

    def bad_rate(sub: pd.DataFrame) -> str:
        rate = (sub["best_cos"] < 0.7).mean()
        return f"{100 * rate:.1f}%"

    # ── Table 1: overall dataset overview ────────────────────────────────────
    ax1 = fig.add_axes([0.05, 0.70, 0.90, 0.19])
    ax1.axis("off")
    ax1.set_title("Dataset Overview", fontsize=11, fontweight="bold", loc="left", pad=6)
    overview_rows = [
        ["Total spectrum pairs", f"{n_total:,}"],
        ["Instrument / frag mode", "QTOF / CID"],
        ["Adducts", "[M+H]⁺  77%   |   [M-H]⁻  23%"],
        ["Collision energies (ACE)", "10 eV  37%   |   20 eV  35%   |   40 eV  28%"],
        ["Median cosine similarity", f"{df['best_cos'].median():.4f}"],
        ["Mean cosine similarity", f"{df['best_cos'].mean():.4f}"],
        ["Bad matches (cos < 0.70)", f"{(df['best_cos'] < 0.7).sum():,}  ({bad_rate(df)})"],
        ["Median NIST candidates per pair", f"{df['n_candidates'].median():.0f}"],
    ]
    tbl1 = ax1.table(
        cellText=overview_rows, colLabels=["Metric", "Value"], loc="center", cellLoc="left"
    )
    tbl1.auto_set_font_size(False)
    tbl1.set_fontsize(9)
    tbl1.scale(1, 1.55)
    for (row, col), cell in tbl1.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f8ff")
        if col == 0:
            cell.set_text_props(fontweight="bold")

    # ── Table 2: cosine bin breakdown ────────────────────────────────────────
    ax2 = fig.add_axes([0.05, 0.445, 0.90, 0.235])
    ax2.axis("off")
    ax2.set_title(
        "Cosine Similarity Breakdown by Bin", fontsize=11, fontweight="bold", loc="left", pad=6
    )
    bin_rows = []
    for lbl in COS_LABELS:
        n = cos_bins_count.get(lbl, 0)
        bin_rows.append([lbl, f"{n:,}", f"{100 * n / n_total:.1f}%"])
    bin_rows.append(["Total", f"{n_total:,}", "100.0%"])
    tbl2 = ax2.table(
        cellText=bin_rows,
        colLabels=["Cosine bin", "Count", "% of total"],
        loc="center",
        cellLoc="center",
    )
    tbl2.auto_set_font_size(False)
    tbl2.set_fontsize(9)
    tbl2.scale(1, 1.52)
    for (row, col), cell in tbl2.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row == len(bin_rows):
            cell.set_facecolor("#ddeeff")
            cell.set_text_props(fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f8ff")

    # ── Table 3: breakdown by CE × adduct ────────────────────────────────────
    ax3 = fig.add_axes([0.05, 0.14, 0.90, 0.285])
    ax3.axis("off")
    ax3.set_title(
        "Cosine Statistics by CE and Adduct", fontsize=11, fontweight="bold", loc="left", pad=6
    )
    ce_rows = []
    for ce in main_ces:
        for ad in adducts:
            sub = df[(df["ace"] == ce) & (df["prec_type"] == ad)]
            if len(sub) == 0:
                continue
            ce_rows.append(
                [
                    f"{ce:.0f} eV",
                    ad,
                    f"{len(sub):,}",
                    f"{sub['best_cos'].median():.4f}",
                    f"{sub['best_cos'].mean():.4f}",
                    f"{sub['best_cos'].quantile(0.10):.4f}",
                    bad_rate(sub),
                ]
            )
    tbl3 = ax3.table(
        cellText=ce_rows,
        colLabels=["CE", "Adduct", "N", "Median cos", "Mean cos", "p10 cos", "Bad (<0.70)"],
        loc="center",
        cellLoc="center",
    )
    tbl3.auto_set_font_size(False)
    tbl3.set_fontsize(9)
    tbl3.scale(1, 1.52)
    for (row, col), cell in tbl3.get_celld().items():
        cell.set_edgecolor("#cccccc")
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f8ff")

    # footer
    foot_ax = fig.add_axes([0, 0, 1, 0.04])
    foot_ax.axis("off")
    foot_ax.text(
        0.5,
        0.5,
        "Generated by scripts/misc/pcdl_nist20_report.py  |  "
        "Data: figs/pcdl_nist20_overlap_mztol0015_with_mol/",
        ha="center",
        va="center",
        fontsize=7,
        color="#888888",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 – Cosine similarity overview
# ═══════════════════════════════════════════════════════════════════════════════
def make_page_cos_overview(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Cosine Similarity Overview", fontsize=14, fontweight="bold", y=0.97)

    gs = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.45, wspace=0.35, left=0.1, right=0.93, top=0.92, bottom=0.06
    )

    # 1a – histogram full range
    ax1 = fig.add_subplot(gs[0, :])
    ax1.hist(df["best_cos"], bins=100, color=BLUE, edgecolor="none", alpha=0.85)
    ax1.axvline(
        df["best_cos"].median(),
        color=RED,
        lw=1.5,
        ls="--",
        label=f"Median = {df['best_cos'].median():.3f}",
    )
    ax1.axvline(
        df["best_cos"].mean(),
        color=ORANGE,
        lw=1.5,
        ls="--",
        label=f"Mean = {df['best_cos'].mean():.3f}",
    )
    ax1.set_xlabel("Cosine similarity")
    ax1.set_ylabel("Count")
    ax1.set_title("(a) Distribution of best cosine similarity (all pairs, n=7,052)")
    ax1.legend()

    # 1b – cos bin bar chart
    ax2 = fig.add_subplot(gs[1, 0])
    bin_counts = df["cos_bin"].value_counts(sort=False)
    colors = [
        GREEN if lbl == "≥0.95" else (ORANGE if lbl in ("0.90–0.95", "0.85–0.90") else RED)
        for lbl in bin_counts.index
    ]
    bars = ax2.bar(bin_counts.index, bin_counts.values, color=colors, edgecolor="none")
    ax2.set_xlabel("Cosine bin")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Pairs per cosine bin")
    ax2.tick_params(axis="x", rotation=35)
    add_value_labels(ax2, bars)

    # 1c – cumulative fraction
    ax3 = fig.add_subplot(gs[1, 1])
    sorted_cos = np.sort(df["best_cos"].values)
    cdf = np.arange(1, len(sorted_cos) + 1) / len(sorted_cos)
    ax3.plot(sorted_cos, cdf, color=BLUE, lw=1.5)
    for thresh, color in [(0.5, RED), (0.7, ORANGE), (0.85, PURPLE), (0.95, GREEN)]:
        frac = (df["best_cos"] >= thresh).mean()
        ax3.axvline(thresh, color=color, lw=1, ls=":", alpha=0.8)
        ax3.text(
            thresh + 0.005, 0.15, f"{thresh:.2f}\n{frac * 100:.1f}%≥", fontsize=6.5, color=color
        )
    ax3.set_xlabel("Cosine threshold")
    ax3.set_ylabel("Cumulative fraction")
    ax3.set_title("(c) CDF – fraction of pairs ≥ threshold")
    ax3.set_xlim(0, 1.01)
    ax3.set_ylim(0, 1.05)
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))

    # 1d – percentile table
    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    pcts = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    vals = [np.percentile(df["best_cos"], p) for p in pcts]
    table_data = [
        [f"p{p}" for p in pcts],
        [f"{v:.4f}" for v in vals],
    ]
    tbl = ax4.table(
        cellText=table_data,
        rowLabels=["Percentile", "Cosine"],
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.6)
    for (row, col), cell in tbl.get_celld().items():
        if col == -1:
            cell.set_facecolor("#eaf2ff")
            cell.set_text_props(fontweight="bold")
        elif row == 0:
            cell.set_facecolor("#d0e8ff")
    ax4.set_title("(d) Cosine similarity percentile table", pad=12, fontsize=9)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 – Breakdown by CE and adduct
# ═══════════════════════════════════════════════════════════════════════════════
def make_page_ce_adduct(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(
        "Cosine by Collision Energy and Adduct Type", fontsize=14, fontweight="bold", y=0.97
    )
    gs = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.5, wspace=0.35, left=0.1, right=0.93, top=0.92, bottom=0.06
    )

    main_ces = [10.0, 20.0, 40.0]
    df_main = df[df["ace"].isin(main_ces)]

    # 2a – violin by CE
    ax1 = fig.add_subplot(gs[0, :])
    data_by_ce = [df_main[df_main["ace"] == ce]["best_cos"].values for ce in main_ces]
    parts = ax1.violinplot(data_by_ce, positions=[1, 2, 3], showmedians=True, showextrema=False)
    for i, (pc, ce) in enumerate(zip(parts["bodies"], main_ces)):
        pc.set_facecolor(CE_COLORS[ce])
        pc.set_alpha(0.7)
    parts["cmedians"].set_colors(["black"] * 3)
    ax1.set_xticks([1, 2, 3])
    ax1.set_xticklabels(
        [f"{ce:.0f} eV\nn={df_main[df_main['ace'] == ce].shape[0]:,}" for ce in main_ces]
    )
    ax1.set_ylabel("Cosine similarity")
    ax1.set_title("(a) Cosine distribution by collision energy (violin)")
    for i, (ce, data) in enumerate(zip(main_ces, data_by_ce)):
        med = np.median(data)
        ax1.text(i + 1 + 0.15, med, f" {med:.3f}", va="center", fontsize=8, color="black")

    # 2b – stacked bar of cos bins by CE
    ax2 = fig.add_subplot(gs[1, 0])
    bin_colors = [RED, RED, ORANGE, ORANGE, GREEN, GREEN]
    bin_colors_map = dict(zip(COS_LABELS, [RED, RED, ORANGE, ORANGE, GREEN, GREEN]))
    bottom = np.zeros(len(main_ces))
    for lbl, color in zip(COS_LABELS, [RED, "#e08080", ORANGE, "#ffe0a0", "#a8d8a8", GREEN]):
        fracs = [
            (
                df_main[(df_main["ace"] == ce) & (df_main["cos_bin"] == lbl)].shape[0]
                / df_main[df_main["ace"] == ce].shape[0]
            )
            for ce in main_ces
        ]
        ax2.bar(
            [f"{ce:.0f}eV" for ce in main_ces],
            fracs,
            bottom=bottom,
            color=color,
            label=lbl,
            edgecolor="none",
        )
        bottom += np.array(fracs)
    ax2.set_ylabel("Fraction")
    ax2.set_title("(b) Cos bin breakdown by CE")
    ax2.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax2.legend(loc="upper right", fontsize=7, title="Cos bin", title_fontsize=7)

    # 2c – box by adduct
    ax3 = fig.add_subplot(gs[1, 1])
    adducts = ["[M+H]+", "[M-H]-"]
    data_by_adduct = [df[df["prec_type"] == a]["best_cos"].values for a in adducts]
    bp = ax3.boxplot(
        data_by_adduct, patch_artist=True, widths=0.45, medianprops=dict(color="black", lw=1.5)
    )
    for patch, color in zip(bp["boxes"], [BLUE, ORANGE]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax3.set_xticks([1, 2])
    ax3.set_xticklabels([f"{a}\nn={len(d):,}" for a, d in zip(adducts, data_by_adduct)])
    ax3.set_ylabel("Cosine similarity")
    ax3.set_title("(c) Cosine by adduct type (box)")
    for i, data in enumerate(data_by_adduct):
        ax3.text(i + 1 + 0.22, np.median(data), f" {np.median(data):.3f}", va="center", fontsize=8)

    # 2d – heatmap: median cos by CE × adduct
    ax4 = fig.add_subplot(gs[2, 0])
    heat = np.zeros((len(main_ces), len(adducts)))
    for i, ce in enumerate(main_ces):
        for j, ad in enumerate(adducts):
            sub = df_main[(df_main["ace"] == ce) & (df_main["prec_type"] == ad)]["best_cos"]
            heat[i, j] = sub.median() if len(sub) > 0 else np.nan
    im = ax4.imshow(heat, vmin=0.85, vmax=1.0, cmap="RdYlGn", aspect="auto")
    ax4.set_xticks(range(len(adducts)))
    ax4.set_xticklabels(adducts)
    ax4.set_yticks(range(len(main_ces)))
    ax4.set_yticklabels([f"{ce:.0f} eV" for ce in main_ces])
    for i in range(len(main_ces)):
        for j in range(len(adducts)):
            ax4.text(
                j, i, f"{heat[i, j]:.3f}", ha="center", va="center", fontsize=9, fontweight="bold"
            )
    plt.colorbar(im, ax=ax4, label="Median cosine")
    ax4.set_title("(d) Median cosine: CE × adduct")

    # 2e – n_candidates histogram
    ax5 = fig.add_subplot(gs[2, 1])
    nc = df["n_candidates"].clip(upper=10)
    ax5.hist(nc, bins=range(1, 12), color=PURPLE, edgecolor="white", align="left")
    ax5.set_xlabel("n NIST candidates (clipped at 10)")
    ax5.set_ylabel("Count")
    ax5.set_title("(e) NIST candidates per PCDL spectrum")
    ax5.text(
        0.95,
        0.95,
        f"Median={df['n_candidates'].median():.0f}\nMean={df['n_candidates'].mean():.2f}",
        transform=ax5.transAxes,
        ha="right",
        va="top",
        fontsize=8,
        color="#333333",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 – Residual / failure mode analysis
# ═══════════════════════════════════════════════════════════════════════════════
def make_page_residuals(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle(
        "Failure Mode Analysis (dominant_residual)", fontsize=14, fontweight="bold", y=0.97
    )
    gs = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.5, wspace=0.4, left=0.1, right=0.93, top=0.92, bottom=0.06
    )

    res_order = list(RESIDUAL_LABELS.keys())
    res_colors = [RESIDUAL_COLORS[r] for r in res_order]
    res_short = [RESIDUAL_LABELS[r] for r in res_order]

    # 3a – pie: overall residual distribution
    ax1 = fig.add_subplot(gs[0, 0])
    counts = df["dominant_residual"].value_counts().reindex(res_order).fillna(0)
    wedges, texts, autotexts = ax1.pie(
        counts.values,
        labels=None,
        colors=res_colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.0),
    )
    for at in autotexts:
        at.set_fontsize(8)
    ax1.legend(
        wedges,
        [f"{s}\n(n={int(c):,})" for s, c in zip(res_short, counts.values)],
        loc="lower left",
        fontsize=6.5,
        bbox_to_anchor=(-0.3, -0.35),
    )
    ax1.set_title("(a) Overall dominant residual (all pairs)")

    # 3b – pie: bad matches only (<0.7)
    ax2 = fig.add_subplot(gs[0, 1])
    bad = df[df["best_cos"] < 0.7]
    counts_bad = bad["dominant_residual"].value_counts().reindex(res_order).fillna(0)
    wedges2, _, autotexts2 = ax2.pie(
        counts_bad.values,
        labels=None,
        colors=res_colors,
        autopct="%1.1f%%",
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=1.0),
    )
    for at in autotexts2:
        at.set_fontsize(8)
    ax2.legend(
        wedges2,
        [f"{s}\n(n={int(c):,})" for s, c in zip(res_short, counts_bad.values)],
        loc="lower left",
        fontsize=6.5,
        bbox_to_anchor=(-0.3, -0.35),
    )
    ax2.set_title(f"(b) Dominant residual – bad matches only\n(cos < 0.70, n={len(bad):,})")

    # 3c – residual breakdown by cos bin (stacked)
    ax3 = fig.add_subplot(gs[1, :])
    bottom = np.zeros(len(COS_LABELS))
    for res, color in zip(res_order, res_colors):
        fracs = []
        for lbl in COS_LABELS:
            sub = df[df["cos_bin"] == lbl]
            fracs.append(sub[sub["dominant_residual"] == res].shape[0] / max(len(sub), 1))
        ax3.bar(
            COS_LABELS,
            fracs,
            bottom=bottom,
            color=color,
            label=RESIDUAL_LABELS[res],
            edgecolor="none",
        )
        bottom += np.array(fracs)
    ax3.set_ylabel("Fraction")
    ax3.set_xlabel("Cosine bin")
    ax3.set_title("(c) Dominant residual fraction by cosine bin")
    ax3.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax3.legend(loc="upper left", fontsize=7.5, title="Residual type", title_fontsize=7.5)

    # 3d – unmatched L2 scatter: pcdl_unmatched vs nist_unmatched
    ax4 = fig.add_subplot(gs[2, 0])
    sc = ax4.scatter(
        df["pcdl_unmatched_l2sq"],
        df["nist_unmatched_noniso_l2sq"],
        c=df["best_cos"],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=5,
        alpha=0.4,
    )
    plt.colorbar(sc, ax=ax4, label="best_cos")
    ax4.set_xlabel("PCDL unmatched L²")
    ax4.set_ylabel("NIST unmatched (non-iso) L²")
    ax4.set_title("(d) Unmatched energy scatter\n(coloured by cosine)")
    ax4.set_xlim(-0.02, 1.05)
    ax4.set_ylim(-0.02, 1.05)
    diag = np.linspace(0, 1, 50)
    ax4.plot(diag, diag, "k--", lw=0.7, alpha=0.5, label="equal")
    ax4.legend(fontsize=7)

    # 3e – n_pairs=0 analysis
    ax5 = fig.add_subplot(gs[2, 1])
    zero_pairs = df[df["n_pairs"] == 0]
    nonzero_pairs = df[df["n_pairs"] > 0]
    labels = ["n_pairs = 0\n(no shared peaks)", "n_pairs > 0"]
    vals = [len(zero_pairs), len(nonzero_pairs)]
    bars = ax5.bar(labels, vals, color=[RED, BLUE], edgecolor="none")
    add_value_labels(ax5, bars)
    ax5.set_ylabel("Count")
    ax5.set_title("(e) Pairs with zero matched peaks")
    ax5.text(
        0,
        vals[0] * 0.5,
        f"{100 * vals[0] / n_total:.1f}%\ncos always 0",
        ha="center",
        va="center",
        fontsize=9,
        color="white",
        fontweight="bold",
    )

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 – Peak count analysis
# ═══════════════════════════════════════════════════════════════════════════════
def make_page_peaks(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Peak Count Analysis", fontsize=14, fontweight="bold", y=0.97)
    gs = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.5, wspace=0.4, left=0.1, right=0.93, top=0.92, bottom=0.06
    )

    # 4a – PCDL vs NIST peak counts scatter
    ax1 = fig.add_subplot(gs[0, :])
    sc = ax1.scatter(
        df["pcdl_n_peaks"],
        df["nist_n_peaks"],
        c=df["best_cos"],
        cmap="RdYlGn",
        vmin=0,
        vmax=1,
        s=6,
        alpha=0.35,
    )
    plt.colorbar(sc, ax=ax1, label="best cosine")
    maxv = max(df["pcdl_n_peaks"].max(), df["nist_n_peaks"].max())
    ax1.plot([0, maxv], [0, maxv], "k--", lw=0.8, alpha=0.5, label="equal")
    ax1.set_xlabel("PCDL n_peaks")
    ax1.set_ylabel("NIST n_peaks")
    ax1.set_title("(a) PCDL vs NIST peak count (coloured by cosine)")
    ax1.legend(fontsize=7)

    # 4b – n_pairs histogram
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.hist(df["n_pairs"].clip(upper=60), bins=30, color=BLUE, edgecolor="none", alpha=0.85)
    ax2.set_xlabel("Matched peak pairs (clipped at 60)")
    ax2.set_ylabel("Count")
    ax2.set_title("(b) Distribution of matched peak pairs per spectrum pair")
    med = df["n_pairs"].median()
    ax2.axvline(med, color=RED, lw=1.5, ls="--", label=f"Median={med:.0f}")
    ax2.legend()

    # 4c – n_pairs vs best_cos
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.scatter(df["n_pairs"].clip(upper=80), df["best_cos"], s=4, alpha=0.25, color=BLUE)
    ax3.set_xlabel("n matched pairs (clipped at 80)")
    ax3.set_ylabel("Best cosine")
    ax3.set_title("(c) Matched pairs vs cosine similarity")

    # 4d – peak count by cos bin (PCDL)
    ax4 = fig.add_subplot(gs[2, 0])
    data_by_bin = [df[df["cos_bin"] == lbl]["pcdl_n_peaks"].values for lbl in COS_LABELS]
    bp = ax4.boxplot(
        data_by_bin, patch_artist=True, widths=0.55, medianprops=dict(color="black", lw=1.5)
    )
    for patch, lbl in zip(bp["boxes"], COS_LABELS):
        color = GREEN if lbl == "≥0.95" else (ORANGE if lbl in ("0.90–0.95", "0.85–0.90") else RED)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax4.set_xticklabels(COS_LABELS, rotation=30, fontsize=7)
    ax4.set_ylabel("PCDL n_peaks")
    ax4.set_title("(d) PCDL peak count by cosine bin")

    # 4e – peak count by cos bin (NIST)
    ax5 = fig.add_subplot(gs[2, 1])
    data_by_bin_n = [df[df["cos_bin"] == lbl]["nist_n_peaks"].values for lbl in COS_LABELS]
    bp2 = ax5.boxplot(
        data_by_bin_n, patch_artist=True, widths=0.55, medianprops=dict(color="black", lw=1.5)
    )
    for patch, lbl in zip(bp2["boxes"], COS_LABELS):
        color = GREEN if lbl == "≥0.95" else (ORANGE if lbl in ("0.90–0.95", "0.85–0.90") else RED)
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax5.set_xticklabels(COS_LABELS, rotation=30, fontsize=7)
    ax5.set_ylabel("NIST n_peaks")
    ax5.set_title("(e) NIST peak count by cosine bin")

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 – Bad matches deep-dive
# ═══════════════════════════════════════════════════════════════════════════════
def make_page_bad_matches(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Bad Match Deep-Dive (cos < 0.70)", fontsize=14, fontweight="bold", y=0.97)
    gs = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.5, wspace=0.4, left=0.1, right=0.93, top=0.92, bottom=0.06
    )

    bad = df[df["best_cos"] < 0.7].copy()

    # 5a – bad match count by CE
    ax1 = fig.add_subplot(gs[0, 0])
    bad_by_ce = bad.groupby("ace").size().reindex(main_ces := [10.0, 20.0, 40.0], fill_value=0)
    total_by_ce = df.groupby("ace").size().reindex(main_ces, fill_value=1)
    pct_bad = 100 * bad_by_ce / total_by_ce
    bars = ax1.bar(
        [f"{ce:.0f}eV" for ce in main_ces],
        pct_bad.values,
        color=[CE_COLORS[ce] for ce in main_ces],
        edgecolor="none",
    )
    for bar, pct in zip(bars, pct_bad.values):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax1.set_ylabel("% bad matches (cos < 0.70)")
    ax1.set_title("(a) Bad match rate by CE")
    ax1.set_ylim(0, pct_bad.max() * 1.25)

    # 5b – bad match count by adduct
    ax2 = fig.add_subplot(gs[0, 1])
    adducts = ["[M+H]+", "[M-H]-"]
    bad_by_ad = bad.groupby("prec_type").size().reindex(adducts, fill_value=0)
    total_by_ad = df.groupby("prec_type").size().reindex(adducts, fill_value=1)
    pct_bad_ad = 100 * bad_by_ad / total_by_ad
    bars2 = ax2.bar(adducts, pct_bad_ad.values, color=[BLUE, ORANGE], edgecolor="none")
    for bar, pct in zip(bars2, pct_bad_ad.values):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f"{pct:.1f}%",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax2.set_ylabel("% bad matches (cos < 0.70)")
    ax2.set_title("(b) Bad match rate by adduct")
    ax2.set_ylim(0, pct_bad_ad.max() * 1.25)

    # 5c – bad: residual vs CE heatmap
    ax3 = fig.add_subplot(gs[1, 0])
    res_order = list(RESIDUAL_LABELS.keys())
    heat = np.zeros((len(main_ces), len(res_order)))
    for i, ce in enumerate(main_ces):
        sub = bad[bad["ace"] == ce]
        total = max(len(sub), 1)
        for j, res in enumerate(res_order):
            heat[i, j] = sub[sub["dominant_residual"] == res].shape[0] / total
    im = ax3.imshow(heat, cmap="YlOrRd", aspect="auto", vmin=0, vmax=heat.max())
    ax3.set_xticks(range(len(res_order)))
    ax3.set_xticklabels(
        [RESIDUAL_LABELS[r] for r in res_order], rotation=25, fontsize=6.5, ha="right"
    )
    ax3.set_yticks(range(len(main_ces)))
    ax3.set_yticklabels([f"{ce:.0f} eV" for ce in main_ces])
    for i in range(len(main_ces)):
        for j in range(len(res_order)):
            ax3.text(
                j,
                i,
                f"{100 * heat[i, j]:.0f}%",
                ha="center",
                va="center",
                fontsize=8,
                fontweight="bold",
                color="white" if heat[i, j] > 0.5 * heat.max() else "black",
            )
    plt.colorbar(im, ax=ax3, label="Fraction of bad matches")
    ax3.set_title("(c) Residual type by CE\n(among bad matches, cos<0.70)")

    # 5d – pcdl_unmatched histogram for bad matches
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.hist(
        bad["pcdl_unmatched_l2sq"],
        bins=40,
        color=RED,
        edgecolor="none",
        alpha=0.8,
        label="pcdl_unmatched",
    )
    ax4.hist(
        bad["nist_unmatched_noniso_l2sq"],
        bins=40,
        color=ORANGE,
        edgecolor="none",
        alpha=0.7,
        label="nist_unmatched_noniso",
    )
    ax4.set_xlabel("Unmatched L² energy")
    ax4.set_ylabel("Count")
    ax4.set_title("(d) Unmatched energy distribution\n(bad matches only)")
    ax4.legend(fontsize=7.5)

    # 5e – worst matches table
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")
    w = (
        worst.sort_values("best_cos")
        .head(15)[
            [
                "pcdl_dset_spec_id",
                "inchikey_s",
                "formula",
                "prec_type",
                "ace",
                "best_cos",
                "dominant_residual",
            ]
        ]
        .copy()
    )
    w["ace"] = w["ace"].astype(int).astype(str) + " eV"
    w["best_cos"] = w["best_cos"].map("{:.4f}".format)
    w["dominant_residual"] = w["dominant_residual"].map(
        lambda x: RESIDUAL_LABELS.get(x, x).replace(" ", "\n")
    )
    w["pcdl_dset_spec_id"] = w["pcdl_dset_spec_id"].str[-20:]
    w.columns = ["PCDL ID (tail)", "InChIKey", "Formula", "Adduct", "CE", "Cos", "Dom. Residual"]
    tbl = ax5.table(
        cellText=w.values,
        colLabels=w.columns,
        loc="center",
        cellLoc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(6.5)
    tbl.scale(1, 1.35)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
        elif row % 2 == 0:
            cell.set_facecolor("#f5f5f5")
    ax5.set_title("(e) 15 worst-matching spectrum pairs", fontsize=9, pad=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 – Isotope-like peak analysis
# ═══════════════════════════════════════════════════════════════════════════════
def make_page_isotope(pdf: PdfPages) -> None:
    fig = plt.figure(figsize=(8.5, 11))
    fig.suptitle("Isotope-Like Peak Analysis", fontsize=14, fontweight="bold", y=0.97)
    gs = gridspec.GridSpec(
        3, 2, figure=fig, hspace=0.5, wspace=0.4, left=0.1, right=0.93, top=0.92, bottom=0.06
    )

    df2 = df.copy()
    df2["pcdl_iso_frac"] = df2["pcdl_unmatched_isotope_like_l2sq"] / (
        df2["pcdl_unmatched_l2sq"].clip(lower=1e-9)
    )
    df2["nist_iso_frac"] = df2["nist_unmatched_isotope_like_l2sq"] / (
        df2["nist_unmatched_l2sq"].clip(lower=1e-9)
    )
    # only rows with some unmatched energy
    pcdl_has = df2[df2["pcdl_unmatched_l2sq"] > 1e-6]
    nist_has = df2[df2["nist_unmatched_l2sq"] > 1e-6]

    # 6a – PCDL: histogram of iso-like fraction among unmatched
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(pcdl_has["pcdl_iso_frac"].clip(0, 1), bins=40, color=BLUE, edgecolor="none", alpha=0.8)
    ax1.set_xlabel("Isotope-like fraction of unmatched PCDL energy")
    ax1.set_ylabel("Count")
    ax1.set_title(f"(a) PCDL unmatched: iso fraction\n(n={len(pcdl_has):,} with unmatched > 0)")
    ax1.axvline(
        pcdl_has["pcdl_iso_frac"].median(),
        color=RED,
        ls="--",
        lw=1.5,
        label=f"Median={pcdl_has['pcdl_iso_frac'].median():.3f}",
    )
    ax1.legend(fontsize=7)

    # 6b – NIST: histogram of iso-like fraction among unmatched
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(
        nist_has["nist_iso_frac"].clip(0, 1), bins=40, color=ORANGE, edgecolor="none", alpha=0.8
    )
    ax2.set_xlabel("Isotope-like fraction of unmatched NIST energy")
    ax2.set_ylabel("Count")
    ax2.set_title(f"(b) NIST unmatched: iso fraction\n(n={len(nist_has):,} with unmatched > 0)")
    ax2.axvline(
        nist_has["nist_iso_frac"].median(),
        color=RED,
        ls="--",
        lw=1.5,
        label=f"Median={nist_has['nist_iso_frac'].median():.3f}",
    )
    ax2.legend(fontsize=7)

    # 6c – scatter: iso-like vs cos, NIST side
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.scatter(
        nist_has["nist_unmatched_isotope_like_l2sq"].clip(upper=0.5),
        nist_has["best_cos"],
        s=5,
        alpha=0.3,
        color=ORANGE,
    )
    ax3.set_xlabel("NIST isotope-like unmatched L² (clipped)")
    ax3.set_ylabel("Best cosine")
    ax3.set_title("(c) NIST iso-like unmatched energy vs cosine")

    # 6d – count of spectra where iso-like dominates NIST unmatched
    ax4 = fig.add_subplot(gs[1, 1])
    iso_dominant = df["dominant_residual"] == "nist_unmatched_isotope_like_l2sq"
    by_ce = {ce: df[(df["ace"] == ce) & iso_dominant].shape[0] for ce in [10.0, 20.0, 40.0]}
    bars = ax4.bar(
        [f"{k:.0f}eV" for k in by_ce],
        list(by_ce.values()),
        color=[CE_COLORS[k] for k in by_ce],
        edgecolor="none",
    )
    add_value_labels(ax4, bars)
    ax4.set_ylabel("Count")
    ax4.set_title("(d) NIST iso-like dominant residual by CE")

    # 6e – summary table of iso stats
    ax5 = fig.add_subplot(gs[2, :])
    ax5.axis("off")

    rows = []
    for ce in [10.0, 20.0, 40.0]:
        sub = df[df["ace"] == ce]
        n_iso_nist = (sub["dominant_residual"] == "nist_unmatched_isotope_like_l2sq").sum()
        n_iso_pcdl = (sub["pcdl_unmatched_isotope_like_l2sq"] > 1e-4).sum()
        median_nist_iso = sub["nist_unmatched_isotope_like_l2sq"].median()
        rows.append(
            [
                f"{ce:.0f} eV",
                str(len(sub)),
                str(n_iso_nist),
                f"{100 * n_iso_nist / len(sub):.1f}%",
                str(n_iso_pcdl),
                f"{100 * n_iso_pcdl / len(sub):.1f}%",
                f"{median_nist_iso:.5f}",
            ]
        )

    cols = [
        "CE",
        "N pairs",
        "NIST iso dominant",
        "%",
        "PCDL has iso-like unmatched",
        "%",
        "Median NIST iso L²",
    ]
    tbl = ax5.table(cellText=rows, colLabels=cols, loc="center", cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.8)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#2c3e50")
            cell.set_text_props(color="white", fontweight="bold")
    ax5.set_title("(e) Isotope-like peak summary by CE", fontsize=9, pad=8)

    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# PAGES 7+ – Mirror plots (2 per page, large image + metadata text)
# ═══════════════════════════════════════════════════════════════════════════════
def make_mirror_plot_pages(pdf: PdfPages) -> None:
    import re

    from matplotlib.image import imread

    mirror_dir = DATA_DIR / "mirror_plots"
    if not mirror_dir.exists():
        print("    [mirror_plots directory not found, skipping mirror plot pages]")
        return
    all_pngs = sorted(mirror_dir.glob("*.png"))
    if not all_pngs:
        return
    pngs = [
        p
        for p in all_pngs
        if not (
            (m2 := re.match(r"\d+_cos[\d.]+_pcdl(\d+)_nist20\d+", p.stem))
            and int(m2.group(1)) in mona_pcdl_ids
        )
    ]
    if not pngs:
        return

    # build a lookup: pcdl_spec_id → row from main CSV (non-MoNA)
    lookup = df.set_index("pcdl_spec_id")

    per_page = 2
    n_pages = (len(pngs) + per_page - 1) // per_page

    for page_idx in range(n_pages):
        batch = pngs[page_idx * per_page : (page_idx + 1) * per_page]
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(
            f"NIST vs PCDL – Bad Matches  "
            f"[{page_idx * per_page + 1}–{min((page_idx + 1) * per_page, len(pngs))} of {len(pngs)}]",
            fontsize=11,
            fontweight="bold",
            y=0.99,
        )

        # 2 rows; each row = image (left 80%) + metadata text (right 20% hidden via text box)
        gs = gridspec.GridSpec(
            len(batch),
            1,
            figure=fig,
            hspace=0.12,
            left=0.01,
            right=0.99,
            top=0.96,
            bottom=0.02,
        )

        for i, png_path in enumerate(batch):
            stem = png_path.stem
            m = re.match(r"(\d+)_cos([\d.]+)_pcdl(\d+)_nist20(\d+)", stem)

            # ── image axis (left 78%) ────────────────────────────────────────
            img_ax = fig.add_subplot(gs[i])
            img = imread(str(png_path))
            img_ax.imshow(img)
            img_ax.axis("off")

            # ── overlay metadata text block (bottom-left of the image area) ──
            if m:
                idx_str, cos_str, pcdl_id, nist_id = m.groups()
                pcdl_id_int = int(pcdl_id)

                # pull row from CSV
                meta_lines = [f"#{idx_str}  NIST20 spec {nist_id}  ↔  PCDL spec {pcdl_id}"]
                if pcdl_id_int in lookup.index:
                    row = lookup.loc[pcdl_id_int]
                    # some pcdl_spec_ids may appear multiple times (different CE); take first
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    meta_lines += [
                        f"InChIKey:  {row['inchikey_s']}",
                        f"Formula:   {row['formula']}",
                        f"Adduct:    {row['prec_type']}   CE: {row['ace']:.0f} eV   Inst: {row['inst_type']}",
                        f"PCDL src:  {row['pcdl_dset_spec_id']}",
                        f"NIST src:  {row['nist_dset_spec_id']}",
                        f"cos:       {row['best_cos']:.4f}   n_pairs: {int(row['n_pairs'])}   "
                        f"PCDL peaks: {int(row['pcdl_n_peaks'])}   NIST peaks: {int(row['nist_n_peaks'])}",
                        f"Dom. residual:  {RESIDUAL_LABELS.get(row['dominant_residual'], row['dominant_residual'])}",
                    ]
                else:
                    meta_lines.append(f"cos = {cos_str}  (no CSV row found)")

                meta_str = "\n".join(meta_lines)
                img_ax.text(
                    0.01,
                    0.01,
                    meta_str,
                    transform=img_ax.transAxes,
                    fontsize=6.5,
                    va="bottom",
                    ha="left",
                    fontfamily="monospace",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.82,
                        "edgecolor": "#cccccc",
                        "boxstyle": "round,pad=0.3",
                    },
                )
            else:
                img_ax.set_title(stem[:60], fontsize=7, pad=2)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"    mirror plots page {page_idx + 1}/{n_pages}")


# ═══════════════════════════════════════════════════════════════════════════════
# PCDL vs MoNA mirror plots section
# ═══════════════════════════════════════════════════════════════════════════════
def make_pcdl_vs_mona_pages(pdf: PdfPages) -> None:
    import re

    from matplotlib.image import imread

    mirror_dir = DATA_DIR / "mirror_plots"
    if not mirror_dir.exists():
        print("    [mirror_plots directory not found, skipping NIST vs MoNA mirror plot pages]")
        return
    all_pngs = sorted(mirror_dir.glob("*.png"))
    if not all_pngs:
        return
    pngs = [
        p
        for p in all_pngs
        if (
            (m2 := re.match(r"\d+_cos[\d.]+_pcdl(\d+)_nist20\d+", p.stem))
            and int(m2.group(1)) in mona_pcdl_ids
        )
    ]
    if not pngs:
        return

    # lookup using unfiltered df_all for MoNA rows
    lookup = df_all[
        df_all["nist_source"].eq(MONA_DSET) & df_all["pcdl_source"].eq(PCDL_DSET)
    ].set_index("pcdl_spec_id")

    per_page = 2
    n_pages = (len(pngs) + per_page - 1) // per_page

    for page_idx in range(n_pages):
        batch = pngs[page_idx * per_page : (page_idx + 1) * per_page]
        fig = plt.figure(figsize=(8.5, 11))
        fig.suptitle(
            f"NIST vs MoNA – Bad Matches  "
            f"[{page_idx * per_page + 1}–{min((page_idx + 1) * per_page, len(pngs))} of {len(pngs)}]",
            fontsize=11,
            fontweight="bold",
            y=0.99,
        )
        gs = gridspec.GridSpec(
            len(batch),
            1,
            figure=fig,
            hspace=0.12,
            left=0.01,
            right=0.99,
            top=0.96,
            bottom=0.02,
        )
        for i, png_path in enumerate(batch):
            stem = png_path.stem
            m = re.match(r"(\d+)_cos([\d.]+)_pcdl(\d+)_nist20(\d+)", stem)
            img_ax = fig.add_subplot(gs[i])
            img = imread(str(png_path))
            img_ax.imshow(img)
            img_ax.axis("off")
            if m:
                idx_str, cos_str, pcdl_id, nist_id = m.groups()
                pcdl_id_int = int(pcdl_id)
                meta_lines = [f"#{idx_str}  NIST20 spec {nist_id}  ↔  MoNA spec {pcdl_id}"]
                if pcdl_id_int in lookup.index:
                    row = lookup.loc[pcdl_id_int]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    meta_lines += [
                        f"InChIKey:  {row['inchikey_s']}",
                        f"Formula:   {row['formula']}",
                        f"Adduct:    {row['prec_type']}   CE: {row['ace']:.0f} eV   Inst: {row['inst_type']}",
                        f"PCDL src:  {row['pcdl_dset_spec_id']}",
                        f"NIST src:  {row['nist_dset_spec_id']}",
                        f"cos:       {row['best_cos']:.4f}   n_pairs: {int(row['n_pairs'])}   "
                        f"PCDL peaks: {int(row['pcdl_n_peaks'])}   NIST peaks: {int(row['nist_n_peaks'])}",
                        f"Dom. residual:  {RESIDUAL_LABELS.get(row['dominant_residual'], row['dominant_residual'])}",
                    ]
                else:
                    meta_lines.append(f"cos = {cos_str}  (no CSV row found)")
                img_ax.text(
                    0.01,
                    0.01,
                    "\n".join(meta_lines),
                    transform=img_ax.transAxes,
                    fontsize=6.5,
                    va="bottom",
                    ha="left",
                    fontfamily="monospace",
                    bbox={
                        "facecolor": "white",
                        "alpha": 0.82,
                        "edgecolor": "#cccccc",
                        "boxstyle": "round,pad=0.3",
                    },
                )
            else:
                img_ax.set_title(stem[:60], fontsize=7, pad=2)

        pdf.savefig(fig, bbox_inches="tight")
        plt.close(fig)
        print(f"    NIST vs MoNA page {page_idx + 1}/{n_pages}")


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main() -> None:
    print(f"Generating report → {OUT_PDF}")
    with PdfPages(OUT_PDF) as pdf:
        meta = pdf.infodict()
        meta["Title"] = "PCDL vs NIST20 Spectral Overlap Report"
        meta["Author"] = "frag-gnn project"
        meta["Subject"] = "Spectral library quality analysis"

        make_cover(pdf)
        print("  [1/7] Cover page")
        make_page_cos_overview(pdf)
        print("  [2/7] Cosine overview")
        make_page_ce_adduct(pdf)
        print("  [3/7] CE & adduct breakdown")
        make_page_residuals(pdf)
        print("  [4/7] Residual analysis")
        make_page_peaks(pdf)
        print("  [5/6] Peak count analysis")
        make_page_bad_matches(pdf)
        print("  [6/6] Bad match deep-dive")
        make_mirror_plot_pages(pdf)
        print("  [7/8] Mirror plots (NIST vs PCDL)")
        make_pcdl_vs_mona_pages(pdf)
        print("  [8/8] Mirror plots (NIST vs MoNA)")

    print(f"\nDone. PDF saved to:\n  {OUT_PDF}")


if __name__ == "__main__":
    main()
