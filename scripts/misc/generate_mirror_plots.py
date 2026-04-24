"""Generate placeholder mirror plots for PCDL vs NIST20 bad matches."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

DATA_DIR = Path("/home/feiw/gitRepos/frag-gnn/figs/pcdl_nist20_overlap_mztol0015_with_mol")
CSV_FILE = DATA_DIR / "pcdl_vs_nist20_best_matches.csv"
OUTPUT_DIR = DATA_DIR / "mirror_plots"

# Create output directory
OUTPUT_DIR.mkdir(exist_ok=True)

# Load data and get worst matches
df = pd.read_csv(CSV_FILE)
worst = df.nsmallest(50, "best_cos")

print(f"Generating {len(worst)} placeholder mirror plots → {OUTPUT_DIR}/")

for idx, (_, row) in enumerate(worst.iterrows(), 1):
    pcdl_id = int(row["pcdl_spec_id"])
    nist_id = int(row["nist_spec_id"])
    cos = row["best_cos"]
    formula = row["formula"]
    prec_type = row["prec_type"]

    # Create a simple placeholder figure (two spectra, empty for now)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"PCDL spec {pcdl_id} vs NIST20 spec {nist_id}\n"
        f"{formula}  |  {prec_type}  |  cos={cos:.4f}",
        fontsize=10, fontweight="bold"
    )

    # Top plot: PCDL spectrum (placeholder)
    ax1.bar([0], [1], width=0.5, color="blue", alpha=0.5, label="PCDL")
    ax1.set_ylabel("PCDL Intensity")
    ax1.set_ylim(0, 1.2)
    ax1.legend(loc="upper right")
    ax1.text(0.5, 0.5, "PCDL spectrum\n(placeholder)",
             ha="center", va="center", fontsize=12, color="gray", alpha=0.5,
             transform=ax1.transAxes)

    # Bottom plot: NIST spectrum (placeholder)
    ax2.bar([0], [1], width=0.5, color="red", alpha=0.5, label="NIST20")
    ax2.set_xlabel("m/z")
    ax2.set_ylabel("NIST Intensity")
    ax2.set_ylim(0, 1.2)
    ax2.legend(loc="upper right")
    ax2.text(0.5, 0.5, "NIST20 spectrum\n(placeholder)",
             ha="center", va="center", fontsize=12, color="gray", alpha=0.5,
             transform=ax2.transAxes)

    plt.tight_layout()

    # Save with naming convention: NNN_cosX.XXX_pcdlNNNN_nist20NNNNN.png
    filename = f"{idx:03d}_cos{cos:.3f}_pcdl{pcdl_id}_nist20{nist_id}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close(fig)

    if idx % 10 == 0:
        print(f"  Generated {idx}/{len(worst)} plots...")

print(f"Done. {len(worst)} mirror plots created in {OUTPUT_DIR}/")
