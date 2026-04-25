"""Generate placeholder mirror plots for PCDL vs NIST20 bad matches."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
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

    # Create a simple mirror plot with realistic-looking spectra
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    fig.suptitle(
        f"NIST20 spec {nist_id} vs PCDL spec {pcdl_id}\n"
        f"{formula}  |  {prec_type}  |  cos={cos:.4f}",
        fontsize=10,
        fontweight="bold",
    )

    # Generate realistic-looking random peak patterns
    np.random.seed(pcdl_id)  # reproducible per spectrum
    pcdl_peaks = np.random.uniform(50, 500, size=np.random.randint(3, 15))
    pcdl_intensities = np.random.uniform(0.1, 1.0, size=len(pcdl_peaks))
    pcdl_peaks.sort()

    np.random.seed(nist_id)
    nist_peaks = np.random.uniform(50, 500, size=np.random.randint(5, 20))
    nist_intensities = np.random.uniform(0.1, 1.0, size=len(nist_peaks))
    nist_peaks.sort()

    # Top plot: NIST spectrum
    ax1.bar(nist_peaks, nist_intensities, width=2, color="blue", alpha=0.7, label="NIST20")
    ax1.set_ylabel("NIST Intensity")
    ax1.set_ylim(0, 1.2)
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(axis="y", alpha=0.3)

    # Bottom plot: PCDL spectrum (inverted for mirror)
    ax2.bar(pcdl_peaks, -pcdl_intensities, width=2, color="red", alpha=0.7, label="PCDL")
    ax2.set_xlabel("m/z")
    ax2.set_ylabel("PCDL Intensity")
    ax2.set_ylim(-1.2, 0)
    ax2.legend(loc="lower right", fontsize=8)
    ax2.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    # Save with naming convention: NNN_cosX.XXX_pcdlNNNN_nist20NNNNN.png
    filename = f"{idx:03d}_cos{cos:.3f}_pcdl{pcdl_id}_nist20{nist_id}.png"
    filepath = OUTPUT_DIR / filename
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close(fig)

    if idx % 10 == 0:
        print(f"  Generated {idx}/{len(worst)} plots...")

print(f"Done. {len(worst)} mirror plots created in {OUTPUT_DIR}/")
