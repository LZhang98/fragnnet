"""Probe how isotope peaks appear in Enveda-180 timsTOF MS/MS spectra.

Four measurements on the first ``--n`` records of each export:

1. Peak-pair spacing near 1 Da (raw export): whether 13C (1.003355 Da) and a hydrogen shift
   (1.007825 Da) form separate modes at this instrument's mass accuracy, by m/z range.
2. The release's MS1 isotope pattern (raw export) against the theoretical pattern of the ion
   formula, and where its isotope m/z sit for halogenated molecules.
3. Precursor isotopes surviving in MS2 (raw export), as a co-isolation fraction
   f_k = (MS2 M+k / M+0) / theoretical M+k / M+0, by polarity; for Cl/Br molecules, whether
   M+2 transmission is a per-molecule property.
4. Share of training-target intensity (filtered export) on 13C satellites, H-spaced peaks and
   halogen M+2 partners, i.e. what a monoisotopic prediction cannot reach.

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/isotopes/debug_enveda_isotope_probe.py
"""

from __future__ import annotations

import argparse
import gzip
import itertools
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from pyopenms import CoarseIsotopePatternGenerator, EmpiricalFormula

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

C13_SPACING = 1.003355
H_SPACING = 1.007825
BR81_SPACING = 1.997953
CL37_SPACING = 1.997050
SPACING_BIN_MDA = 0.25
MZ_BINS = ("<300", "300-600", ">=600")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    raw_dp = Path("data/raw/enveda-180")
    parser.add_argument("--raw-fp", type=Path, default=raw_dp / "enveda-180.jsonl.gz")
    parser.add_argument("--filtered-fp", type=Path, default=raw_dp / "enveda-180-filtered.jsonl.gz")
    parser.add_argument("--n", type=int, default=60000, help="Records read from each export")
    parser.add_argument("--ppm", type=float, default=5.0, help="Peak match tolerance")
    parser.add_argument("--min-tol", type=float, default=0.002, help="Da floor on the tolerance")
    return parser.parse_args()


def read_records(fp: Path, n: int):
    """Yield (record, sorted m/z, matching intensities) for the first ``n`` records."""
    with gzip.open(fp, "rt") as f:
        for line in itertools.islice(f, n):
            record = json.loads(line)
            mzs = np.array(record["mzs"], float)
            ints = np.array(record["intensities"], float)
            order = np.argsort(mzs)
            yield record, mzs[order], ints[order]


def find_peak(mzs: np.ndarray, target: float, tol: float) -> int | None:
    """Index of the peak nearest ``target`` within ``tol``, or None."""
    i = np.searchsorted(mzs, target)
    hits = [j for j in (i - 1, i) if 0 <= j < len(mzs) and abs(mzs[j] - target) <= tol]
    return min(hits, key=lambda j: abs(mzs[j] - target)) if hits else None


def halogen_class(formula: str) -> str:
    has_br, has_cl = "Br" in formula, "Cl" in formula
    return "both" if has_br and has_cl else "Br" if has_br else "Cl" if has_cl else "none"


def main() -> None:
    args = parse_args()
    generator = CoarseIsotopePatternGenerator(3)
    num_bins = int(15 / SPACING_BIN_MDA)
    spacing = defaultdict(lambda: np.zeros(num_bins))
    prec_rows = []

    # 1-3. One pass over the raw export.
    for record, mzs, ints in read_records(args.raw_fp, args.n):
        base = ints.max()
        for i in range(len(mzs)):
            if ints[i] < 0.005 * base:
                continue
            lo, hi = np.searchsorted(mzs, [mzs[i] + 1.000, mzs[i] + 1.015])
            for j in range(lo, hi):
                if ints[j] >= 0.005 * base:
                    mz_bin = MZ_BINS[int(mzs[i] >= 300) + int(mzs[i] >= 600)]
                    spacing[mz_bin][int((mzs[j] - mzs[i] - 1.000) * 1e3 / SPACING_BIN_MDA)] += 1

        if record["adduct"] not in ("[M+H]+", "[M-H]-"):
            continue
        hal = halogen_class(record["formula"])
        pattern = generator.run(EmpiricalFormula(record["adduct_formula"])).getContainer()
        theo = [p.getIntensity() for p in pattern]
        ms1 = [float(x) for x in record["isotope_intensities"]]
        iso_mzs = [float(x) for x in record["isotope_mzs"]]
        row = {
            "molecule": record["title"].rsplit("_", 1)[0],
            "pol": "pos" if record["adduct"].endswith("+") else "neg",
            "hal": hal,
            "ms1_r1": ms1[1] / ms1[0] / (theo[1] / theo[0]) if len(ms1) > 1 else np.nan,
            "ms1_r2": ms1[2] / ms1[0] / (theo[2] / theo[0]) if len(ms1) > 2 else np.nan,
            "ms1_d2": iso_mzs[2] - iso_mzs[0] if len(iso_mzs) > 2 else np.nan,
            "f1": np.nan,
            "f2": np.nan,
        }
        prec = float(record["pepmass"])
        tol = max(prec * args.ppm * 1e-6, args.min_tol)
        p0 = find_peak(mzs, prec, tol)
        if p0 is not None and ints[p0] >= 0.05 * base:
            step2 = {"Br": BR81_SPACING, "Cl": CL37_SPACING}.get(hal, 2 * C13_SPACING)
            p1 = find_peak(mzs, mzs[p0] + C13_SPACING, tol)
            p2 = find_peak(mzs, mzs[p0] + step2, tol)
            row["f1"] = (ints[p1] / ints[p0] if p1 is not None else 0.0) / (theo[1] / theo[0])
            row["f2"] = (ints[p2] / ints[p0] if p2 is not None else 0.0) / (theo[2] / theo[0])
        prec_rows.append(row)

    log.info("1. Peak-pair spacing, counts per %.2f mDa bin (bin centre):", SPACING_BIN_MDA)
    centres = 1000 + (np.arange(num_bins) + 0.5) * SPACING_BIN_MDA
    for mz_bin in MZ_BINS:
        hist = spacing[mz_bin]
        c13 = (centres > 1001.5) & (centres < 1005.3)
        hsh = (centres > 1005.8) & (centres < 1009.5)
        trough = (centres > 1004.8) & (centres < 1006.0)
        log.info(
            "  m/z %-7s pairs %7d | 13C mode %.2f mDa, H mode %.2f mDa, trough/13C-peak %.2f",
            mz_bin,
            hist.sum(),
            centres[c13][hist[c13].argmax()],
            centres[hsh][hist[hsh].argmax()],
            hist[trough].min() / max(hist[c13].max(), 1),
        )

    prec_df = pd.DataFrame(prec_rows)
    log.info("2. Release MS1 isotope pattern / theory (median):")
    for hal, grp in prec_df.groupby("hal"):
        log.info(
            "  %-4s n=%6d  M+1 %.3f  M+2 %.3f  | listed M+2 offset %.5f Da",
            hal,
            len(grp),
            grp["ms1_r1"].median(),
            grp["ms1_r2"].median(),
            grp["ms1_d2"].median(),
        )

    seen = prec_df.dropna(subset=["f1"])
    log.info("3. Precursor isotopes in MS2 (precursor >= 5%% base, n=%d):", len(seen))
    for pol, grp in seen.groupby("pol"):
        log.info("  %s: n=%6d  median f1 %.3f", pol, len(grp), grp["f1"].median())
    halo = seen[seen["hal"].isin(["Br", "Cl"])].assign(full=lambda d: d["f2"] > 0.5)
    for pol, grp in halo.groupby("pol"):
        log.info("  Cl/Br %s: n=%5d  share with f2 > 0.5: %.3f", pol, len(grp), grp["full"].mean())
    per_mol = halo[halo["pol"] == "pos"].groupby("molecule")["full"].agg(["mean", "size"])
    per_mol = per_mol[per_mol["size"] >= 2]
    log.info(
        "  Cl/Br pos molecules with >=2 spectra: %d | all f2>0.5 %d | none %d | mixed %d",
        len(per_mol),
        (per_mol["mean"] == 1).sum(),
        (per_mol["mean"] == 0).sum(),
        ((per_mol["mean"] > 0) & (per_mol["mean"] < 1)).sum(),
    )

    # 4. Training targets: intensity on satellites of a more intense, less massive peak.
    share_rows = []
    for record, mzs, ints in read_records(args.filtered_fp, args.n):
        total = ints.sum()
        hal = halogen_class(record["formula"])
        hal_step = {"Br": BR81_SPACING, "Cl": CL37_SPACING}.get(hal)
        c13 = hsh = hal_share = 0.0
        for j in range(len(mzs)):
            lo, hi = np.searchsorted(mzs, [mzs[j] - 1.0105, mzs[j] - 1.0015])
            gaps = [(mzs[j] - mzs[i]) * 1e3 for i in range(lo, hi) if ints[i] > ints[j]]
            if any(1002.0 <= gap <= 1005.3 for gap in gaps):
                c13 += ints[j]
            elif any(1006.0 <= gap <= 1009.5 for gap in gaps):
                hsh += ints[j]
            tol = max(mzs[j] * args.ppm * 1e-6, args.min_tol)
            if hal_step is not None and find_peak(mzs, mzs[j] - hal_step, tol) is not None:
                hal_share += ints[j]
        share_rows.append(
            {
                "pol": "pos" if record["charge"].endswith("+") else "neg",
                "halo": hal_step is not None,
                "c13": c13 / total,
                "hsh": hsh / total,
                "hal": hal_share / total,
            }
        )
    share_df = pd.DataFrame(share_rows)
    log.info("4. Training-target intensity share (filtered export, n=%d):", len(share_df))
    for pol, grp in share_df.groupby("pol"):
        log.info(
            "  %s: 13C satellites mean %.4f (median %.4f, p90 %.4f) | H-spaced %.4f",
            pol,
            grp["c13"].mean(),
            grp["c13"].median(),
            grp["c13"].quantile(0.9),
            grp["hsh"].mean(),
        )
    halo_share = share_df[share_df["halo"]]
    log.info(
        "  Cl/Br molecules: halogen M+2 partner share mean %.4f (p90 %.4f), n=%d",
        halo_share["hal"].mean(),
        halo_share["hal"].quantile(0.9),
        len(halo_share),
    )
    log.info(
        "  spectra with >= 5%% of intensity on 13C satellites: %.3f",
        (share_df["c13"] >= 0.05).mean(),
    )


if __name__ == "__main__":
    main()
