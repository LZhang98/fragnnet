"""Probe how isotope peaks appear in MassSpecGym 1.5 MS/MS spectra.

The MSGYM counterpart of ``debug_enveda_isotope_probe.py``. MSGYM is positive mode only and has
no MS1 isotope fields, so Enveda test 2 has no analogue; everything is split by instrument.

1. Peak-pair spacing near 1 Da: whether 13C (1.003355 Da) and a hydrogen shift (1.007825 Da)
   form separate modes, by instrument and m/z range.
2. Precursor isotopes surviving in MS2, as a co-isolation fraction
   f_k = (MS2 M+k / M+0) / theoretical M+k / M+0, by instrument and adduct; for Cl/Br molecules,
   whether M+2 transmission is a per-molecule property.
3. Share of target intensity on 13C satellites, H-spaced peaks and halogen M+2 partners, i.e.
   what a monoisotopic prediction cannot reach.

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/debug_msgym_isotope_probe.py
"""

from __future__ import annotations

import argparse
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
    parser.add_argument("--tsv-fp", type=Path, default=Path("data/raw/msgym_1_5/MassSpecGym1.5.tsv"))
    parser.add_argument("--num", type=int, default=None, help="Records read (default: all)")
    parser.add_argument("--ppm", type=float, default=10.0, help="Peak match tolerance")
    parser.add_argument("--min-tol", type=float, default=0.003, help="Da floor on the tolerance")
    return parser.parse_args()


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
    df = pd.read_csv(args.tsv_fp, sep="\t", nrows=args.num)
    df["instrument_type"] = df["instrument_type"].fillna("unknown")
    log.info("Read %d spectra", len(df))

    generator = CoarseIsotopePatternGenerator(3)
    num_bins = int(15 / SPACING_BIN_MDA)
    spacing = defaultdict(lambda: np.zeros(num_bins))
    rows = []
    for rec in df.itertuples(index=False):
        mzs = np.array(rec.mzs.split(","), float)
        ints = np.array(rec.intensities.split(","), float)
        order = np.argsort(mzs)
        mzs, ints = mzs[order], ints[order]
        inst = rec.instrument_type
        base = ints.max()
        total = ints.sum()

        # 1. Spacing histogram.
        for i in range(len(mzs)):
            if ints[i] < 0.005 * base:
                continue
            lo, hi = np.searchsorted(mzs, [mzs[i] + 1.000, mzs[i] + 1.015])
            for j in range(lo, hi):
                if ints[j] >= 0.005 * base:
                    mz_bin = MZ_BINS[int(mzs[i] >= 300) + int(mzs[i] >= 600)]
                    spacing[(inst, mz_bin)][int((mzs[j] - mzs[i] - 1.000) * 1e3 / SPACING_BIN_MDA)] += 1

        hal = halogen_class(rec.formula)
        hal_step = {"Br": BR81_SPACING, "Cl": CL37_SPACING}.get(hal)

        # 2. Co-isolation fraction at the precursor.
        row = {
            "molecule": rec.inchikey,
            "inst": inst,
            "adduct": rec.adduct,
            "hal": hal,
            "n_peaks": len(mzs),
            "f1": np.nan,
            "f2": np.nan,
        }
        theo = [p.getIntensity() for p in generator.run(EmpiricalFormula(rec.precursor_formula)).getContainer()]
        prec = float(rec.precursor_mz)
        tol = max(prec * args.ppm * 1e-6, args.min_tol)
        p0 = find_peak(mzs, prec, tol)
        row["prec_present"] = p0 is not None
        if p0 is not None and ints[p0] >= 0.05 * base:
            step2 = hal_step if hal_step is not None else 2 * C13_SPACING
            p1 = find_peak(mzs, mzs[p0] + C13_SPACING, tol)
            p2 = find_peak(mzs, mzs[p0] + step2, tol)
            row["f1"] = (ints[p1] / ints[p0] if p1 is not None else 0.0) / (theo[1] / theo[0])
            row["f2"] = (ints[p2] / ints[p0] if p2 is not None else 0.0) / (theo[2] / theo[0])

        # 3. Intensity on satellites of a more intense, less massive peak.
        c13 = hsh = hal_share = 0.0
        for j in range(len(mzs)):
            lo, hi = np.searchsorted(mzs, [mzs[j] - 1.0105, mzs[j] - 1.0015])
            gaps = [(mzs[j] - mzs[i]) * 1e3 for i in range(lo, hi) if ints[i] > ints[j]]
            if any(1002.0 <= gap <= 1005.3 for gap in gaps):
                c13 += ints[j]
            elif any(1006.0 <= gap <= 1009.5 for gap in gaps):
                hsh += ints[j]
            ptol = max(mzs[j] * args.ppm * 1e-6, args.min_tol)
            if hal_step is not None and find_peak(mzs, mzs[j] - hal_step, ptol) is not None:
                hal_share += ints[j]
        row.update(c13=c13 / total, hsh=hsh / total, hal_share=hal_share / total)
        rows.append(row)

    log.info("1. Peak-pair spacing, counts per %.2f mDa bin (bin centre):", SPACING_BIN_MDA)
    centres = 1000 + (np.arange(num_bins) + 0.5) * SPACING_BIN_MDA
    c13_win = (centres > 1001.5) & (centres < 1005.3)
    h_win = (centres > 1005.8) & (centres < 1009.5)
    trough = (centres > 1004.8) & (centres < 1006.0)
    for (inst, mz_bin), hist in sorted(spacing.items()):
        log.info(
            "  %-8s m/z %-7s pairs %7d | 13C mode %.2f mDa (n %d), H mode %.2f mDa (n %d), trough/13C-peak %.2f",
            inst,
            mz_bin,
            hist.sum(),
            centres[c13_win][hist[c13_win].argmax()],
            hist[c13_win].sum(),
            centres[h_win][hist[h_win].argmax()],
            hist[h_win].sum(),
            hist[trough].min() / max(hist[c13_win].max(), 1),
        )

    res = pd.DataFrame(rows)
    log.info("2. Precursor in MS2 and co-isolation fraction:")
    for (inst, adduct), grp in res.groupby(["inst", "adduct"]):
        seen = grp.dropna(subset=["f1"])
        log.info(
            "  %-8s %-8s n=%6d  precursor present %.3f  >=5%% base %d | median f1 %.3f  share f1>0 %.3f",
            inst,
            adduct,
            len(grp),
            grp["prec_present"].mean(),
            len(seen),
            seen["f1"].median() if len(seen) else np.nan,
            (seen["f1"] > 0).mean() if len(seen) else np.nan,
        )
    halo = res.dropna(subset=["f1"])
    halo = halo[halo["hal"].isin(["Br", "Cl"])].assign(full=lambda d: d["f2"] > 0.5)
    for inst, grp in halo.groupby("inst"):
        log.info("  Cl/Br %-8s n=%5d  share with f2 > 0.5: %.3f  median f2 %.3f", inst, len(grp), grp["full"].mean(), grp["f2"].median())
    per_mol = halo.groupby("molecule")["full"].agg(["mean", "size"])
    per_mol = per_mol[per_mol["size"] >= 2]
    log.info(
        "  Cl/Br molecules with >=2 spectra: %d | all f2>0.5 %d | none %d | mixed %d",
        len(per_mol),
        (per_mol["mean"] == 1).sum(),
        (per_mol["mean"] == 0).sum(),
        ((per_mol["mean"] > 0) & (per_mol["mean"] < 1)).sum(),
    )

    log.info("3. Target intensity share:")
    for inst, grp in res.groupby("inst"):
        log.info(
            "  %-8s n=%6d  13C satellites mean %.4f (median %.4f, p90 %.4f) | H-spaced %.4f | spectra >=5%% on 13C %.3f",
            inst,
            len(grp),
            grp["c13"].mean(),
            grp["c13"].median(),
            grp["c13"].quantile(0.9),
            grp["hsh"].mean(),
            (grp["c13"] >= 0.05).mean(),
        )
    halo_all = res[res["hal"].isin(["Br", "Cl"])]
    for inst, grp in halo_all.groupby("inst"):
        log.info(
            "  Cl/Br %-8s halogen M+2 partner share mean %.4f (p90 %.4f), n=%d",
            inst,
            grp["hal_share"].mean(),
            grp["hal_share"].quantile(0.9),
            len(grp),
        )


if __name__ == "__main__":
    main()
