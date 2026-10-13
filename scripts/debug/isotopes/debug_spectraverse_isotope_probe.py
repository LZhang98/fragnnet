"""Probe how isotope peaks appear in Spectraverse (SPV) MS/MS spectra.

The SPV counterpart of ``debug_msgym_isotope_probe.py``. SPV mixes both polarities and four
instrument labels (ORB, QTOF, Other, NA) and has no MS1 isotope fields, so everything is split by
instrument and polarity. It reads the processed training targets (``spec_df.pkl``).

1. Peak-pair spacing near 1 Da: whether 13C (1.003355 Da) and a hydrogen shift (1.007825 Da)
   form separate modes, by instrument and m/z range.
2. Precursor isotopes surviving in MS2, as a co-isolation fraction
   f_k = (MS2 M+k / M+0) / theoretical M+k / M+0, by instrument and adduct; for Cl/Br molecules,
   whether M+2 transmission is a per-molecule property.
3. Share of target intensity on 13C satellites, H-spaced peaks and halogen M+2 partners, i.e.
   what a monoisotopic prediction cannot reach, and the cosine a perfect monoisotopic prediction
   reaches (gate 0a step 2 of the isotope envelope plan, at peak level).

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/isotopes/debug_spectraverse_isotope_probe.py
"""

from __future__ import annotations

import argparse
import logging
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from pyopenms import CoarseIsotopePatternGenerator, EmpiricalFormula
from pyteomics.mass import Composition
from rdkit import RDLogger

from fragnnet.utils.formula_utils import PREC_TYPE_TO_FORMULA_DIFF

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

C13_SPACING = 1.003355
H_SPACING = 1.007825
BR81_SPACING = 1.997953
CL37_SPACING = 1.997050
SPACING_BIN_MDA = 0.25
NUM_SPACING_BINS = int(15 / SPACING_BIN_MDA)
MZ_BINS = ("<300", "300-600", ">=600")
INST_PPM = {"ORB": 10.0, "QTOF": 20.0, "Other": 20.0, "NA": 20.0}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    proc_dp = Path("data/proc/spectraverse")
    parser.add_argument("--spec-fp", type=Path, default=proc_dp / "spec_df.pkl")
    parser.add_argument("--mol-fp", type=Path, default=proc_dp / "mol_df.pkl")
    parser.add_argument("--num", type=int, default=None, help="Spectra sampled (default: all)")
    parser.add_argument("--min-tol", type=float, default=0.003, help="Da floor on the tolerance")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out-fp", type=Path, default=None, help="Optional per-spectrum pickle")
    return parser.parse_args()


def find_peak(mzs: np.ndarray, target: float, tol: float) -> int | None:
    """Index of the peak nearest ``target`` within ``tol``, or None."""
    i = np.searchsorted(mzs, target)
    hits = [j for j in (i - 1, i) if 0 <= j < len(mzs) and abs(mzs[j] - target) <= tol]
    return min(hits, key=lambda j: abs(mzs[j] - target)) if hits else None


def halogen_class(formula: str) -> str:
    has_br, has_cl = "Br" in formula, "Cl" in formula
    return "both" if has_br and has_cl else "Br" if has_br else "Cl" if has_cl else "none"


def precursor_formula(mol_formula: str, prec_type: str) -> str:
    comp = Composition(formula=mol_formula.rstrip("+-")) + Composition(formula=PREC_TYPE_TO_FORMULA_DIFF[prec_type])
    return "".join(f"{el}{n}" for el, n in sorted(comp.items()) if n)


def probe_one(task: tuple) -> tuple[dict, dict]:
    spec_id, mol_id, inst, ion_mode, adduct, prec_mz, formula, peaks, min_tol = task
    arr = np.asarray(peaks, float)
    order = np.argsort(arr[:, 0])
    mzs, ints = arr[order, 0], arr[order, 1]
    ppm = INST_PPM[inst]
    base = ints.max()
    total = ints.sum()

    # 1. Spacing histogram.
    spacing = defaultdict(lambda: np.zeros(NUM_SPACING_BINS))
    for i in range(len(mzs)):
        if ints[i] < 0.005 * base:
            continue
        lo, hi = np.searchsorted(mzs, [mzs[i] + 1.000, mzs[i] + 1.015])
        for j in range(lo, hi):
            if ints[j] >= 0.005 * base:
                mz_bin = MZ_BINS[int(mzs[i] >= 300) + int(mzs[i] >= 600)]
                spacing[mz_bin][int((mzs[j] - mzs[i] - 1.000) * 1e3 / SPACING_BIN_MDA)] += 1

    hal = halogen_class(formula)
    hal_step = {"Br": BR81_SPACING, "Cl": CL37_SPACING}.get(hal)

    # 2. Co-isolation fraction at the precursor.
    row = {
        "spec_id": spec_id,
        "molecule": mol_id,
        "inst": inst,
        "ion_mode": ion_mode,
        "adduct": adduct,
        "hal": hal,
        "n_peaks": len(mzs),
        "f1": np.nan,
        "f2": np.nan,
    }
    theo = [
        p.getIntensity()
        for p in CoarseIsotopePatternGenerator(3).run(EmpiricalFormula(precursor_formula(formula, adduct))).getContainer()
    ]
    tol = max(prec_mz * ppm * 1e-6, min_tol)
    p0 = find_peak(mzs, prec_mz, tol)
    row["prec_present"] = p0 is not None
    if p0 is not None and ints[p0] >= 0.05 * base:
        step2 = hal_step if hal_step is not None else 2 * C13_SPACING
        p1 = find_peak(mzs, mzs[p0] + C13_SPACING, tol)
        p2 = find_peak(mzs, mzs[p0] + step2, tol)
        row["f1"] = (ints[p1] / ints[p0] if p1 is not None else 0.0) / (theo[1] / theo[0])
        row["f2"] = (ints[p2] / ints[p0] if p2 is not None else 0.0) / (theo[2] / theo[0])

    # 3. Intensity on satellites of a more intense, less massive peak.
    is_c13 = np.zeros(len(mzs), bool)
    is_hal = np.zeros(len(mzs), bool)
    hsh = 0.0
    for j in range(len(mzs)):
        lo, hi = np.searchsorted(mzs, [mzs[j] - 1.0105, mzs[j] - 1.0015])
        gaps = [(mzs[j] - mzs[i]) * 1e3 for i in range(lo, hi) if ints[i] > ints[j]]
        if any(1002.0 <= gap <= 1005.3 for gap in gaps):
            is_c13[j] = True
        elif any(1006.0 <= gap <= 1009.5 for gap in gaps):
            hsh += ints[j]
        ptol = max(mzs[j] * ppm * 1e-6, min_tol)
        if hal_step is not None:
            k = find_peak(mzs, mzs[j] - hal_step, ptol)
            is_hal[j] = k is not None and ints[k] > ints[j]
    norm = np.sqrt((ints**2).sum())
    row.update(
        c13=ints[is_c13].sum() / total,
        hsh=hsh / total,
        hal_share=ints[is_hal & ~is_c13].sum() / total,
        # 1 - cos(target, monoisotopic part) = 1 - ||mono|| / ||target||, at peak level.
        cos_loss_c13=1 - np.sqrt((ints[~is_c13] ** 2).sum()) / norm,
        cos_loss_all=1 - np.sqrt((ints[~(is_c13 | is_hal)] ** 2).sum()) / norm,
    )
    return row, {(inst, k): v for k, v in spacing.items()}


def main() -> None:
    args = parse_args()
    spec_df = pd.read_pickle(args.spec_fp)
    mol_df = pd.read_pickle(args.mol_fp)[["mol_id", "formula"]]
    if args.num is not None:
        spec_df = spec_df.sample(n=args.num, random_state=0)
    spec_df = spec_df.assign(mol_id=spec_df["mol_id"].astype(int)).merge(mol_df, on="mol_id", how="left")
    log.info("Read %d spectra, %d molecules", len(spec_df), spec_df["mol_id"].nunique())

    tasks = [
        (r.spec_id, r.mol_id, r.inst_type, r.ion_mode, r.prec_type, float(r.prec_mz), r.formula, r.peaks, args.min_tol)
        for r in spec_df.itertuples(index=False)
    ]
    del spec_df
    spacing = defaultdict(lambda: np.zeros(NUM_SPACING_BINS))
    rows = []
    with Pool(args.workers) as pool:
        for row, hists in pool.imap_unordered(probe_one, tasks, chunksize=512):
            rows.append(row)
            for key, hist in hists.items():
                spacing[key] += hist
    res = pd.DataFrame(rows)
    if args.out_fp is not None:
        res.to_pickle(args.out_fp)
        log.info("Wrote %s", args.out_fp)

    log.info("1. Peak-pair spacing, counts per %.2f mDa bin (bin centre):", SPACING_BIN_MDA)
    centres = 1000 + (np.arange(NUM_SPACING_BINS) + 0.5) * SPACING_BIN_MDA
    c13_win = (centres > 1001.5) & (centres < 1005.3)
    h_win = (centres > 1005.8) & (centres < 1009.5)
    trough = (centres > 1004.8) & (centres < 1006.0)
    for (inst, mz_bin), hist in sorted(spacing.items()):
        log.info(
            "  %-6s m/z %-7s pairs %7d | 13C mode %.2f mDa (n %d), H mode %.2f mDa (n %d), trough/13C-peak %.2f",
            inst,
            mz_bin,
            hist.sum(),
            centres[c13_win][hist[c13_win].argmax()],
            hist[c13_win].sum(),
            centres[h_win][hist[h_win].argmax()],
            hist[h_win].sum(),
            hist[trough].min() / max(hist[c13_win].max(), 1),
        )

    log.info("2. Precursor in MS2 and co-isolation fraction:")
    for (inst, adduct), grp in res.groupby(["inst", "adduct"]):
        seen = grp.dropna(subset=["f1"])
        log.info(
            "  %-6s %-15s n=%6d  precursor present %.3f  >=5%% base %6d | median f1 %.3f  share f1>0 %.3f",
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
    for (inst, ion_mode), grp in halo.groupby(["inst", "ion_mode"]):
        log.info(
            "  Cl/Br %-6s %s n=%5d  share with f2 > 0.5: %.3f  median f2 %.3f",
            inst,
            ion_mode,
            len(grp),
            grp["full"].mean(),
            grp["f2"].median(),
        )
    per_mol = halo.groupby(["molecule", "ion_mode"])["full"].agg(["mean", "size"])
    per_mol = per_mol[per_mol["size"] >= 2]
    log.info(
        "  Cl/Br molecule x polarity with >=2 spectra: %d | all f2>0.5 %d | none %d | mixed %d",
        len(per_mol),
        (per_mol["mean"] == 1).sum(),
        (per_mol["mean"] == 0).sum(),
        ((per_mol["mean"] > 0) & (per_mol["mean"] < 1)).sum(),
    )

    log.info("3. Target intensity share and perfect-monoisotopic cosine loss (peak level):")
    for (inst, ion_mode), grp in res.groupby(["inst", "ion_mode"]):
        log.info(
            "  %-6s %s n=%6d  13C sat. mean %.4f (median %.4f, p90 %.4f) | H-spaced %.4f | "
            ">=5%% on 13C %.3f | 1-cos 13C %.4f, 13C+hal %.4f",
            inst,
            ion_mode,
            len(grp),
            grp["c13"].mean(),
            grp["c13"].median(),
            grp["c13"].quantile(0.9),
            grp["hsh"].mean(),
            (grp["c13"] >= 0.05).mean(),
            grp["cos_loss_c13"].mean(),
            grp["cos_loss_all"].mean(),
        )
    for ion_mode, grp in res.groupby("ion_mode"):
        log.info(
            "  all    %s n=%6d  13C sat. mean %.4f | 1-cos 13C %.4f, 13C+hal %.4f",
            ion_mode,
            len(grp),
            grp["c13"].mean(),
            grp["cos_loss_c13"].mean(),
            grp["cos_loss_all"].mean(),
        )
    halo_all = res[res["hal"].isin(["Br", "Cl"])]
    for (inst, ion_mode), grp in halo_all.groupby(["inst", "ion_mode"]):
        log.info(
            "  Cl/Br %-6s %s halogen M+2 partner share mean %.4f (p90 %.4f), 1-cos 13C+hal %.4f, n=%d",
            inst,
            ion_mode,
            grp["hal_share"].mean(),
            grp["hal_share"].quantile(0.9),
            grp["cos_loss_all"].mean(),
            len(grp),
        )


if __name__ == "__main__":
    main()
