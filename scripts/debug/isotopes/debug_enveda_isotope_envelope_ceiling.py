"""Stage 0a of the MASSter isotope-envelope plan: how much cosine a natural envelope recovers.

For each val spectrum, the target splits into satellite peaks (13C and halogen M+2 partners, the
rule of ``debug_enveda_isotope_probe.py``) and the rest, the monoisotopic part. A perfect
monoisotopic model predicts the monoisotopic part exactly. The envelope model adds, for every
monoisotopic peak with a precursor-bounded subformula, its natural M+1 and M+2 isotopologues
from that composition, scaled by co-isolation factors ``f1`` and ``f2``:

    prediction = mono + f1 * env1 + f2 * env2

Cosine is the val metric's: raw intensities summed into floor(m/z / 0.01) bins. Six dot products
per spectrum give the cosine for any (f1, f2) in closed form. ``f`` per polarity is chosen on even
``mol_id`` and reported on odd ``mol_id``.

Run from repo root:
    conda run -n FRAGNNET-GPU python scripts/debug/isotopes/debug_enveda_isotope_envelope_ceiling.py
"""

from __future__ import annotations

import argparse
import logging
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from fragnnet.utils.clean_spectra_utils import subformula_candidates

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

C13_SPACING = 1.003355
HALOGEN_M2 = {"Br": 1.997953, "Cl": 1.997050}
# Ratio of the M+1 / M+2 isotope to the lightest one, and its mass offset (Da).
M1_ISOTOPES = {
    "C": (0.010816, 1.003355),
    "H": (0.000115, 1.006277),
    "N": (0.003654, 0.997035),
    "O": (0.000381, 1.004217),
    "S": (0.007895, 0.999387),
    "Si": (0.050778, 0.999568),
}
M2_ISOTOPES = {
    "O": (0.002055, 2.004245),
    "S": (0.044536, 1.995796),
    "Cl": (0.319781, 1.997050),
    "Br": (0.972760, 1.997953),
    "Si": (0.033630, 1.996845),
    "K": (0.072166, 1.998117),
}
F_GRID = np.round(np.arange(0.0, 1.0001, 0.05), 2)
DOT_KEYS = ("tt", "tm", "t1", "t2", "mm", "11", "22", "m1", "m2", "12")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proc-dp", type=Path, default=Path("data/proc/enveda_180"))
    parser.add_argument("--split-dp", type=Path, default=Path("data/split/enveda_180_inchikey_811"))
    parser.add_argument("--split", default="val")
    parser.add_argument("--prec-types", nargs="+", default=["[M+H]+", "[M-H]-"])
    parser.add_argument("--ppm", type=float, default=10.0, help="Match and decomposition tolerance")
    parser.add_argument("--min-tol", type=float, default=0.002, help="Da floor on the tolerance")
    parser.add_argument("--bin", type=float, default=0.01, help="Cosine bin width (Da)")
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=0, help="Spectra to use; 0 = all")
    parser.add_argument("--out-fp", type=Path, default=None, help="Optional per-spectrum pickle")
    return parser.parse_args()


ARGS: argparse.Namespace | None = None


def tolerance(mz: float) -> float:
    return max(mz * ARGS.ppm * 1e-6, ARGS.min_tol)


def satellite_mask(mzs: np.ndarray, ints: np.ndarray, halogen_step: float | None) -> np.ndarray:
    """Flag peaks with a more intense 13C parent or, for halogenated molecules, a halogen parent."""
    mask = np.zeros(len(mzs), bool)
    for j in range(len(mzs)):
        lo, hi = np.searchsorted(mzs, [mzs[j] - 1.0053, mzs[j] - 1.0020])
        if np.any(ints[lo:hi] > ints[j]):
            mask[j] = True
            continue
        if halogen_step is not None:
            target = mzs[j] - halogen_step
            i = np.searchsorted(mzs, target)
            if any(
                0 <= k < len(mzs) and abs(mzs[k] - target) <= tolerance(mzs[j]) for k in (i - 1, i)
            ):
                mask[j] = True
    return mask


def envelope(counts: dict[str, int]) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return (offset, ratio-to-M+0) rows for the M+1 and M+2 isotopologues of one composition.

    M+1 / M+0 is exactly sum_e n_e r_e1. M+2 / M+0 is the pair term (R1^2 - sum_e n_e r_e1^2) / 2,
    placed at two 13C spacings, plus sum_e n_e r_e2 at each heavy isotope's own offset.
    """
    m1 = [(off, counts.get(e, 0) * r) for e, (r, off) in M1_ISOTOPES.items() if counts.get(e, 0)]
    r1 = sum(r for _, r in m1)
    pair = (r1**2 - sum(counts.get(e, 0) * r**2 for e, (r, _) in M1_ISOTOPES.items())) / 2
    m2 = [(2 * C13_SPACING, pair)] if pair > 0 else []
    m2 += [(off, counts.get(e, 0) * r) for e, (r, off) in M2_ISOTOPES.items() if counts.get(e, 0)]
    return m1, m2


def binned(mzs: np.ndarray, ints: np.ndarray) -> dict[int, float]:
    out: dict[int, float] = {}
    for b, v in zip(np.floor(mzs / ARGS.bin).astype(int), ints):
        out[b] = out.get(b, 0.0) + v
    return out


def dot(a: dict[int, float], b: dict[int, float]) -> float:
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b.get(k, 0.0) for k, v in a.items())


def spectrum_dots(row: tuple) -> dict:
    """Dot products between target, monoisotopic part and the two envelope parts."""
    spec_id, mol_id, prec_type, formula, peaks = row
    peaks = np.asarray(peaks, float)
    order = np.argsort(peaks[:, 0])
    mzs, ints = peaks[order, 0], peaks[order, 1]
    hal = [h for h in HALOGEN_M2 if h in formula]
    halogen_step = HALOGEN_M2[hal[0]] if len(hal) == 1 else None
    sat = satellite_mask(mzs, ints, halogen_step)

    env1_mz, env1_int, env2_mz, env2_int = [], [], [], []
    explained = 0.0
    for mz, inten in zip(mzs[~sat], ints[~sat]):
        cands = subformula_candidates(float(mz), formula, prec_type, tolerance(mz))
        if not cands:
            continue
        explained += inten
        weight = inten / len(cands)  # candidates share the peak equally
        for counts in cands:
            m1, m2 = envelope(counts)
            env1_mz += [mz + off for off, _ in m1]
            env1_int += [weight * r for _, r in m1]
            env2_mz += [mz + off for off, _ in m2]
            env2_int += [weight * r for _, r in m2]

    t = binned(mzs, ints)
    m = binned(mzs[~sat], ints[~sat])
    e1 = binned(np.array(env1_mz), np.array(env1_int)) if env1_mz else {}
    e2 = binned(np.array(env2_mz), np.array(env2_int)) if env2_mz else {}
    parts = {"t": t, "m": m, "1": e1, "2": e2}
    out = {"spec_id": spec_id, "mol_id": mol_id, "prec_type": prec_type}
    out["halogen"] = halogen_step is not None
    out["mono_explained_share"] = explained / max(ints[~sat].sum(), 1e-12)
    out["satellite_share"] = ints[sat].sum() / ints.sum()
    for key in DOT_KEYS:
        out[key] = dot(parts[key[0]], parts[key[1]])
    return out


def cosine(df: pd.DataFrame, f1: np.ndarray | float, f2: np.ndarray | float) -> np.ndarray:
    num = df["tm"] + f1 * df["t1"] + f2 * df["t2"]
    den = (
        df["mm"]
        + f1**2 * df["11"]
        + f2**2 * df["22"]
        + 2 * f1 * df["m1"]
        + 2 * f2 * df["m2"]
        + 2 * f1 * f2 * df["12"]
    )
    return (num / np.sqrt(df["tt"] * den)).to_numpy()


def init_worker(args: argparse.Namespace) -> None:
    global ARGS
    ARGS = args


def main() -> None:
    args = parse_args()
    init_worker(args)

    # 1. Load the split's spectra with their molecular formula.
    spec = pd.read_pickle(args.proc_dp / "spec_df.pkl")[["spec_id", "mol_id", "prec_type", "peaks"]]
    mol = pd.read_pickle(args.proc_dp / "mol_df.pkl")[["mol_id", "formula"]]
    ids = pd.read_csv(args.split_dp / f"{args.split}_ids.csv")[["spec_id"]]
    df = ids.merge(spec, on="spec_id").merge(mol, on="mol_id")
    df = df[df["prec_type"].isin(args.prec_types)]
    if args.limit:
        df = df.iloc[: args.limit]
    cols = ["spec_id", "mol_id", "prec_type", "formula", "peaks"]
    rows = list(df[cols].itertuples(index=False, name=None))  # plain tuples pickle to workers
    log.info("%d %s spectra, %d molecules", len(rows), args.split, df["mol_id"].nunique())

    # 2. Per-spectrum dot products (the expensive part: subformula decomposition per peak).
    with Pool(args.workers, initializer=init_worker, initargs=(args,)) as pool:
        res = pd.DataFrame(pool.map(spectrum_dots, rows, chunksize=64))
    res["pol"] = np.where(res["prec_type"].str.endswith("+"), "pos", "neg")
    if args.out_fp is not None:
        res.to_pickle(args.out_fp)

    # 3. Fit f per polarity on even molecules, report on odd ones.
    even = res["mol_id"] % 2 == 0
    log.info("1 - cos against the full target (binned %.2f Da, raw intensity):", args.bin)
    header = "  %-10s %6s | mono %.4f | natural %.4f | fitted f1=%.2f f2=%.2f %.4f | oracle f2 %.4f"
    for pol, grp in res.groupby("pol"):
        fit = grp[even[grp.index]]
        grid = [(a, b, cosine(fit, a, b).mean()) for a in F_GRID for b in F_GRID]
        f1, f2, _ = max(grid, key=lambda x: x[2])
        rep = grp[~even[grp.index]]
        for label, sub in (("odd, all", rep), ("odd, Cl/Br", rep[rep["halogen"]])):
            if sub.empty:
                continue
            oracle = np.maximum(cosine(sub, f1, 0.0), cosine(sub, f1, 1.0)).mean()
            log.info(
                header,
                f"{pol} {label}",
                len(sub),
                1 - cosine(sub, 0.0, 0.0).mean(),
                1 - cosine(sub, 1.0, 1.0).mean(),
                f1,
                f2,
                1 - cosine(sub, f1, f2).mean(),
                1 - oracle,
            )
    log.info(
        "  satellite intensity share %.4f; monoisotopic intensity with a subformula %.4f",
        res["satellite_share"].mean(),
        res["mono_explained_share"].mean(),
    )


if __name__ == "__main__":
    main()
