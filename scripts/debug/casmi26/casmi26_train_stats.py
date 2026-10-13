"""Per-spectrum quality and subformula statistics for the CASMI 2026 training set.

Writes one row per ``train.parquet`` row, in file order, with:

- the peak-position heuristics of ``notebooks/data_notebooks/nist23_data_quality.ipynb``
  (same definitions, so CASMI and NIST23 numbers are comparable);
- subformula coverage: the fraction of peaks and intensity whose m/z matches a
  precursor-bounded subformula of the labelled molecule, using the candidate set of
  ``fragnnet.utils.clean_spectra_utils.filter_subformula_peaks`` (charge migration included);
- the precursor m/z implied by the formula and adduct, and its ppm error.

It also writes one row per normalized SMILES with RDKit's formula, charge, heavy-atom count
and component count, compared with the release's ``molecular_formula`` label.

No DAGs are built. ``notebooks/data_notebooks/casmi26_train.ipynb`` reads both tables.

Usage:
    python scripts/debug/casmi26/casmi26_train_stats.py \
        --train_fp data/raw/casmi2026/train.parquet \
        --out_dp output/casmi26_train_check_2026_10_05 --workers 10 --verify_n 200
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import subprocess
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolDescriptors

from fragnnet.utils.clean_spectra_utils import (
    INSTRUMENT_TOLERANCES,
    _subformula_masses,
    filter_subformula_peaks,
)
from fragnnet.utils.formula_utils import (
    ELEMENT_TO_MASS,
    PREC_TYPE_TO_MASS_DIFF,
    VALID_ELEMENTS,
    parse_formula,
)

logger = logging.getLogger(__name__)

# CASMI adduct string -> (project precursor type, number of M in the precursor).
# A dimer [2M+X] is checked as [M'+X] with M' = 2M: every fragment of a dimer is a
# subformula of the doubled formula, so this is the same precursor bound.
ADDUCT_TO_PROJECT = {
    "[M+H]+": ("[M+H]+", 1),
    "[M-H]-": ("[M-H]-", 1),
    "[M+Na]+": ("[M+Na]+", 1),
    "[M+NH4]+": ("[M+NH4]+", 1),
    "[M+Cl]-": ("[M+Cl]-", 1),
    "[M+K]+": ("[M+K]+", 1),
    "[M+Br]-": ("[M+Br]-", 1),
    "[M]+": ("[M]+", 1),
    "[M-H2O+H]+": ("[M-H2O+H]+", 1),
    "[M-2H2O+H]+": ("[M-2H2O+H]+", 1),
    "[M+CH2O2-H]-": ("[M+HCOOH-H]-", 1),
    "[M+C2H4O2-H]-": ("[M+CH3COOH-H]-", 1),
    "[M-H2O-H]-": ("[M-H-H2O]-", 1),
    "[M-CO2-H]-": ("[M-H-CO2]-", 1),
    "[2M+H]+": ("[M+H]+", 2),
    "[2M+Na]+": ("[M+Na]+", 2),
    "[2M+K]+": ("[M+K]+", 2),
    "[2M+NH4]+": ("[M+NH4]+", 2),
    "[2M-H]-": ("[M-H]-", 2),
    "[2M+CH2O2-H]-": ("[M+HCOOH-H]-", 2),
    "[2M+C2H4O2-H]-": ("[M+CH3COOH-H]-", 2),
}

# Matching tolerance per instrument class: (ppm, tolerance floor m/z), from the project's
# INSTRUMENT_TOLERANCES. Low-resolution and unknown instruments get the QTOF tolerance and are
# reported separately; a low coverage there reflects m/z precision, not wrong labels.
INSTRUMENT_CLASS_TOLERANCE = {
    "ORB": (INSTRUMENT_TOLERANCES["ORB"]["relative"] * 1e6, INSTRUMENT_TOLERANCES["ORB"]["min_mz"]),
    "TOF": (
        INSTRUMENT_TOLERANCES["QTOF"]["relative"] * 1e6,
        INSTRUMENT_TOLERANCES["QTOF"]["min_mz"],
    ),
    "LR": (
        INSTRUMENT_TOLERANCES["QTOF"]["relative"] * 1e6,
        INSTRUMENT_TOLERANCES["QTOF"]["min_mz"],
    ),
    "UNK": (
        INSTRUMENT_TOLERANCES["QTOF"]["relative"] * 1e6,
        INSTRUMENT_TOLERANCES["QTOF"]["min_mz"],
    ),
}

C13_SHIFT = 1.0033548378  # 13C - 12C
NULL_OFFSET_DA = 0.05  # chance-match control: shift every peak by +/- this before matching
# A second, tight tier for every instrument (the project's Orbitrap tolerance). At the QTOF
# tolerance the shifted-peak control already matches most intensity, so only the tight tier
# separates a right formula from chance.
TIGHT_TOLERANCE = (
    INSTRUMENT_TOLERANCES["ORB"]["relative"] * 1e6,
    INSTRUMENT_TOLERANCES["ORB"]["min_mz"],
)
MAX_GRID_SIZE = 20_000_000  # skip formulas whose subformula grid would exceed this

SPEC_COLUMNS = [
    "molecular_formula",
    "adduct",
    "instrument_type",
    "precursor_mz",
    "ms2_mzs",
    "ms2_normalized_intensities",
]


def instrument_class(instrument: str | None) -> str:
    """Map a free-text instrument label to ORB, TOF, LR (low resolution) or UNK."""
    if instrument is None or (isinstance(instrument, float) and math.isnan(instrument)):
        return "UNK"
    text = instrument.strip().lower()
    if text in ("", "nan", "n/a", "in source cid"):
        return "UNK"
    if re.search(r"orbitrap|exactive|qehf|qft|itft|hybrid ft|-ft\b", text):
        return "ORB"
    if re.search(r"tof|maxis|impact|synapt", text):
        return "TOF"
    if re.search(r"qq|trap|quattro|ltq|(^|-)q?it$|^lit$|-q$|be$|ebeb", text):
        return "LR"
    return "UNK"


def _strip_charge(formula: str) -> tuple[str, bool]:
    stripped = re.sub(r"[+-]\d*$", "", formula.strip())
    return stripped, stripped != formula.strip()


def _counts_to_formula(counts: dict[str, int]) -> str:
    return "".join(f"{element}{count}" for element, count in sorted(counts.items()) if count)


def _grid_size(counts: dict[str, int]) -> int:
    size = 1
    for count in counts.values():
        size *= count + 1
    return size * 4  # headroom for up to two carrier budgets with a few carrier levels


def _match(sorted_masses: np.ndarray, mzs: np.ndarray, tol_da: np.ndarray) -> np.ndarray:
    """Return a mask of peaks within ``tol_da`` of any candidate mass (as the filter does)."""
    if sorted_masses.size == 0 or mzs.size == 0:
        return np.zeros(mzs.shape, dtype=bool)
    right = np.searchsorted(sorted_masses, mzs)
    left = np.clip(right - 1, 0, sorted_masses.size - 1)
    right = np.clip(right, 0, sorted_masses.size - 1)
    distance = np.minimum(np.abs(sorted_masses[left] - mzs), np.abs(sorted_masses[right] - mzs))
    return distance <= tol_da


def _frac(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def _peak_heuristics(mzs: np.ndarray, ints: np.ndarray, prec_mz: float) -> dict:
    """Peak-position heuristics with the definitions of nist23_data_quality.ipynb."""
    total = float(ints.sum())
    out = {
        "num_peaks": int(mzs.size),
        "n_peaks_1pct": int((ints >= 0.01).sum()),
        "min_rel_int": float(ints.min()) if ints.size else np.nan,
        "frac_not_far_above_prec": np.nan,
        "n_far_above_prec": np.nan,
        "has_prec_peak": False,
        "intensity_entropy": np.nan,
    }
    if mzs.size == 0:
        return out
    if np.isfinite(prec_mz):
        out["frac_not_far_above_prec"] = float((mzs <= prec_mz + 1.5).mean())
        out["n_far_above_prec"] = int((mzs > prec_mz + 2.0).sum())
        out["has_prec_peak"] = bool((np.abs(mzs - prec_mz) <= 0.02).any())
    if total > 0:
        p = ints[ints > 0] / total
        out["intensity_entropy"] = float(-np.sum(p * np.log(p)))
    return out


def _subformula_stats(
    masses: np.ndarray,
    mzs: np.ndarray,
    ints: np.ndarray,
    prec_mz: float,
    ppm: float,
    min_mz: float,
) -> dict:
    """Coverage of one spectrum by one candidate mass set."""
    total = float(ints.sum())
    tol_da = ppm * 1e-6 * np.maximum(mzs, min_mz)
    explained = _match(masses, mzs, tol_da)

    # 13C satellites of explained peaks are not subformulas but are not noise either.
    iso_parents = np.sort(mzs[explained] + C13_SHIFT)
    isotope = ~explained & _match(iso_parents, mzs, tol_da)

    null_plus = _match(masses, mzs + NULL_OFFSET_DA, tol_da)
    null_minus = _match(masses, mzs - NULL_OFFSET_DA, tol_da)

    big = ints >= 0.01
    if np.isfinite(prec_mz):
        cutoff = prec_mz + ppm * 1e-6 * max(prec_mz, min_mz)
        le_prec = mzs <= cutoff
    else:
        le_prec = np.ones(mzs.shape, dtype=bool)
    base_idx = int(np.argmax(ints))
    tight_da = TIGHT_TOLERANCE[0] * 1e-6 * np.maximum(mzs, TIGHT_TOLERANCE[1])
    tight = _match(masses, mzs, tight_da)
    tight_null = 0.5 * (
        _frac(ints[_match(masses, mzs + NULL_OFFSET_DA, tight_da)].sum(), total)
        + _frac(ints[_match(masses, mzs - NULL_OFFSET_DA, tight_da)].sum(), total)
    )
    return {
        "sf_peak_frac": float(explained.mean()),
        "sf_int_frac": _frac(ints[explained].sum(), total),
        "sf_int_frac_1pct": _frac(ints[explained & big].sum(), ints[big].sum()),
        "sf_int_frac_le_prec": _frac(ints[explained & le_prec].sum(), ints[le_prec].sum()),
        "sf_int_frac_iso": _frac(ints[explained | isotope].sum(), total),
        "sf_base_explained": bool(explained[base_idx]),
        "sf_int_frac_null": 0.5
        * (_frac(ints[null_plus].sum(), total) + _frac(ints[null_minus].sum(), total)),
        "int_frac_above_prec": _frac(ints[~le_prec].sum(), total),
        "sf_peak_frac_tight": float(tight.mean()),
        "sf_int_frac_tight": _frac(ints[tight].sum(), total),
        "sf_base_explained_tight": bool(tight[base_idx]),
        "sf_int_frac_null_tight": tight_null,
    }


def _neutral_mass(counts: dict[str, int]) -> float:
    return float(sum(ELEMENT_TO_MASS[element] * count for element, count in counts.items()))


def _prepare_key(formula: str, adduct: str) -> dict:
    """Resolve a (formula, adduct) pair to a candidate-set key, or a skip status."""
    if adduct not in ADDUCT_TO_PROJECT:
        return {"status": "unsupported_adduct", "adduct_proj": None, "mult": np.nan}
    adduct_proj, mult = ADDUCT_TO_PROJECT[adduct]
    stripped, charged = _strip_charge(formula)
    try:
        counts = parse_formula(stripped)
    except (ValueError, AssertionError):
        return {"status": "bad_formula", "adduct_proj": adduct_proj, "mult": mult}
    if set(counts) - set(VALID_ELEMENTS):
        return {"status": "unsupported_element", "adduct_proj": adduct_proj, "mult": mult}
    counts = {element: count * mult for element, count in counts.items()}
    prec_calc = _neutral_mass(counts) + PREC_TYPE_TO_MASS_DIFF[adduct_proj]
    out = {
        "adduct_proj": adduct_proj,
        "mult": mult,
        "charged_formula": charged,
        "prec_calc_mz": prec_calc,
        "sf_formula": _counts_to_formula(counts),
    }
    if _grid_size(counts) > MAX_GRID_SIZE:
        out["status"] = "too_large"
    else:
        out["status"] = "ok"
    return out


def _list_column(table: pa.Table, name: str) -> tuple[np.ndarray, np.ndarray]:
    column = table.column(name).combine_chunks()
    return column.offsets.to_numpy(), column.values.to_numpy(zero_copy_only=False)


def process_row_group(task: tuple[str, int, int, int, int]) -> tuple[pd.DataFrame, int, int]:
    """Compute per-spectrum statistics for one parquet row group.

    Args:
        task: (train path, row-group index, global row offset, verify count, seed).

    Returns:
        One row per spectrum, with the global ``row`` index into ``train.parquet``; the number
        of spectra re-checked against ``filter_subformula_peaks``; and how many disagreed.
    """
    train_fp, rg_idx, row_offset, verify_n, seed = task
    table = pq.ParquetFile(train_fp).read_row_group(rg_idx, columns=SPEC_COLUMNS)
    meta = table.select(["molecular_formula", "adduct", "instrument_type", "precursor_mz"])
    meta = meta.to_pandas()
    mz_off, mz_val = _list_column(table, "ms2_mzs")
    int_off, int_val = _list_column(table, "ms2_normalized_intensities")
    if not np.array_equal(mz_off, int_off):
        raise ValueError(f"Row group {rg_idx}: m/z and intensity lists differ in length")

    n_rows = len(meta)
    key_cache: dict[tuple[str, str], dict] = {}
    rows: list[dict] = []
    order = meta.sort_values(["molecular_formula", "adduct"], kind="stable").index.to_numpy()
    formulas = meta["molecular_formula"].to_numpy()
    adducts = meta["adduct"].to_numpy()
    prec_mzs = meta["precursor_mz"].to_numpy(dtype=np.float64)
    inst_classes = [instrument_class(value) for value in meta["instrument_type"].to_numpy()]
    rng = np.random.default_rng(seed + rg_idx)
    verify_rows = set(rng.choice(n_rows, size=min(verify_n, n_rows), replace=False).tolist())
    verify_checked = 0
    verify_mismatch = 0

    for i in order:
        formula = formulas[i]
        adduct = adducts[i]
        prec_mz = float(prec_mzs[i])
        inst = inst_classes[i]
        ppm, min_mz = INSTRUMENT_CLASS_TOLERANCE[inst]
        mzs = mz_val[mz_off[i] : mz_off[i + 1]].astype(np.float64)
        ints = int_val[int_off[i] : int_off[i + 1]].astype(np.float64)

        key = (formula, adduct)
        if key not in key_cache:
            key_cache.clear()  # rows are sorted by key, so only the current key is reused
            key_cache[key] = _prepare_key(formula, adduct)
        info = key_cache[key]

        row = {"row": row_offset + int(i), "inst_class": inst, "tol_ppm": ppm, "tol_min_mz": min_mz}
        row.update(_peak_heuristics(mzs, ints, prec_mz))
        row["adduct_proj"] = info["adduct_proj"]
        row["mult"] = info["mult"]
        row["charged_formula"] = info.get("charged_formula", False)
        prec_calc = info.get("prec_calc_mz", np.nan)
        row["prec_calc_mz"] = prec_calc
        row["prec_calc_ppm"] = (prec_mz - prec_calc) / prec_calc * 1e6 if prec_calc else np.nan
        status = info["status"]
        if status == "ok" and mzs.size == 0:
            status = "no_peaks"
        row["sf_status"] = status
        if status == "ok":
            masses = _subformula_masses(info["sf_formula"], info["adduct_proj"])
            row.update(_subformula_stats(masses, mzs, ints, prec_mz, ppm, min_mz))
            if i in verify_rows:
                reference = filter_subformula_peaks(
                    list(zip(mzs.tolist(), ints.tolist())),
                    info["sf_formula"],
                    info["adduct_proj"],
                    tolerance_ppm=ppm,
                    tolerance_min_mz=min_mz,
                )
                tol_da = ppm * 1e-6 * np.maximum(mzs, min_mz)
                mine = mzs[_match(masses, mzs, tol_da)]
                verify_checked += 1
                verify_mismatch += int(
                    not np.array_equal(np.sort(mine), np.sort([p[0] for p in reference]))
                )
        rows.append(row)

    out = pd.DataFrame(rows).sort_values("row").reset_index(drop=True)
    logger.info("row group %d: %d rows, verified %d", rg_idx, n_rows, verify_checked)
    return out, verify_checked, verify_mismatch


def _mol_row(item: tuple[str, str, int]) -> dict:
    smiles, formula_label, n_labels = item
    RDLogger.DisableLog("rdApp.*")
    out = {
        "normalized_smiles": smiles,
        "formula_label": formula_label,
        "n_formula_labels": n_labels,
        "parse_ok": False,
        "formula_rdkit": None,
        "formal_charge": np.nan,
        "heavy_atoms": np.nan,
        "n_components": np.nan,
        "formula_match": False,
    }
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return out
    formula_rdkit = rdMolDescriptors.CalcMolFormula(mol)
    out.update(
        parse_ok=True,
        formula_rdkit=formula_rdkit,
        formal_charge=Chem.GetFormalCharge(mol),
        heavy_atoms=mol.GetNumHeavyAtoms(),
        n_components=len(Chem.GetMolFrags(mol)),
        formula_match=formula_rdkit.replace(" ", "") == formula_label.replace(" ", ""),
    )
    return out


def molecule_stats(train_fp: Path, workers: int) -> pd.DataFrame:
    """RDKit formula, charge and size per normalized SMILES, against the release label."""
    labels = pd.read_parquet(train_fp, columns=["normalized_smiles", "molecular_formula"])
    grouped = labels.groupby("normalized_smiles")["molecular_formula"]
    first = grouped.first()
    n_labels = grouped.nunique()
    items = list(zip(first.index, first.to_numpy(), n_labels.reindex(first.index).to_numpy()))
    with Pool(workers) as pool:
        rows = pool.map(_mol_row, items, chunksize=2000)
    return pd.DataFrame(rows)


def _git_state() -> dict:
    def run(*cmd: str) -> str:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()

    return {
        "revision": run("git", "rev-parse", "HEAD"),
        "dirty": bool(run("git", "status", "--short")),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train_fp", type=Path, default=Path("data/raw/casmi2026/train.parquet"))
    parser.add_argument("--out_dp", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument(
        "--verify_n",
        type=int,
        default=200,
        help="Spectra per row group re-checked against filter_subformula_peaks",
    )
    parser.add_argument("--seed", type=int, default=20261005)
    parser.add_argument("--max_row_groups", type=int, default=None, help="Smoke runs only")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dp.mkdir(parents=True, exist_ok=True)
    start = time.time()

    parquet = pq.ParquetFile(args.train_fp)
    n_groups = parquet.metadata.num_row_groups
    if args.max_row_groups is not None:
        n_groups = min(n_groups, args.max_row_groups)
    offsets = np.concatenate(
        [[0], np.cumsum([parquet.metadata.row_group(g).num_rows for g in range(n_groups)])]
    )
    tasks = [
        (str(args.train_fp), g, int(offsets[g]), args.verify_n, args.seed) for g in range(n_groups)
    ]
    with Pool(args.workers, maxtasksperchild=1) as pool:
        parts = pool.map(process_row_group, tasks, chunksize=1)
    checked = sum(part[1] for part in parts)
    mismatched = sum(part[2] for part in parts)
    spec_stats = pd.concat([part[0] for part in parts], ignore_index=True)
    if mismatched:
        raise RuntimeError(
            f"{mismatched}/{checked} verified spectra disagree with filter_subformula_peaks"
        )
    spec_fp = args.out_dp / "casmi26_train_spec_stats.parquet"
    spec_stats.to_parquet(spec_fp, index=False)
    logger.info("wrote %s (%d rows)", spec_fp, len(spec_stats))

    mol_stats = molecule_stats(args.train_fp, args.workers)
    mol_fp = args.out_dp / "casmi26_train_mol_stats.parquet"
    mol_stats.to_parquet(mol_fp, index=False)
    logger.info("wrote %s (%d rows)", mol_fp, len(mol_stats))

    run_info = {
        "args": {key: str(value) for key, value in vars(args).items()},
        "git": _git_state(),
        "train_rows": int(offsets[-1]),
        "instrument_class_tolerance": INSTRUMENT_CLASS_TOLERANCE,
        "adduct_to_project": ADDUCT_TO_PROJECT,
        "null_offset_da": NULL_OFFSET_DA,
        "tight_tolerance": TIGHT_TOLERANCE,
        "max_grid_size": MAX_GRID_SIZE,
        "verify_checked": checked,
        "verify_mismatched": mismatched,
        "sf_status_counts": spec_stats["sf_status"].value_counts().to_dict(),
        "elapsed_s": round(time.time() - start, 1),
    }
    (args.out_dp / "run_info.json").write_text(json.dumps(run_info, indent=2))
    logger.info("done in %.0f s", run_info["elapsed_s"])


if __name__ == "__main__":
    main()
