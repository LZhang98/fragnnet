"""Br/Cl/I carriers, their spectrum quality, and measured halogen-swap pairs in CASMI 2026 train.

Stage 1 scores every Br/Cl/I-carrying train row (formula label) with a halogen-isotope-aware
subformula coverage at the tight (ORB) tolerance: a peak is explained when it matches a
monoisotopic subformula ion, or is a 13C, 81Br or 37Cl (one or two heavy atoms) satellite of a
matched peak, so the heavy-isotope peaks of a halogenated fragment count as explained. When the
precursor is the M+2 isotopologue, the candidates also include the heavy-halogen shift. The same
rule scores the +/-0.05 Da shifted-peak chance control. The plain monoisotopic coverage of ``casmi26_train_stats.py`` is
kept beside it for comparison. The precursor is checked against the monoisotopic ion and the
M+2 isotopologue (some libraries isolate the 81Br/37Cl ion).

Stage 2 counts carrier molecules and spectra per ``ingest_lib`` (all rows and high-quality
rows), how many are absent from NIST20 HR, and Br->Cl, I->Br, I->Cl swap pairs (single-site and
all-site swaps, 2D InChIKey block) inside library pools, with and without NIST20 HR.

Usage (from the repository root):
    PYTHONPATH=src python scripts/debug/casmi_enveda/casmi26_halogen_census.py \
        --out_dp output/casmi26_halogen_2026_10_16 --workers 4
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "masster_memorization"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from casmi26_train_stats import (  # noqa: E402
    C13_SHIFT,
    NULL_OFFSET_DA,
    TIGHT_TOLERANCE,
    _list_column,
    _match,
    _prepare_key,
    _subformula_masses,
    instrument_class,
)
from element_swap_census import swapped  # noqa: E402

from fragnnet.utils.formula_utils import parse_formula  # noqa: E402

RDLogger.DisableLog("rdApp.*")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

BR_SHIFT = 80.9162897 - 78.9183371  # 81Br - 79Br
CL_SHIFT = 36.9659026 - 34.9688527  # 37Cl - 35Cl
HALOGENS = ("Br", "Cl", "I")
META_COLS = [
    "ingest_lib",
    "normalized_smiles",
    "inchikey14",
    "molecular_formula",
    "ionization_mode",
    "instrument_type",
    "adduct",
    "precursor_mz",
    "collision_energy_ev",
]
SWAPS = [("Br", "Cl"), ("I", "Br"), ("I", "Cl")]
POOLS = {
    "msnlib": ["pluskal_ms2"],
    "enveda180": ["enveda-180"],
    "msnlib+enveda180": ["pluskal_ms2", "enveda-180"],
    "casmi_all": None,
}
# High-quality row: HR monomer, precursor within 10 ppm (mono or M+2), >= 3 peaks, and the
# isotope-aware tight coverage both high and well above its chance control.
HQ_MIN_COV = 0.8
HQ_MIN_GAP = 0.3
HQ_MAX_PPM = 10.0


def _formula_counts(formula: str) -> dict[str, int]:
    try:
        return parse_formula(str(formula).rstrip("+-"))
    except (ValueError, AssertionError):
        return {}


def _iso_shifts(counts: dict[str, int]) -> list[float]:
    shifts = [C13_SHIFT]
    for el, shift in (("Br", BR_SHIFT), ("Cl", CL_SHIFT)):
        if counts.get(el, 0) >= 1:
            shifts.append(shift)
        if counts.get(el, 0) >= 2:
            shifts.append(2 * shift)
    if counts.get("Br", 0) and counts.get("Cl", 0):
        shifts.append(BR_SHIFT + CL_SHIFT)
    return shifts


def _cov(masses, mzs, ints, tol_da):
    total = float(ints.sum())
    hit = _match(masses, mzs, tol_da)
    frac = float(ints[hit].sum() / total) if total > 0 else np.nan
    return frac, bool(hit[int(np.argmax(ints))]) if mzs.size else False


def _cov_iso(masses, mzs, ints, tol_da, shifts):
    """Monoisotopic match, plus peaks that are a 13C/81Br/37Cl satellite of a matched peak."""
    total = float(ints.sum())
    hit = _match(masses, mzs, tol_da)
    parents = np.sort(mzs[hit])
    for shift in shifts:
        hit = hit | _match(parents, mzs - shift, tol_da)
    frac = float(ints[hit].sum() / total) if total > 0 else np.nan
    return frac, bool(hit[int(np.argmax(ints))]) if mzs.size else False


def score_row_group(task):
    """Score one row group in batches (a whole decoded row group needs ~7 GB)."""
    train_fp, rg = task
    pf = pq.ParquetFile(train_fp)
    cols = META_COLS + ["ms2_mzs", "ms2_normalized_intensities"]
    parts, offset = [], 0
    for batch in pf.iter_batches(batch_size=8192, row_groups=[rg], columns=cols):
        table = pa.Table.from_batches([batch])
        parts.append(_score_table(table, rg, offset))
        offset += table.num_rows
    logger.info("row group %d: %d rows", rg, offset)
    return pd.concat(parts, ignore_index=True)


def _score_table(table, rg, offset):
    meta = table.select(META_COLS).to_pandas()
    counts = {f: _formula_counts(f) for f in meta["molecular_formula"].unique()}
    keep = np.array([any(counts[f].get(el, 0) for el in HALOGENS) for f in meta["molecular_formula"]])
    mz_off, mz_val = _list_column(table, "ms2_mzs")
    int_off, int_val = _list_column(table, "ms2_normalized_intensities")
    rows = []
    key_cache: dict = {}
    for i in np.flatnonzero(keep):
        formula, adduct = meta.at[i, "molecular_formula"], meta.at[i, "adduct"]
        c = counts[formula]
        mzs = mz_val[mz_off[i] : mz_off[i + 1]].astype(np.float64)
        ints = int_val[int_off[i] : int_off[i + 1]].astype(np.float64)
        prec = float(meta.at[i, "precursor_mz"])
        row = {"rg": rg, "idx": offset + int(i), "num_peaks": int(mzs.size), "n_peaks_1pct": int((ints >= 0.01).sum()),
               "inst_class": instrument_class(meta.at[i, "instrument_type"])}
        for el in HALOGENS:
            row[el] = c.get(el, 0)
        key = (formula, adduct)
        if key not in key_cache:
            info = _prepare_key(formula, adduct)
            if info["status"] == "ok":
                info["masses"] = _subformula_masses(info["sf_formula"], info["adduct_proj"])
                info["shifts"] = _iso_shifts(c)
            key_cache[key] = info
        info = key_cache[key]
        row["status"] = info["status"] if mzs.size else "no_peaks"
        row["mult"] = info["mult"]
        pc = info.get("prec_calc_mz", np.nan)
        if np.isfinite(pc) and pc:
            heavy = BR_SHIFT if c.get("Br", 0) else (CL_SHIFT if c.get("Cl", 0) else 0.0)
            ppm0 = (prec - pc) / pc * 1e6
            ppm2 = (prec - pc - heavy) / (pc + heavy) * 1e6 if heavy else np.inf
            row["prec_ppm"] = ppm0 if abs(ppm0) <= abs(ppm2) else ppm2
            row["prec_is_m2"] = bool(abs(ppm2) < abs(ppm0))
        else:
            row["prec_ppm"], row["prec_is_m2"] = np.nan, False
        if row["status"] == "ok":
            tol = TIGHT_TOLERANCE[0] * 1e-6 * np.maximum(mzs, TIGHT_TOLERANCE[1])
            row["cov_mono"], row["base_mono"] = _cov(info["masses"], mzs, ints, tol)
            masses = info["masses"]
            if row["prec_is_m2"]:  # 81Br/37Cl precursor isolated: halogen-retaining fragments shift
                masses = np.unique(np.concatenate([masses, masses + info["shifts"][1]]))
            shifts = info["shifts"]
            row["cov_iso"], row["base_iso"] = _cov_iso(masses, mzs, ints, tol, shifts)
            row["null_iso"] = 0.5 * (_cov_iso(masses, mzs + NULL_OFFSET_DA, ints, tol, shifts)[0]
                                     + _cov_iso(masses, mzs - NULL_OFFSET_DA, ints, tol, shifts)[0])
        rows.append(row)
    out = meta.loc[keep].reset_index(drop=True)
    return pd.concat([out, pd.DataFrame(rows)], axis=1)


def nist20_hr(proc_dp: Path) -> pd.DataFrame:
    spec = pd.read_pickle(proc_dp / "spec_df.pkl")
    spec = spec.loc[spec["dset"].astype(str).str.startswith("nist20_hr"), ["mol_id", "prec_type", "inst_type"]]
    mol = pd.read_pickle(proc_dp / "mol_df.pkl")[["mol_id", "smiles"]]
    mol = mol[mol["mol_id"].isin(spec["mol_id"])].copy()
    mol["ik14"] = [Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in mol["smiles"]]
    return spec.merge(mol, on="mol_id")


def swap_pairs(carriers: dict[str, str], pool_keys: set[str], src: str, dst: str) -> set[tuple[str, str]]:
    pairs = set()
    for ik14, smi in carriers.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        for _mode, key in swapped(mol, src, dst):
            if key in pool_keys and key != ik14:
                pairs.add((ik14, key))
    return pairs


def has_el(smi: str, el: str) -> bool:
    mol = Chem.MolFromSmiles(smi)
    return mol is not None and any(a.GetSymbol() == el for a in mol.GetAtoms())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_fp", type=Path, default=Path("data/raw/casmi2026/train.parquet"))
    ap.add_argument("--nist20_proc_dp", type=Path, default=Path("data/proc/nist20"))
    ap.add_argument("--out_dp", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out_dp.mkdir(parents=True, exist_ok=True)

    rows_fp = args.out_dp / "casmi_halogen_rows.parquet"
    if rows_fp.exists():
        hal = pd.read_parquet(rows_fp)
    else:
        n_rg = pq.ParquetFile(args.train_fp).num_row_groups
        with Pool(args.workers, maxtasksperchild=1) as pool:
            parts = pool.map(score_row_group, [(str(args.train_fp), rg) for rg in range(n_rg)])
        hal = pd.concat(parts, ignore_index=True)
        hal.to_parquet(rows_fp)
    hal["gap_iso"] = hal["cov_iso"] - hal["null_iso"]
    hal["hq"] = (
        hal["inst_class"].isin(["TOF", "ORB"])
        & (hal["mult"] == 1)
        & (hal["prec_ppm"].abs() <= HQ_MAX_PPM)
        & (hal["num_peaks"] >= 3)
        & (hal["cov_iso"] >= HQ_MIN_COV)
        & (hal["gap_iso"] >= HQ_MIN_GAP)
        & hal["base_iso"].fillna(False).astype(bool)
    )
    logger.info("halogen rows %d, hq %d", len(hal), int(hal["hq"].sum()))

    # Quality by library and halogen.
    q = []
    for (lib, el), g in [((lib, el), g[g[el] > 0]) for lib, g in hal.groupby("ingest_lib") for el in ("Br", "Cl")]:
        hr = g[g["inst_class"].isin(["TOF", "ORB"]) & (g["mult"] == 1) & (g["status"] == "ok")]
        q.append({"lib": lib, "el": el, "spectra": len(g), "hr_monomer": len(hr),
                  "cov_mono_med": hr["cov_mono"].median(), "cov_iso_med": hr["cov_iso"].median(),
                  "null_iso_med": hr["null_iso"].median(), "prec_m2_frac": hr["prec_is_m2"].mean(),
                  "hq_spectra": int(g["hq"].sum()), "hq_frac": g["hq"].mean()})
    q = pd.DataFrame(q)
    q.to_csv(args.out_dp / "quality_by_lib.csv", index=False)
    print(q.round(3).to_string(index=False))

    nist = nist20_hr(args.nist20_proc_dp)
    nist_smiles = nist.drop_duplicates("ik14").set_index("ik14")["smiles"].to_dict()
    nist_keys = set(nist_smiles)
    nist_el = {el: {k for k, s in nist_smiles.items() if has_el(s, el)} for el in HALOGENS}
    report = {"nist20_hr": {"molecules": len(nist_keys)}}
    for el in HALOGENS:
        report["nist20_hr"][el] = {"molecules": len(nist_el[el]), "spectra": int(nist["ik14"].isin(nist_el[el]).sum())}

    tiers = {"all": hal, "hq": hal[hal["hq"]]}
    per_lib = []
    for tier, sub in tiers.items():
        for lib, g in sub.groupby("ingest_lib"):
            row = {"tier": tier, "lib": lib}
            for el in HALOGENS:
                c = g[g[el] > 0]
                mols = set(c["inchikey14"])
                row[f"{el}_mols"], row[f"{el}_spec"], row[f"{el}_mols_new_vs_nist20"] = (
                    len(mols), len(c), len(mols - nist_keys))
            per_lib.append(row)
        for name, libs in POOLS.items():
            g = sub if libs is None else sub[sub["ingest_lib"].isin(libs)]
            row = {"tier": tier, "lib": f"POOL:{name}"}
            for el in HALOGENS:
                c = g[g[el] > 0]
                mols = set(c["inchikey14"])
                row[f"{el}_mols"], row[f"{el}_spec"], row[f"{el}_mols_new_vs_nist20"] = (
                    len(mols), len(c), len(mols - nist_keys))
            per_lib.append(row)
    per_lib = pd.DataFrame(per_lib)
    per_lib.to_csv(args.out_dp / "carriers_by_lib.csv", index=False)
    print(per_lib.to_string(index=False))

    # Swap pairs. Destination molecules may be non-halogen-swapped partners (e.g. a Cl partner
    # of a Br source is itself a halogen carrier, so it is in ``hal``).
    pair_rows, pair_table = [], []
    for tier, sub in tiers.items():
        for name, libs in list(POOLS.items()) + [("nist20_hr+msnlib+enveda180", ["pluskal_ms2", "enveda-180"])]:
            g = sub if libs is None else sub[sub["ingest_lib"].isin(libs)]
            smiles = g.drop_duplicates("inchikey14").set_index("inchikey14")["normalized_smiles"].to_dict()
            conds = defaultdict(set)
            for ik, adduct, inst, lib in g[["inchikey14", "adduct", "inst_class", "ingest_lib"]].itertuples(index=False):
                conds[ik].add((adduct, inst))
            with_nist = name.startswith("nist20")
            if with_nist:
                for ik, adduct, inst in nist[["ik14", "prec_type", "inst_type"]].itertuples(index=False):
                    conds[ik].add((adduct, "ORB" if inst in ("FT", "ORB", "ORB_F") else "TOF"))
                smiles = {**nist_smiles, **smiles}
            keys = set(smiles)
            for src, dst in SWAPS:
                carriers = {k: s for k, s in smiles.items() if has_el(s, src)}
                pairs = swap_pairs(carriers, keys, src, dst)
                coarse = {p for p in pairs if conds[p[0]] & conds[p[1]]}
                new = {p for p in pairs if not (p[0] in nist_keys and p[1] in nist_keys)}
                pair_rows.append({"tier": tier, "pool": name, "swap": f"{src}->{dst}", "src_carriers": len(carriers),
                                  "pairs": len(pairs), "coarse_compatible": len(coarse),
                                  "pairs_not_both_in_nist20": len(new)})
                for a, b in sorted(pairs):
                    pair_table.append({"tier": tier, "pool": name, "swap": f"{src}->{dst}", "src_ik14": a,
                                       "dst_ik14": b, "coarse": (a, b) in coarse, "src_in_nist20": a in nist_keys,
                                       "dst_in_nist20": b in nist_keys, "src_smiles": smiles[a],
                                       "dst_smiles": smiles[b]})
            logger.info("pairs: tier %s pool %s", tier, name)
    pair_rows = pd.DataFrame(pair_rows)
    pair_rows.to_csv(args.out_dp / "swap_pairs_summary.csv", index=False)
    pd.DataFrame(pair_table).to_csv(args.out_dp / "swap_pairs.csv", index=False)
    print(pair_rows.to_string(index=False))
    report["hq_rule"] = {"inst": ["TOF", "ORB"], "monomer": True, "max_abs_ppm_mono_or_m2": HQ_MAX_PPM,
                         "min_peaks": 3, "min_cov_iso_tight": HQ_MIN_COV, "min_cov_minus_null": HQ_MIN_GAP,
                         "base_peak_explained": True}
    (args.out_dp / "report.json").write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))


if __name__ == "__main__":
    main()
