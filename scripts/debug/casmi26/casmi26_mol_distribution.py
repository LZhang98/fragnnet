"""Molecule-distribution bias in CASMI 2026 train against the test proxies.

The hidden CASMI test is natural products; train is two-thirds Enamine synthetics (Enveda-180).
This script describes every train molecule (and NIST20/NIST23 HR, for reference) and measures how
well train covers two test proxies:

- ``enveda-np-examples`` (250 timsTOF natural products and drugs), the closest proxy to the
  hidden test;
- the placeholder ``test.parquet`` molecules (400 Enamine compounds, all also in train).

Per molecule (2D InChIKey block from RDKit): Ertl NP-likeness (RDKit Contrib ``NP_Score``),
Fsp3, heavy atoms, exact mass, ring and aromatic-ring counts, potential stereocentres (SMILES are
stereo-free), O/C and N/C ratios, halogen/S/P flags, a pyranose/furanose sugar flag, and the
number of spectra per library.

Coverage: for every proxy molecule, the maximum Tanimoto similarity (Morgan radius 2, 2048 bits)
to train molecules **with that proxy's own molecules removed**, i.e. as if the proxy were a
molecule-disjoint test set. The same is computed per train library as a reference.

Outputs in ``--out_dp``: ``molecules.parquet``, ``nn_similarity.parquet``, ``summary.json``.

Usage:
    PYTHONPATH=src python scripts/debug/casmi26/casmi26_mol_distribution.py \
        --out_dp output/casmi26_mol_distribution_2026_10_06 --workers 10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem, DataStructs, RDConfig, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator, rdMolDescriptors

# RDConfig.RDContribDir can point at a stale install; the conda rdkit ships Contrib in the package.
_NP_DIRS = [os.path.join(os.path.dirname(rdkit.__file__), "Contrib", "NP_Score"),
            os.path.join(RDConfig.RDContribDir, "NP_Score")]
_NP_DIR = next((d for d in _NP_DIRS if os.path.isfile(os.path.join(d, "npscorer.py"))), None)
if _NP_DIR is None:
    raise ImportError(f"RDKit Contrib NP_Score not found in {_NP_DIRS}")
sys.path.append(_NP_DIR)
import npscorer  # noqa: E402  (RDKit Contrib module, not a package)

logger = logging.getLogger(__name__)

SUGAR = Chem.MolFromSmarts("[OX2;R1]1[CX4;R1][CX4;R1][CX4;R1][CX4;R1][CX4;R1]1")  # pyranose ring
FURANOSE = Chem.MolFromSmarts("[OX2;R1]1[CX4;R1][CX4;R1][CX4;R1][CX4;R1]1")
_NP_MODEL = None
_FPGEN = None


def _init_worker() -> None:
    global _NP_MODEL, _FPGEN
    RDLogger.DisableLog("rdApp.*")
    _NP_MODEL = npscorer.readNPModel()
    _FPGEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)


def describe(smiles: str) -> dict:
    """Descriptors and a packed Morgan fingerprint for one SMILES."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"smiles": smiles, "ik14": None}
    counts: dict[str, int] = {}
    for atom in mol.GetAtoms():
        counts[atom.GetSymbol()] = counts.get(atom.GetSymbol(), 0) + 1
    n_c = counts.get("C", 0)
    fp = _FPGEN.GetFingerprint(mol)
    return {
        "smiles": smiles,
        "ik14": Chem.MolToInchiKey(mol)[:14],
        "np_score": npscorer.scoreMol(mol, _NP_MODEL),
        "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        "heavy_atoms": mol.GetNumHeavyAtoms(),
        "exact_mass": Descriptors.ExactMolWt(mol),
        "rings": rdMolDescriptors.CalcNumRings(mol),
        "aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "stereocentres": len(Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)),
        "o_per_c": counts.get("O", 0) / n_c if n_c else np.nan,
        "n_per_c": counts.get("N", 0) / n_c if n_c else np.nan,
        "has_halogen": any(counts.get(x, 0) for x in ("F", "Cl", "Br", "I")),
        "has_S": counts.get("S", 0) > 0,
        "has_P": counts.get("P", 0) > 0,
        "has_sugar": mol.HasSubstructMatch(SUGAR) or mol.HasSubstructMatch(FURANOSE),
        "fp": DataStructs.BitVectToText(fp),
    }


def max_similarity(task: tuple[list[str], int, int]) -> np.ndarray:
    """Max Tanimoto of each query fingerprint (by text) to the reference set held in a global."""
    queries, start, _ = task
    out = np.empty(len(queries))
    for i, text in enumerate(queries):
        fp = DataStructs.CreateFromBitString(text)
        out[i] = max(DataStructs.BulkTanimotoSimilarity(fp, _REF)) if _REF else np.nan
    return out


_REF: list = []


def nn_similarity(query_fps: list[str], ref_fps: list[str], workers: int) -> np.ndarray:
    """Max Tanimoto similarity from each query to the reference fingerprints."""
    global _REF
    _REF = [DataStructs.CreateFromBitString(t) for t in ref_fps]
    chunks = [query_fps[i : i + 50] for i in range(0, len(query_fps), 50)]
    with Pool(workers) as pool:  # forked after _REF is set, so workers share it
        parts = pool.map(max_similarity, [(c, 0, 0) for c in chunks])
    return np.concatenate(parts) if parts else np.array([])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--train_fp", type=Path, default=Path("data/raw/casmi2026/train.parquet"))
    parser.add_argument("--test_fp", type=Path, default=Path("data/raw/casmi2026/test.parquet"))
    parser.add_argument("--nist20_proc_dp", type=Path, default=Path("data/proc/nist20"))
    parser.add_argument("--nist23_proc_dp", type=Path, default=Path("data/proc/nist23"))
    parser.add_argument("--out_dp", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--library_sample", type=int, default=500,
                        help="Molecules sampled per train library for the NN-similarity reference")
    parser.add_argument("--seed", type=int, default=20261006)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.out_dp.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet(args.train_fp, columns=["ingest_lib", "normalized_smiles", "inchikey14"])
    nist_smiles = []
    for proc_dp, prefix in ((args.nist20_proc_dp, "nist20_hr"), (args.nist23_proc_dp, "nist23_hr")):
        spec = pd.read_pickle(proc_dp / "spec_df.pkl")[["dset", "mol_id"]]
        mol_ids = spec.loc[spec["dset"].astype(str).str.startswith(prefix), "mol_id"].unique()
        mol = pd.read_pickle(proc_dp / "mol_df.pkl")[["mol_id", "smiles"]]
        nist_smiles.append(mol.loc[mol["mol_id"].isin(mol_ids), "smiles"])
    nist_smiles = pd.unique(pd.concat(nist_smiles))

    all_smiles = pd.unique(pd.concat([train["normalized_smiles"], pd.Series(nist_smiles)]))
    with Pool(args.workers, initializer=_init_worker) as pool:
        desc = pd.DataFrame(pool.map(describe, all_smiles, chunksize=1000))
    logger.info("described %d SMILES, %d unparsable", len(desc), desc["ik14"].isna().sum())
    # Several SMILES can share one 2D key; map every SMILES before collapsing to one row per key.
    desc = desc.dropna(subset=["ik14"])
    smiles_to_ik = desc.set_index("smiles")["ik14"]
    desc = desc.drop_duplicates("ik14").set_index("ik14")

    train["ik14"] = train["normalized_smiles"].map(smiles_to_ik)
    if train["ik14"].isna().any():
        raise ValueError("some train SMILES have no InChIKey")
    spectra = train.groupby(["ik14", "ingest_lib"]).size().unstack(fill_value=0)
    mols = desc.drop(columns=["smiles"]).join(spectra.add_prefix("spectra__"), how="left")
    lib_cols = [f"spectra__{lib}" for lib in spectra.columns]
    mols[lib_cols] = mols[lib_cols].fillna(0).astype(int)
    mols["in_train"] = mols[lib_cols].sum(axis=1) > 0
    mols["in_nist_hr"] = mols.index.isin(
        pd.Series(nist_smiles).map(smiles_to_ik)
    )
    mols.drop(columns=["fp"]).rename_axis("ik14").reset_index().to_parquet(args.out_dp / "molecules.parquet", index=False)

    # Placeholder test molecules: the 400 are Enveda-180 molecules whose spectra are copied
    # verbatim into train (docs/findings/casmi2026_dataset_2026_10_05.md section 3). Recover them by
    # (precursor m/z, base-peak intensity, peak count, adduct). The adduct is needed: Enveda-180 files
    # some co-isolated acquisitions under two molecules with different adducts (e.g. m/z 379.152).
    test = pd.read_parquet(args.test_fp, columns=["ms2_mzs", "precursor_mz", "base_peak_intensity", "adduct"])
    env = pd.read_parquet(
        args.train_fp,
        columns=["ingest_lib", "precursor_mz", "base_peak_intensity", "num_peaks", "adduct",
                 "normalized_smiles"],
        filters=[("ingest_lib", "==", "enveda-180")],
    )
    env_key: dict[tuple, set[str]] = {}
    for p, b, n, a, smi in zip(env.precursor_mz, env.base_peak_intensity, env.num_peaks, env.adduct,
                               env.normalized_smiles):
        env_key.setdefault((round(p, 4), round(b, 1), int(n), a), set()).add(smi)
    test_smiles: set[str] = set()
    for p, b, mzs, a in zip(test.precursor_mz, test.base_peak_intensity, test.ms2_mzs, test.adduct):
        hits = env_key.get((round(p, 4), round(b, 1), len(mzs), a))
        if not hits or len(hits) != 1:
            raise ValueError(f"placeholder spectrum at m/z {p} does not map to exactly one molecule")
        test_smiles |= hits
    placeholder_ik = {smiles_to_ik[s] for s in test_smiles if s in smiles_to_ik}
    logger.info("placeholder test molecules recovered: %d", len(placeholder_ik))

    # Nearest-neighbour coverage of each proxy by train minus the proxy's own molecules.
    rng = np.random.default_rng(args.seed)
    train_ik = mols.index[mols["in_train"]]
    np_ik = set(mols.index[mols["spectra__enveda-np-examples"] > 0])
    queries = {"enveda-np-examples": sorted(np_ik), "placeholder test (Enamine)": sorted(placeholder_ik)}
    for lib in [c.removeprefix("spectra__") for c in lib_cols if c != "spectra__enveda-np-examples"]:
        members = mols.index[mols[f"spectra__{lib}"] > 0]
        take = min(args.library_sample, len(members))
        queries[f"library: {lib}"] = sorted(rng.choice(members, size=take, replace=False))
    rows = []
    for name, q in queries.items():
        excluded = set(q)
        ref = [ik for ik in train_ik if ik not in excluded]
        sims = nn_similarity(mols.loc[q, "fp"].tolist(), mols.loc[ref, "fp"].tolist(), args.workers)
        rows.extend({"query_set": name, "ik14": ik, "max_tanimoto": s} for ik, s in zip(q, sims))
        logger.info("%s: n=%d median NN Tanimoto %.3f", name, len(q), float(np.median(sims)))
    nn = pd.DataFrame(rows)
    nn.to_parquet(args.out_dp / "nn_similarity.parquet", index=False)

    # Summary: descriptor medians per group, molecule- and spectrum-weighted shares.
    groups = {
        "train, all": mols["in_train"],
        "train, enveda-180": mols["spectra__enveda-180"] > 0,
        "train, not enveda-180": mols["in_train"] & (mols["spectra__enveda-180"] == 0),
        "enveda-np-examples": mols["spectra__enveda-np-examples"] > 0,
        "NIST20/23 HR": mols["in_nist_hr"],
    }
    desc_cols = ["np_score", "fsp3", "heavy_atoms", "exact_mass", "rings", "aromatic_rings",
                 "stereocentres", "o_per_c", "n_per_c", "has_halogen", "has_S", "has_sugar"]
    summary = {
        "medians": {g: mols.loc[m, desc_cols].astype(float).median().round(3).to_dict() for g, m in groups.items()},
        "means": {g: mols.loc[m, desc_cols].astype(float).mean().round(3).to_dict() for g, m in groups.items()},
        "np_score_gt_0_share": {g: round(float((mols.loc[m, "np_score"] > 0).mean()), 4) for g, m in groups.items()},
        "molecule_share_by_library": (mols.loc[mols.in_train, lib_cols] > 0).mean().round(4).to_dict(),
        "spectrum_share_by_library": (train["ingest_lib"].value_counts(normalize=True)).round(4).to_dict(),
        "nn_tanimoto": nn.groupby("query_set")["max_tanimoto"].describe(percentiles=[0.1, 0.5, 0.9]).round(3).to_dict("index"),
        "nn_tanimoto_eq_1_share": nn.groupby("query_set")["max_tanimoto"].apply(lambda s: round(float((s >= 0.999).mean()), 4)).to_dict(),
        "nn_tanimoto_ge_0p7_share": nn.groupby("query_set")["max_tanimoto"].apply(lambda s: round(float((s >= 0.7).mean()), 4)).to_dict(),
        "placeholder_test_molecules": len(placeholder_ik),
    }
    (args.out_dp / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    logger.info("done")


if __name__ == "__main__":
    main()
