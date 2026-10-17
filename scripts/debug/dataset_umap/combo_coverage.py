"""Coverage of a candidate open training set: Spectraverse + MSnLib + Enveda-180 (with and without
Spectraverse's MS-DIAL subset).

Reports molecules and Br/Cl/I carriers per combination and pairwise overlaps, then the fraction of
HMDB (the ``hmdb_fit.py`` sample) and of a seeded NIST20 ∪ NIST23 HR sample whose nearest molecule
(Morgan r=2, 2048 bits, Tanimoto) in the combination is ≥ 0.7, identical, or < 0.4. A Spectraverse
molecule is "MS-DIAL only" when every one of its spectra has source MS-DIAL.

Usage (from the repository root, ``umap-viz`` env):
    python -I scripts/debug/dataset_umap/combo_coverage.py \
        --dp output/dataset_umap_hmdb_2026_10_17 --n_nist 5000 --workers 6
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))
from open_vs_nist import nearest  # noqa: E402

RDLogger.DisableLog("rdApp.*")


def ik14(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    return Chem.MolToInchiKey(mol)[:14] if mol is not None else None


def cov(nn: np.ndarray) -> dict:
    return {"ge_0.7": round(float(np.mean(nn >= 0.7)), 3), "identical": round(float(np.mean(nn >= 0.999)), 3),
            "lt_0.4": round(float(np.mean(nn < 0.4)), 3), "median": round(float(np.median(nn)), 3)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dp", type=Path, required=True)
    ap.add_argument("--n_nist", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=20261017)
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()

    mols = pd.read_parquet(args.dp / "mols.parquet")
    src = pd.read_parquet(args.dp / "spectraverse_mol_sources.parquet")
    msdial_only = src.groupby("smiles")["source"].agg(lambda s: set(s) == {"MS-DIAL"})
    spv = mols[mols["dataset"] == "Spectraverse"].copy()
    spv_smiles_only = set(msdial_only[msdial_only].index)
    # mols.parquet keeps one SMILES per 2D key; map MS-DIAL-only status through the 2D key.
    src_keys = pd.DataFrame({"smiles": msdial_only.index, "msdial_only": msdial_only.values})
    src_keys["ik14"] = [ik14(s) for s in src_keys["smiles"]]
    key_only = src_keys.groupby("ik14")["msdial_only"].all()
    spv["msdial_only"] = spv["ik14"].map(key_only).fillna(False).astype(bool)

    parts = {
        "Spectraverse": spv,
        "Spectraverse w/o MS-DIAL": spv[~spv["msdial_only"]],
        "MSnLib": mols[mols["dataset"] == "MSnLib"],
        "Enveda-180": mols[mols["dataset"] == "Enveda-180"],
    }
    combos = {
        "SPV+MSnLib+Enveda": ["Spectraverse", "MSnLib", "Enveda-180"],
        "SPV(no MS-DIAL)+MSnLib+Enveda": ["Spectraverse w/o MS-DIAL", "MSnLib", "Enveda-180"],
        "SPV+MSnLib": ["Spectraverse", "MSnLib"],
    }
    report = {"msdial_only_spv_molecules": int(spv["msdial_only"].sum()),
              "msdial_only_smiles": len(spv_smiles_only), "sizes": {}, "overlap": {}}
    keys = {k: set(v["ik14"]) for k, v in parts.items()}
    for a in ("Spectraverse", "MSnLib", "Enveda-180"):
        for b in ("Spectraverse", "MSnLib", "Enveda-180"):
            if a < b:
                report["overlap"][f"{a} & {b}"] = len(keys[a] & keys[b])
    nist = mols[mols["dataset"].isin(["NIST20 HR", "NIST23 HR"])].drop_duplicates("ik14")
    for name, members in {**{k: [k] for k in parts}, **combos}.items():
        u = pd.concat([parts[m] for m in members]).drop_duplicates("ik14")
        report["sizes"][name] = {"molecules": len(u), "Br": int(u["Br"].sum()), "Cl": int(u["Cl"].sum()),
                                 "I": int(u["I"].sum()), "in_NIST20∪23": int(u["ik14"].isin(set(nist["ik14"])).sum())}

    hmdb = pd.read_parquet(args.dp / "hmdb_nn.parquet")
    nn_spv_nomsdial = nearest(hmdb["smiles"].tolist(), parts["Spectraverse w/o MS-DIAL"]["smiles"].tolist(),
                              args.workers)
    hmdb_nn = {
        "Spectraverse": hmdb["nn_Spectraverse"].to_numpy(),
        "Spectraverse w/o MS-DIAL": nn_spv_nomsdial,
        "MSnLib": hmdb["nn_MSnLib"].to_numpy(),
        "Enveda-180": hmdb["nn_Enveda-180"].to_numpy(),
    }
    hmdb_nn["NIST20∪23"] = hmdb["nn_NIST20∪23"].to_numpy()
    for name, members in combos.items():
        hmdb_nn[name] = np.max([hmdb_nn[m] for m in members], axis=0)
    hmdb_nn["SPV+MSnLib+Enveda+NIST"] = np.maximum(hmdb_nn["SPV+MSnLib+Enveda"], hmdb_nn["NIST20∪23"])
    lc = hmdb["long_chain"].to_numpy()
    report["hmdb"] = {k: {"all": cov(v), "long_chain": cov(v[lc]), "other": cov(v[~lc])} for k, v in hmdb_nn.items()}

    nist_s = nist.sample(n=min(args.n_nist, len(nist)), random_state=args.seed)
    nist_nn = {m: nearest(nist_s["smiles"].tolist(), parts[m]["smiles"].tolist(), args.workers) for m in parts}
    for name, members in combos.items():
        nist_nn[name] = np.max([nist_nn[m] for m in members], axis=0)
    report["nist_sample"] = {"n": len(nist_s), **{k: cov(v) for k, v in nist_nn.items()}}
    for el in ("Br",):
        mask = nist_s[el].to_numpy()
        report[f"nist_sample_{el}"] = {"n": int(mask.sum()), **{k: cov(v[mask]) for k, v in nist_nn.items()}}

    (args.dp / "combo_coverage.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
