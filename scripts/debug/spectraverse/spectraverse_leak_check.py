"""Check a Spectraverse CV split for structure leaks between train, val and test.

For every ``<split-dp>/cv{k}`` the split files are read as training reads them (by ``spec_id``),
and each pair of splits is checked for a shared molecule under five identity keys:

1. ``mol_id`` (one per canonical SMILES in ``mol_df.pkl``).
2. ``inchikey_s``: RDKit 2D InChIKey (first block) from ``mol_df.pkl``.
3. ``mgf_ik14``: first block of the MGF's own ``INCHIKEY`` field (step-1 CSV).
4. ``parent_ik14``: 2D InChIKey of the standardized parent (cleanup, largest fragment,
   uncharge, canonical tautomer), which merges salts, charge forms and tautomers.
5. ``stereo-free canonical SMILES`` of the parent, a second spelling of key 4.

Then, per cv, the maximum ECFP6 (Morgan radius 3, 2048 bits) Tanimoto of every val and test
molecule to any train molecule is summarized, with the share of val/test spectra whose molecule
has a train partner at 1.0; cross-fold pairs at 1.0 are listed. That is a structural-similarity
read, not a leak test: the Spectraverse folds are MCES-based, not Tanimoto-based.

    python scripts/debug/spectraverse/spectraverse_leak_check.py \
        --proc-dp data/proc/spectraverse_1_0_3 \
        --df-fp data/df_spectraverse_1_0_3/spectraverse_df.csv \
        --split-dp data/split/spectraverse_1_0_3_benchmark \
        --out-dp output/spv_1_0_3/leak_check

Exits 1 if any identity key is shared between two splits of any cv.
"""

from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize

RDLogger.DisableLog("rdApp.*")

SPLITS = ("train", "val", "test")
PAIRS = (("train", "val"), ("train", "test"), ("val", "test"))
KEYS = ("mol_id", "inchikey_s", "mgf_ik14", "parent_ik14", "parent_smiles")
SIM_BINS = (1.0, 0.95, 0.9, 0.8, 0.7)


def parent_keys(smiles: str) -> tuple[str, str]:
    """Return (2D InChIKey first block, stereo-free canonical SMILES) of the standardized parent.

    Args:
        smiles: Molecule SMILES from ``mol_df.pkl``.

    Returns:
        The two keys; both empty strings if RDKit cannot parse or standardize the molecule.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "", ""
    try:
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.FragmentParent(mol)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        mol = rdMolStandardize.TautomerEnumerator().Canonicalize(mol)
    except Exception:  # noqa: BLE001 - a failed standardization is reported, not fatal
        return "", ""
    ik = Chem.MolToInchiKey(mol)
    Chem.RemoveStereochemistry(mol)
    return ik[:14], Chem.MolToSmiles(mol)


_FPS: list = []
_FOLD_OF: np.ndarray = np.empty(0, dtype=int)


def _init(fps: list, fold_of: np.ndarray) -> None:
    global _FPS, _FOLD_OF
    _FPS, _FOLD_OF = fps, fold_of


def max_sim_by_fold(i: int) -> np.ndarray:
    """Max Tanimoto of molecule ``i`` to the molecules of each fold 1..n (itself excluded)."""
    sims = np.asarray(DataStructs.BulkTanimotoSimilarity(_FPS[i], _FPS))
    sims[i] = -1.0
    out = np.full(int(_FOLD_OF.max()) + 1, -1.0)
    np.maximum.at(out, _FOLD_OF, sims)
    return out[1:]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--proc-dp", type=Path, default=Path("data/proc/spectraverse_1_0_3"))
    parser.add_argument(
        "--df-fp", type=Path, default=Path("data/df_spectraverse_1_0_3/spectraverse_df.csv")
    )
    parser.add_argument(
        "--split-dp", type=Path, default=Path("data/split/spectraverse_1_0_3_benchmark")
    )
    parser.add_argument("--out-dp", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    args.out_dp.mkdir(parents=True, exist_ok=True)

    spec = pd.read_pickle(args.proc_dp / "spec_df.pkl")[["spec_id", "mol_id", "dset_spec_id"]]
    mol = pd.read_pickle(args.proc_dp / "mol_df.pkl")[["mol_id", "smiles", "inchikey_s"]]
    raw = pd.read_csv(args.df_fp, usecols=["dset_spec_id", "inchikey"])
    spec["dset_spec_id"] = spec["dset_spec_id"].astype(int)
    spec = spec.merge(raw, on="dset_spec_id", how="left", validate="one_to_one")
    if spec["inchikey"].isna().any():
        raise ValueError(f"{spec['inchikey'].isna().sum()} spectra lack an MGF INCHIKEY")
    spec["mgf_ik14"] = spec["inchikey"].str[:14]

    with Pool(args.workers) as pool:
        parents = pool.map(parent_keys, mol["smiles"].tolist(), chunksize=256)
    mol["parent_ik14"] = [p[0] for p in parents]
    mol["parent_smiles"] = [p[1] for p in parents]
    n_fail = int((mol["parent_ik14"] == "").sum())
    spec = spec.merge(mol, on="mol_id", how="left", validate="many_to_one")

    report: dict = {"n_spectra": len(spec), "n_mols": len(mol), "parent_failures": n_fail}
    leaks = []
    cv_dirs = sorted(args.split_dp.glob("cv*"), key=lambda p: int(p.name[2:]))
    split_of_cv = {}
    for cv_dp in cv_dirs:
        ids = {s: pd.read_csv(cv_dp / f"{s}_ids.csv")["spec_id"] for s in SPLITS}
        split = pd.Series(index=spec["spec_id"], dtype=object)
        for s in SPLITS:
            split.loc[ids[s].to_numpy()] = s
        if split.isna().any() or sum(len(v) for v in ids.values()) != len(spec):
            raise ValueError(f"{cv_dp}: splits do not cover every spectrum exactly once")
        sp = spec.assign(split=split.loc[spec["spec_id"]].to_numpy())
        split_of_cv[cv_dp.name] = sp.groupby("mol_id")["split"].first()
        cv_report = {}
        for key in KEYS:
            vals = sp[sp[key] != ""] if sp[key].dtype == object else sp
            by_split = {s: set(vals.loc[vals["split"] == s, key]) for s in SPLITS}
            for a, b in PAIRS:
                shared = by_split[a] & by_split[b]
                cv_report[f"{key}:{a}-{b}"] = len(shared)
                for k in sorted(shared, key=str)[:20]:
                    rows = vals[vals[key] == k]
                    leaks.append(
                        {
                            "cv": cv_dp.name,
                            "key": key,
                            "value": k,
                            "pair": f"{a}-{b}",
                            "smiles": sorted(set(rows["smiles"])),
                            "splits": sorted(set(rows["split"])),
                        }
                    )
        report[cv_dp.name] = cv_report
        print(cv_dp.name, {k: v for k, v in cv_report.items() if v} or "no shared keys")

    # ECFP6 near-identity: per-molecule max similarity to each MGF fold, then per cv.
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=3, fpSize=2048)
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in mol["smiles"]]
    fold_csv = args.df_fp.parent / "spectraverse_fold.csv"
    fold = pd.read_csv(fold_csv, usecols=["dset_spec_id", "fold"])
    mol_fold = spec.merge(fold, on="dset_spec_id").groupby("mol_id")["fold"].agg(set)
    if (mol_fold.map(len) > 1).any():
        raise ValueError("a mol_id spans several MGF folds")
    fold_of = mol["mol_id"].map(mol_fold.map(min)).to_numpy(dtype=int)
    with Pool(args.workers, initializer=_init, initargs=(fps, fold_of)) as pool:
        best = np.stack(pool.map(max_sim_by_fold, range(len(mol)), chunksize=256))  # (n_mols, n_folds)
    mol_idx = pd.Series(np.arange(len(mol)), index=mol["mol_id"])
    n_spec_of_mol = spec.groupby("mol_id").size()
    sim_rows = []
    for name, mol_split in split_of_cv.items():
        train_folds = sorted({fold_of[mol_idx[m]] for m in mol_split.index[mol_split == "train"]})
        to_train = best[:, [f - 1 for f in train_folds]].max(axis=1)
        for s in ("val", "test"):
            sel = mol_idx[mol_split.index[mol_split == s]].to_numpy()
            v = to_train[sel]
            row = {"cv": name, "split": s, "n_mols": len(sel), "median": float(np.median(v))}
            row.update({f">={t}": int((v >= t).sum()) for t in SIM_BINS})
            n_spec = n_spec_of_mol.loc[mol_split.index[mol_split == s]].to_numpy()
            row["n_spectra"] = int(n_spec.sum())
            row["spectra_with_1.0_partner"] = int(n_spec[v >= 1.0].sum())
            sim_rows.append(row)
    sim_df = pd.DataFrame(sim_rows)
    sim_df.to_csv(args.out_dp / "max_tanimoto_to_train.csv", index=False)
    print(sim_df.to_string(index=False))

    # list the cross-fold molecule pairs at Tanimoto 1.0
    ident = []
    for i in np.where((best >= 1.0).any(axis=1))[0]:
        sims = np.asarray(DataStructs.BulkTanimotoSimilarity(fps[i], fps))
        for j in np.where(sims >= 1.0)[0]:
            if j > i and fold_of[j] != fold_of[i]:
                ident.append(
                    {
                        "mol_id_a": int(mol["mol_id"].iat[i]),
                        "fold_a": int(fold_of[i]),
                        "smiles_a": mol["smiles"].iat[i],
                        "mol_id_b": int(mol["mol_id"].iat[j]),
                        "fold_b": int(fold_of[j]),
                        "smiles_b": mol["smiles"].iat[j],
                    }
                )
    pd.DataFrame(ident).to_csv(args.out_dp / "cross_fold_tanimoto_1.csv", index=False)
    report["cross_fold_tanimoto_1_pairs"] = len(ident)
    print(f"cross-fold molecule pairs at ECFP6 Tanimoto 1.0: {len(ident)}")

    (args.out_dp / "leaks.json").write_text(json.dumps(leaks, indent=2))
    (args.out_dp / "report.json").write_text(json.dumps(report, indent=2))
    n_leak = sum(sum(report[name].values()) for name in split_of_cv)
    print(f"parent standardization failures: {n_fail}; shared identity keys: {n_leak}")
    return 1 if n_leak else 0


if __name__ == "__main__":
    raise SystemExit(main())
