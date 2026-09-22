"""Is charge retention decided by a shortcut, and would a mechanistic feature fix it?

A shortcut is a feature whose relationship with the target holds inside the training
distribution and breaks outside it, so the diagnostic is not whether a feature is available or
predictive -- it is how much of its skill SURVIVES a harder split.

Target: for a cleavage that produced an observed ion, which half keeps the charge. Built from
NIST's own peak annotations with no model in the loop. For each annotated product A of an
[M+H]+ spectrum the complementary ion B is computed exactly from the formulas
(B = precursor + proton - A) and the spectrum is searched for B at its own m/z, allowing one
hydrogen transfer either way; B scores ZERO when absent, so outright wins count. Restricted to
precursors with exactly one nitrogen, which holds composition fixed while letting the
BASICITY CLASS of that nitrogen -- a structural property no element count can express -- vary.

Two earlier designs were biased and are recorded here so they are not repeated. Treating every
annotated product as "the keeper" is circular: the annotated ion is the keeper by construction.
Requiring BOTH halves to be annotated is worse, because it conditions on splits where neither
half dominated, which drives every rate to 0.5.

Three feature groups are compared under three splits: a random molecule-grouped split, the
production InChIKey split, and the Murcko scaffold split.

Results and their interpretation:
``docs/findings/feature_headroom_2026_09_22.md`` (Shortcut learning section).

Usage:
    python scripts/debug/debug_charge_retention_shortcut.py
"""

import os

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score

from fragnnet.utils.formula_utils import parse_formula

RDLogger.DisableLog("rdApp.*")
PROTON, H_MASS = 1.007276, 1.0078250319
MASSES = {"C": 12.0, "H": H_MASS, "N": 14.0030740052, "O": 15.9949146221, "S": 31.97207069,
          "P": 30.97376151, "F": 18.99840320, "Cl": 34.96885271, "Br": 78.9183376,
          "I": 126.904473, "Si": 27.9769265327, "Se": 79.9165196, "B": 11.0093055}
ELEMENTS = ("C", "N", "O", "S", "P", "F", "Cl", "Br", "I")
N_CLASSES = [
    ("guanidine/amidine", Chem.MolFromSmarts("[NX3][CX3]=[NX2,NX3+]")),
    ("nitro", Chem.MolFromSmarts("[$([NX3](=O)=O),$([NX3+](=O)[O-])]")),
    ("nitrile", Chem.MolFromSmarts("[NX1]#[CX2]")),
    ("amide/sulfonamide", Chem.MolFromSmarts("[NX3][CX3,SX4]=[OX1]")),
    ("aromatic N", Chem.MolFromSmarts("[n]")),
    ("aliphatic amine", Chem.MolFromSmarts("[NX3;H2,H1,H0;!$(N=*);!$(N#*)]")),
]
CLASS_INDEX = {name: i for i, (name, _) in enumerate(N_CLASSES)}
CLASS_INDEX["other"] = len(N_CLASSES)
CACHE = os.environ.get("CHARGE_SHORTCUT_CACHE", "")


def classify(mol):
    for name, patt in N_CLASSES:
        if patt is not None and mol.HasSubstructMatch(patt):
            return name
    return "other"


def build(num_spectra=60000, tolerance=0.01):
    cache: dict = {}

    def parsed(formula):
        if formula not in cache:
            try:
                cache[formula] = parse_formula(formula)
            except Exception:
                cache[formula] = None
        return cache[formula]

    mol_df = pd.read_pickle("data/proc/nist20/mol_df.pkl")
    n_class, n_count = {}, {}
    for mol_id, mol in zip(mol_df["mol_id"], mol_df["mol"]):
        if mol is None:
            continue
        count = sum(1 for a in mol.GetAtoms() if a.GetSymbol() == "N")
        n_count[mol_id] = count
        if count == 1:
            n_class[mol_id] = classify(mol)
    del mol_df

    ann = pd.read_pickle("data/proc/nist20/ann_df.pkl")
    ann = ann[(ann["prec_type"] == "[M+H]+") & ann["mol_id"].map(n_count).eq(1)]
    rng = np.random.default_rng(0)
    if len(ann) > num_spectra:
        ann = ann.iloc[rng.choice(len(ann), size=num_spectra, replace=False)]
    spec = pd.read_pickle("data/proc/nist20/spec_df.pkl")
    peaks_by_id = dict(zip(spec["spec_id"].astype(str), spec["peaks"]))
    del spec

    rows, labels, groups = [], [], []
    for spec_id, mol_id, precursor, products, mzs, isotopes in zip(
        ann["spec_id"].astype(str), ann["mol_id"], ann["formula"], ann["ann_products"],
        ann["ann_peak_mzs"], ann["ann_isotopes"],
    ):
        peaks = peaks_by_id.get(spec_id)
        if not peaks:
            continue
        mz_array = np.array([p[0] for p in peaks], float)
        int_array = np.array([p[1] for p in peaks], float)
        order = np.argsort(mz_array)
        mz_array, int_array = mz_array[order], int_array[order]
        prec = parsed(precursor)
        if prec is None:
            continue

        def intensity_at(mass):
            lo = np.searchsorted(mz_array, mass - tolerance)
            hi = np.searchsorted(mz_array, mass + tolerance)
            return float(int_array[lo:hi].max()) if hi > lo else 0.0

        label_index = CLASS_INDEX[n_class.get(mol_id, "other")]
        seen = set()
        for product, mz, isotope in zip(products, mzs, isotopes):
            if isotope != 0:
                continue
            a = parsed(product)
            if a is None:
                continue
            b = {e: prec.get(e, 0) + (1 if e == "H" else 0) - a.get(e, 0)
                 for e in set(prec) | set(a) | {"H"}}
            if any(v < 0 for v in b.values()):
                continue
            heavy_a = sum(v for k, v in a.items() if k != "H")
            heavy_b = sum(v for k, v in b.items() if k != "H")
            if heavy_a == 0 or heavy_b == 0:
                continue
            key = tuple(sorted(b.items()))
            if key in seen:
                continue
            seen.add(key)
            int_a = intensity_at(float(mz))
            if int_a == 0.0:
                int_a = float(int_array[np.argmin(np.abs(mz_array - float(mz)))])
            mass_b = sum(MASSES[e] * v for e, v in b.items() if e in MASSES)
            int_b = max(intensity_at(mass_b + PROTON - H_MASS + s * H_MASS) for s in (-1, 0, 1))
            if int_a == int_b:
                continue
            mass_a = sum(MASSES[e] * v for e, v in a.items() if e in MASSES)
            rdbe_a = 1 + a.get("C", 0) - 0.5 * a.get("H", 0) + 0.5 * a.get("N", 0)
            rdbe_b = 1 + b.get("C", 0) - 0.5 * b.get("H", 0) + 0.5 * b.get("N", 0)
            rows.append(
                [heavy_a, heavy_b, heavy_a - heavy_b, mass_a, mass_b,
                 mass_a / (mass_a + mass_b), rdbe_a, rdbe_b]
                + [a.get(e, 0) for e in ELEMENTS] + [b.get(e, 0) for e in ELEMENTS]
                + [float(a.get("N", 0) > 0), float(label_index)]
            )
            labels.append(int(int_a > int_b))
            groups.append(mol_id)

    X = np.asarray(rows, np.float32)
    y = np.asarray(labels)
    g = np.asarray(groups)
    if CACHE:
        np.savez_compressed(CACHE, X=X, y=y, g=g)
    return X, y, g


def load_split(name):
    base = f"data/split/{name}"
    out = {}
    for part in ("train", "val"):
        ids = pd.read_csv(f"{base}/{part}_ids.csv")["mol_id"].unique()
        out[part] = set(ids.tolist())
    return out


def main():
    if CACHE and os.path.exists(CACHE):
        data = np.load(CACHE)
        X, y, g = data["X"], data["y"], data["g"]
    else:
        X, y, g = build()
    print(f"splits={len(y):,}  molecules={len(np.unique(g)):,}  base rate={y.mean():.3f}\n")

    groups_def = {
        "size and mass only (the shortcut)": [0, 1, 2, 3, 4, 5],
        "composition of both halves": list(range(X.shape[1] - 1)),
        "composition + nitrogen class": list(range(X.shape[1])),
    }

    regimes = {}
    rng = np.random.default_rng(0)
    unique = np.unique(g)
    held = set(rng.choice(unique, size=len(unique) // 4, replace=False).tolist())
    regimes["random molecules"] = (~np.isin(g, list(held)), np.isin(g, list(held)))
    for name, label in (("nist20_ma_mi_inchikey", "InChIKey split"),
                        ("nist20_ma_mi_scaffold", "scaffold split")):
        try:
            split = load_split(name)
        except Exception as error:
            print(f"  ({label} unavailable: {type(error).__name__})")
            continue
        train_mask = np.isin(g, list(split["train"]))
        val_mask = np.isin(g, list(split["val"]))
        if train_mask.sum() and val_mask.sum():
            regimes[label] = (train_mask, val_mask)

    header = f"{'feature group':36s}" + "".join(f"{k:>20s}" for k in regimes)
    print(header)
    for label, columns in groups_def.items():
        cells = []
        for train_mask, val_mask in regimes.values():
            model = HistGradientBoostingClassifier(
                max_iter=200, random_state=0, early_stopping=False
            )
            model.fit(X[train_mask][:, columns], y[train_mask])
            score = roc_auc_score(y[val_mask], model.predict_proba(X[val_mask][:, columns])[:, 1])
            cells.append(f"{score:20.4f}")
        print(f"{label:36s}" + "".join(cells))
    for name, (train_mask, val_mask) in regimes.items():
        print(f"  {name}: train rows {train_mask.sum():,} / eval rows {val_mask.sum():,}")


if __name__ == "__main__":
    main()
