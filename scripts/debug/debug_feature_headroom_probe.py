"""Is a candidate atom/bond feature information the model does not already have?

Every 2D-topological descriptor is a deterministic function of the molecular graph, so the
question is never "is it predictive of fragmentation" -- most chemistry descriptors are -- but
whether the encoder can already compute it from its current inputs. This probe answers that
directly: for each candidate descriptor, fit a predictor from the CURRENT feature set (the
2026-09-22 parity set) plus 1- and 2-hop neighbourhood means, the information a message-passing
encoder holds after two rounds, and report held-out skill on a molecule-grouped split.

High skill means the feature is derivable, so adding it buys inductive bias at best and costs
input width; low skill means it carries information the graph does not contain. The probe is a
lower bound on what the production encoders can derive: FraGNNet runs 4 GINE layers and
MASSter adds exact per-graph global attention, both strictly more expressive than a
two-hop mean.

Results and their interpretation: ``docs/findings/feature_headroom_2026_09_22.md``.

Usage:
    python scripts/debug/debug_feature_headroom_probe.py
"""

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolDescriptors
from rdkit.Chem.EState import EStateIndices
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, r2_score

from fragnnet.utils.feat_utils import MolGraph

RDLogger.DisableLog("rdApp.*")

ATOM_FEATS = [
    "a_onehot", "a_degree", "a_hybrid", "a_formal", "a_radical", "a_ring",
    "a_ring_size", "a_num_hs", "a_mass", "a_chiral", "a_charge", "a_fused_ring",
]
BOND_FEATS = ["b_degree", "b_ring", "b_conj", "b_ring_size", "b_fused_ring"]

BENZYLIC = Chem.MolFromSmarts("[CX4;!$(C=*)][c]")
ALLYLIC = Chem.MolFromSmarts("[CX4;!$(C=*)][C]=[C]")
ALPHA_CARBONYL = Chem.MolFromSmarts("[#6][CX3]=[OX1]")
ALPHA_HETERO = Chem.MolFromSmarts("[#6][!#6;!#1]")
AMIDE = Chem.MolFromSmarts("[CX3](=[OX1])[NX3]")
ROTATABLE = Chem.MolFromSmarts("[!$(*#*)&!D1]-&!@[!$(*#*)&!D1]")


def matched_atoms(mol, patt, position=0):
    hits = set()
    for match in mol.GetSubstructMatches(patt):
        hits.add(match[position])
    return hits


def molecule_rows(mol, mg):
    """Return (atom features, atom candidates, bond features, bond candidates) or None."""
    graph = mg.get_mol_graph(mol, bigraph=False)
    atom_x = graph["atom_feats"].astype(np.float32)
    num_atoms = mol.GetNumAtoms()

    props = AllChem.MMFFGetMoleculeProperties(mol)
    if props is None:
        return None
    estate = np.asarray(EStateIndices(mol), dtype=np.float32)
    crippen = np.asarray(rdMolDescriptors._CalcCrippenContribs(mol), dtype=np.float32)
    tpsa = np.asarray(rdMolDescriptors._CalcTPSAContribs(mol), dtype=np.float32)
    asa = np.asarray(rdMolDescriptors._CalcLabuteASAContribs(mol)[0], dtype=np.float32)[:num_atoms]

    benzylic = matched_atoms(mol, BENZYLIC)
    allylic = matched_atoms(mol, ALLYLIC)
    alpha_carbonyl = matched_atoms(mol, ALPHA_CARBONYL)
    alpha_hetero = matched_atoms(mol, ALPHA_HETERO)

    atom_cand = {
        "mmff_atom_type": np.array([props.GetMMFFAtomType(i) for i in range(num_atoms)], float),
        "mmff_partial_charge": np.array(
            [props.GetMMFFPartialCharge(i) for i in range(num_atoms)], float
        ),
        "estate_index": estate,
        "crippen_logp": crippen[:, 0],
        "crippen_mr": crippen[:, 1],
        "tpsa_contrib": tpsa,
        "labute_asa": asa,
        "is_benzylic": np.array([float(i in benzylic) for i in range(num_atoms)]),
        "is_allylic": np.array([float(i in allylic) for i in range(num_atoms)]),
        "is_alpha_carbonyl": np.array([float(i in alpha_carbonyl) for i in range(num_atoms)]),
        "is_alpha_hetero": np.array([float(i in alpha_hetero) for i in range(num_atoms)]),
    }

    # 1- and 2-hop neighbourhood means: what two rounds of message passing can average.
    adjacency = Chem.GetAdjacencyMatrix(mol).astype(np.float32)
    degree = np.maximum(adjacency.sum(1, keepdims=True), 1.0)
    hop1 = (adjacency @ atom_x) / degree
    hop2 = (adjacency @ hop1) / degree
    atom_X = np.hstack([atom_x, hop1, hop2])

    bond_x = graph["bond_feats"].astype(np.float32)
    rot = {tuple(sorted(m)) for m in mol.GetSubstructMatches(ROTATABLE)}
    amide_bonds = set()
    for match in mol.GetSubstructMatches(AMIDE):
        amide_bonds.add(tuple(sorted((match[0], match[2]))))
    bond_rows, bond_cand_rows = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        key = tuple(sorted((i, j)))
        # A bond sees its own features and both endpoints' 2-hop-contextualized features.
        bond_rows.append(
            np.concatenate(
                [bond_x[bond.GetIdx()], atom_X[i] + atom_X[j], np.abs(atom_X[i] - atom_X[j])]
            )
        )
        bond_cand_rows.append(
            [
                float(key in rot),
                float(key in amide_bonds),
                float(i in benzylic or j in benzylic),
                float(i in alpha_carbonyl or j in alpha_carbonyl),
            ]
        )
    return atom_X, atom_cand, np.asarray(bond_rows, np.float32), np.asarray(
        bond_cand_rows, np.float32
    )


def skill(X, y, groups, categorical):
    """Held-out skill of predicting y from X, split by molecule.

    Categorical targets use a random forest, which handles many-class targets with rare
    levels; continuous targets use gradient boosting. Classes with fewer than 20 atoms are
    folded into one bucket so balanced accuracy is not dominated by singletons.
    """
    unique = np.unique(groups)
    rng = np.random.default_rng(0)
    test_groups = set(rng.choice(unique, size=max(1, len(unique) // 4), replace=False).tolist())
    test = np.array([g in test_groups for g in groups])
    if categorical:
        values, counts = np.unique(y, return_counts=True)
        rare = values[counts < 20]
        y = np.where(np.isin(y, rare), -999.0, y)
        classes, y_enc = np.unique(y, return_inverse=True)
        if len(classes) < 2 or y_enc[~test].std() == 0:
            return None, None, len(classes)
        model = RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0)
        model.fit(X[~test], y_enc[~test])
        pred = model.predict(X[test])
        majority = np.bincount(y_enc[test]).argmax()
        base = balanced_accuracy_score(y_enc[test], np.full(test.sum(), majority))
        return balanced_accuracy_score(y_enc[test], pred), base, len(classes)
    if y[~test].std() == 0:
        return None, None, 0
    model = HistGradientBoostingRegressor(max_iter=150, random_state=0, early_stopping=False)
    model.fit(X[~test], y[~test])
    return r2_score(y[test], model.predict(X[test])), 0.0, 0


def main(num_molecules=1500):
    df = pd.read_pickle("data/proc/nist20/mol_df.pkl")
    rng = np.random.default_rng(0)
    sample = df.iloc[rng.choice(len(df), size=num_molecules, replace=False)]
    mg = MolGraph(ATOM_FEATS, BOND_FEATS, 0)

    atom_X, atom_groups, bond_X, bond_groups = [], [], [], []
    atom_y = {}
    bond_y = []
    failures: dict[str, int] = {}
    for mol_index, mol in enumerate(sample["mol"]):
        if mol is None:
            continue
        try:
            result = molecule_rows(mol, mg)
        except Exception as error:  # noqa: BLE001 - a skipped molecule is reported, not fatal
            failures[type(error).__name__] = failures.get(type(error).__name__, 0) + 1
            continue
        if result is None:
            continue
        aX, acand, bX, bcand = result
        atom_X.append(aX)
        atom_groups.append(np.full(aX.shape[0], mol_index))
        for name, values in acand.items():
            atom_y.setdefault(name, []).append(values)
        if bX.size:
            bond_X.append(bX)
            bond_y.append(bcand)
            bond_groups.append(np.full(bX.shape[0], mol_index))

    atom_X = np.vstack(atom_X)
    atom_groups = np.concatenate(atom_groups)
    bond_X = np.vstack(bond_X)
    bond_y = np.vstack(bond_y)
    bond_groups = np.concatenate(bond_groups)
    if failures:
        print("skipped molecules by error:", failures)
    print(f"molecules={len(set(atom_groups))}  atoms={atom_X.shape[0]}  bonds={bond_X.shape[0]}")
    print(f"atom predictor width={atom_X.shape[1]}  bond predictor width={bond_X.shape[1]}\n")

    categorical = {"mmff_atom_type", "is_benzylic", "is_allylic", "is_alpha_carbonyl",
                   "is_alpha_hetero"}
    print(f"{'atom candidate':24s} {'metric':8s} {'skill':>8s} {'baseline':>9s}  classes")
    for name, chunks in atom_y.items():
        y = np.concatenate(chunks)
        is_cat = name in categorical
        value, base, classes = skill(atom_X, y, atom_groups, is_cat)
        metric = "bal.acc" if is_cat else "R2"
        if value is None:
            print(f"{name:24s} {metric:8s} {'constant':>8s}")
            continue
        extra = f"  {classes}" if is_cat else ""
        print(f"{name:24s} {metric:8s} {value:8.3f} {base:9.3f}{extra}")

    print()
    bond_names = ["is_rotatable", "is_amide_bond", "is_benzylic_bond", "alpha_carbonyl_bond"]
    print(f"{'bond candidate':24s} {'metric':8s} {'skill':>8s} {'baseline':>9s}  classes")
    for column, name in enumerate(bond_names):
        y = bond_y[:, column]
        value, base, classes = skill(bond_X, y, bond_groups, True)
        if value is None:
            print(f"{name:24s} {'bal.acc':8s} {'constant':>8s}")
            continue
        print(f"{name:24s} {'bal.acc':8s} {value:8.3f} {base:9.3f}  {classes}")


if __name__ == "__main__":
    main()
