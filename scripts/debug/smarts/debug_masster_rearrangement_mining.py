"""Mine peaks that no connected atom set reaches, to find candidate SMARTS families.

Default: Orbitrap [M+H]+ of MassSpecGym + NIST20. ``--precursor`` selects another precursor type
(adducts are handled as a group kept or lost whole); ``--dataset nist23`` uses NIST23 with a
molecule-level split and every held-out test molecule removed.

For every Orbitrap [M+H]+ spectrum of MassSpecGym benchmark and NIST20 (``nist20_inchikey``,
``i1``'s split), train and val only, each peak at >= 5% of the base peak is classified:

- ``connected``: some composition within tolerance is reachable by a connected atom set of
  the same heavy-atom composition within +-4 H (MASSter's ordinary channels);
- ``h_limit``: a composition's heavy atoms form a connected set, but only beyond +-4 H;
- ``rearrangement``: compositions exist, and none of their heavy-atom sets is connected;
- ``capped``: the connected-set search hit its cap, undecided;
- ``isotope``: a more intense peak sits one 13C (or one 37Cl/81Br/34S) below;
- ``no_formula``: no sub-formula of the precursor within tolerance (background, above
  precursor, other molecule).

Rearrangement peaks get their lowest-error composition, the implied neutral loss from
[M+H]+, and, for losses of at most ``MAX_MOTIF`` heavy atoms, the connected atom sets of the
molecule with the loss's heavy composition, written as a fragment SMILES with its first
neighbor shell. Losses are then ranked by molecule count, with train/val and dataset
replication.

Run from the repository root:
    python scripts/debug/smarts/debug_masster_rearrangement_mining.py [--workers 10]
"""

import argparse
import sys
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).resolve().parent))

from debug_masster_smarts_rule_gate import edge_tuple, reachable_h  # noqa: E402

OUT = Path("output/rearrangement_mining_2026_10_14")
SOURCES = {
    "msgym": (Path("data/proc/ms_gym_benchmark_1_5"), Path("data/split/ms_gym_benchmark_1_5")),
    "nist20": (Path("data/proc/nist20"), Path("data/split/nist20_inchikey")),
}
# [M-H]- lives in NIST20 only; nist20_inchikey holds [M+H]+ spectra, so use the ma_mi split
# (the [M-H]- G0 gate's split, an independent draw from i1's).
NEGATIVE_SOURCES = {
    "nist20": (Path("data/proc/nist20"), Path("data/split/nist20_ma_mi_inchikey")),
}
PRECURSOR = "[M+H]+"
DATASET = "msgym_nist20"
# precursor type -> (charge, adduct atoms kept or lost as a group, H carried by the adduct group,
# precursor-ion H minus the neutral molecule's H). The adduct is not bonded in the molecular
# graph, so only the molecular part of an ion is tested for connectivity.
ADDUCTS = {
    "[M+H]+": (1, {}, 0, 1),
    "[M-H]-": (-1, {}, 0, -1),
    "[M+Na]+": (1, {"Na": 1}, 0, 0),
    "[M+K]+": (1, {"K": 1}, 0, 0),
    "[M+NH4]+": (1, {"N": 1}, 4, 4),
    "[M+Cl]-": (-1, {"Cl": 1}, 0, 0),
    "[M+CHO2]-": (-1, {"C": 1, "O": 2}, 1, 1),
    "[Cat]+": (1, {}, 0, 0),
    "[M]+": (1, {}, 0, 0),
}
# NIST23 split: molecule-level from nist23_ma_mi_inchikey; molecules whose InChIKey block 1 is
# in any test split we use are dropped, so no held-out molecule informs rule design.
NIST23 = (Path("data/proc/nist23"), Path("data/split/nist23_ma_mi_inchikey"))
TEST_SPLITS = (
    (Path("data/proc/nist23"), Path("data/split/nist23_ma_mi_inchikey")),
    (Path("data/proc/nist20"), Path("data/split/nist20_ma_mi_inchikey")),
    (Path("data/proc/nist20"), Path("data/split/nist20_inchikey")),
    (Path("data/proc/ms_gym_benchmark_1_5"), Path("data/split/ms_gym_benchmark_1_5")),
)
INSTRUMENTS = ("ORB", "ORB_F")
RELATIVE, FLOOR = 1e-5, 200.0
MIN_REL_INTENSITY = 0.05
H_MASS = 1.00782503207
ELECTRON = 0.00054858
C13_SHIFT = 1.0033548
M2_SHIFTS = (1.9970499, 1.9979534, 1.9957959)  # 37Cl, 81Br, 34S
MONO = {
    "C": 12.0, "N": 14.0030740048, "O": 15.99491461956, "S": 31.97207100, "P": 30.97376163,
    "F": 18.99840322, "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473, "Si": 27.9769265325,
    "B": 11.0093054, "Se": 79.9165213, "As": 74.9215965, "Na": 22.9897692809, "K": 38.96370668,
}
HALOGEN = {"F", "Cl", "Br", "I"}
TRIVALENT = {"N", "P", "B", "As"}
TETRAVALENT = {"C", "Si"}
REACH_CAP = 50_000
# Per-molecule budget: after this many capped searches, new compositions of the molecule are
# marked capped without searching (large molecules otherwise run for hours). Capped is never
# counted as rearrangement, so this only grows the undecided class.
MAX_CAPPED_SEARCHES = 30
MAX_MOTIF = 6
_ORDER = ["C", "H", "N", "O", "P", "S", "F", "Cl", "Br", "I", "Si", "B", "Se", "As", "Na", "K"]


def formula_string(heavy: dict[str, int], h: int) -> str:
    counts = dict(heavy)
    counts["H"] = h
    parts = []
    for el in _ORDER:
        n = counts.get(el, 0)
        if n > 0:
            parts.append(el if n == 1 else f"{el}{n}")
        elif n < 0:
            parts.append(f"{el}{n}")
    return "".join(parts) or "-"


def sub_compositions(mol_counts: Counter) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Every heavy sub-composition, its mass and its maximum RDBE-limited H."""
    elements = sorted(mol_counts)
    grids = np.meshgrid(*[np.arange(mol_counts[el] + 1) for el in elements], indexing="ij")
    comps = np.stack([g.ravel() for g in grids], axis=1)
    comps = comps[comps.sum(axis=1) > 0]
    masses = comps @ np.asarray([MONO[el] for el in elements])
    return elements, comps, masses


def rdbe_ok(elements: list[str], comps: np.ndarray, hs: np.ndarray) -> np.ndarray:
    tetra = sum(comps[:, i] for i, el in enumerate(elements) if el in TETRAVALENT)
    tri = sum(comps[:, i] for i, el in enumerate(elements) if el in TRIVALENT)
    hal = sum(comps[:, i] for i, el in enumerate(elements) if el in HALOGEN)
    return (tetra - (hs + hal) / 2 + tri / 2 + 1) >= -0.5


def is_isotope(mz: float, inten: float, peaks: np.ndarray, tol: float, heavy_m2: bool) -> bool:
    shifts = (C13_SHIFT, *M2_SHIFTS) if heavy_m2 else (C13_SHIFT,)
    for k, shift in enumerate(shifts):
        near = np.abs(peaks[:, 0] - (mz - shift)) <= tol
        if near.any() and peaks[near, 1].max() >= (inten if k == 0 else 0.25 * inten):
            return True
    return False


def connected_sets(mol, edges, composition: Counter, size: int, cap: int = 20_000) -> list[tuple[int, ...]]:
    """Connected atom sets with exactly this heavy composition (small sizes only)."""
    n = mol.GetNumAtoms()
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    adjacency = defaultdict(list)
    for left, right in zip(*edges):
        adjacency[left].append(right)
        adjacency[right].append(left)
    found: set[frozenset[int]] = set()
    seen: set[frozenset[int]] = set()
    checks = 0

    def grow(current: frozenset[int], counts: Counter) -> None:
        nonlocal checks
        checks += 1
        if checks > cap:
            return
        if len(current) == size:
            found.add(current)
            return
        frontier = {j for i in current for j in adjacency[i]} - current
        for atom in frontier:
            if atom < min(current) or counts[symbols[atom]] >= composition.get(symbols[atom], 0):
                continue
            grown = current | {atom}
            if grown in seen:
                continue
            seen.add(grown)
            counts[symbols[atom]] += 1
            grow(grown, counts)
            counts[symbols[atom]] -= 1

    for start in range(n):
        if composition.get(symbols[start], 0):
            grow(frozenset([start]), Counter({symbols[start]: 1}))
    return [tuple(sorted(s)) for s in found]


def motif_smiles(mol, atoms: tuple[int, ...]) -> str:
    """Fragment SMILES of the lost atoms plus their first neighbor shell (neighbors as [*:el])."""
    inner = set(atoms)
    shell = {nb.GetIdx() for i in atoms for nb in mol.GetAtomWithIdx(i).GetNeighbors()} - inner
    rw = Chem.RWMol(mol)
    for i in shell:
        atom = rw.GetAtomWithIdx(i)
        atom.SetAtomMapNum(1)
    bonds = [
        b.GetIdx() for b in mol.GetBonds()
        if b.GetBeginAtomIdx() in inner or b.GetEndAtomIdx() in inner
    ]
    try:
        return Chem.MolFragmentToSmiles(
            rw, atomsToUse=sorted(inner | shell), bondsToUse=bonds or None, canonical=True,
            isomericSmiles=False,
        )
    except Exception:  # noqa: BLE001
        return "?"


def process_molecule(task) -> dict:
    RDLogger.DisableLog("rdApp.*")
    dataset, mol_id, smiles, spectra = task
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or any(a.GetSymbol() not in MONO for a in mol.GetAtoms()):
        return {"peaks": [], "rearr": []}
    charge, adduct_atoms, adduct_h, h_delta = ADDUCTS[PRECURSOR]
    edges = edge_tuple(mol)
    symbols = [a.GetSymbol() for a in mol.GetAtoms()]
    mol_counts = Counter(symbols)
    total_h = sum(a.GetTotalNumHs() for a in mol.GetAtoms())
    heavy_m2 = any(el in mol_counts or el in adduct_atoms for el in ("Cl", "Br", "S"))
    elements, comps, masses = sub_compositions(mol_counts)
    h_range = np.arange(0, total_h + 6)
    electron = -charge * ELECTRON

    # Ion table: molecular sub-composition (index into comps, -1 = none) x molecular H, without
    # the adduct and, for adduct precursors, with the whole adduct group retained.
    parts = []
    comp_ids = np.repeat(np.arange(len(comps)), len(h_range))
    h_ids = np.tile(h_range, len(comps))
    ok = rdbe_ok(elements, comps[comp_ids], h_ids)
    base_mass = masses[comp_ids] + h_ids * H_MASS
    parts.append((base_mass[ok] + electron, comp_ids[ok], h_ids[ok], np.zeros(ok.sum(), bool)))
    if adduct_atoms or adduct_h:
        adduct_mass = sum(MONO[el] * n for el, n in adduct_atoms.items()) + adduct_h * H_MASS
        parts.append((base_mass[ok] + adduct_mass + electron, comp_ids[ok], h_ids[ok], np.ones(ok.sum(), bool)))
        if adduct_atoms:
            # The bare adduct ion (e.g. Cl-), no molecular atoms.
            parts.append((np.array([adduct_mass + electron]), np.array([-1]), np.array([0]), np.array([True])))
    ion = np.concatenate([p[0] for p in parts])
    comp_index = np.concatenate([p[1] for p in parts])
    h_index = np.concatenate([p[2] for p in parts])
    with_adduct = np.concatenate([p[3] for p in parts])
    order = np.argsort(ion)
    ion, comp_index, h_index, with_adduct = ion[order], comp_index[order], h_index[order], with_adduct[order]
    reach_cache: dict[int, set[int] | None] = {}
    precursor_heavy = dict(mol_counts)
    for el, n in adduct_atoms.items():
        precursor_heavy[el] = precursor_heavy.get(el, 0) + n
    precursor_h = total_h + h_delta
    peak_rows, rearr_rows = [], []
    reached: set[tuple[int, int]] = set()
    capped_searches = 0

    def classify(ci: int, h: int) -> str:
        """Connected within +-4 H, stopping at the first twin; otherwise an exhaustive search."""
        nonlocal capped_searches
        if ci < 0:
            return "adduct"
        if (ci, h) in reached:
            return "connected"
        if ci not in reach_cache and capped_searches >= MAX_CAPPED_SEARCHES:
            return "capped"
        if ci not in reach_cache:
            composition = Counter({el: int(c) for el, c in zip(elements, comps[ci]) if c})
            size = int(comps[ci].sum())
            found = reachable_h(
                mol, edges, composition, size, cap=REACH_CAP, required_h=frozenset([h])
            )
            if found is not None and any(abs(h - value) <= 4 for value in found):
                reached.add((ci, h))
                return "connected"
            # The search did not stop early, so ``found`` is exhaustive (or None if capped).
            reach_cache[ci] = found
            capped_searches += found is None
        found = reach_cache[ci]
        if found is None:
            return "capped"
        if not found:
            return "rearrangement"
        return "connected" if any(abs(h - value) <= 4 for value in found) else "h_limit"

    for row in spectra.itertuples():
        peaks = np.asarray(row.peaks, float).reshape(-1, 2)
        if len(peaks) == 0:
            continue
        weights = peaks[:, 1] / peaks[:, 1].sum()
        base = peaks[:, 1].max()
        for p in np.flatnonzero(peaks[:, 1] >= MIN_REL_INTENSITY * base):
            mz, inten = peaks[p]
            tol = max(mz, FLOOR) * RELATIVE
            if is_isotope(mz, inten, peaks, tol, heavy_m2):
                label = "isotope"
                matches = []
            else:
                lo, hi = np.searchsorted(ion, [mz - tol, mz + tol + 1e-12])
                matches = list(range(lo, hi))
                label = "no_formula" if not matches else None
            best = None
            if matches:
                classes = set()
                for m in sorted(matches, key=lambda m: abs(ion[m] - mz)):
                    classes.add(classify(int(comp_index[m]), int(h_index[m])))
                    if "connected" in classes:
                        break
                for name in ("connected", "adduct", "capped", "h_limit", "rearrangement"):
                    if name in classes:
                        label = name
                        break
                best = min(matches, key=lambda m: abs(ion[m] - mz))
            peak_rows.append(
                {
                    "dataset": dataset, "mol_id": mol_id, "spec_id": row.spec_id, "split": row.split,
                    "mz": mz, "weight": weights[p], "rel": inten / base, "label": label,
                    "n_formulas": len(matches),
                }
            )
            if label == "rearrangement":
                ci, h, kept = int(comp_index[best]), int(h_index[best]), bool(with_adduct[best])
                heavy = {el: int(c) for el, c in zip(elements, comps[ci]) if c}
                ion_heavy = dict(heavy)
                if kept:
                    for el, n in adduct_atoms.items():
                        ion_heavy[el] = ion_heavy.get(el, 0) + n
                ion_h = h + (adduct_h if kept else 0)
                loss_heavy = {el: precursor_heavy[el] - ion_heavy.get(el, 0) for el in precursor_heavy}
                loss_heavy = {el: c for el, c in loss_heavy.items() if c}
                # The motif search looks for the lost MOLECULAR atoms only.
                loss_mol = {el: mol_counts[el] - heavy.get(el, 0) for el in mol_counts}
                loss_mol = {el: c for el, c in loss_mol.items() if c}
                rearr_rows.append(
                    {
                        "dataset": dataset, "mol_id": mol_id, "spec_id": row.spec_id,
                        "split": row.split, "mz": mz, "weight": weights[p], "rel": inten / base,
                        "ion": formula_string(ion_heavy, ion_h),
                        "ion_mol": formula_string(heavy, h),
                        "adduct_kept": kept,
                        "loss": formula_string(loss_heavy, precursor_h - ion_h),
                        "loss_heavy": formula_string(loss_mol, 0),
                        "loss_size": sum(loss_mol.values()),
                        "ppm": 1e6 * (ion[best] - mz) / mz,
                        "n_formulas": len(matches),
                    }
                )

    # Motifs: connected sets with the loss's heavy composition, once per (molecule, loss).
    motifs: dict[str, str] = {}
    for loss_heavy, size in {(r["loss_heavy"], r["loss_size"]) for r in rearr_rows}:
        if size > MAX_MOTIF or size == 0:
            motifs[loss_heavy] = "large" if size else "none"
            continue
        composition = Counter(_parse(loss_heavy))
        sets = connected_sets(mol, edges, composition, size)
        excisions = [
            s for s in sets
            if len(Chem.GetMolFrags(Chem.RWMol(_remove_atoms(mol, s)))) >= 2
        ]
        chosen = excisions or sets
        motifs[loss_heavy] = ";".join(sorted({motif_smiles(mol, s) for s in chosen})[:3]) or "disconnected"
    for r in rearr_rows:
        r["motif"] = motifs[r["loss_heavy"]]
        r["smiles"] = smiles
    return {"peaks": peak_rows, "rearr": rearr_rows}


def _parse(formula: str) -> dict[str, int]:
    import re

    return {el: int(n or 1) for el, n in re.findall(r"([A-Z][a-z]?)(\d*)", formula) if el != "H"}


def _remove_atoms(mol, atoms) -> Chem.Mol:
    rw = Chem.RWMol(mol)
    for i in sorted(atoms, reverse=True):
        rw.RemoveAtom(i)
    return rw.GetMol()


def _test_inchikeys() -> set[str]:
    keys = set()
    for data, split_dir in TEST_SPLITS:
        mol_df = pd.read_pickle(data / "mol_df.pkl").set_index("mol_id")
        test = pd.read_csv(split_dir / "test_ids.csv")
        keys |= set(mol_df.loc[test.mol_id.unique(), "inchikey_s"].dropna())
    return keys


def _nist23_spectra() -> tuple[pd.DataFrame, pd.DataFrame]:
    """NIST23 spectra of PRECURSOR with a molecule-level train/val split, test molecules dropped."""
    data, split_dir = NIST23
    mol_df = pd.read_pickle(data / "mol_df.pkl").set_index("mol_id")
    spec = pd.read_pickle(data / "spec_df.pkl")
    split = pd.concat(
        [pd.read_csv(split_dir / f"{name}_ids.csv").assign(split=name) for name in ("train", "val", "test")]
    )
    mol_split = split.drop_duplicates("mol_id").set_index("mol_id").split
    spec = spec[(spec.prec_type == PRECURSOR) & spec.inst_type.isin(INSTRUMENTS)].copy()
    # Molecules with no split row (adducts the split files omit) count as train.
    spec["split"] = spec.mol_id.map(mol_split).fillna("train")
    held_out = _test_inchikeys()
    spec = spec[(spec.split != "test") & ~spec.mol_id.map(mol_df.inchikey_s).isin(held_out)]
    return spec, mol_df


def load_tasks(limit: int | None) -> list:
    tasks = []
    if DATASET == "nist23":
        spec, mol_df = _nist23_spectra()
        frames = [("nist23", spec, mol_df)]
    else:
        frames = []
        sources = SOURCES if PRECURSOR == "[M+H]+" else NEGATIVE_SOURCES
        for dataset, (data, split_dir) in sources.items():
            mol_df = pd.read_pickle(data / "mol_df.pkl").set_index("mol_id")
            spec = pd.read_pickle(data / "spec_df.pkl")
            spec = spec[(spec.prec_type == PRECURSOR) & spec.inst_type.isin(INSTRUMENTS)]
            split = pd.concat(
                [pd.read_csv(split_dir / f"{name}_ids.csv").assign(split=name) for name in ("train", "val")]
            )
            spec = spec.merge(split[["spec_id", "split"]], on="spec_id", how="inner")
            frames.append((dataset, spec, mol_df))
    for dataset, spec, mol_df in frames:
        groups = list(spec.groupby("mol_id"))
        if limit:
            groups = groups[:limit]
        print(f"{dataset} {PRECURSOR}: {len(spec)} spectra, {len(groups)} molecules", flush=True)
        tasks += [
            (dataset, mol_id, mol_df.at[mol_id, "smiles"], group[["spec_id", "split", "peaks"]])
            for mol_id, group in groups
        ]
    # Large molecules first, so the pool does not end on a straggler.
    tasks.sort(key=lambda t: -len(t[2]))
    return tasks


def main() -> None:
    global PRECURSOR, OUT, DATASET
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None, help="molecules per dataset (smoke)")
    parser.add_argument("--precursor", choices=sorted(ADDUCTS), default="[M+H]+")
    parser.add_argument("--dataset", choices=["msgym_nist20", "nist23"], default="msgym_nist20")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    # Workers are forked after these are set, so they inherit them.
    args.out = args.out or OUT
    PRECURSOR, OUT, DATASET = args.precursor, args.out, args.dataset
    OUT.mkdir(parents=True, exist_ok=True)
    tasks = load_tasks(args.limit)
    suffix = "_smoke" if args.limit else ""
    # Every molecule in scope, for rule precision in debug_masster_rewire_rule_gate.py.
    pd.DataFrame(
        [
            {
                "key": f"{dataset}:{mol_id}", "dataset": dataset, "mol_id": mol_id,
                "split": group.split.iloc[0], "n_spectra": len(group), "smiles": smiles,
            }
            for dataset, mol_id, smiles, group in tasks
        ]
    ).to_csv(OUT / f"molecules{suffix}.csv", index=False)
    peaks, rearr = [], []
    with Pool(args.workers, maxtasksperchild=500) as pool:
        for i, result in enumerate(pool.imap_unordered(process_molecule, tasks, chunksize=4)):
            peaks += result["peaks"]
            rearr += result["rearr"]
            if i % 2000 == 0:
                print(f"{i}/{len(tasks)} molecules", flush=True)
    peaks = pd.DataFrame(peaks)
    rearr = pd.DataFrame(rearr)
    peaks.to_pickle(OUT / f"peaks{suffix}.pkl")
    rearr.to_pickle(OUT / f"rearrangement{suffix}.pkl")
    pd.set_option("display.width", 250)
    pd.set_option("display.max_colwidth", 80)

    # 1. Intensity by class, per dataset and split (share of total spectrum intensity).
    n_spec = peaks.groupby(["dataset", "split"]).spec_id.nunique()
    share = peaks.groupby(["dataset", "split", "label"]).weight.sum().unstack("label").fillna(0)
    share = 100 * share.div(n_spec, axis=0)
    print("\nMean % of spectrum intensity by class (peaks >= 5% of base)\n" + share.round(2).to_string())
    if rearr.empty:
        return
    rearr["key"] = rearr.dataset + ":" + rearr.mol_id.astype(str)

    # 2. Rearrangement losses ranked by molecules, with replication.
    def summarize(d: pd.DataFrame) -> pd.Series:
        out = {"mols": d.key.nunique()}
        for (dataset, split), g in d.groupby(["dataset", "split"]):
            out[f"mols_{dataset}_{split}"] = g.mol_id.nunique()
        out.update(
            {
                "spectra": d.spec_id.nunique(),
                "mean_rel": d.rel.mean(),
                "ambiguous": (d.n_formulas > 1).mean(),
                "adduct_kept": d.adduct_kept.mean(),
                "top_motif": d.motif.value_counts().index[0],
            }
        )
        return pd.Series(out)

    losses = pd.DataFrame({loss: summarize(d) for loss, d in rearr.groupby("loss")}).T.fillna(0)
    losses = losses.sort_values("mols", ascending=False)
    losses.to_csv(OUT / f"losses{suffix}.csv")
    print(f"\nTop rearrangement losses from {PRECURSOR}\n" + losses.head(40).round(2).to_string())


if __name__ == "__main__":
    main()
