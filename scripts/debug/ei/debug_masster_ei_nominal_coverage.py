"""Nominal-mass reach of MASSter's ``[M]+`` candidate space on NIST23 EI (unit resolution).

For a seeded sample of validation molecules of ``data/split/nist23_ei_inchikey`` (one spectrum
each), the candidate stand-in pool is the connected components of every 0-, 1- and 2-cut mask
(the pool of the SMARTS G0 gate, vendored below). Channels come from the production call:
``[M]+``, CRF only, H shifts -4..+4, valence bound. A channel's nominal m/z is the integer mass sum over its atoms
(most abundant isotope per element, implicit H included) plus the H shift; the electron is
dropped, as in NIST nominal spectra.

Per spectrum (eval-convention target: intensity > 0, m/z < 1500, L1-normalised):

- ``l1_hK`` / ``l2_hK``: share of sum(w) / sum(w^2) at an integer some valid channel with
  |H shift| <= K reaches. ``sqrt(l2_h4)`` bounds the unit-resolution cosine.
- ``*_iso``: the same with the +1 and +2 of every reachable integer added.
- ``null_*``: coverage by the channel set of a different sampled molecule whose nominal
  precursor mass is closest (mismatched-structure null).
- ``round_agree`` / ``floor_agree``: share of valid channels whose production (exact) m/z,
  rounded or floored, equals the nominal m/z. ``floor`` is what ``batched_bin_func`` does at a
  1 Da bin width.
- ``channels_per_peak_w``: intensity-weighted count of valid channels at each observed integer.

Run from the repository root:

    python scripts/debug/ei/debug_masster_ei_nominal_coverage.py --n_molecules 5000 --workers 6
"""

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, RDLogger
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from torch_geometric.data import Batch

from fragnnet.utils.feat_utils import get_mol_graph

# The test model factory provides the MASSter whose channel call this probe reuses.
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from tests.test_masster_model import _build_model  # noqa: E402

RDLogger.DisableLog("rdApp.*")
PERIODIC_TABLE = Chem.GetPeriodicTable()
H_SHIFTS = np.arange(-4, 5)
MZ_MAX = 1500.0
POOL_CHUNK = 4000
SPEC_FP = Path("data/proc/nist23_ei/spec_df.pkl")
MOL_FP = Path("data/proc/nist23_ei/mol_df.pkl")
SPLIT_FP = Path("data/split/nist23_ei_inchikey/val_ids.csv")
CALC = None


# 1. Stand-in pool and production channels, vendored from debug_masster_smarts_g0_ceiling.py
# (pool_sets, ChannelCalculator, channel_mzs) without its SMARTS formed-bond input, which is 0
# for every connected pool set.


def pool_sets(mol: Chem.Mol, max_cuts: int = 2) -> dict[tuple[int, ...], int]:
    """Return connected components of every 0- to ``max_cuts``-cut mask with their cut counts.

    Args:
        mol: Precursor molecule.
        max_cuts: Largest number of simultaneously severed bonds, at most 2.

    Returns:
        Map from sorted atom tuple to the number of severed bonds crossing its boundary. A
        connected set's boundary is fixed by the molecule, so the count does not depend on
        the mask that produced it.
    """
    n_atoms = mol.GetNumAtoms()
    bonds = [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]
    left = np.asarray([b[0] for b in bonds], dtype=np.int64)
    right = np.asarray([b[1] for b in bonds], dtype=np.int64)
    result: dict[tuple[int, ...], int] = {}
    masks: list[tuple[int, ...]] = [()]
    if max_cuts >= 1:
        masks += [(i,) for i in range(len(bonds))]
    if max_cuts >= 2:
        masks += [(i, j) for i in range(len(bonds)) for j in range(i + 1, len(bonds))]
    if max_cuts > 2:
        raise ValueError("the stand-in pool supports at most two cuts")
    for cut in masks:
        keep = np.ones(len(bonds), dtype=bool)
        keep[list(cut)] = False
        graph = coo_matrix(
            (np.ones(int(keep.sum())), (left[keep], right[keep])), shape=(n_atoms, n_atoms)
        )
        n_components, labels = connected_components(graph, directed=False)
        if n_components == 1:
            result.setdefault(tuple(range(n_atoms)), 0)
            continue
        for label in range(n_components):
            atoms = tuple(np.flatnonzero(labels == label).tolist())
            if atoms in result:
                continue
            inside = set(atoms)
            result[atoms] = sum((bonds[i][0] in inside) != (bonds[i][1] in inside) for i in cut)
    return result


class ChannelCalculator:
    """MASSter's own ``[M]+`` ion channels for arbitrary atom sets of one molecule."""

    def __init__(self) -> None:
        self.model = None

    def __call__(
        self, mol: Chem.Mol, atom_sets: list[tuple[int, ...]], cut_counts: list[int]
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return valid exact-mass ion channels of atom sets, with the production valence bound.

        Args:
            mol: Precursor molecule.
            atom_sets: Molecule-local atoms of each fragment.
            cut_counts: ``fragment_cut_counts`` per fragment.

        Returns:
            ``(m/z, valid)``, both ``(len(atom_sets), 9)`` over H shifts -4..+4 (CRF only).
        """
        graph = Batch.from_data_list(
            [get_mol_graph(mol, ["a_onehot", "a_degree"], ["b_degree"], 0, True)]
        )
        if self.model is None:
            self.model = _build_model(
                graph,
                include_cmf=False,
                max_h_transfer=int(H_SHIFTS.max()),
                h_transfer_bound="valence",
                precursor_types=("[M]+",),
            ).eval()
        atoms = th.tensor([a for s in atom_sets for a in s], dtype=th.long)
        fragments = th.tensor([i for i, s in enumerate(atom_sets) for _ in s], dtype=th.long)
        with th.no_grad():
            mzs, valid = self.model._fragment_masses_and_valid_channels(
                graph,
                atoms,
                fragments,
                th.zeros(len(atom_sets), dtype=th.long),
                len(atom_sets),
                th.tensor([0]),
                th.tensor(cut_counts, dtype=th.long),
            )
        return mzs.double().numpy()[:, 0, :], valid.numpy()[:, 0, :]


def channel_mzs(
    calc: ChannelCalculator, mol: Chem.Mol, sets: list[tuple[int, ...]], cuts: list[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Call the channel calculator in bounded chunks."""
    parts = [
        calc(mol, sets[start : start + POOL_CHUNK], cuts[start : start + POOL_CHUNK])
        for start in range(0, len(sets), POOL_CHUNK)
    ]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])


# 2. Nominal channels and coverage.


def _init_worker() -> None:
    global CALC
    RDLogger.DisableLog("rdApp.*")
    CALC = ChannelCalculator()


def nominal_atom_masses(mol: Chem.Mol) -> np.ndarray:
    """Integer mass of each atom with its implicit hydrogens (most abundant isotopes)."""
    return np.array(
        [
            round(PERIODIC_TABLE.GetMostCommonIsotopeMass(a.GetSymbol())) + a.GetTotalNumHs()
            for a in mol.GetAtoms()
        ],
        dtype=np.int64,
    )


def eval_target(peaks: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return integer m/z, L1 weights and L2-normalised squared weights of one spectrum."""
    array = np.asarray(peaks, dtype=np.float64).reshape(-1, 2)
    array = array[(array[:, 1] > 0) & (array[:, 0] < MZ_MAX)]
    mz = np.rint(array[:, 0]).astype(np.int64)
    w = array[:, 1] / array[:, 1].sum()
    return mz, w, w**2 / (w**2).sum()


def molecule_channels(task: tuple[int, Chem.Mol]) -> dict:
    """Valid nominal channels of one molecule and the exact-vs-nominal agreement."""
    spec_id, mol = task
    pool = pool_sets(mol, 2)
    sets = list(pool)
    exact, valid = channel_mzs(CALC, mol, sets, [pool[s] for s in sets])
    atom_nominal = nominal_atom_masses(mol)
    nominal = np.array([atom_nominal[list(s)].sum() for s in sets])[:, None] + H_SHIFTS[None, :]
    return {
        "spec_id": spec_id,
        "n_sets": len(sets),
        "prec_nominal": int(atom_nominal.sum()),
        "nominal": nominal[valid],
        "abs_shift": np.broadcast_to(np.abs(H_SHIFTS), valid.shape)[valid],
        "round_agree": float(np.mean(np.rint(exact[valid]) == nominal[valid])),
        "floor_agree": float(np.mean(np.floor(exact[valid]) == nominal[valid])),
    }


def coverage(
    mz: np.ndarray, w: np.ndarray, w2: np.ndarray, reach: np.ndarray
) -> tuple[float, float]:
    hit = np.isin(mz, reach)
    return float(w[hit].sum()), float(w2[hit].sum())


def score(target: tuple, channels: dict, null_channels: dict) -> dict:
    """Coverage of one spectrum by its own channels and by a mismatched molecule's."""
    mz, w, w2 = target
    row = {
        k: channels[k] for k in ("spec_id", "n_sets", "prec_nominal", "round_agree", "floor_agree")
    }
    row["n_peaks"] = len(mz)
    for k in (0, 1, 2, 4):
        reach = np.unique(channels["nominal"][channels["abs_shift"] <= k])
        row[f"l1_h{k}"], row[f"l2_h{k}"] = coverage(mz, w, w2, reach)
    reach = np.unique(channels["nominal"])
    iso = np.unique(np.concatenate([reach, reach + 1, reach + 2]))
    row["l1_h4_iso"], row["l2_h4_iso"] = coverage(mz, w, w2, iso)
    row["null_l1_h4"], row["null_l2_h4"] = coverage(mz, w, w2, np.unique(null_channels["nominal"]))
    row["l1_above_M"] = float(w[mz > channels["prec_nominal"]].sum())
    uncovered = ~np.isin(mz, reach)
    row["l1_uncovered_below50"] = float(w[uncovered & (mz < 50)].sum())
    values, counts = np.unique(channels["nominal"], return_counts=True)
    index = np.clip(np.searchsorted(values, mz), 0, max(len(values) - 1, 0))
    per_peak = np.where(values[index] == mz, counts[index], 0) if len(values) else np.zeros_like(mz)
    row["channels_per_peak_w"] = float((per_peak * w).sum())
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--n_molecules", type=int, default=5000)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--out_dp", type=Path, default=Path("output/masster_ei_nominal_2026_09_30"))
    args = parser.parse_args()

    # 1. Sample one validation spectrum per molecule.
    spec_df = pd.read_pickle(SPEC_FP)
    mol_df = pd.read_pickle(MOL_FP).set_index("mol_id")
    val_ids = pd.read_csv(SPLIT_FP)["spec_id"]
    sample = (
        spec_df[spec_df["spec_id"].isin(val_ids)]
        .drop_duplicates("mol_id")
        .sample(args.n_molecules, random_state=args.seed)
    )

    # 2. Channels per molecule, in parallel.
    tasks = [(r.spec_id, mol_df.loc[r.mol_id, "mol"]) for r in sample.itertuples()]
    with mp.Pool(args.workers, initializer=_init_worker) as pool:
        channels = pool.map(molecule_channels, tasks, chunksize=8)

    # 3. Score each spectrum against its own channels and the nearest-mass other molecule's.
    order = np.argsort([c["prec_nominal"] for c in channels], kind="stable")
    null_of = {}
    for rank, i in enumerate(order):
        neighbours = [order[j] for j in (rank - 1, rank + 1) if 0 <= j < len(order)]
        null_of[i] = min(
            neighbours,
            key=lambda j: abs(channels[j]["prec_nominal"] - channels[i]["prec_nominal"]),
        )
    rows = [
        score(eval_target(r.peaks), channels[i], channels[null_of[i]])
        for i, r in enumerate(sample.itertuples())
    ]

    # 4. Write rows and the summary.
    frame = pd.DataFrame(rows)
    args.out_dp.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_dp / "nominal_coverage_rows.csv", index=False)
    summary = frame.drop(columns="spec_id").describe(percentiles=[0.1, 0.5, 0.9]).T
    summary["cos_ceiling_mean"] = np.nan
    for column in [c for c in frame.columns if c.startswith(("l2_", "null_l2_"))]:
        summary.loc[column, "cos_ceiling_mean"] = np.sqrt(frame[column]).mean()
    summary.to_csv(args.out_dp / "nominal_coverage_summary.csv")
    print(f"{len(frame)} spectra; rows in {args.out_dp}")
    print(summary.round(4).to_string())


if __name__ == "__main__":
    main()
