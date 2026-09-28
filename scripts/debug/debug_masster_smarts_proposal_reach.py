"""How reachable are MASSter's SMARTS template cuts under a trained proposal?

With ``masster_smarts_templates`` on, every eligible template's cut mask is forced into the
candidate pool. This probe asks what the learned proposal would do without the forcing. It
loads a checkpoint trained with SMARTS off, runs it in eval mode on the spectra whose molecule
has an eligible template, and captures each spectrum's bond logits and retained cut masks from
the model's own candidate call. Per (spectrum, template) it records:

- ``retained``: the template cut is already in the model's retained mask pool;
- ``learned_rank``: 1-based rank of the cut in the exact learned enumeration (Bernoulli
  probability order, the ranking the candidate builder uses), or 0 past ``RANK_LIMIT``;
- ``log_p``: the cut's Bernoulli log-probability, and its gaps to the most likely mask
  (``gap_map``) and to the least likely retained multi-cut mask (``gap_cutoff``; the forced
  empty and single-cut masks are excluded, since they are kept however unlikely);
- ``pool_share``: the cut's probability over the retained pool's total mask probability, the
  chance that a draw from the proposal restricted to the pool is this cut;
- ``product_explained``: target intensity the template-only side mass-matches (FT tolerance,
  10 ppm with a 200 m/z floor, H shifts -4..+4). Connected sides are ordinary fragments, so
  only disconnected sides are new; templates with none are marked ``redundant``;
- added coverage, each a fraction of the spectrum's intensity that no pool fragment's model
  channel covers and a new fragment's does (``pool_covered`` is the pool's own coverage):
  ``ordinary_added`` for the forced cut's ordinary pieces missing from the pool,
  ``template_added`` for the template-only sides, and ``null_*`` for random connected atom
  sets of the same sizes, missing from the pool, as the chance baseline. New fragments get
  the model's own channels (adduct, H-shift window and valence bound).

A template-only fragment's proposal probability is at most its mask's, since it multiplies the
mask probability by a mechanism and a charge probability.

Run from the repository root:
    python scripts/debug/debug_masster_smarts_proposal_reach.py <custom_fp> <ckpt> <out_pkl> \
        [max_spectra]
"""

import pickle
import sys
from collections import defaultdict

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import fragnnet.model.masster_model as masster_model
from fragnnet.frag.masster_core import top_mixture_bond_sets_raw
from fragnnet.frag.masster_templates import (
    _is_connected,
    build_masster_templates,
    filter_masster_templates,
)
from fragnnet.runner import load_config
from fragnnet.utils.misc_utils import to_device
from fragnnet.utils.script_utils import apply_inference_config, init_model_from_ckpt, setup_device
from fragnnet.utils.setup_utils import get_dataset_cls

RANK_LIMIT = 65536
PROTON = 1.007276
H_MASS = 1.007825


def _log_sigmoid(values: np.ndarray) -> np.ndarray:
    return -np.logaddexp(0.0, -values)


def _neutral_mass(mol: Chem.Mol, atoms: tuple[int, ...]) -> float:
    table = Chem.GetPeriodicTable()
    return sum(
        table.GetMostCommonIsotopeMass(mol.GetAtomWithIdx(atom).GetSymbol())
        + mol.GetAtomWithIdx(atom).GetTotalNumHs() * H_MASS
        for atom in atoms
    )


def _explained(mz: np.ndarray, intensity: np.ndarray, neutral: float) -> float:
    """Target intensity fraction within FT tolerance of any [M+H]+ H-shift channel."""
    hit = np.zeros(mz.shape, dtype=bool)
    for shift in range(-4, 5):
        ion = neutral + PROTON + shift * H_MASS
        hit |= np.abs(mz - ion) <= max(ion, 200.0) * 1e-5
    return float(intensity[hit].sum() / intensity.sum()) if intensity.sum() > 0 else 0.0


def _covered(mz: np.ndarray, channels: np.ndarray) -> np.ndarray:
    """Target peaks within FT tolerance (10 ppm, 200 m/z floor) of any sorted channel m/z."""
    if channels.size == 0:
        return np.zeros(mz.shape, dtype=bool)
    right = np.clip(np.searchsorted(channels, mz), 0, channels.size - 1)
    left = np.clip(right - 1, 0, channels.size - 1)
    nearest = np.minimum(np.abs(channels[right] - mz), np.abs(channels[left] - mz))
    return nearest <= np.maximum(mz, 200.0) * 1e-5


def _cut_components(num_atoms: int, edges: tuple, cut: set[int]) -> list[tuple[int, ...]]:
    """Connected components of the precursor after removing the ``cut`` bond indices."""
    parent = list(range(num_atoms))

    def find(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    for bond, (left, right) in enumerate(zip(*edges)):
        if bond not in cut:
            parent[find(left)] = find(right)
    groups = defaultdict(list)
    for atom in range(num_atoms):
        groups[find(atom)].append(atom)
    return [tuple(group) for group in groups.values()]


def _random_connected(
    rng: np.random.Generator, adjacency: list[list[int]], size: int, exclude: set
) -> tuple[int, ...] | None:
    """A random connected atom set of ``size`` atoms not in ``exclude``, by random growth."""
    for _ in range(50):
        chosen = {int(rng.integers(len(adjacency)))}
        frontier = set(adjacency[next(iter(chosen))])
        while len(chosen) < size and frontier:
            atom = sorted(frontier)[int(rng.integers(len(frontier)))]
            chosen.add(atom)
            frontier |= set(adjacency[atom])
            frontier -= chosen
        key = tuple(sorted(chosen))
        if len(key) == size and frozenset(key) not in exclude:
            return key
    return None


def main(custom_fp: str, ckpt_fp: str, out_fp: str, max_spectra: str = "0") -> None:
    # 1. Load the checkpoint's config and the val split, restricted to template molecules.
    th.manual_seed(0)
    config_d = load_config("config", custom_fp, [])
    config_d = apply_inference_config(
        config_d, auxiliary_scores=["cos_sim"], skip_extra_losses=True, disable_preproc=True,
        output_formula_str=False, train_batch_size=64, eval_batch_size=64,
        dynamic_batch_sampler=False, eval_mz_bin_res=None,
    )
    if config_d["masster_smarts_templates"]:
        raise ValueError("probe the proposal of a checkpoint trained with SMARTS off")
    ds = get_dataset_cls(config_d["model_type"])(split="val", **config_d)
    spec_df = ds.spec_df.reset_index(drop=True)
    mol_df = pd.read_pickle(config_d["mol_fp"]).set_index("mol_id")
    templates_by_key: dict[tuple[int, str], tuple] = {}
    keep = []
    for index, row in enumerate(spec_df.itertuples()):
        key = (row.mol_id, row.prec_type)
        if key not in templates_by_key:
            templates_by_key[key] = filter_masster_templates(
                build_masster_templates(Chem.MolFromSmiles(mol_df.loc[row.mol_id, "smiles"])),
                precursor_type=row.prec_type,
                fragmentation_mode=None,
            )
        if templates_by_key[key]:
            keep.append(index)
    if int(max_spectra) > 0:
        keep = keep[: int(max_spectra)]
    print(f"val spectra {len(spec_df)}, with eligible templates {len(keep)}")
    dl = DataLoader(Subset(ds, keep), batch_size=64, shuffle=False,
                    collate_fn=ds.get_collate_fn(), num_workers=config_d["num_workers"])
    device = setup_device(config_d)
    model = init_model_from_ckpt(ckpt_fp, config_d, device)

    # 2. Capture the proposal's bond logits and retained masks from the model's own call.
    captured = {}
    original = masster_model.build_masster_candidates

    def capture(bond_logits, chemical_bond_index, atom_batch_indices, atom_ptr, **kwargs):
        candidates = original(bond_logits, chemical_bond_index, atom_batch_indices, atom_ptr,
                              **kwargs)
        captured["logits"] = bond_logits.detach().double().cpu().numpy()
        captured["bond_batch"] = atom_batch_indices[chemical_bond_index[0]].cpu().numpy()
        captured["mask_logp"] = candidates.mask_log_probabilities.detach().double().cpu().numpy()
        captured["mask_batch"] = candidates.mask_batch_indices.cpu().numpy()
        captured["boundary_bonds"] = candidates.mechanism_boundary_bond_indices.cpu().numpy()
        captured["boundary_masks"] = candidates.boundary_mechanism_indices.cpu().numpy()
        return candidates

    # The pool's own fragments and ion channels, from the model's mass step.
    channel_step = model.model._fragment_masses_and_valid_channels

    def capture_channels(mol_pyg, atom_indices, atom_fragment_indices, fragment_batch_indices,
                         num_fragments, precursor_type_indices, fragment_cut_counts=None):
        ion_mzs, valid = channel_step(mol_pyg, atom_indices, atom_fragment_indices,
                                      fragment_batch_indices, num_fragments,
                                      precursor_type_indices, fragment_cut_counts)
        captured["pool_atoms"] = atom_indices.cpu().numpy()
        captured["pool_atom_fragments"] = atom_fragment_indices.cpu().numpy()
        captured["pool_batch"] = fragment_batch_indices.cpu().numpy()
        captured["pool_mzs"] = ion_mzs.double().cpu().numpy()
        captured["pool_valid"] = valid.cpu().numpy()
        captured["precursor_types"] = precursor_type_indices
        return ion_mzs, valid

    masster_model.build_masster_candidates = capture
    model.model._fragment_masses_and_valid_channels = capture_channels
    rng = np.random.default_rng(0)
    rows = []
    cursor = 0
    with th.inference_mode():
        for batch in tqdm(dl, desc="val"):
            batch = to_device(batch, device, non_blocking=False)
            out = model.inference_step(batch, metrics="test")
            spec_rows = spec_df.iloc[keep[cursor : cursor + len(batch["mol_smiles"])]]
            cursor += len(spec_rows)
            true_mz = out["true_mzs"].cpu().numpy()
            true_int = out["true_logprobs"].exp().cpu().numpy()
            true_batch = out["true_batch_idxs"].cpu().numpy()
            bond_starts = np.searchsorted(captured["bond_batch"], np.arange(len(spec_rows)))
            mask_bonds = defaultdict(list)
            for bond, mask in zip(captured["boundary_bonds"], captured["boundary_masks"]):
                mask_bonds[mask].append(bond)
            atom_ptr = batch["mol_pyg"].ptr.cpu().numpy()
            pool_sets: dict[int, set] = defaultdict(set)
            pool_channels: dict[int, list] = defaultdict(list)
            order = np.argsort(captured["pool_atom_fragments"], kind="stable")
            fragment_ids, starts = np.unique(
                captured["pool_atom_fragments"][order], return_index=True
            )
            for fragment, atoms in zip(
                fragment_ids, np.split(captured["pool_atoms"][order], starts[1:]), strict=True
            ):
                view = int(captured["pool_batch"][fragment])
                pool_sets[view].add(frozenset((atoms - atom_ptr[view]).tolist()))
                pool_channels[view].extend(
                    captured["pool_mzs"][fragment][captured["pool_valid"][fragment]].tolist()
                )
            requests: list[tuple[int, tuple[int, ...], int]] = []
            request_index: dict[tuple[int, tuple[int, ...]], int] = {}
            pending = []

            def request(view: int, atoms: tuple[int, ...], edges: tuple) -> int:
                key = (view, tuple(sorted(atoms)))
                if key not in request_index:
                    atom_set = set(key[1])
                    boundary = sum((a in atom_set) != (b in atom_set) for a, b in zip(*edges))
                    request_index[key] = len(requests)
                    requests.append((view, key[1], boundary))
                return request_index[key]

            # 3. Score every eligible template against the spectrum's own proposal.
            for view, row in enumerate(spec_rows.itertuples()):
                templates = templates_by_key[(row.mol_id, row.prec_type)]
                logits = captured["logits"][captured["bond_batch"] == view]
                start = bond_starts[view]
                in_view = captured["mask_batch"] == view
                retained_logp = captured["mask_logp"][in_view]
                retained_cuts = [
                    tuple(sorted(int(bond) - start for bond in mask_bonds[mask]))
                    for mask in np.flatnonzero(in_view)
                ]
                retained = set(retained_cuts)
                multi_cut_logp = retained_logp[[len(cut) >= 2 for cut in retained_cuts]]
                cutoff = multi_cut_logp.min() if multi_cut_logp.size else np.nan
                pool_logp = np.logaddexp.reduce(retained_logp)
                base = _log_sigmoid(-logits).sum()
                map_logp = np.maximum(_log_sigmoid(logits), _log_sigmoid(-logits)).sum()
                ptr, bonds = top_mixture_bond_sets_raw(
                    np.ascontiguousarray(logits[:, None]), np.zeros(1), RANK_LIMIT, 0.0, 0.0
                )
                rank_of = {
                    tuple(bonds[ptr[i] : ptr[i + 1]].tolist()): i + 1 for i in range(len(ptr) - 1)
                }
                mol = mol_df.loc[row.mol_id, "mol"]
                edges = tuple(zip(*[(b.GetBeginAtomIdx(), b.GetEndAtomIdx())
                                    for b in mol.GetBonds()]))
                spectrum = true_batch == view
                adjacency = [[] for _ in range(mol.GetNumAtoms())]
                for left, right in zip(*edges):
                    adjacency[left].append(right)
                    adjacency[right].append(left)
                for template in templates:
                    cut = tuple(template.broken_bond_indices)
                    log_p = base + logits[list(cut)].sum()
                    new_sides = [c for c in template.components if not _is_connected(c, edges)]
                    new_ordinary = [
                        piece
                        for piece in _cut_components(mol.GetNumAtoms(), edges, set(cut))
                        if frozenset(piece) not in pool_sets[view]
                    ]
                    exclude = pool_sets[view] | {frozenset(c) for c in new_ordinary + new_sides}
                    nulls = {
                        name: [
                            request(view, match, edges)
                            for match in (
                                _random_connected(rng, adjacency, len(piece), exclude)
                                for piece in pieces
                            )
                            if match is not None
                        ]
                        for name, pieces in (("ordinary", new_ordinary), ("template", new_sides))
                    }
                    pending.append((
                        len(rows),
                        view,
                        [request(view, piece, edges) for piece in new_ordinary],
                        [request(view, side, edges) for side in new_sides],
                        nulls["ordinary"],
                        nulls["template"],
                    ))
                    rows.append({
                        "spec_id": row.spec_id,
                        "mol_id": row.mol_id,
                        "rule": template.rule_name,
                        "num_cuts": len(cut),
                        "num_bonds": len(logits),
                        "retained": cut in retained,
                        "learned_rank": rank_of.get(cut, 0),
                        "log_p": log_p,
                        "gap_map": log_p - map_logp,
                        "gap_cutoff": log_p - cutoff,
                        "pool_share": float(np.exp(log_p - pool_logp)),
                        "retained_masks": int(in_view.sum()),
                        "redundant": not new_sides,
                        "n_ordinary_new": len(new_ordinary),
                        "product_explained": max(
                            (_explained(true_mz[spectrum], true_int[spectrum],
                                        _neutral_mass(mol, side)) for side in new_sides),
                            default=0.0,
                        ),
                    })

            # 4. Intensity newly covered: target peaks within tolerance of a new fragment's
            # model channels and of no pool channel, as a fraction of the spectrum.
            if requests:
                atoms, fragments, views, boundaries = [], [], [], []
                for index, (view, request_atoms, boundary) in enumerate(requests):
                    atoms.extend(int(atom_ptr[view]) + atom for atom in request_atoms)
                    fragments.extend([index] * len(request_atoms))
                    views.append(view)
                    boundaries.append(boundary)
                ion_mzs, valid = channel_step(
                    batch["mol_pyg"],
                    th.tensor(atoms, device=device),
                    th.tensor(fragments, device=device),
                    th.tensor(views, device=device),
                    len(requests),
                    captured["precursor_types"],
                    th.tensor(boundaries, device=device),
                )
                ion_mzs, valid = ion_mzs.double().cpu().numpy(), valid.cpu().numpy()
            peaks = {}
            for view in {entry[1] for entry in pending}:
                spectrum = true_batch == view
                mz, weight = true_mz[spectrum], true_int[spectrum] / true_int[spectrum].sum()
                pool = _covered(mz, np.sort(np.asarray(pool_channels[view])))
                peaks[view] = (mz, weight, pool)
            for row_index, view, *groups in pending:
                mz, weight, pool = peaks[view]
                rows[row_index]["pool_covered"] = float(weight[pool].sum())
                for name, group in zip(
                    ("ordinary_added", "template_added", "null_ordinary_added",
                     "null_template_added"),
                    groups,
                    strict=True,
                ):
                    hit = np.zeros(mz.shape, dtype=bool)
                    for index in group:
                        hit |= _covered(mz, np.sort(ion_mzs[index][valid[index]]))
                    rows[row_index][name] = float(weight[hit & ~pool].sum())
    masster_model.build_masster_candidates = original
    model.model._fragment_masses_and_valid_channels = channel_step
    frame = pd.DataFrame(rows)
    with open(out_fp, "wb") as handle:
        pickle.dump(frame, handle)

    # 5. Summaries by rule and by cut count, then added coverage by template kind.
    frame["in_budget"] = (frame.learned_rank > 0) & (frame.learned_rank <= 4096)
    frame["useful"] = frame.product_explained >= 0.01
    agg = {
        "n": ("rule", "size"),
        "retained": ("retained", "mean"),
        "in_4096": ("in_budget", "mean"),
        "rank_med": ("learned_rank", lambda r: np.median(np.where(r > 0, r, np.inf))),
        "gap_cutoff_med": ("gap_cutoff", "median"),
        "gap_map_med": ("gap_map", "median"),
        "share_med": ("pool_share", "median"),
        "redundant": ("redundant", "mean"),
        "useful": ("useful", "mean"),
    }
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 20)
    print(frame.groupby("rule").agg(**agg).sort_values("n", ascending=False).round(3))
    print(frame.groupby("num_cuts").agg(**agg).round(3))
    print(frame.groupby("useful").agg(**agg).round(3))
    print("overall", frame.agg({"retained": "mean", "in_budget": "mean"}).round(4).to_dict())
    frame["kind"] = np.select(
        [frame.redundant, frame.num_cuts >= 4], ["duplicate", "ring"], "neutral loss"
    )
    coverage = {
        "n": ("rule", "size"),
        "new_ordinary": ("n_ordinary_new", "mean"),
        "pool_covered": ("pool_covered", "mean"),
        **{
            f"{name}_mean": (name, "mean")
            for name in ("ordinary_added", "null_ordinary_added", "template_added",
                         "null_template_added")
        },
        **{
            f"{name}_ge1pct": (name, lambda v: (v >= 0.01).mean())
            for name in ("ordinary_added", "null_ordinary_added", "template_added",
                         "null_template_added")
        },
    }
    print(frame.groupby("kind").agg(**coverage).round(4).T)


if __name__ == "__main__":
    main(*sys.argv[1:])
