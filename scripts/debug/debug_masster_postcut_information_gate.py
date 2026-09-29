"""Does an explicit description of each candidate after cutting carry information a MASSter
checkpoint does not use? A cross-fitted, per-candidate correction gate on validation molecules.

Step 2 of the wrong-peak analysis (2026-09-29), after
``docs/findings/masster/masster_i1_base_peak_errors_2026_09_28.md``. Two subcommands:

``extract`` runs one annotated eval-mode forward per batch, as
``debug_masster_base_peak_errors.py`` does. For every candidate peak with predicted share
>= 1e-5 or target intensity it saves descriptors in families:

- ``basics``: H shift, |H shift|, intact precursor, heavy atoms, fraction of precursor atoms,
  cut count (the size and depth terms every family shares);
- ``existing``: the eight structure attributes of the base-peak tables (a reference);
- ``postcut``: the fragment after cutting. Retained bonds, the cyclomatic number, SSSR and
  aromatic rings kept and broken, conjugation kept, ring / aromatic / multiple bonds cut,
  retained and lost cut-site atom types, cation-stabilizing cut sites (acyl, benzylic,
  alpha-heteroatom, allylic, tertiary carbon, heteroatom), cut-bond classes, ion composition
  and RDBE;
- ``charge``: basic site kept and its class, kept basic matches, graph distance from the
  retained cut sites to the kept basic site, Gasteiger charge sum / min / max of the fragment
  and the complement's sum;
- ``context``: NCE, instrument and precursor m/z, given to every family.

Structure-level descriptors are averaged over a peak's structures with the model's shares.

``fit`` fits ``logit' = log p + g(x)``, with ``g`` a small MLP over ``log p`` and one family's
descriptors, by listwise softmax cross-entropy against the target over the saved candidates. It
is fitted on validation molecules of one ``mol_id`` parity and scored on the other, then
swapped. The held-out cosine gain over the raw prediction, per family, measures how much of the
model's validation error the family predicts on molecules the correction never saw. The
``context`` family (``g`` of ``log p`` and context alone) is the calibration baseline; a family
earns information only by its gain over that.

The correction learns how the checkpoint errs on new molecules. A branch trained on training
molecules cannot learn that directly, so a gain is evidence that the information exists and
transfers across molecules, not that a branch will recover it. A null says an explicit
description of this kind carries nothing the model does not already use.

Run from the repository root:

    python scripts/debug/debug_masster_postcut_information_gate.py extract \
        --checkpoint "saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt" \
        --custom_fp config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml \
        --output output/postcut_gate/i1_e059_val.npz
    python scripts/debug/debug_masster_postcut_information_gate.py fit \
        --input output/postcut_gate/i1_e059_val.npz --output output/postcut_gate/fit.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader

from fragnnet.runner import load_config
from fragnnet.utils.script_utils import apply_inference_config, init_model_from_ckpt
from fragnnet.utils.setup_utils import get_dataset_cls

# Support both the script entry point and imports from pytest.
try:
    from .debug_masster_base_peak_errors import SAVE_FLOOR, map_targets, merge_channels
    from .debug_masster_base_peak_tables import BASE_SMARTS, MoleculeChemistry, clustered_bootstrap
except ImportError:
    from debug_masster_base_peak_errors import SAVE_FLOOR, map_targets, merge_channels
    from debug_masster_base_peak_tables import BASE_SMARTS, MoleculeChemistry, clustered_bootstrap

RDLogger.DisableLog("rdApp.*")
LOGGER = logging.getLogger(__name__)

# Cut-site atom types: element, aromaticity and hybridization.
ATOM_TYPES = (
    "C_ar",
    "C_sp3",
    "C_carbonyl",
    "C_sp2",
    "C_sp",
    "N_ar",
    "N_sp3",
    "N_other",
    "O",
    "S",
    "halogen",
    "other",
)
# Cut sites that stabilize a charge or radical on the retained side after heterolysis.
STABILIZING_SMARTS = (
    ("acyl", "[CX3]=[OX1]"),
    ("benzylic", "[CX4][c]"),
    ("alpha_hetero", "[CX4][#7,#8,#16]"),
    ("allylic", "[CX4][CX3]=[CX3]"),
    ("tertiary", "[CX4]([#6])([#6])[#6]"),
    ("heteroatom", "[#7,#8,#16]"),
)
# Named cut-bond classes of the base-peak tables; anything else is one "other" column.
BOND_CLASSES = (
    "amide_CN",
    "ester_acylO",
    "acid_CO",
    "sulfonyl_SN",
    "sulfonyl_SC",
    "phosph_PO",
    "anomeric_CO",
    "aryl_O",
    "alkyl_O",
    "aryl_N",
    "alkyl_N",
    "C_halogen",
    "C_S",
    "N_N",
    "N_O",
    "acyl_CC",
    "biaryl_CC",
    "benzylic_CC",
    "alkyl_CC",
    "other",
)
ELEMENT_COLUMNS = ("C", "N", "O", "S", "P", "halogen")
HALOGENS = {"F", "Cl", "Br", "I"}
DISTANCE_CAP = 10.0

# Descriptor columns per family, in the saved column order.
BASICS = ("h_shift", "h_shift_abs", "is_precursor", "n_atoms", "frac_atoms", "n_cuts")
EXISTING = (
    "ring_cut",
    "loses_aromatic",
    "breaks_aromatic",
    "keeps_base",
)
POSTCUT = (
    ("kept_bonds", "cyclomatic", "rings_kept", "aromatic_rings_kept", "rings_broken")
    + ("aromatic_rings_broken", "conj_kept", "ring_bonds_cut", "aromatic_bonds_cut")
    + ("multiple_bonds_cut",)
    + tuple(f"cut_in_{t}" for t in ATOM_TYPES)
    + tuple(f"cut_out_{t}" for t in ATOM_TYPES)
    + tuple(f"stab_in_{name}" for name, _ in STABILIZING_SMARTS)
    + tuple(f"stab_out_{name}" for name, _ in STABILIZING_SMARTS)
    + tuple(f"bond_{c}" for c in BOND_CLASSES)
    + tuple(f"n_{e}" for e in ELEMENT_COLUMNS)
    + ("n_H", "rdbe")
)
CHARGE = (
    ("base_kept_matches", "base_distance", "base_distance_missing")
    + tuple(f"base_kept_{name}" for name, _ in BASE_SMARTS)
    + ("q_sum", "q_min", "q_max", "q_complement")
)
CONTEXT = ("nce", "is_ob_f", "prec_mz")
DESCRIPTORS = BASICS + EXISTING + POSTCUT + CHARGE
FAMILIES = {
    "context": (),
    "existing": BASICS + EXISTING,
    "postcut": BASICS + POSTCUT,
    "charge": BASICS + CHARGE,
    "postcut+charge": BASICS + POSTCUT + CHARGE,
    "all": DESCRIPTORS,
}


def atom_type(atom: Chem.Atom) -> str:
    """Cut-site type of one atom (element, aromaticity, hybridization)."""
    symbol = atom.GetSymbol()
    if symbol == "C":
        if atom.GetIsAromatic():
            return "C_ar"
        hybrid = atom.GetHybridization()
        if hybrid == Chem.HybridizationType.SP3:
            return "C_sp3"
        if hybrid == Chem.HybridizationType.SP:
            return "C_sp"
        carbonyl = any(
            b.GetBondType() == Chem.BondType.DOUBLE and b.GetOtherAtom(atom).GetSymbol() == "O"
            for b in atom.GetBonds()
        )
        return "C_carbonyl" if carbonyl else "C_sp2"
    if symbol == "N":
        if atom.GetIsAromatic():
            return "N_ar"
        return "N_sp3" if atom.GetHybridization() == Chem.HybridizationType.SP3 else "N_other"
    if symbol in ("O", "S"):
        return symbol
    if symbol in HALOGENS:
        return "halogen"
    return "other"


class PostCutLookups:
    """Per-molecule arrays for vectorized post-cut descriptors (RDKit atom and bond order)."""

    def __init__(self, mol: Chem.Mol, chem: MoleculeChemistry):
        self.n_atoms = chem.n_atoms
        n = self.n_atoms
        self.bonds = np.asarray(chem.bonds, dtype=np.int64).reshape(-1, 2)
        bonds = list(mol.GetBonds())
        self.bond_ring = np.asarray(chem.bond_ring, dtype=float)
        self.bond_aromatic = np.asarray([b.GetIsAromatic() for b in bonds], dtype=float)
        self.bond_multiple = np.asarray(
            [b.GetBondType() != Chem.BondType.SINGLE for b in bonds], dtype=float
        )
        classes = [c.removeprefix("ring:") for c in chem.bond_class]
        self.bond_class = np.zeros((len(bonds), len(BOND_CLASSES)))
        for i, name in enumerate(classes):
            column = BOND_CLASSES.index(name) if name in BOND_CLASSES else len(BOND_CLASSES) - 1
            self.bond_class[i, column] = 1.0
        # Bond endpoint incidence, for marking each fragment's cut-site atoms on either side.
        self.incidence_u = np.zeros((len(bonds), n))
        self.incidence_v = np.zeros((len(bonds), n))
        if len(bonds):
            self.incidence_u[np.arange(len(bonds)), self.bonds[:, 0]] = 1.0
            self.incidence_v[np.arange(len(bonds)), self.bonds[:, 1]] = 1.0
        self.atom_type = np.zeros((n, len(ATOM_TYPES)))
        for atom in mol.GetAtoms():
            self.atom_type[atom.GetIdx(), ATOM_TYPES.index(atom_type(atom))] = 1.0
        self.stabilizing = np.zeros((n, len(STABILIZING_SMARTS)))
        for column, (_, smarts) in enumerate(STABILIZING_SMARTS):
            for match in mol.GetSubstructMatches(Chem.MolFromSmarts(smarts)):
                self.stabilizing[match[0], column] = 1.0
        self.elements = np.zeros((n, len(ELEMENT_COLUMNS)))
        for atom in mol.GetAtoms():
            symbol = "halogen" if atom.GetSymbol() in HALOGENS else atom.GetSymbol()
            if symbol in ELEMENT_COLUMNS:
                self.elements[atom.GetIdx(), ELEMENT_COLUMNS.index(symbol)] = 1.0
        self.hydrogens = np.asarray(chem.hydrogens, dtype=float)
        self.rings = [np.asarray(sorted(r)) for r in mol.GetRingInfo().AtomRings()]
        self.ring_aromatic = [
            all(mol.GetAtomWithIdx(int(a)).GetIsAromatic() for a in r) for r in self.rings
        ]
        self.conjugated = np.asarray(sorted(chem.conjugated), dtype=np.int64)
        self.base_class = next(
            (i for i, (name, _) in enumerate(BASE_SMARTS) if name == chem.base_class), None
        )
        self.base_matches = [np.asarray(sorted(m), dtype=np.int64) for m in chem.base_matches]
        distance = Chem.GetDistanceMatrix(mol) if n > 1 else np.zeros((1, 1))
        # Graph distance from every atom to the nearest atom of each basic-site match.
        self.base_distance = (
            np.stack([distance[:, m].min(axis=1) for m in self.base_matches], axis=1)
            if self.base_matches
            else np.zeros((n, 0))
        )


def fragment_descriptors(inside: np.ndarray, look: PostCutLookups, charges: np.ndarray) -> dict:
    """Post-cut and charge descriptors of each fragment row of a membership matrix.

    Args:
        inside: Boolean membership, shape ``(F, n_atoms)``.
        look: The molecule's lookups.
        charges: Gasteiger charge per atom, shape ``(n_atoms,)``.

    Returns:
        Name -> float array of shape ``(F,)`` for every H-independent descriptor.
    """
    fragments = inside.shape[0]
    x = inside.astype(float)
    u, v = look.bonds[:, 0], look.bonds[:, 1]
    crossing = (inside[:, u] != inside[:, v]).astype(float)
    kept = (inside[:, u] & inside[:, v]).astype(float)
    # A cut bond's inside endpoint is a retained cut site, its outside endpoint a lost one.
    in_u, in_v = crossing * x[:, u], crossing * x[:, v]
    cut_in = ((in_u @ look.incidence_u + in_v @ look.incidence_v) > 0).astype(float)
    cut_out = ((crossing - in_u) @ look.incidence_u + (crossing - in_v) @ look.incidence_v) > 0
    cut_out = cut_out.astype(float)
    n_atoms = x.sum(axis=1)
    out = {
        "n_atoms": n_atoms,
        "frac_atoms": n_atoms / look.n_atoms,
        "n_cuts": crossing.sum(axis=1),
        "kept_bonds": kept.sum(axis=1),
        "ring_bonds_cut": crossing @ look.bond_ring,
        "aromatic_bonds_cut": crossing @ look.bond_aromatic,
        "multiple_bonds_cut": crossing @ look.bond_multiple,
    }
    out["cyclomatic"] = out["kept_bonds"] - n_atoms + 1.0
    out["ring_cut"] = (out["ring_bonds_cut"] > 0).astype(float)
    rings_kept = np.zeros(fragments)
    rings_broken = np.zeros(fragments)
    aromatic_kept = np.zeros(fragments)
    aromatic_broken = np.zeros(fragments)
    aromatic_lost = np.zeros(fragments, dtype=bool)
    for ring, aromatic in zip(look.rings, look.ring_aromatic, strict=True):
        count = x[:, ring].sum(axis=1)
        whole = count == ring.size
        partial = (count > 0) & ~whole
        rings_kept += whole
        rings_broken += partial
        if aromatic:
            aromatic_kept += whole
            aromatic_broken += partial
            aromatic_lost |= count == 0
    out.update(
        rings_kept=rings_kept,
        rings_broken=rings_broken,
        aromatic_rings_kept=aromatic_kept,
        aromatic_rings_broken=aromatic_broken,
        loses_aromatic=aromatic_lost.astype(float),
        breaks_aromatic=(aromatic_broken > 0).astype(float),
        conj_kept=(
            x[:, look.conjugated].mean(axis=1) if look.conjugated.size else np.ones(fragments)
        ),
    )
    for column, name in enumerate(ATOM_TYPES):
        out[f"cut_in_{name}"] = cut_in @ look.atom_type[:, column]
        out[f"cut_out_{name}"] = cut_out @ look.atom_type[:, column]
    for column, (name, _) in enumerate(STABILIZING_SMARTS):
        out[f"stab_in_{name}"] = cut_in @ look.stabilizing[:, column]
        out[f"stab_out_{name}"] = cut_out @ look.stabilizing[:, column]
    bond_counts = crossing @ look.bond_class
    for column, name in enumerate(BOND_CLASSES):
        out[f"bond_{name}"] = bond_counts[:, column]
    element_counts = x @ look.elements
    for column, name in enumerate(ELEMENT_COLUMNS):
        out[f"n_{name}"] = element_counts[:, column]
    out["n_H"] = x @ look.hydrogens
    # 2. Charge: the kept basic site, and how far the retained cut sites are from it.
    kept_matches = (
        np.stack([inside[:, m].all(axis=1) for m in look.base_matches], axis=1)
        if look.base_matches
        else np.zeros((fragments, 0), dtype=bool)
    )
    out["keeps_base"] = kept_matches.any(axis=1).astype(float)
    out["base_kept_matches"] = kept_matches.sum(axis=1).astype(float)
    for column, (name, _) in enumerate(BASE_SMARTS):
        out[f"base_kept_{name}"] = (
            out["keeps_base"] if look.base_class == column else np.zeros(fragments)
        )
    if kept_matches.shape[1]:
        # Nearest (retained cut site, kept basic match) pair; inf where either is missing.
        per_match = np.where(cut_in[:, :, None] > 0, look.base_distance[None], np.inf).min(axis=1)
        nearest = np.where(kept_matches, per_match, np.inf).min(axis=1)
    else:
        nearest = np.full(fragments, np.inf)
    out["base_distance_missing"] = (~np.isfinite(nearest)).astype(float)
    out["base_distance"] = np.minimum(
        np.where(np.isfinite(nearest), nearest, DISTANCE_CAP), DISTANCE_CAP
    )
    q = charges[None, :]
    out["q_sum"] = x @ charges
    out["q_complement"] = charges.sum() - out["q_sum"]
    out["q_min"] = np.where(inside, q, np.inf).min(axis=1)
    out["q_max"] = np.where(inside, q, -np.inf).max(axis=1)
    return out


def channel_descriptors(
    fragment: dict, local: np.ndarray, h_shifts: np.ndarray, n_atoms: int
) -> np.ndarray:
    """Descriptor matrix of output channels, columns in ``DESCRIPTORS`` order.

    Adds the H-dependent columns: the shift, its magnitude, the intact precursor flag, the H
    count of the neutral species (fragment atoms' H plus the shift) and its RDBE,
    ``C - (H + X) / 2 + (N + P) / 2 + 1``.

    Args:
        fragment: ``fragment_descriptors`` of the spectrum's fragments.
        local: Each channel's fragment row, shape ``(C,)``.
        h_shifts: Each channel's H shift, shape ``(C,)``.
        n_atoms: Heavy atoms of the molecule.

    Returns:
        Float array of shape ``(C, len(DESCRIPTORS))``.
    """
    h = h_shifts.astype(float)
    hydrogens = fragment["n_H"][local] + h
    columns = {
        "h_shift": h,
        "h_shift_abs": np.abs(h),
        "is_precursor": ((fragment["n_atoms"][local] == n_atoms) & (h_shifts == 0)).astype(float),
        "n_H": hydrogens,
        "rdbe": fragment["n_C"][local]
        - (hydrogens + fragment["n_halogen"][local]) / 2.0
        + (fragment["n_N"][local] + fragment["n_P"][local]) / 2.0
        + 1.0,
    }
    return np.column_stack(
        [columns[name] if name in columns else fragment[name][local] for name in DESCRIPTORS]
    )


def _identity(path: str) -> dict:
    data = Path(path).read_bytes()
    return {"path": path, "bytes": len(data), "md5": hashlib.md5(data).hexdigest()}


def extract(args: argparse.Namespace) -> None:
    """Run the annotated forward over one split and save the candidate descriptor arrays."""
    th.manual_seed(args.seed)
    np.random.seed(args.seed)  # noqa: NPY002 - seed legacy dependencies as well
    random.seed(args.seed)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()

    # 1. Model and dataset with the checkpoint's own config.
    config = load_config(args.template_fp, args.custom_fp, list(args.override))
    if config["model_type"] != "masster":
        raise ValueError(f"expected model_type masster, got {config['model_type']}")
    config = apply_inference_config(
        config, disable_preproc=True, num_workers=0, eval_batch_size=args.batch_size
    )
    device = th.device(args.device)
    if device.type == "cuda" and not th.cuda.is_available():
        raise RuntimeError("CUDA unavailable; choose --device cpu explicitly")
    if device.type == "cpu":
        config["masster_candidate_backend"] = "cpu"
    model = init_model_from_ckpt(args.checkpoint, config, device)
    model.eval()
    dataset = get_dataset_cls(config["model_type"])(split=args.split, **config)
    specs = dataset.spec_df.reset_index(drop=True)
    molecules = pd.read_pickle(config["mol_fp"]).set_index("mol_id")["mol"]
    inst_types = list(config["spec_params"]["inst_types"])
    tolerance = {
        i: (config["inst_type_loss_tol"][name]["rel"], config["inst_type_loss_tol"][name]["min_mz"])
        for i, name in enumerate(inst_types)
    }
    if args.max_spectra:
        dataset = th.utils.data.Subset(dataset, list(range(min(args.max_spectra, len(specs)))))
        specs = specs.iloc[: len(dataset)]
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=get_dataset_cls(config["model_type"]).get_collate_fn(),
        num_workers=0,
    )
    provenance = {
        "args": vars(args),
        "started_unix": started,
        "torch": th.__version__,
        "gpu": th.cuda.get_device_name(device) if device.type == "cuda" else None,
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": [_identity(p) for p in (args.checkpoint, config["spec_fp"], config["mol_fp"])],
        "spectra": len(specs),
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))

    lookups: dict[int, PostCutLookups] = {}
    saved = {
        k: [] for k in ("spec_id", "mol_id", "count", "pred", "tgt", "x", "target_norm", "context")
    }
    cursor = 0
    with th.inference_mode():
        for batch in loader:
            size = int(batch["batch_size"])
            rows = specs.iloc[cursor : cursor + size]
            if not np.array_equal(batch["spec_unique_id"].numpy(), rows.spec_id.to_numpy()):
                raise RuntimeError("batch order does not match the spectrum table")
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            out = model(**batch, return_explainability_tensors=True)
            # 2. Eval-convention target, as in the base-peak probe.
            true_mzs, true_ints, true_idx = model.preproc_spec(
                batch["spec_mzs"],
                batch["spec_ints"],
                batch["spec_batch_idxs"],
                train=False,
                pred=False,
                batch_size=size,
            )
            true_mzs = true_mzs.double().cpu().numpy()
            true_ints = true_ints.double().cpu().numpy()
            true_idx = true_idx.cpu().numpy()
            pred_mzs = out["pred_mzs"].double().cpu().numpy()
            pred_probs = out["pred_logprobs"].double().exp().cpu().numpy()
            pred_idx = out["pred_batch_idxs"].cpu().numpy()
            pred_frag = out["pred_pathway_fragment_idxs"].cpu().numpy()
            pred_h = out["pred_pathway_h_shifts"].cpu().numpy()
            member_atoms = out["fragment_atom_indices"].cpu().numpy()
            member_frag = out["fragment_atom_fragment_idxs"].cpu().numpy()
            frag_view = out["ion_h_fragment_batch_idxs"].cpu().numpy()
            atom_ptr = batch["mol_pyg"].ptr.cpu().numpy()
            charges_all = batch["mol_pyg"].atom_gasteiger_charges.double().cpu().numpy()
            inst_idx = batch["spec_inst_type"].reshape(-1).cpu().numpy()
            order = np.argsort(member_frag, kind="stable")
            member_frag, member_atoms = member_frag[order], member_atoms[order]
            frag_start = np.searchsorted(member_frag, np.arange(frag_view.size))
            frag_stop = np.searchsorted(member_frag, np.arange(frag_view.size), side="right")
            if np.any(np.diff(frag_view) < 0):
                raise RuntimeError("fragments are not grouped by spectrum")
            view_start = np.searchsorted(frag_view, np.arange(size))
            view_stop = np.searchsorted(frag_view, np.arange(size), side="right")

            for view, row in enumerate(rows.itertuples()):
                # 3. Candidate peaks and the target on them.
                channels = np.flatnonzero(pred_idx == view)
                probs = pred_probs[channels] / pred_probs[channels].sum()
                cand_mzs, pred, inverse = merge_channels(pred_mzs[channels], probs)
                peaks = true_idx == view
                peak_ints = true_ints[peaks]
                rel, min_mz = tolerance[int(inst_idx[view])]
                on_cand, _ = map_targets(true_mzs[peaks], peak_ints, cand_mzs, rel, min_mz)
                # 4. Descriptors per fragment, per channel, then per candidate peak.
                if row.mol_id not in lookups:
                    mol = molecules.loc[row.mol_id]
                    lookups[row.mol_id] = PostCutLookups(mol, MoleculeChemistry(mol))
                look = lookups[row.mol_id]
                if atom_ptr[view + 1] - atom_ptr[view] != look.n_atoms:
                    raise RuntimeError(f"spectrum {row.spec_id}: graph and molecule atoms differ")
                first, last = view_start[view], view_stop[view]
                members = slice(frag_start[first], frag_stop[last - 1])
                inside = np.zeros((last - first, look.n_atoms), dtype=bool)
                inside[member_frag[members] - first, member_atoms[members] - atom_ptr[view]] = True
                x = channel_descriptors(
                    fragment_descriptors(
                        inside, look, charges_all[atom_ptr[view] : atom_ptr[view + 1]]
                    ),
                    pred_frag[channels] - first,
                    pred_h[channels],
                    look.n_atoms,
                )
                weight = np.maximum(pred, 1e-300)[:, None]
                cand_x = (
                    np.stack(
                        [
                            np.bincount(inverse, probs * x[:, a], cand_mzs.size)
                            for a in range(x.shape[1])
                        ],
                        axis=1,
                    )
                    / weight
                )
                keep = (pred >= SAVE_FLOOR) | (on_cand > 0)
                saved["spec_id"].append(int(row.spec_id))
                saved["mol_id"].append(int(row.mol_id))
                saved["count"].append(int(keep.sum()))
                saved["pred"].append(pred[keep].astype(np.float32))
                saved["tgt"].append(on_cand[keep].astype(np.float32))
                saved["x"].append(cand_x[keep].astype(np.float32))
                saved["target_norm"].append(float(np.linalg.norm(peak_ints)))
                saved["context"].append(
                    [
                        float(np.mean(row.nce)) / 100.0,
                        float(row.inst_type == "OB_F"),
                        float(row.prec_mz) / 1000.0,
                    ]
                )
            cursor += size
            if cursor % (args.batch_size * 20) == 0 or cursor == len(specs):
                LOGGER.info("%d/%d spectra, %.0f s", cursor, len(specs), time.time() - started)

    # 5. Save the arrays and close the provenance record.
    np.savez(
        output,
        spec_id=np.asarray(saved["spec_id"]),
        mol_id=np.asarray(saved["mol_id"]),
        offsets=np.concatenate([[0], np.cumsum(saved["count"])]),
        pred=np.concatenate(saved["pred"]),
        tgt=np.concatenate(saved["tgt"]),
        x=np.concatenate(saved["x"]),
        target_norm=np.asarray(saved["target_norm"]),
        context=np.asarray(saved["context"], dtype=np.float32),
        descriptors=np.asarray(DESCRIPTORS),
        context_names=np.asarray(CONTEXT),
    )
    provenance["elapsed_seconds"] = time.time() - started
    provenance["status"] = "complete"
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))


class Correction(th.nn.Module):
    """``logit' = log p + g([log p, context, x])``, zero at start.

    ``g`` is a two-layer MLP of width ``hidden``, or linear when ``hidden`` is 0.
    """

    def __init__(self, width: int, hidden: int, dropout: float):
        super().__init__()
        if hidden == 0:
            self.net = th.nn.Sequential(th.nn.Linear(width, 1))
        else:
            self.net = th.nn.Sequential(
                th.nn.Linear(width, hidden),
                th.nn.ReLU(),
                th.nn.Dropout(dropout),
                th.nn.Linear(hidden, hidden),
                th.nn.ReLU(),
                th.nn.Linear(hidden, 1),
            )
        th.nn.init.zeros_(self.net[-1].weight)
        th.nn.init.zeros_(self.net[-1].bias)

    def forward(self, log_pred: th.Tensor, features: th.Tensor) -> th.Tensor:
        """Corrected logits, shape ``(N,)``, from ``log_pred (N,)`` and ``features (N, W)``."""
        return log_pred + self.net(features).squeeze(-1)


def segment_scores(
    logits: th.Tensor, tgt: th.Tensor, segment: th.Tensor, target_norm: th.Tensor, spectra: int
) -> tuple[th.Tensor, th.Tensor, th.Tensor]:
    """Per-spectrum listwise cross-entropy, cosine and top-candidate agreement.

    Args:
        logits: Candidate logits, shape ``(N,)``.
        tgt: Target intensity per candidate, shape ``(N,)``.
        segment: Spectrum index per candidate in ``[0, spectra)``, shape ``(N,)``.
        target_norm: L2 norm of each whole target, shape ``(spectra,)``.
        spectra: Number of spectra.

    Returns:
        Cross-entropy against the covered target distribution (0 where nothing is covered),
        cosine with the whole target, and 1.0 where the top predicted candidate is the top
        target candidate; each shape ``(spectra,)``.
    """
    top = th.full((spectra,), -th.inf, device=logits.device).scatter_reduce(
        0, segment, logits, reduce="amax"
    )
    e = th.exp(logits - top[segment])
    z = th.zeros(spectra, device=logits.device).index_add(0, segment, e)
    log_softmax = logits - top[segment] - th.log(z[segment])
    covered = th.zeros(spectra, device=logits.device).index_add(0, segment, tgt)
    weight = tgt / covered.clamp_min(1e-30)[segment]
    cross_entropy = -th.zeros(spectra, device=logits.device).index_add(
        0, segment, weight * log_softmax
    )
    p = e / z[segment]
    dot = th.zeros(spectra, device=logits.device).index_add(0, segment, p * tgt)
    norm = th.zeros(spectra, device=logits.device).index_add(0, segment, p * p).sqrt()
    cosine = dot / (norm * target_norm)
    tgt_top = th.full((spectra,), -th.inf, device=logits.device).scatter_reduce(
        0, segment, tgt, reduce="amax"
    )
    is_pred_top = logits == top[segment]
    agree = th.zeros(spectra, device=logits.device).index_add(
        0, segment, (is_pred_top & (tgt == tgt_top[segment]) & (tgt > 0)).float()
    )
    return cross_entropy, cosine, (agree > 0).float()


def _rows(offsets: np.ndarray, spectra: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Candidate rows of the given spectra and their local spectrum index."""
    counts = np.diff(offsets)[spectra]
    rows = np.concatenate([np.arange(offsets[s], offsets[s + 1]) for s in spectra])
    return rows, np.repeat(np.arange(spectra.size), counts)


def fit_family(
    data: dict,
    columns: list[int],
    fit_spectra: np.ndarray,
    test_spectra: np.ndarray,
    args: argparse.Namespace,
    seed: int,
) -> np.ndarray:
    """Fit the correction on ``fit_spectra`` and return held-out per-spectrum metrics.

    Returns:
        Array ``(len(test_spectra), 2)``: cosine and top-candidate agreement after correction.
    """
    device = th.device(args.device)
    rng = np.random.default_rng(seed)
    th.manual_seed(seed)
    # 1. Inner early-stopping split by molecule, inside the fit half.
    mols = np.unique(data["mol_id"][fit_spectra])
    inner = set(rng.choice(mols, max(1, int(0.1 * mols.size)), replace=False).tolist())
    in_inner = np.array([m in inner for m in data["mol_id"][fit_spectra]])
    train_spectra, stop_spectra = fit_spectra[~in_inner], fit_spectra[in_inner]

    def features(spectra: np.ndarray) -> tuple:
        rows, segment = _rows(data["offsets"], spectra)
        log_pred = np.log(np.maximum(data["pred"][rows], 1e-12))
        parts = [log_pred[:, None], data["context"][spectra][segment]]
        if columns:
            parts.append(data["x"][rows][:, columns])
        return (
            np.concatenate(parts, axis=1).astype(np.float32),
            log_pred.astype(np.float32),
            data["tgt"][rows].astype(np.float32),
            segment,
        )

    x_train, _, _, _ = features(train_spectra)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    del x_train
    model = Correction(1 + len(CONTEXT) + len(columns), args.hidden, args.dropout).to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    def batch_tensors(spectra: np.ndarray) -> tuple:
        x, log_pred, tgt, segment = features(spectra)
        return (
            th.as_tensor((x - mean) / std, device=device),
            th.as_tensor(log_pred, device=device),
            th.as_tensor(tgt, device=device),
            th.as_tensor(segment, device=device),
            th.as_tensor(data["target_norm"][spectra], dtype=th.float32, device=device),
        )

    def evaluate(spectra: np.ndarray) -> np.ndarray:
        model.eval()
        outputs = []
        with th.no_grad():
            for start in range(0, spectra.size, args.batch_spectra):
                chunk = spectra[start : start + args.batch_spectra]
                x, log_pred, tgt, segment, norm = batch_tensors(chunk)
                ce, cos, agree = segment_scores(model(log_pred, x), tgt, segment, norm, chunk.size)
                outputs.append(th.stack([ce, cos, agree], dim=1).cpu().numpy())
        return np.concatenate(outputs)

    # 2. Minibatches of whole spectra; keep the weights with the best inner loss.
    best, best_state, stale = np.inf, None, 0
    for epoch in range(args.epochs):
        model.train()
        order = rng.permutation(train_spectra)
        for start in range(0, order.size, args.batch_spectra):
            chunk = order[start : start + args.batch_spectra]
            x, log_pred, tgt, segment, norm = batch_tensors(chunk)
            ce, _, _ = segment_scores(model(log_pred, x), tgt, segment, norm, chunk.size)
            loss = ce.mean()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        inner_loss = evaluate(stop_spectra)[:, 0].mean()
        LOGGER.info("  epoch %d inner cross-entropy %.5f", epoch, inner_loss)
        if inner_loss < best - 1e-5:
            best, stale = inner_loss, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= args.patience:
                break
    model.load_state_dict(best_state)
    return evaluate(test_spectra)[:, 1:]


def fit(args: argparse.Namespace) -> None:
    """Cross-fit every family's correction and report held-out gains."""
    loaded = np.load(args.input)
    data = {k: loaded[k] for k in loaded.files}
    names = [str(n) for n in data["descriptors"]]
    spectra = np.arange(data["spec_id"].size)
    fold = data["mol_id"] % 2
    device = th.device(args.device)

    # 1. The raw prediction on the saved candidates, as the reference.
    raw = np.zeros((spectra.size, 2))
    for start in range(0, spectra.size, args.batch_spectra):
        chunk = spectra[start : start + args.batch_spectra]
        rows, segment = _rows(data["offsets"], chunk)
        logits = th.as_tensor(np.log(np.maximum(data["pred"][rows], 1e-12)), device=device)
        _, cos, agree = segment_scores(
            logits,
            th.as_tensor(data["tgt"][rows], device=device),
            th.as_tensor(segment, device=device),
            th.as_tensor(data["target_norm"][chunk], dtype=th.float32, device=device),
            chunk.size,
        )
        raw[chunk] = th.stack([cos, agree], dim=1).cpu().numpy()

    # 2. Cross-fit each family: fit on one parity, score the other, then swap.
    rows = []
    held = {}
    for family in args.families:
        columns = [names.index(n) for n in FAMILIES[family]]
        result = np.full((spectra.size, 2), np.nan)
        for k in (0, 1):
            LOGGER.info("%s: fit on parity %d", family, k)
            test = spectra[fold != k]
            result[test] = fit_family(data, columns, spectra[fold == k], test, args, seed=17 + k)
        held[family] = result
    for family, result in held.items():
        gain = result[:, 0] - raw[:, 0]
        row = {
            "family": family,
            "columns": len(FAMILIES[family]),
            "raw_cos": raw[:, 0].mean(),
            "cos": result[:, 0].mean(),
            "gain": gain.mean(),
            "raw_top_agree": raw[:, 1].mean(),
            "top_agree": result[:, 1].mean(),
        }
        sums = clustered_bootstrap(
            data["mol_id"], {"g": gain, "n": np.ones(gain.size)}, args.reps, seed=0
        )
        row["gain_lo"], row["gain_hi"] = np.percentile(sums[:, 0] / sums[:, 1], [2.5, 97.5])
        if family != "context" and "context" in held:
            over = result[:, 0] - held["context"][:, 0]
            sums = clustered_bootstrap(
                data["mol_id"], {"g": over, "n": np.ones(over.size)}, args.reps, seed=0
            )
            row["over_context"] = over.mean()
            row["over_context_lo"], row["over_context_hi"] = np.percentile(
                sums[:, 0] / sums[:, 1], [2.5, 97.5]
            )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    per_spectrum = pd.DataFrame(
        {"spec_id": data["spec_id"], "mol_id": data["mol_id"], "raw_cos": raw[:, 0]}
        | {f"cos_{f}": r[:, 0] for f, r in held.items()}
        | {f"agree_{f}": r[:, 1] for f, r in held.items()}
    )
    per_spectrum.to_pickle(Path(args.output).with_suffix(".per_spectrum.pkl"))
    LOGGER.info("\n%s", table.round(4).to_string(index=False))


def main(argv: list[str] | None = None) -> None:
    """Dispatch the ``extract`` and ``fit`` subcommands."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    ex = sub.add_parser("extract")
    ex.add_argument("--checkpoint", required=True)
    ex.add_argument("--custom_fp", required=True)
    ex.add_argument("--template_fp", default="config")
    ex.add_argument("--override", nargs="*", default=[])
    ex.add_argument("--split", choices=("train", "val", "test"), default="val")
    ex.add_argument("--batch_size", type=int, default=128)
    ex.add_argument("--seed", type=int, default=0)
    ex.add_argument("--max_spectra", type=int, default=0, help="first N spectra; 0 = all")
    ex.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ex.add_argument("--output", required=True, help=".npz; .provenance.json beside it")
    ft = sub.add_parser("fit")
    ft.add_argument("--input", required=True)
    ft.add_argument("--output", required=True, help="csv; .per_spectrum.pkl beside it")
    ft.add_argument("--families", nargs="+", default=list(FAMILIES))
    ft.add_argument("--hidden", type=int, default=128)
    ft.add_argument("--dropout", type=float, default=0.1)
    ft.add_argument("--lr", type=float, default=1e-3)
    ft.add_argument("--weight_decay", type=float, default=1e-5)
    ft.add_argument("--epochs", type=int, default=30)
    ft.add_argument("--patience", type=int, default=3)
    ft.add_argument("--batch_spectra", type=int, default=512)
    ft.add_argument("--reps", type=int, default=200)
    ft.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "extract":
        extract(args)
    else:
        fit(args)


if __name__ == "__main__":
    main()
