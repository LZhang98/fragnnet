"""Does an explicit description of each candidate after cutting carry information a MASSter
checkpoint does not use? A cross-fitted, per-candidate correction gate on validation molecules.

Step 2 of the wrong-peak analysis (2026-09-29), after
``docs/findings/masster/masster_i1_base_peak_errors_2026_09_28.md``. Subcommands ``extract`` and
``fit`` (plus ``ceiling``, below):

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

Rule provenance (G1 of ``95341fce:docs/plans/masster/masster_smarts_pathway_plan.md, removed 2026-10-18``). ``extract`` also
saves, per candidate peak, the float64 ``mz`` and ``rule_x``: whether the peak's m/z is a valid
production ion channel (``MassterModel._fragment_masses_and_valid_channels`` with the template's
cut count and formed-bond count) of either side of an eligible SMARTS template of the molecule.
Columns: ``rule_any`` (primary marker), ``rule_disc`` / ``rule_conn`` (the side is disconnected /
connected), ``rule_atomset`` (share of the peak's channels whose fragment IS a template side),
``rule_any_tol`` (instrument-tolerance sensitivity marker) and one multi-hot ``rule_<name>`` per
active rule. ``fit`` adds the families ``rules``, ``rules+existing``, ``rules_tol`` and the
negative control ``rules_permuted`` (rule columns permuted across the candidates of each
spectrum), and reports gains on the marked spectra. ``ceiling`` is the CPU headroom read: set the
marked peaks' shares to their target shares, against matched unmarked control peaks.
``extract --mol_ids`` / ``--template_bearing_only`` restrict the spectra for smoke tests.

The correction learns how the checkpoint errs on new molecules. A branch trained on training
molecules cannot learn that directly, so a gain is evidence that the information exists and
transfers across molecules, not that a branch will recover it. A null says an explicit
description of this kind carries nothing the model does not already use.

Run from the repository root:

    python scripts/debug/masster_memorization/debug_masster_postcut_information_gate.py extract \
        --checkpoint "saved_ckpts/i1_inst_ob_obf_i50ig7rf/ckpt/model-epoch=059.ckpt" \
        --custom_fp config/masster/mh_ft_memorization_route_2026_09_24/i1_inst_ob_obf.yml \
        --output output/postcut_gate/i1_e059_val.npz
    python scripts/debug/masster_memorization/debug_masster_postcut_information_gate.py fit \
        --input output/postcut_gate/i1_e059_val.npz --output output/postcut_gate/fit.csv
    python scripts/debug/masster_memorization/debug_masster_postcut_information_gate.py ceiling \
        --input output/postcut_gate/i1_e059_val.npz --output output/postcut_gate/ceiling.csv \
        --fit_csv output/postcut_gate/fit.csv
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import re
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch as th
from rdkit import Chem, RDLogger
from torch.utils.data import DataLoader

from fragnnet.inference.checkpoint import apply_inference_config, init_model_from_ckpt
from fragnnet.models.masster.candidates.smarts_templates import (
    MASSTER_RULE_NAMES,
    build_masster_templates,
    filter_masster_templates,
    formed_bond_count,
)
from fragnnet.runner import load_config
from fragnnet.utils.setup_utils import get_dataset_cls

# Support both the script entry point and imports from pytest.
try:
    from .debug_masster_base_peak_errors import SAVE_FLOOR, _identity, map_targets, merge_channels
    from .debug_masster_base_peak_tables import BASE_SMARTS, MoleculeChemistry, clustered_bootstrap
except ImportError:
    from debug_masster_base_peak_errors import SAVE_FLOOR, _identity, map_targets, merge_channels
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

# Rule provenance columns, saved apart from ``x`` (``rule_x``) so old caches and the 98
# descriptor columns keep their meaning. The per-rule columns follow the registry read at import
# time; the cache records the names it was built with.
RULE_NAMES = tuple(MASSTER_RULE_NAMES)
RULE_SUMMARY = ("rule_any", "rule_disc", "rule_conn", "rule_atomset", "rule_any_tol")
RULE_COLUMNS = RULE_SUMMARY + tuple(f"rule_{name}" for name in RULE_NAMES)
# The features mirroring a rule-pathway token: every rule column except the sensitivity marker.
RULE_FAMILY_COLUMNS = tuple(c for c in RULE_COLUMNS if c != "rule_any_tol")
FAMILIES |= {
    "rules": RULE_FAMILY_COLUMNS,
    "rules+existing": BASICS + EXISTING + RULE_FAMILY_COLUMNS,
    "rules_tol": ("rule_any_tol",),
    # Negative control: the rules family with its columns permuted across each spectrum's rows.
    "rules_permuted": RULE_FAMILY_COLUMNS,
}
PERMUTED_FAMILIES = frozenset({"rules_permuted"})
# The families that need ``rule_x`` in the cache.
RULE_FAMILIES = ("rules", "rules+existing", "rules_tol", "rules_permuted")
# Primary marker match: candidate m/z (rounded to 1e-6) within this many Da of a channel.
MARKER_TOL_DA = 1e-5
# Headroom decision thresholds of the plan (candidate-space cosine).
HEADROOM_TARGET = 0.003


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


@dataclass(frozen=True)
class TemplateSide:
    """One side of an eligible SMARTS template, as the production candidate builder sees it.

    Attributes:
        rule_name: Registered rule that produced the template.
        rule_position: Position of ``rule_name`` in ``RULE_NAMES``.
        atoms: Molecule-local atom indices of the side.
        connected: Whether the side is one connected piece of the precursor graph.
        cut_count: Severed bonds the builder assigns the side: ALL of the template's broken
            bonds, not only those crossing this side.
        formed_bonds: Bonds the assembly forms inside the side (pieces minus one).
    """

    rule_name: str
    rule_position: int
    atoms: tuple[int, ...]
    connected: bool
    cut_count: int
    formed_bonds: int


def template_sides(mol: Chem.Mol, precursor_type: str) -> tuple[TemplateSide, ...]:
    """Both sides of every eligible template of a molecule.

    Templates come from ``filter_masster_templates`` directly (a templates-off checkpoint has no
    rule-hash buffer for the model's own eligibility method), with the production values for
    each side's cut count and formed-bond count (``candidates/builder.py``).

    Args:
        mol: The molecule as ``Chem.MolFromSmiles`` builds it, which is what training matches.
        precursor_type: The spectrum's adduct, e.g. ``"[M+H]+"``.

    Returns:
        Every side of every eligible template, in template then component order.
    """
    edges = np.asarray(
        [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=np.int32
    ).reshape(-1, 2)
    endpoints = tuple(tuple(int(atom) for atom in column) for column in edges.T)
    templates = filter_masster_templates(
        build_masster_templates(mol), precursor_type=precursor_type, fragmentation_mode=None
    )
    sides = []
    for template in templates:
        for component in template.components:
            formed = formed_bond_count(component, endpoints)
            sides.append(
                TemplateSide(
                    rule_name=template.rule_name,
                    rule_position=RULE_NAMES.index(template.rule_name),
                    atoms=tuple(int(a) for a in component),
                    connected=formed == 0,
                    cut_count=len(template.broken_bond_indices),
                    formed_bonds=formed,
                )
            )
    return tuple(sides)


def side_channel_table(
    core: th.nn.Module,
    mol_pyg: object,
    sides: Sequence[TemplateSide],
    side_views: Sequence[int],
    precursor_type_indices: th.Tensor,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Production ion channels of template sides, in one batched call.

    Calls ``MassterModel._fragment_masses_and_valid_channels`` with the atom offsets of the
    batched graph, so the m/z and the valence bound (including the formed-bond correction) are
    the checkpoint's own.

    Args:
        core: The ``MassterModel``.
        mol_pyg: The batched precursor graphs (``batch["mol_pyg"]``).
        sides: Sides to score, molecule-local atoms.
        side_views: Spectrum index in the batch of each side.
        precursor_type_indices: Adduct index per spectrum, shape ``(batch_size,)``.

    Returns:
        ``(mzs, valid, h_shifts)``: float64 m/z and validity of shape ``(S, C)`` over the ion
        modes and H shifts, and each channel's H shift ``(C,)``.
    """
    ptr = mol_pyg.ptr.cpu().numpy()
    device = mol_pyg.ptr.device
    atoms = np.concatenate(
        [np.asarray(s.atoms) + ptr[v] for s, v in zip(sides, side_views, strict=True)]
    )
    fragments = np.repeat(np.arange(len(sides)), [len(s.atoms) for s in sides])
    ion_mzs, valid = core._fragment_masses_and_valid_channels(
        mol_pyg,
        th.as_tensor(atoms, dtype=th.long, device=device),
        th.as_tensor(fragments, dtype=th.long, device=device),
        th.as_tensor(np.asarray(side_views), dtype=th.long, device=device),
        len(sides),
        precursor_type_indices.reshape(-1),
        th.as_tensor([s.cut_count for s in sides], dtype=th.long, device=device),
        th.as_tensor([s.formed_bonds for s in sides], dtype=th.long, device=device),
    )
    modes = ion_mzs.shape[1]
    max_h = int(core.max_h_transfer)
    h_shifts = np.tile(np.arange(-max_h, max_h + 1), modes)
    return (
        ion_mzs.double().reshape(len(sides), -1).cpu().numpy(),
        valid.reshape(len(sides), -1).cpu().numpy(),
        h_shifts,
    )


def _window(sorted_mzs: np.ndarray, centre: np.ndarray, half_width: np.ndarray | float) -> tuple:
    """Index range of ``sorted_mzs`` within ``centre +- half_width``, one per centre."""
    lower = np.searchsorted(sorted_mzs, centre - half_width, side="left")
    upper = np.searchsorted(sorted_mzs, centre + half_width, side="right")
    return lower, upper


def rule_marker_columns(
    cand_mzs: np.ndarray,
    side_mzs: np.ndarray,
    side_valid: np.ndarray,
    sides: Sequence[TemplateSide],
    *,
    tol_da: float,
    rel: float,
    min_mz: float,
) -> np.ndarray:
    """Rule provenance columns of one spectrum's candidate peaks, ``rule_atomset`` left at 0.

    A candidate is marked by a side when its m/z lies within ``tol_da`` of a VALID production
    channel of that side (H shifts and ion modes included). ``rule_any_tol`` widens the match to
    the instrument tolerance ``rel * max(m/z, min_mz)``, which admits near isobars.

    Args:
        cand_mzs: Sorted unique candidate m/z (rounded to 1e-6), shape ``(K,)``.
        side_mzs: Channel m/z of each side, shape ``(S, C)``.
        side_valid: Channel validity, shape ``(S, C)``.
        sides: The sides, for their rule and connectivity.
        tol_da: Primary match tolerance in Da.
        rel: Relative instrument tolerance.
        min_mz: m/z below which the instrument tolerance is held at ``rel * min_mz``.

    Returns:
        Float array ``(K, len(RULE_COLUMNS))``.
    """
    column = {name: i for i, name in enumerate(RULE_COLUMNS)}
    out = np.zeros((cand_mzs.size, len(RULE_COLUMNS)), dtype=np.float32)
    for side, mzs, valid in zip(sides, side_mzs, side_valid, strict=True):
        mzs = mzs[valid]
        if mzs.size == 0:
            continue
        kind = "rule_conn" if side.connected else "rule_disc"
        for lower, upper in zip(*_window(cand_mzs, mzs, tol_da), strict=True):
            out[
                lower:upper, [column["rule_any"], column[kind], column[f"rule_{side.rule_name}"]]
            ] = 1
        wide = rel * np.maximum(mzs, min_mz)
        for lower, upper in zip(*_window(cand_mzs, mzs, wide), strict=True):
            out[lower:upper, column["rule_any_tol"]] = 1
    return out


def atomset_channel_match(
    inside: np.ndarray, sides: Sequence[TemplateSide], n_atoms: int
) -> np.ndarray:
    """Whether each fragment's atom set is exactly the atom set of some template side.

    Args:
        inside: Boolean membership of the spectrum's fragments, shape ``(F, n_atoms)``.
        sides: Template sides of the molecule.
        n_atoms: Heavy atoms of the molecule.

    Returns:
        Bool array ``(F,)``.
    """
    masks = np.zeros((len(sides), n_atoms), dtype=bool)
    for i, side in enumerate(sides):
        masks[i, list(side.atoms)] = True
    return (inside[:, None, :] == masks[None]).all(axis=2).any(axis=1)


def parse_mol_ids(value: str) -> np.ndarray:
    """Molecule ids from a comma-separated list or the path of a file holding one.

    A file may separate ids by commas or whitespace; a leading ``mol_id`` header is skipped.
    """
    text = Path(value).read_text() if Path(value).is_file() else value
    tokens = [t for t in re.split(r"[,\s]+", text.strip()) if t and t != "mol_id"]
    try:
        return np.asarray([int(t) for t in tokens], dtype=np.int64)
    except ValueError as error:
        raise ValueError(f"--mol_ids holds a non-integer token: {error}") from error


def select_spectra(specs: pd.DataFrame, sides_of: object, args: argparse.Namespace) -> list[int]:
    """Row positions of the spectra to extract, in order.

    Applies ``--mol_ids``, ``--template_bearing_only`` (with ``--max_molecules`` drawing that many
    of the bearing molecules at random from ``--seed``) and ``--max_spectra`` in that order.

    Args:
        specs: The split's spectrum table, reset to positional order.
        sides_of: Callable ``(mol_id, precursor_type) -> tuple[TemplateSide, ...]``.
        args: Parsed extract arguments.

    Returns:
        Ascending row positions into ``specs``.
    """
    keep = np.ones(len(specs), dtype=bool)
    if args.mol_ids:
        wanted = parse_mol_ids(args.mol_ids)
        absent = np.setdiff1d(wanted, specs["mol_id"].to_numpy())
        if absent.size:
            raise ValueError(f"--mol_ids not in the {args.split} split: {absent.tolist()[:10]}")
        keep &= specs["mol_id"].isin(wanted).to_numpy()
    if args.template_bearing_only:
        pairs = specs.loc[keep, ["mol_id", "prec_type"]].drop_duplicates()
        bearing = {(int(m), p) for m, p in pairs.itertuples(index=False) if sides_of(int(m), p)}
        if args.max_molecules:
            molecules = sorted({m for m, _ in bearing})
            rng = np.random.default_rng(args.seed)
            chosen = set(rng.permutation(molecules)[: args.max_molecules].tolist())
            bearing = {(m, p) for m, p in bearing if m in chosen}
        keep &= np.asarray(
            [
                (int(m), p) in bearing
                for m, p in zip(specs["mol_id"], specs["prec_type"], strict=True)
            ]
        )
    indices = np.flatnonzero(keep)
    if args.max_spectra:
        indices = indices[: args.max_spectra]
    if indices.size == 0:
        raise ValueError("no spectra selected")
    return indices.tolist()


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
    core = getattr(model, "model", model)
    precursor_types = tuple(core.precursor_types)
    dataset = get_dataset_cls(config["model_type"])(split=args.split, **config)
    specs = dataset.spec_df.reset_index(drop=True)
    mol_table = pd.read_pickle(config["mol_fp"]).set_index("mol_id")
    molecules = mol_table["mol"]
    inst_types = list(config["spec_params"]["inst_types"])
    tolerance = {
        i: (config["inst_type_loss_tol"][name]["rel"], config["inst_type_loss_tol"][name]["min_mz"])
        for i, name in enumerate(inst_types)
    }

    # 2. Eligible template sides per (molecule, adduct), built as training builds them.
    sides_cache: dict[tuple[int, str], tuple[TemplateSide, ...]] = {}

    def sides_of(mol_id: int, precursor_type: str) -> tuple[TemplateSide, ...]:
        if (mol_id, precursor_type) not in sides_cache:
            mol = Chem.MolFromSmiles(mol_table["smiles"].loc[mol_id])
            if mol is None:
                raise ValueError(f"molecule {mol_id}: invalid SMILES for SMARTS matching")
            sides_cache[(mol_id, precursor_type)] = template_sides(mol, precursor_type)
        return sides_cache[(mol_id, precursor_type)]

    indices = select_spectra(specs, sides_of, args)
    if len(indices) != len(specs):
        dataset = th.utils.data.Subset(dataset, indices)
        specs = specs.iloc[indices].reset_index(drop=True)
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
        "masster_rule_names": list(RULE_NAMES),
        "rule_columns": list(RULE_COLUMNS),
        "marker_tol_da": args.marker_tol,
        "precursor_types": list(precursor_types),
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))

    lookups: dict[int, PostCutLookups] = {}
    saved = {
        k: []
        for k in (
            "spec_id",
            "mol_id",
            "count",
            "pred",
            "tgt",
            "x",
            "target_norm",
            "context",
            "mz",
            "rule_x",
            "n_sides",
            "n_marked_all",
            "n_cands_all",
        )
    }
    atomset_column = RULE_COLUMNS.index("rule_atomset")
    any_column = RULE_COLUMNS.index("rule_any")
    cursor = 0
    with th.inference_mode():
        for batch in loader:
            size = int(batch["batch_size"])
            rows = specs.iloc[cursor : cursor + size]
            if not np.array_equal(batch["spec_unique_id"].numpy(), rows.spec_id.to_numpy()):
                raise RuntimeError("batch order does not match the spectrum table")
            batch = model.transfer_batch_to_device(batch, device, dataloader_idx=0)
            out = model(**batch, return_explainability_tensors=True)
            if "pred_pathway_scores" in out:  # merged rows: use the per-pathway rows
                out = {
                    **out,
                    "pred_mzs": out["pred_pathway_mzs"],
                    "pred_ints": out["pred_pathway_scores"],
                    "pred_logprobs": out["pred_pathway_scores"].log(),
                    "pred_batch_idxs": out["pred_pathway_batch_idxs"],
                }
            # 3. Eval-convention target, as in the base-peak probe.
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
            # 4. The eligible template sides of every spectrum, scored in one batched call.
            prec_idx = batch["spec_prec_type"].reshape(-1).cpu().numpy()
            view_sides = [
                sides_of(int(row.mol_id), precursor_types[int(prec_idx[v])])
                for v, row in enumerate(rows.itertuples())
            ]
            flat = [(v, side) for v, group in enumerate(view_sides) for side in group]
            side_start = np.cumsum([0] + [len(group) for group in view_sides])
            if flat:
                side_mzs, side_valid, _ = side_channel_table(
                    core,
                    batch["mol_pyg"],
                    [side for _, side in flat],
                    [v for v, _ in flat],
                    batch["spec_prec_type"],
                )
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
                # 5. Candidate peaks and the target on them.
                channels = np.flatnonzero(pred_idx == view)
                probs = pred_probs[channels] / pred_probs[channels].sum()
                cand_mzs, pred, inverse = merge_channels(pred_mzs[channels], probs)
                peaks = true_idx == view
                peak_ints = true_ints[peaks]
                rel, min_mz = tolerance[int(inst_idx[view])]
                on_cand, _ = map_targets(true_mzs[peaks], peak_ints, cand_mzs, rel, min_mz)
                # 6. Descriptors per fragment, per channel, then per candidate peak.
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
                # 7. Rule provenance: candidates at a template side's production channel.
                sides = view_sides[view]
                rule_x = np.zeros((cand_mzs.size, len(RULE_COLUMNS)), dtype=np.float32)
                if sides:
                    if max(max(side.atoms) for side in sides) >= look.n_atoms:
                        raise RuntimeError(f"spectrum {row.spec_id}: template atoms outside graph")
                    mine = slice(side_start[view], side_start[view + 1])
                    rule_x = rule_marker_columns(
                        cand_mzs,
                        side_mzs[mine],
                        side_valid[mine],
                        sides,
                        tol_da=args.marker_tol,
                        rel=rel,
                        min_mz=min_mz,
                    )
                    is_side = atomset_channel_match(inside, sides, look.n_atoms)
                    rule_x[:, atomset_column] = (
                        np.bincount(
                            inverse,
                            probs * is_side[pred_frag[channels] - first],
                            cand_mzs.size,
                        )
                        / weight[:, 0]
                    )
                keep = (pred >= SAVE_FLOOR) | (on_cand > 0)
                saved["mz"].append(cand_mzs[keep].astype(np.float64))
                saved["rule_x"].append(rule_x[keep])
                saved["n_sides"].append(len(sides))
                saved["n_marked_all"].append(int((rule_x[:, any_column] > 0).sum()))
                saved["n_cands_all"].append(int(cand_mzs.size))
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
                        float(row.inst_type == "ORB_F"),
                        float(row.prec_mz) / 1000.0,
                    ]
                )
            cursor += size
            if cursor % (args.batch_size * 20) == 0 or cursor == len(specs):
                LOGGER.info("%d/%d spectra, %.0f s", cursor, len(specs), time.time() - started)

    # 8. Save the arrays and close the provenance record.
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
        mz=np.concatenate(saved["mz"]),
        rule_x=np.concatenate(saved["rule_x"]),
        rule_columns=np.asarray(RULE_COLUMNS),
        n_sides=np.asarray(saved["n_sides"]),
        n_marked_all=np.asarray(saved["n_marked_all"]),
        n_cands_all=np.asarray(saved["n_cands_all"]),
    )
    n_sides = np.asarray(saved["n_sides"])
    provenance["templated_spectra"] = int((n_sides > 0).sum())
    provenance["templated_molecules"] = len(
        {m for m, s in zip(saved["mol_id"], n_sides, strict=True) if s}
    )
    provenance["marked_spectra"] = int((np.asarray(saved["n_marked_all"]) > 0).sum())
    saved_marks = np.concatenate([r[:, any_column] for r in saved["rule_x"]]) > 0
    provenance["marked_rows_saved"] = int(saved_marks.sum())
    provenance["marked_spectra_saved"] = int(
        (
            np.bincount(
                np.repeat(np.arange(len(saved["count"])), saved["count"]),
                weights=saved_marks,
                minlength=len(saved["count"]),
            )
            > 0
        ).sum()
    )
    provenance["marked_peaks"] = int(sum(saved["n_marked_all"]))
    provenance["rows"] = int(sum(saved["count"]))
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
    rule_columns: Sequence[int] = (),
    rule_key: str = "rule_x",
) -> np.ndarray:
    """Fit the correction on ``fit_spectra`` and return held-out per-spectrum metrics.

    Args:
        data: The cache arrays.
        columns: Descriptor columns of ``data["x"]`` given to the correction.
        fit_spectra: Spectra to fit on.
        test_spectra: Spectra to score.
        args: Fit arguments.
        seed: Seed of the inner split, the minibatch order and the initialisation.
        rule_columns: Columns of ``data[rule_key]`` given to the correction.
        rule_key: Which rule array to read: ``rule_x``, or ``rule_x_permuted`` for the control.

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
        if rule_columns:
            parts.append(data[rule_key][rows][:, list(rule_columns)])
        return (
            np.concatenate(parts, axis=1).astype(np.float32),
            log_pred.astype(np.float32),
            data["tgt"][rows].astype(np.float32),
            segment,
        )

    x_train, _, _, _ = features(train_spectra)
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1e-6
    if rule_columns and not args.standardize_rules:
        # Rule flags are 0/1 on well under 1% of rows; z-scoring would inflate a flag to ~100 sd.
        mean[-len(rule_columns) :] = 0.0
        std[-len(rule_columns) :] = 1.0
    del x_train
    width = 1 + len(CONTEXT) + len(columns) + len(rule_columns)
    model = Correction(width, args.hidden, args.dropout).to(device)
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


def resolve_family(
    family: str, descriptor_names: Sequence[str], rule_names: Sequence[str]
) -> tuple[list[int], list[int]]:
    """Column indices of a family in the cache's ``x`` and ``rule_x`` arrays.

    Raises:
        ValueError: A column of the family is in neither array, e.g. a rule family on a cache
            extracted before rule provenance existed.
    """
    x_columns, rule_columns = [], []
    for name in FAMILIES[family]:
        if name in descriptor_names:
            x_columns.append(list(descriptor_names).index(name))
        elif name in rule_names:
            rule_columns.append(list(rule_names).index(name))
        else:
            raise ValueError(f"family {family!r} needs column {name!r}, absent from the cache")
    return x_columns, rule_columns


def available_families(descriptor_names: Sequence[str], rule_names: Sequence[str]) -> list[str]:
    """Families whose columns the cache holds, in ``FAMILIES`` order."""
    known = set(descriptor_names) | set(rule_names)
    return [f for f, columns in FAMILIES.items() if set(columns) <= known]


def _segments(offsets: np.ndarray) -> np.ndarray:
    """Spectrum index of every saved row."""
    return np.repeat(np.arange(offsets.size - 1), np.diff(offsets))


def marked_spectra(
    offsets: np.ndarray, rule_x: np.ndarray, rule_names: Sequence[str]
) -> np.ndarray:
    """Whether each spectrum has a saved candidate marked by ``rule_any``, shape ``(N,)``."""
    flag = rule_x[:, list(rule_names).index("rule_any")] > 0
    return np.bincount(_segments(offsets), weights=flag, minlength=offsets.size - 1) > 0


def permute_rule_columns(offsets: np.ndarray, rule_x: np.ndarray, seed: int) -> np.ndarray:
    """Permute the rule rows among the candidates of each spectrum (the negative control).

    One permutation per spectrum moves whole rows, so each spectrum keeps its marker counts and
    every rule multi-hot pattern, while the pairing with a candidate's own size, depth and
    probability is destroyed.

    Args:
        offsets: Row offsets of the spectra, shape ``(N + 1,)``.
        rule_x: Rule columns, shape ``(rows, R)``.
        seed: Seed of the permutations.

    Returns:
        A permuted copy of ``rule_x``.
    """
    rng = np.random.default_rng(seed)
    out = rule_x.copy()
    flagged = np.bincount(_segments(offsets), weights=(rule_x != 0).any(axis=1)) > 0
    for spectrum in np.flatnonzero(flagged):
        first, last = offsets[spectrum], offsets[spectrum + 1]
        out[first:last] = rule_x[first:last][rng.permutation(last - first)]
    return out


def marked_gain_row(
    prefix: str, values: np.ndarray, marked: np.ndarray, mol_ids: np.ndarray, reps: int
) -> dict:
    """Mean of per-spectrum ``values`` over the marked spectra, in two units, with 95% CIs.

    ``{prefix}_whole`` divides the marked spectra's sum by ALL spectra (unmarked contribute 0,
    the whole-split unit of the plan's +0.003 threshold); ``{prefix}_subset`` divides by the
    marked spectra only. Intervals resample molecules.
    """
    masked = np.where(marked, values, 0.0)
    sums = clustered_bootstrap(
        mol_ids,
        {"g": masked, "n": np.ones(values.size), "m": marked.astype(float)},
        reps,
        seed=0,
    )
    whole = sums[:, 0] / sums[:, 1]
    subset = sums[:, 0] / np.maximum(sums[:, 2], 1.0)
    return {
        f"{prefix}_whole": masked.sum() / values.size,
        f"{prefix}_whole_lo": np.percentile(whole, 2.5),
        f"{prefix}_whole_hi": np.percentile(whole, 97.5),
        f"{prefix}_subset": masked.sum() / max(marked.sum(), 1),
        f"{prefix}_subset_lo": np.percentile(subset, 2.5),
        f"{prefix}_subset_hi": np.percentile(subset, 97.5),
    }


def fit(args: argparse.Namespace) -> None:
    """Cross-fit every family's correction and report held-out gains."""
    loaded = np.load(args.input)
    data = {k: loaded[k] for k in loaded.files}
    names = [str(n) for n in data["descriptors"]]
    rule_names = [str(n) for n in data["rule_columns"]] if "rule_columns" in data else []
    families = args.families or available_families(names, rule_names)
    resolved = {f: resolve_family(f, names, rule_names) for f in families}
    spectra = np.arange(data["spec_id"].size)
    fold = data["mol_id"] % 2
    device = th.device(args.device)
    marked = marked_spectra(data["offsets"], data["rule_x"], rule_names) if rule_names else None
    if PERMUTED_FAMILIES & set(families):
        data["rule_x_permuted"] = permute_rule_columns(
            data["offsets"], data["rule_x"], args.permute_seed
        )

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
    for family in families:
        columns, rule_columns = resolved[family]
        rule_key = "rule_x_permuted" if family in PERMUTED_FAMILIES else "rule_x"
        result = np.full((spectra.size, 2), np.nan)
        for k in (0, 1):
            LOGGER.info("%s: fit on parity %d", family, k)
            test = spectra[fold != k]
            result[test] = fit_family(
                data,
                columns,
                spectra[fold == k],
                test,
                args,
                seed=17 + k,
                rule_columns=rule_columns,
                rule_key=rule_key,
            )
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
        if marked is not None:
            row["n_marked_spectra"] = int(marked.sum())
            row |= marked_gain_row("marked_gain", gain, marked, data["mol_id"], args.reps)
        if family != "context" and "context" in held:
            over = result[:, 0] - held["context"][:, 0]
            sums = clustered_bootstrap(
                data["mol_id"], {"g": over, "n": np.ones(over.size)}, args.reps, seed=0
            )
            row["over_context"] = over.mean()
            row["over_context_lo"], row["over_context_hi"] = np.percentile(
                sums[:, 0] / sums[:, 1], [2.5, 97.5]
            )
            if marked is not None:
                row |= marked_gain_row(
                    "marked_over_context", over, marked, data["mol_id"], args.reps
                )
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(args.output, index=False)
    per_spectrum = pd.DataFrame(
        {"spec_id": data["spec_id"], "mol_id": data["mol_id"], "raw_cos": raw[:, 0]}
        | ({"marked": marked} if marked is not None else {})
        | {f"cos_{f}": r[:, 0] for f, r in held.items()}
        | {f"agree_{f}": r[:, 1] for f, r in held.items()}
    )
    per_spectrum.to_pickle(Path(args.output).with_suffix(".per_spectrum.pkl"))
    LOGGER.info("\n%s", table.round(4).to_string(index=False))


def cosine_with_target(p: np.ndarray, tgt: np.ndarray, target_norm: float) -> float:
    """Cosine of a candidate distribution with the WHOLE target (uncovered peaks included).

    Same definition as ``segment_scores``: ``p . tgt / (|p| * target_norm)``.
    """
    norm = float(np.linalg.norm(p))
    if norm == 0.0 or target_norm == 0.0:
        return 0.0
    return float(p @ tgt / (norm * target_norm))


def swap_ceiling(
    pred: np.ndarray,
    tgt: np.ndarray,
    swap: np.ndarray,
    target_norm: float,
    *,
    mass_preserving: bool,
) -> float:
    """Cosine when the predicted shares on the ``swap`` candidates are replaced by the target's.

    ``literal`` (the plan's text): each swapped candidate takes its share of the covered target,
    ``tgt / sum(tgt)``, the others keep their prediction, and the result is renormalised (which
    the cosine is invariant to). ``mass_preserving``: the swapped candidates keep their total
    predicted mass but spread it in proportion to the target, isolating the allocation inside
    the set; a set with no target intensity is left unchanged.

    Args:
        pred: Saved predicted share per candidate, shape ``(K,)``.
        tgt: Target intensity per candidate, shape ``(K,)``.
        swap: Bool mask of the swapped candidates, shape ``(K,)``.
        target_norm: L2 norm of the whole target.
        mass_preserving: Which variant.

    Returns:
        The cosine with the whole target.
    """
    p = pred / pred.sum()
    q = p.copy()
    if swap.any():
        if mass_preserving:
            swapped_target = tgt[swap].sum()
            if swapped_target > 0:
                q[swap] = p[swap].sum() * tgt[swap] / swapped_target
        elif tgt.sum() > 0:
            q[swap] = tgt[swap] / tgt.sum()
    return cosine_with_target(q, tgt, target_norm)


def rank_classes(tgt: np.ndarray) -> np.ndarray:
    """Target-rank class of each candidate: ``1 + floor(log2(rank))`` of the covered ones.

    Candidates with no target intensity form the last class, one above the lowest covered one
    (class 0 when nothing is covered), so class distance orders candidates by target size.
    """
    classes = np.zeros(tgt.size, dtype=np.int64)
    covered = np.flatnonzero(tgt > 0)
    if covered.size:
        ranks = np.empty(covered.size, dtype=np.int64)
        ranks[np.argsort(-tgt[covered], kind="stable")] = np.arange(1, covered.size + 1)
        classes[covered] = 1 + np.floor(np.log2(ranks)).astype(np.int64)
        classes[tgt <= 0] = classes[covered].max() + 1
    return classes


def match_controls(
    mz: np.ndarray, classes: np.ndarray, marked: np.ndarray, rng: np.random.Generator, band: float
) -> tuple[np.ndarray, np.ndarray]:
    """One draw of an unmarked control peak per marked peak, without replacement.

    Each marked peak, in random order, takes an unmarked candidate of its spectrum matched on
    m/z band (within ``band`` of its m/z, relative) and target-rank class. The tiers relax in
    order: (0) band and the same class, (1) band and the nearest class, (2) the same class
    anywhere, (3) the nearest class anywhere.

    Args:
        mz: Candidate m/z, shape ``(K,)``.
        classes: ``rank_classes`` of the candidates.
        marked: Bool marker per candidate.
        rng: Random generator.
        band: Relative m/z window.

    Returns:
        Control row per marked peak (-1 when no unmarked candidate is left) and the tier that
        matched it (-1 when none), both of shape ``(marked.sum(),)``.
    """
    free = ~marked
    order = rng.permutation(np.flatnonzero(marked))
    controls = np.full(order.size, -1, dtype=np.int64)
    tiers = np.full(order.size, -1, dtype=np.int64)
    for i, k in enumerate(order):
        near = free & (np.abs(mz - mz[k]) <= band * mz[k])
        distance = np.abs(classes - classes[k])
        for tier, (pool, exact) in enumerate(
            ((near, True), (near, False), (free, True), (free, False))
        ):
            if not pool.any():
                continue
            wanted = 0 if exact else distance[pool].min()
            chosen = pool & (distance == wanted)
            if chosen.any():
                controls[i] = rng.choice(np.flatnonzero(chosen))
                tiers[i] = tier
                free[controls[i]] = False
                break
    return controls, tiers


def spectrum_headroom(
    pred: np.ndarray,
    tgt: np.ndarray,
    mz: np.ndarray,
    marked: np.ndarray,
    target_norm: float,
    rng: np.random.Generator,
    *,
    band: float,
    draws: int,
) -> dict:
    """Ceiling gains of one spectrum for its marked peaks and for matched control peaks.

    Returns:
        ``base`` cosine, ``rule_*`` and ``control_*`` gains over it for both variants (control
        averaged over ``draws``), ``headroom_*`` = rule minus control, and the share of marked
        peaks matched at each tier.
    """
    classes = rank_classes(tgt)
    base = cosine_with_target(pred / pred.sum(), tgt, target_norm)
    out = {"base": base}
    control = {"literal": 0.0, "mass_preserving": 0.0}
    tier_counts = np.zeros(4)
    for _ in range(draws):
        controls, tiers = match_controls(mz, classes, marked, rng, band)
        swap = np.zeros(mz.size, dtype=bool)
        swap[controls[controls >= 0]] = True
        tier_counts += np.bincount(tiers[tiers >= 0], minlength=4)
        for variant in control:
            control[variant] += swap_ceiling(
                pred, tgt, swap, target_norm, mass_preserving=variant == "mass_preserving"
            )
    for variant in control:
        rule = swap_ceiling(
            pred, tgt, marked, target_norm, mass_preserving=variant == "mass_preserving"
        )
        out[f"rule_{variant}"] = rule - base
        out[f"control_{variant}"] = control[variant] / draws - base
        out[f"headroom_{variant}"] = out[f"rule_{variant}"] - out[f"control_{variant}"]
    out["marked_peaks"] = int(marked.sum())
    out |= {f"tier{t}": tier_counts[t] / (draws * marked.sum()) for t in range(4)}
    return out


def g1_decision(
    headroom: float,
    headroom_hi: float,
    gain_lo: float,
    gain_hi: float,
    target: float = HEADROOM_TARGET,
) -> str:
    """The pre-registered G1 rule.

    CONTINUE to p1 iff the whole-split headroom is at least ``target`` and the cross-fitted gain
    of ``rules`` over ``context`` has CI lower > 0. STOP iff not CONTINUE, the gain CI half-width
    is below ``target`` and the headroom CI upper is below ``target``. Otherwise UNDERPOWERED.
    """
    if headroom >= target and gain_lo > 0:
        return "CONTINUE"
    if (gain_hi - gain_lo) / 2 < target and headroom_hi < target:
        return "STOP"
    return "UNDERPOWERED"


def marker_masks(rule_x: np.ndarray, rule_names: Sequence[str]) -> dict[str, np.ndarray]:
    """Bool marker per candidate row for each marker the ceiling reports.

    ``rule_atomset`` is a share, so a peak counts as that fragment when at least half of its
    channels are one; every other column marks a peak when it is above zero.
    """
    names = list(rule_names)
    masks = {}
    for name in names:
        column = rule_x[:, names.index(name)]
        masks[name] = column >= 0.5 if name == "rule_atomset" else column > 0
    return masks


def ceiling(args: argparse.Namespace) -> None:
    """Headroom read: swap marked peaks to their target share, against matched controls."""
    started = time.time()
    loaded = np.load(args.input)
    offsets = loaded["offsets"]
    pred, tgt, mz = loaded["pred"], loaded["tgt"].astype(np.float64), loaded["mz"]
    target_norm, mol_ids, spec_ids = loaded["target_norm"], loaded["mol_id"], loaded["spec_id"]
    rule_names = [str(n) for n in loaded["rule_columns"]]
    masks = marker_masks(loaded["rule_x"], rule_names)
    total = spec_ids.size
    segments = _segments(offsets)
    per_spectrum = {"spec_id": spec_ids, "mol_id": mol_ids}
    rows = []
    for name, mask in masks.items():
        flagged = np.bincount(segments, weights=mask, minlength=total) > 0
        columns = {}
        for spectrum in np.flatnonzero(flagged):
            first, last = offsets[spectrum], offsets[spectrum + 1]
            result = spectrum_headroom(
                pred[first:last].astype(np.float64),
                tgt[first:last],
                mz[first:last],
                mask[first:last],
                float(target_norm[spectrum]),
                np.random.default_rng([args.seed, int(spectrum)]),
                band=args.band,
                draws=args.draws,
            )
            for key, value in result.items():
                columns.setdefault(key, np.zeros(total))[spectrum] = value
        row = {"marker": name, "marked_spectra": int(flagged.sum())}
        row["marked_peaks"] = int(mask.sum())
        for variant in ("literal", "mass_preserving"):
            for kind in ("rule", "control", "headroom"):
                key = f"{kind}_{variant}"
                values = columns.get(key, np.zeros(total))
                per_spectrum[f"{name}:{key}"] = values
                if kind == "headroom":
                    row |= marked_gain_row(key, values, flagged, mol_ids, args.reps)
                else:
                    row[f"{key}_whole"] = values.sum() / total
                    row[f"{key}_subset"] = values.sum() / max(flagged.sum(), 1)
        for tier in range(4):
            row[f"tier{tier}"] = columns.get(f"tier{tier}", np.zeros(total)).sum() / max(
                flagged.sum(), 1
            )
        rows.append(row)
        LOGGER.info("%s: %d marked spectra", name, flagged.sum())
    table = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    pd.DataFrame(per_spectrum).to_pickle(output.with_suffix(".per_spectrum.pkl"))

    # Decision from the primary marker and each supplied fit table (default MLP first).
    primary = table[table["marker"] == "rule_any"].iloc[0]
    decisions = {}
    for path in args.fit_csv:
        fit_rows = pd.read_csv(path).set_index("family")
        if "rules" not in fit_rows.index:
            raise ValueError(f"{path} has no 'rules' family row")
        gain = fit_rows.loc["rules"]
        decisions[path] = {
            "decision": g1_decision(
                primary["headroom_literal_whole"],
                primary["headroom_literal_whole_hi"],
                gain["over_context_lo"],
                gain["over_context_hi"],
            ),
            "headroom": float(primary["headroom_literal_whole"]),
            "headroom_ci": [
                float(primary["headroom_literal_whole_lo"]),
                float(primary["headroom_literal_whole_hi"]),
            ],
            "rules_over_context": float(gain["over_context"]),
            "rules_over_context_ci": [
                float(gain["over_context_lo"]),
                float(gain["over_context_hi"]),
            ],
        }
    provenance = {
        "args": vars(args),
        "git_revision": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "git_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "diff", "HEAD"])
        ).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "inputs": [_identity(p) for p in [args.input, *args.fit_csv]],
        "masster_rule_names": list(RULE_NAMES),
        "cache_rule_columns": rule_names,
        "spectra": int(total),
        "decisions": decisions,
        "elapsed_seconds": time.time() - started,
    }
    output.with_suffix(".provenance.json").write_text(json.dumps(provenance, indent=2, default=str))
    LOGGER.info("\n%s", table[["marker", "marked_spectra", "headroom_literal_whole"]].to_string())
    LOGGER.info("decisions: %s", json.dumps(decisions, indent=2))


def main(argv: list[str] | None = None) -> None:
    """Dispatch the ``extract``, ``fit`` and ``ceiling`` subcommands."""
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
    ex.add_argument("--mol_ids", default="", help="comma list or file of molecule ids to keep")
    ex.add_argument(
        "--template_bearing_only",
        action="store_true",
        help="keep spectra whose molecule has an eligible template for the spectrum's adduct",
    )
    ex.add_argument(
        "--max_molecules", type=int, default=0, help="with --template_bearing_only: random N"
    )
    ex.add_argument("--marker_tol", type=float, default=MARKER_TOL_DA, help="Da, primary marker")
    ex.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ex.add_argument("--output", required=True, help=".npz; .provenance.json beside it")
    ft = sub.add_parser("fit")
    ft.add_argument("--input", required=True)
    ft.add_argument("--output", required=True, help="csv; .per_spectrum.pkl beside it")
    ft.add_argument(
        "--families",
        nargs="+",
        default=None,
        choices=list(FAMILIES),
        help="default: every family whose columns the cache holds",
    )
    ft.add_argument("--standardize_rules", action="store_true", help="z-score rule columns too")
    ft.add_argument("--permute_seed", type=int, default=0)
    ft.add_argument("--hidden", type=int, default=128)
    ft.add_argument("--dropout", type=float, default=0.1)
    ft.add_argument("--lr", type=float, default=1e-3)
    ft.add_argument("--weight_decay", type=float, default=1e-5)
    ft.add_argument("--epochs", type=int, default=30)
    ft.add_argument("--patience", type=int, default=3)
    ft.add_argument("--batch_spectra", type=int, default=512)
    ft.add_argument("--reps", type=int, default=200)
    ft.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    ce = sub.add_parser("ceiling")
    ce.add_argument("--input", required=True, help="cache written by extract (with rule_x)")
    ce.add_argument("--output", required=True, help="csv; .per_spectrum.pkl, .provenance.json")
    ce.add_argument("--fit_csv", nargs="*", default=[], help="fit tables for the G1 decision")
    ce.add_argument("--draws", type=int, default=20, help="control draws per spectrum")
    ce.add_argument("--band", type=float, default=0.10, help="relative m/z window of controls")
    ce.add_argument("--reps", type=int, default=1000)
    ce.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.command == "extract":
        extract(args)
    elif args.command == "fit":
        fit(args)
    else:
        ceiling(args)


if __name__ == "__main__":
    main()
