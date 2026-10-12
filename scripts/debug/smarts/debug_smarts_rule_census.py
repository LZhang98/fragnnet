"""Census of what each SMARTS rule adds on NIST20, model-free.

Three read-only measurements behind docs/findings/smarts_usage_audit_2026_09_23.md:

- ``census``: per rule, how many ``mol_df`` molecules match and how many distinct partitions
  the production matcher yields, against the same matcher with ``uniquify=False``. A gap
  means RDKit's default uniquification drops a valid embedding (a mirror direction).
- ``sides``: for every MASSter template, split its two components into the one an
  ordinary cut of the same bonds already produces and the one only the template produces,
  then report the spectrum intensity each explains. This says whether a rule's *new*
  assembly is the chemically meaningful ion or a merged neutral byproduct.
- ``variants``: the same explained-intensity read-out for candidate rule patterns that are
  not in the registry (a charge-tolerant sulfoxide, primary and tertiary sulfonamides).

"Explained" is a mass match, not an attribution: an ion within 0.01 Da of the component's
heavy atoms plus its own hydrogens plus one cap per broken bond, shifted by -6..+4 H and one
proton for ``[M+H]+`` / ``[M-H]-``. A template side that is a merged neutral byproduct gives
the chance floor of this test, so compare rules against that floor rather than zero.

Usage:
    python scripts/debug/smarts/debug_smarts_rule_census.py census
    python scripts/debug/smarts/debug_smarts_rule_census.py sides
    python scripts/debug/smarts/debug_smarts_rule_census.py variants
"""

from __future__ import annotations

import argparse
import collections

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

from fragnnet.fragmentation.smarts_prepass import (
    FRAG_RULES,
    NL_RULES,
    NeutralLossRule,
    _apply_nl_prepass,
    _apply_smarts_prepass,
)
from fragnnet.models.masster.path_selection import templates as mst

RDLogger.DisableLog("rdApp.*")

MOL_FP = "data/proc/nist20/mol_df.pkl"
SPEC_FP = "data/proc/nist20/spec_df.pkl"
MASS = {
    "C": 12.0, "H": 1.00782503207, "N": 14.0030740048, "O": 15.99491461956,
    "S": 31.97207100, "P": 30.97376163, "F": 18.99840322, "Cl": 34.96885268,
    "Br": 78.9183371, "I": 126.904473, "Si": 27.9769265325, "Se": 79.9165213, "B": 11.0093054,
}  # fmt: skip
PROTON = 1.007276467
TOLERANCE_DA = 0.01
H_SHIFTS = range(-6, 5)
VARIANTS = (
    NeutralLossRule(
        "sulfoxide_so_loss_charge_tolerant",
        "[#6:1]-[$([SX3]=[OX1]),$([SX3+]-[OX1-]):2]~[OX1:3]",
        [2, 3],
        47.9669,
        3,
    ),
    NeutralLossRule(
        "aryl_sulfonamide_so2_NH", "[c:1]-[SX4:2](=[O:3])(=[O:4])-[NH:5]", [2, 3, 4], 63.9619, 8
    ),
    NeutralLossRule(
        "aryl_sulfonamide_so2_NH2", "[c:1]-[SX4:2](=[O:3])(=[O:4])-[NH2:5]", [2, 3, 4], 63.9619, 8
    ),
    NeutralLossRule(
        "aryl_sulfonamide_so2_NR2",
        "[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3H0:5]",
        [2, 3, 4],
        63.9619,
        8,
    ),
)


class _AllEmbeddings:
    """Molecule proxy whose substructure search keeps every embedding."""

    def __init__(self, mol: Chem.Mol) -> None:
        self._mol = mol

    def GetSubstructMatches(self, pattern: Chem.Mol):  # noqa: N802 - mirrors the RDKit name
        return self._mol.GetSubstructMatches(pattern, uniquify=False, maxMatches=1_000_000)

    def __getattr__(self, name: str):
        # Everything else (e.g. GetAtomWithIdx for substituent sides) is the molecule's own.
        return getattr(self._mol, name)


def _partition_key(pair) -> tuple:
    return tuple(sorted((tuple(np.flatnonzero(pair[0])), tuple(np.flatnonzero(pair[1])))))


def census(mol_df: pd.DataFrame) -> pd.DataFrame:
    """Count matches and yielded partitions per rule, default vs ``uniquify=False``."""
    frag_rules = list(FRAG_RULES) + list(mst._MASSTER_ONLY_FRAG_RULES)
    nl_rules = list(NL_RULES) + list(mst._MASSTER_ONLY_NL_RULES)
    counts = collections.defaultdict(collections.Counter)
    for mol in mol_df.mol:
        if mol is None:
            continue
        num_atoms, num_bonds = mol.GetNumAtoms(), mol.GetNumBonds()
        edges = np.asarray(
            [(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()], dtype=np.int32
        ).reshape(-1, 2)
        edge_mask = np.ones(num_bonds, dtype=np.uint8)
        for rule in frag_rules:
            if num_atoms < rule.min_num_atoms or not mol.HasSubstructMatch(
                Chem.MolFromSmarts(rule.reactant_smarts)
            ):
                continue
            default = _apply_smarts_prepass([rule], mol, edge_mask, edges, num_atoms, num_bonds)
            every = _apply_smarts_prepass(
                [rule], _AllEmbeddings(mol), edge_mask, edges, num_atoms, num_bonds
            )
            default_keys, every_keys = (
                set(map(_partition_key, default)),
                set(map(_partition_key, every)),
            )
            counts[rule.name].update(
                match_mols=1,
                zero_yield_mols=int(not default_keys),
                parts_default=len(default_keys),
                parts_all_embeddings=len(every_keys),
                mols_losing_parts=int(bool(every_keys - default_keys)),
            )
        for rule in nl_rules:
            if num_atoms < rule.min_num_atoms or not mol.HasSubstructMatch(
                Chem.MolFromSmarts(rule.reactant_smarts)
            ):
                continue
            default = {
                tuple(np.flatnonzero(m)) for m, _ in _apply_nl_prepass([rule], mol, num_atoms)
            }
            every = {
                tuple(np.flatnonzero(m))
                for m, _ in _apply_nl_prepass([rule], _AllEmbeddings(mol), num_atoms)
            }
            counts[rule.name].update(
                match_mols=1,
                zero_yield_mols=int(not default),
                parts_default=len(default),
                parts_all_embeddings=len(every),
                mols_losing_parts=int(bool(every - default)),
            )
    rows = [
        {"rule": rule.name, "in_masster": rule.name in mst.MASSTER_RULE_NAMES}
        | dict(counts[rule.name])
        for rule in frag_rules + nl_rules
    ]
    return pd.DataFrame(rows).fillna(0)


def _component_mass(mol: Chem.Mol, atoms, broken_bonds) -> float:
    atom_set = set(atoms)
    caps = sum(
        (mol.GetBondWithIdx(b).GetBeginAtomIdx() in atom_set)
        != (mol.GetBondWithIdx(b).GetEndAtomIdx() in atom_set)
        for b in broken_bonds
    )
    heavy = sum(MASS[mol.GetAtomWithIdx(a).GetSymbol()] for a in atoms)
    hydrogens = sum(mol.GetAtomWithIdx(a).GetTotalNumHs() for a in atoms) + caps
    return heavy + hydrogens * MASS["H"]


def _explained_fraction(peaks: np.ndarray, neutral_mass: float, sign: int) -> float:
    hit = np.zeros(len(peaks), dtype=bool)
    for shift in H_SHIFTS:
        ion = neutral_mass + shift * MASS["H"] + sign * PROTON
        hit |= np.abs(peaks[:, 0] - ion) <= TOLERANCE_DA
    return float(peaks[hit, 1].sum() / peaks[:, 1].sum())


def _summarize(rows: list[dict], value_columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    aggregations = {"mols": ("mol_id", "nunique"), "spectra": ("mol_id", "size")}
    for column in value_columns:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_frac_gt1pct"] = (column, lambda x: float((x > 0.01).mean()))
    return frame.groupby(["rule", "prec_type"]).agg(**aggregations).reset_index()


def sides(mol_df: pd.DataFrame, spectra_by_mol) -> pd.DataFrame:
    """Explained intensity of each template's template-only vs ordinary component."""
    rows = []
    for mol_id, mol in zip(mol_df.mol_id, mol_df.mol):
        if mol is None or mol_id not in spectra_by_mol.groups:
            continue
        for template in mst.build_masster_templates(mol):
            # 1. Components an ordinary cut of the template's bonds already produces.
            cut = Chem.RWMol(mol)
            for bond_index in sorted(template.broken_bond_indices, reverse=True):
                bond = mol.GetBondWithIdx(bond_index)
                cut.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            ordinary = set(Chem.GetMolFrags(cut.GetMol(), asMols=False, sanitizeFrags=False))
            ordinary = {tuple(sorted(component)) for component in ordinary}
            new = [c for c in template.components if c not in ordinary]
            old = [c for c in template.components if c in ordinary]
            if len(new) != 1 or len(old) != 1:
                continue
            # 2. Score both sides against every [M+H]+ / [M-H]- spectrum of the molecule.
            new_mass = _component_mass(mol, new[0], template.broken_bond_indices)
            old_mass = _component_mass(mol, old[0], template.broken_bond_indices)
            for spectrum in spectra_by_mol.get_group(mol_id).itertuples():
                peaks = np.asarray(spectrum.peaks, dtype=float)
                sign = 1 if spectrum.prec_type == "[M+H]+" else -1
                rows.append(
                    {
                        "rule": template.rule_name,
                        "prec_type": spectrum.prec_type,
                        "mol_id": mol_id,
                        "template_only": _explained_fraction(peaks, new_mass, sign),
                        "ordinary": _explained_fraction(peaks, old_mass, sign),
                    }
                )
    return _summarize(rows, ("template_only", "ordinary"))


def variants(mol_df: pd.DataFrame, spectra_by_mol) -> pd.DataFrame:
    """Explained intensity of the product of each unregistered candidate NL pattern."""
    rows = []
    for mol_id, mol in zip(mol_df.mol_id, mol_df.mol):
        if mol is None or mol_id not in spectra_by_mol.groups:
            continue
        for rule in VARIANTS:
            for product_mask, _ in _apply_nl_prepass([rule], mol, mol.GetNumAtoms()):
                product = np.flatnonzero(product_mask).tolist()
                product_set = set(product)
                broken = [
                    b.GetIdx()
                    for b in mol.GetBonds()
                    if (b.GetBeginAtomIdx() in product_set) != (b.GetEndAtomIdx() in product_set)
                ]
                mass = _component_mass(mol, product, broken)
                for spectrum in spectra_by_mol.get_group(mol_id).itertuples():
                    peaks = np.asarray(spectrum.peaks, dtype=float)
                    sign = 1 if spectrum.prec_type == "[M+H]+" else -1
                    rows.append(
                        {
                            "rule": rule.name,
                            "prec_type": spectrum.prec_type,
                            "mol_id": mol_id,
                            "product": _explained_fraction(peaks, mass, sign),
                        }
                    )
    return _summarize(rows, ("product",))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("mode", choices=("census", "sides", "variants"))
    args = parser.parse_args()
    mol_df = pd.read_pickle(MOL_FP)
    if args.mode == "census":
        table = census(mol_df)
    else:
        spec_df = pd.read_pickle(SPEC_FP)
        spec_df = spec_df[spec_df.prec_type.isin(["[M+H]+", "[M-H]-"])]
        spectra_by_mol = spec_df[["mol_id", "prec_type", "peaks"]].groupby("mol_id")
        table = (sides if args.mode == "sides" else variants)(mol_df, spectra_by_mol)
    with pd.option_context(
        "display.width", 250, "display.max_columns", 20, "display.float_format", "{:.3f}".format
    ):
        print(table.to_string(index=False))


if __name__ == "__main__":
    main()
