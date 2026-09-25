"""Precompute SMARTS-defined atom partitions for MASSter."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from rdkit import Chem

from fragnnet.frag.smarts_prepass import (
    FRAG_RULES,
    NL_RULES,
    NeutralLossRule,
    SmartsFragRule,
    _apply_nl_prepass,
    _apply_smarts_prepass,
)

# MASSter owns these allowlists. They intentionally exclude the generic
# crf12/crf13 patterns, whose broad matches create many unsupported candidates,
# and pure bond-deletion rules already represented by the ordinary mechanism.
_SHARED_MASSTER_FRAG_RULE_NAMES = (
    "anthraquinone_bis_co_loss",
    "xanthone_ring_cleavage",
    "phenoxazine_ring_cleavage",
    "furanocoumarin_ring_cleavage",
    "benzopyranone_ring_cleavage",
    "halogenated_quinoline_ring_cleavage",
    "benzopyranone_bis_ch_loss",
    "anthraquinone_ring_c4_plus_ch_loss",
    "morphinan_amine_arm_loss",
    "desoxymorphinan_ring_cleavage",
    "desoxymorphinan_aromatic_core_release",
)
_SHARED_MASSTER_NL_RULE_NAMES = (
    "aryl_sulfonamide_so2",
    "aryl_sulfonate_so2",
    "aryl_sulfonate_so3",
    "n_nitramine_n2o",
    "sulfoxide_so_loss",
    "diarylvinyl_c2h2_loss",
    "diaryl_ketone_co_loss",
    "aminobenzothiazole_hcn_loss",
)

# These literature-derived rules are MASSter-only. Keeping them out of the
# shared FRAG_RULES/NL_RULES registries avoids changing the established BFS
# preprocessing behavior.
# Evidence and admission gates: docs/reference/masster_smarts_rules.md.
# Unimplemented candidates and mapping limits:
# docs/findings/masster/masster_smarts_review_2026_09_13.md.
# - Phenylalanine oxygen transfer: https://doi.org/10.3389/fchem.2022.1094329
# - Aryl-sulfonamide oxygen transfer: https://doi.org/10.1016/j.ijms.2018.09.033
# - Prazole-sulfone rearrangement: https://pmc.ncbi.nlm.nih.gov/articles/PMC10420917/
_MASSTER_ONLY_FRAG_RULES = (
    # Protonated 2-, 3-, or 4-pyridyl O-carbamates transfer Me, Et, or nBu from
    # the carbamate N to the pyridyl-O residue. Cutting O-C(=O) and N-alkyl
    # produces three components; the pyridyl-O and selected alkyl components
    # form the observed RC5H5NO+ assembly.
    # Paper: https://www.sciencedirect.com/science/article/pii/S004040201401518X
    # Pyrimidine extension is not implemented; pirimicarb evidence (section 2.3):
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC6222809/
    SmartsFragRule(
        name="pyridyl_carbamate_alkyl_transfer",
        reactant_smarts=(
            "[$(c1ncccc1),$(c1cnccc1),$(c1ccncc1):1]-[OX2:2]-"
            "[CX3:3](=[O:6])-[NX3:4]-"
            "[$([CH3]),$([CH2][CH3]),$([CH2][CH2][CH2][CH3]):5]"
        ),
        cut_bond_map_pairs=[(2, 3), (4, 5)],
        merge_outer_map_nums=[2, 5],
        min_num_atoms=11,
    ),
)
_MASSTER_ONLY_NL_RULES = (
    # Deprotonated N-phenylbenzamides lose internal CO through an N-triggered
    # Smiles rearrangement, joining the benzoyl aryl residue to the amide N.
    # Paper: https://pubmed.ncbi.nlm.nih.gov/26377014/
    NeutralLossRule(
        name="n_phenylbenzamide_co_smiles",
        reactant_smarts="[c:1]-[CX3;!R:2](=[O:3])-[NH:4]-[c:5]",
        nl_map_nums=[2, 3],
        nl_mass=27.9949,
        min_num_atoms=14,
    ),
    # Protonated aryl methyl esters expel CO2 while the methyl group transfers
    # from oxygen to the aromatic residue. The methyl guard avoids the broad
    # acyclic-ester rule that MASSter intentionally excludes.
    # Paper: https://pubmed.ncbi.nlm.nih.gov/29441643/
    NeutralLossRule(
        name="methyl_benzoate_co2_transfer",
        reactant_smarts="[c:1]-[CX3;!R:2](=[O:3])-[OX2:4]-[CH3:5]",
        nl_map_nums=[2, 3, 4],
        nl_mass=43.9898,
        min_num_atoms=10,
    ),
    # Ionized o-, m-, and p-anisoyl fluorides lose CO and retain fluorine on
    # the methoxy-substituted aromatic residue. Recursive queries keep the
    # methoxy group on the same ring as the acyl fluoride.
    # Paper (EI): https://doi.org/10.1002/oms.1210261122
    NeutralLossRule(
        name="anisoyl_fluoride_co_f_transfer",
        reactant_smarts=(
            "[$(c1c([OX2][CH3])cccc1),$(c1cc([OX2][CH3])ccc1),"
            "$(c1ccc([OX2][CH3])cc1):1]-[CX3;!R:2](=[O:3])-[F:4]"
        ),
        nl_map_nums=[2, 3],
        nl_mass=27.9949,
        min_num_atoms=10,
    ),
    # Protonated amines and quaternary ammonium ions bearing a two-carbon
    # haloalkyl arm transfer Cl/Br/I to N while the arm leaves as C2H4.
    # Paper: https://doi.org/10.1007/s13361-015-1261-4
    NeutralLossRule(
        name="short_haloalkyl_ammonium_c2_transfer",
        reactant_smarts=("[$([NX3;+0;!$(N-C=O)]),$([NX4;+1]):1]-[CH2:2]-[CH2:3]-[Cl,Br,I:4]"),
        nl_map_nums=[2, 3],
        nl_mass=28.0313,
        min_num_atoms=4,
    ),
    # The three-carbon homologue follows the same pathway but loses C3H6.
    # Paper: https://doi.org/10.1007/s13361-015-1261-4
    NeutralLossRule(
        name="short_haloalkyl_ammonium_c3_transfer",
        reactant_smarts=(
            "[$([NX3;+0;!$(N-C=O)]),$([NX4;+1]):1]-[CH2:2]-[CH2:3]-[CH2:4]-[Cl,Br,I:5]"
        ),
        nl_map_nums=[2, 3, 4],
        nl_mass=42.0470,
        min_num_atoms=5,
    ),
    # Deprotonated N-fluoroarenesulfonamides form diagnostic SO2F-. The neutral
    # ArN complement is variably substituted, so this rule maps the small
    # product directly instead of trying to enumerate every neutral atom.
    # Paper (negative ion): https://doi.org/10.1002/rcm.9722
    NeutralLossRule(
        name="n_fluoro_sulfonamide_f_transfer",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3H1:5]-[F:6]",
        nl_map_nums=[],
        nl_mass=None,
        min_num_atoms=10,
        product_map_nums=[2, 3, 4, 6],
    ),
)
RuleType = TypeVar("RuleType", SmartsFragRule, NeutralLossRule)


def _select_rules_by_name(
    rules: list[RuleType],
    names: tuple[str, ...],
) -> tuple[RuleType, ...]:
    """Resolve a stable allowlist and fail if the shared definition disappears."""
    by_name = {rule.name: rule for rule in rules}
    missing = set(names) - by_name.keys()
    if missing:
        raise ValueError(f"missing MASSter SMARTS rules: {sorted(missing)}")
    if len(names) != len(set(names)):
        raise ValueError("MASSter SMARTS rule names must be unique")
    return tuple(by_name[name] for name in names)


MASSTER_FRAG_RULES: tuple[SmartsFragRule, ...] = (
    _select_rules_by_name(
        FRAG_RULES,
        _SHARED_MASSTER_FRAG_RULE_NAMES,
    )
    + _MASSTER_ONLY_FRAG_RULES
)
MASSTER_NL_RULES: tuple[NeutralLossRule, ...] = (
    _select_rules_by_name(
        NL_RULES,
        _SHARED_MASSTER_NL_RULE_NAMES,
    )
    + _MASSTER_ONLY_NL_RULES
)
MASSTER_FRAG_RULE_NAMES = tuple(rule.name for rule in MASSTER_FRAG_RULES)
MASSTER_NL_RULE_NAMES = tuple(rule.name for rule in MASSTER_NL_RULES)
if any(not rule.merge_outer_map_nums for rule in MASSTER_FRAG_RULES):
    raise ValueError("MASSter FRAG rules must define a new assembly")
MASSTER_RULE_NAMES = (
    _SHARED_MASSTER_FRAG_RULE_NAMES
    + _SHARED_MASSTER_NL_RULE_NAMES
    + tuple(rule.name for rule in _MASSTER_ONLY_FRAG_RULES)
    + tuple(rule.name for rule in _MASSTER_ONLY_NL_RULES)
)
if len(MASSTER_RULE_NAMES) != len(set(MASSTER_RULE_NAMES)):
    raise ValueError("MASSter SMARTS rule names must be globally unique")
MASSTER_NUM_RULES = len(MASSTER_RULE_NAMES)
_MASSTER_RULE_INDEX = {
    name: index for index, name in enumerate(MASSTER_RULE_NAMES, start=1)
}

# New literature rules are admitted only for the ionization conditions in the
# supporting experiments. Existing empirical rules retain their current
# unrestricted behavior until they are audited separately.
_MASSTER_PRECURSOR_TYPES = {
    "pyridyl_carbamate_alkyl_transfer": ("[M+H]+",),
    "n_phenylbenzamide_co_smiles": ("[M-H]-",),
    "methyl_benzoate_co2_transfer": ("[M+H]+",),
    "anisoyl_fluoride_co_f_transfer": ("[M]+",),
    "short_haloalkyl_ammonium_c2_transfer": ("[M+H]+", "[M]+"),
    "short_haloalkyl_ammonium_c3_transfer": ("[M+H]+", "[M]+"),
    "n_fluoro_sulfonamide_f_transfer": ("[M-H]-",),
}
_MASSTER_FRAGMENTATION_MODES = {
    "anisoyl_fluoride_co_f_transfer": ("EI",),
}


@dataclass(frozen=True)
class MassterTemplate:
    """One SMARTS mechanism represented in precursor atom/bond coordinates."""

    broken_bond_indices: tuple[int, ...]
    components: tuple[tuple[int, ...], ...]
    rule_index: int
    rule_name: str
    allowed_precursor_types: tuple[str, ...] | None = None
    allowed_fragmentation_modes: tuple[str, ...] | None = None


def filter_masster_templates(
    templates: tuple[MassterTemplate, ...],
    *,
    precursor_type: str,
    fragmentation_mode: str | None,
) -> tuple[MassterTemplate, ...]:
    """Keep templates supported by the spectrum's ionization conditions."""
    return tuple(
        template
        for template in templates
        if (
            template.allowed_precursor_types is None
            or precursor_type in template.allowed_precursor_types
        )
        and (
            template.allowed_fragmentation_modes is None
            or fragmentation_mode in template.allowed_fragmentation_modes
        )
    )


def _component_atoms(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(index) for index in np.flatnonzero(mask))


def _crossing_bonds(
    component: tuple[int, ...],
    edges: np.ndarray,
) -> tuple[int, ...]:
    atom_set = set(component)
    return tuple(
        bond_index
        for bond_index, (left, right) in enumerate(edges)
        if (int(left) in atom_set) != (int(right) in atom_set)
    )


def build_masster_templates(mol: Chem.Mol) -> tuple[MassterTemplate, ...]:
    """Match the curated SMARTS rules and return deduplicated template mechanisms.

    The returned indices are local to ``mol`` and follow RDKit's atom and bond
    ordering, which is also the ordering used by ``MolGraph.chemical_bond_index``.
    Rule index zero is reserved for MASSter's ordinary connected-component
    mechanism; SMARTS rules therefore use one-based indices here.
    """
    num_atoms = mol.GetNumAtoms()
    num_bonds = mol.GetNumBonds()
    edges = np.asarray(
        [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in mol.GetBonds()],
        dtype=np.int32,
    ).reshape(-1, 2)
    edge_mask = np.ones(num_bonds, dtype=np.uint8)
    templates: list[MassterTemplate] = []

    frag_matches = _apply_smarts_prepass(
        list(MASSTER_FRAG_RULES),
        mol,
        edge_mask,
        edges,
        num_atoms,
        num_bonds,
    )
    for left_mask, right_mask, rule_offset in frag_matches:
        rule = MASSTER_FRAG_RULES[rule_offset]
        components = tuple(sorted((_component_atoms(left_mask), _component_atoms(right_mask))))
        templates.append(
            MassterTemplate(
                broken_bond_indices=_crossing_bonds(components[0], edges),
                components=components,
                rule_index=_MASSTER_RULE_INDEX[rule.name],
                rule_name=rule.name,
                allowed_precursor_types=_MASSTER_PRECURSOR_TYPES.get(rule.name),
                allowed_fragmentation_modes=(_MASSTER_FRAGMENTATION_MODES.get(rule.name)),
            )
        )

    nl_matches = _apply_nl_prepass(list(MASSTER_NL_RULES), mol, num_atoms)
    all_atoms = set(range(num_atoms))
    for product_mask, rule_offset in nl_matches:
        rule = MASSTER_NL_RULES[rule_offset]
        product = _component_atoms(product_mask)
        neutral_loss = tuple(sorted(all_atoms - set(product)))
        if not product or not neutral_loss:
            continue
        components = tuple(sorted((product, neutral_loss)))
        templates.append(
            MassterTemplate(
                broken_bond_indices=_crossing_bonds(product, edges),
                components=components,
                rule_index=_MASSTER_RULE_INDEX[rule.name],
                rule_name=rule.name,
                allowed_precursor_types=_MASSTER_PRECURSOR_TYPES.get(rule.name),
                allowed_fragmentation_modes=(_MASSTER_FRAGMENTATION_MODES.get(rule.name)),
            )
        )

    unique: dict[
        tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int],
        MassterTemplate,
    ] = {}
    for template in templates:
        key = (
            template.broken_bond_indices,
            template.components,
            template.rule_index,
        )
        unique.setdefault(key, template)
    return tuple(unique.values())
