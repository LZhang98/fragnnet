"""Precompute SMARTS-defined atom partitions for MASSter."""

from __future__ import annotations

import zlib
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TypeVar

import numpy as np
from rdkit import Chem

from fragnnet.fragmentation.smarts_prepass import (
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
# The eleven compound ring-cut rules were dropped on 2026-09-27: on c0's val pool their
# forced cuts cover no more target intensity than random fragments of the same size, and
# their merged byproducts cover less. So was aryl_sulfonate_so3, whose single-bond cut only
# ever repeats the ordinary partition. FraGNNet's prepass keeps all of them.
# Evidence: docs/findings/masster/masster_smarts_cuda_audit_2026_09_27.md
#
# DISABLED 2026-09-29: eight rules are commented out below, to re-check on Enveda-180. On
# NIST20, five add no m/z coverage (their product always has a connected formula twin) and
# nothing even on c0's pool; three match no NIST20 spectrum. Their names stay in
# MASSTER_RULE_VOCABULARY, so no rule index shifts; uncomment a rule to restore it.
# Evidence: docs/reference/masster_smarts_rules.md#existing-rules-under-the-gate
_SHARED_MASSTER_FRAG_RULE_NAMES: tuple[str, ...] = ()
_SHARED_MASSTER_NL_RULE_NAMES = (
    "aryl_sulfonamide_so2",
    "aryl_sulfonate_so2",
    "n_nitramine_n2o",
    "sulfoxide_so_loss",
    # DISABLED 2026-09-29: formula twins, no gain on c0's pool.
    # "diarylvinyl_c2h2_loss",
    # "diaryl_ketone_co_loss",
    # "aminobenzothiazole_hcn_loss",
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
    # DISABLED 2026-09-29: matches one NIST20 molecule, which has no spectra.
    # Protonated 2-, 3-, or 4-pyridyl O-carbamates transfer Me, Et, or nBu from
    # the carbamate N to the pyridyl-O residue. Cutting O-C(=O) and N-alkyl
    # produces three components; the pyridyl-O and selected alkyl components
    # form the observed RC5H5NO+ assembly.
    # Paper: https://www.sciencedirect.com/science/article/pii/S004040201401518X
    # Pyrimidine extension is not implemented; pirimicarb evidence (section 2.3):
    # https://pmc.ncbi.nlm.nih.gov/articles/PMC6222809/
    # SmartsFragRule(
    #     name="pyridyl_carbamate_alkyl_transfer",
    #     reactant_smarts=(
    #         "[$(c1ncccc1),$(c1cnccc1),$(c1ccncc1):1]-[OX2:2]-"
    #         "[CX3:3](=[O:6])-[NX3:4]-"
    #         "[$([CH3]),$([CH2][CH3]),$([CH2][CH2][CH2][CH3]):5]"
    #     ),
    #     cut_bond_map_pairs=[(2, 3), (4, 5)],
    #     merge_outer_map_nums=[2, 5],
    #     min_num_atoms=11,
    # ),
    # The rules marked "Gate" below passed the 2026-09-27 per-class gate: on the NIST20
    # spectra of the molecules they match, in the admitted polarity, their product explains
    # target intensity that no connected atom set of the same composition can, beyond a
    # random baseline. Evidence: docs/findings/masster/masster_smarts_cuda_audit_2026_09_27.md
    #
    # Deprotonated N-acyl arylsulfonamides form a phenoxide from the arylsulfonyl ring and
    # the oxygen of the remote amide carbonyl. Cutting Ar-S and the carbonyl C-O leaves
    # three components; the aryl side and that oxygen form the product.
    # Paper: https://pubmed.ncbi.nlm.nih.gov/33586949/
    # Gate: [M-H]-, 10 molecules, +5.7% of intensity (95% CI 0.8-12.8).
    SmartsFragRule(
        name="n_acyl_aryl_sulfonamide_carbonyl_o_transfer",
        reactant_smarts="[c:1]-[SX4:2](=O)(=O)-[#7:3]~[CX3:4](~[OX1,OX2H1:5])-[#6:6]",
        cut_bond_map_pairs=[(1, 2), (4, 5)],
        merge_outer_map_nums=[1, 5],
        min_num_atoms=10,
    ),
    # Deprotonated N'-tert-butyl diacylhydrazines give [R1CO2]-: the opposite amide
    # oxygen transfers to the R1 carbonyl carbon on the tert-butyl nitrogen.
    # Paper: https://doi.org/10.1371/journal.pone.0063097
    # Gate: [M-H]-, 3 molecules, +49.6% of intensity; at least 1% in 59% of spectra.
    SmartsFragRule(
        name="diacylhydrazine_carboxylate_o_transfer",
        reactant_smarts=(
            "[#6]-[CX3:1](=[OX1])-[NX3:2](-[CX4]([CH3])([CH3])[CH3])-[#7]"
            "~[CX3:3](~[OX1,OX2H1:4])-[#6]"
        ),
        cut_bond_map_pairs=[(1, 2), (3, 4)],
        merge_outer_map_nums=[1, 4],
        min_num_atoms=10,
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
    # DISABLED 2026-09-29: formula twins, no gain on c0's pool.
    # Protonated aryl methyl esters expel CO2 while the methyl group transfers
    # from oxygen to the aromatic residue. The methyl guard avoids the broad
    # acyclic-ester rule that MASSter intentionally excludes.
    # Paper: https://pubmed.ncbi.nlm.nih.gov/29441643/
    # NeutralLossRule(
    #     name="methyl_benzoate_co2_transfer",
    #     reactant_smarts="[c:1]-[CX3;!R:2](=[O:3])-[OX2:4]-[CH3:5]",
    #     nl_map_nums=[2, 3, 4],
    #     nl_mass=43.9898,
    #     min_num_atoms=10,
    # ),
    # DISABLED 2026-09-29: EI only, and no NIST20 molecule matches.
    # Ionized o-, m-, and p-anisoyl fluorides lose CO and retain fluorine on
    # the methoxy-substituted aromatic residue. Recursive queries keep the
    # methoxy group on the same ring as the acyl fluoride.
    # Paper (EI): https://doi.org/10.1002/oms.1210261122
    # NeutralLossRule(
    #     name="anisoyl_fluoride_co_f_transfer",
    #     reactant_smarts=(
    #         "[$(c1c([OX2][CH3])cccc1),$(c1cc([OX2][CH3])ccc1),"
    #         "$(c1ccc([OX2][CH3])cc1):1]-[CX3;!R:2](=[O:3])-[F:4]"
    #     ),
    #     nl_map_nums=[2, 3],
    #     nl_mass=27.9949,
    #     min_num_atoms=10,
    # ),
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
    # DISABLED 2026-09-29: formula twins, no gain on c0's pool.
    # The three-carbon homologue follows the same pathway but loses C3H6.
    # Paper: https://doi.org/10.1007/s13361-015-1261-4
    # NeutralLossRule(
    #     name="short_haloalkyl_ammonium_c3_transfer",
    #     reactant_smarts=(
    #         "[$([NX3;+0;!$(N-C=O)]),$([NX4;+1]):1]-[CH2:2]-[CH2:3]-[CH2:4]-[Cl,Br,I:5]"
    #     ),
    #     nl_map_nums=[2, 3, 4],
    #     nl_mass=42.0470,
    #     min_num_atoms=5,
    # ),
    # DISABLED 2026-09-29: no NIST20 molecule matches.
    # Deprotonated N-fluoroarenesulfonamides form diagnostic SO2F-. The neutral
    # ArN complement is variably substituted, so this rule maps the small
    # product directly instead of trying to enumerate every neutral atom.
    # Paper (negative ion): https://doi.org/10.1002/rcm.9722
    # NeutralLossRule(
    #     name="n_fluoro_sulfonamide_f_transfer",
    #     reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3H1:5]-[F:6]",
    #     nl_map_nums=[],
    #     nl_mass=None,
    #     min_num_atoms=10,
    #     product_map_nums=[2, 3, 4, 6],
    # ),
    # Deprotonated primary arylsulfonamides extrude SO2 and join the aryl to N; the shared
    # aryl_sulfonamide_so2 requires NH, so it misses primary sulfonamides.
    # Paper (SO2 extrusion): https://pubmed.ncbi.nlm.nih.gov/18059004/
    # Gate: [M-H]-, 67 molecules, +6.4% (95% CI 3.7-9.6); null in [M+H]+.
    NeutralLossRule(
        name="primary_aryl_sulfonamide_so2",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3H2:5]",
        nl_map_nums=[2, 3, 4],
        nl_mass=63.9619,
        min_num_atoms=8,
    ),
    # Tertiary arylsulfonamides: the same SO2 extrusion onto the dialkylated N.
    # Gate (weak): [M+H]+ +0.11% (CI 0.01-0.27), [M-H]- +0.07% (CI 0.01-0.18).
    NeutralLossRule(
        name="tertiary_aryl_sulfonamide_so2",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3H0:5]",
        nl_map_nums=[2, 3, 4],
        nl_mass=63.9619,
        min_num_atoms=8,
    ),
    # Protonated primary arylsulfonamides transfer one sulfonyl oxygen to the aryl ring,
    # giving a phenoxonium ion; S, the other oxygen and N leave.
    # Paper: https://doi.org/10.1016/j.ijms.2018.09.033
    # Gate: [M+H]+, 175 molecules, +2.6% (95% CI 1.9-3.5).
    NeutralLossRule(
        name="primary_aryl_sulfonamide_o_transfer",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3H2:5]",
        nl_map_nums=[2, 4, 5],
        nl_mass=None,
        min_num_atoms=8,
    ),
    # Protonated phenylalanines lose NH3 and CH2CO while a carboxyl oxygen stays with the
    # benzyl residue, giving a hydroxybenzyl-type ion.
    # Paper: https://doi.org/10.3389/fchem.2022.1094329
    # Gate (weak): [M+H]+, 66 molecules, +0.15% (CI 0.00-0.36).
    NeutralLossRule(
        name="phenylalanine_carboxyl_o_transfer",
        reactant_smarts="[c:1]-[CH2:2]-[CH:3](-[NX3;H2:4])-[C:5](=[O:6])[OX2H1:7]",
        nl_map_nums=[3, 4, 5, 6],
        nl_mass=None,
        min_num_atoms=10,
    ),
    # Protonated hydroxamic acids excise NH, joining the N-hydroxyl oxygen to the carbonyl
    # carbon (AIMe Fig. 2c, m/z 173.000). The standardizer writes the hydroximic tautomer
    # ON=C(O)R, so both tautomers are matched.
    # Paper: https://doi.org/10.64898/2026.08.05.743095
    # Gate (weak): [M+H]+, 47 molecules, +0.04% (CI 0.01-0.08).
    NeutralLossRule(
        name="hydroxamic_acid_nh_excision",
        reactant_smarts="[#6]-[CX3:1](~[OX1,OX2H1:2])~[NX2,NX3H1:3]-[OX2H1:4]",
        nl_map_nums=[3],
        nl_mass=15.0109,
        min_num_atoms=5,
    ),
    # Deprotonated perfluoroalkyl sulfonates lose CF2 and transfer an alpha F to give FSO3-.
    # Paper: https://pubmed.ncbi.nlm.nih.gov/41539986/
    # Gate: [M-H]-, 10 molecules, +1.4% (CI 0.0-3.4); at least 1% in 22% of spectra.
    NeutralLossRule(
        name="pfas_sulfonate_f_transfer",
        reactant_smarts="[F:6]-[CX4:1](-F)-[SX4:2](=[O:3])(=[O:4])-[OX2H1,OX1-:5]",
        nl_map_nums=[],
        nl_mass=None,
        min_num_atoms=7,
        product_map_nums=[2, 3, 4, 5, 6],
    ),
    # Protonated secondary and tertiary arylsulfonamides transfer one sulfonyl oxygen to the
    # aryl ring, as the primary rule above; the N-substituted sulfinylamine R2N-S=O leaves,
    # so the whole N side is removed. Measured losses HNOS, C3H7NOS, CH3BrNOS dominate the
    # uncovered intensity of i1's sulfonamide tail (2026-09-30).
    # Paper: https://doi.org/10.1016/j.ijms.2018.09.033
    # Gate: [M+H]+, 115 val molecules +1.8% (CI 0.7-2.8); 104 train-sample molecules +2.3%
    # (CI 1.6-3.2). docs/findings/masster/masster_i1_worst_predictions_2026_09_30.md.
    NeutralLossRule(
        name="substituted_aryl_sulfonamide_o_transfer",
        reactant_smarts="[c:1]-[SX4:2](=[O:3])(=[O:4])-[NX3;!H2:5]",
        nl_map_nums=[2, 4, 5],
        nl_mass=None,
        min_num_atoms=9,
        nl_side_map_nums=[5],
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

# Checkpoint rule vocabulary. Position p holds rule index p + 1 (index 0 is the ordinary
# mechanism), and every rule-indexed parameter is laid out in this order. APPEND ONLY: add a
# new rule at the end, and keep a removed rule's name here so later indices never shift. A
# model records these names' hashes, so a checkpoint trained on a shorter prefix still loads
# and ignores the rules appended after it (MassterModel._restore_smarts_rule_vocabulary).
MASSTER_RULE_VOCABULARY: tuple[str, ...] = (
    "aryl_sulfonamide_so2",
    "aryl_sulfonate_so2",
    "n_nitramine_n2o",
    "sulfoxide_so_loss",
    "diarylvinyl_c2h2_loss",
    "diaryl_ketone_co_loss",
    "aminobenzothiazole_hcn_loss",
    "pyridyl_carbamate_alkyl_transfer",
    "n_acyl_aryl_sulfonamide_carbonyl_o_transfer",
    "diacylhydrazine_carboxylate_o_transfer",
    "n_phenylbenzamide_co_smiles",
    "methyl_benzoate_co2_transfer",
    "anisoyl_fluoride_co_f_transfer",
    "short_haloalkyl_ammonium_c2_transfer",
    "short_haloalkyl_ammonium_c3_transfer",
    "n_fluoro_sulfonamide_f_transfer",
    "primary_aryl_sulfonamide_so2",
    "tertiary_aryl_sulfonamide_so2",
    "primary_aryl_sulfonamide_o_transfer",
    "phenylalanine_carboxyl_o_transfer",
    "hydroxamic_acid_nh_excision",
    "pfas_sulfonate_f_transfer",
    "substituted_aryl_sulfonamide_o_transfer",
)
if len(MASSTER_RULE_VOCABULARY) != len(set(MASSTER_RULE_VOCABULARY)):
    raise ValueError("the MASSter rule vocabulary must not repeat a name")
_unlisted_rules = sorted(set(MASSTER_RULE_NAMES) - set(MASSTER_RULE_VOCABULARY))
if _unlisted_rules:
    raise ValueError(f"append these MASSter rules to MASSTER_RULE_VOCABULARY: {_unlisted_rules}")
MASSTER_NUM_RULES = len(MASSTER_RULE_VOCABULARY)
_MASSTER_RULE_INDEX = {name: index for index, name in enumerate(MASSTER_RULE_VOCABULARY, start=1)}
# CRC-32 of each vocabulary name, in vocabulary order: the checkpoint's record of its rules.
MASSTER_RULE_NAME_HASHES: tuple[int, ...] = tuple(
    zlib.crc32(name.encode("utf-8")) for name in MASSTER_RULE_VOCABULARY
)

# New literature rules are admitted only for the ionization conditions in the
# supporting experiments. Existing empirical rules retain their current
# unrestricted behavior until they are audited separately.
_MASSTER_PRECURSOR_TYPES = {
    # DISABLED 2026-09-29 with their rules.
    # "pyridyl_carbamate_alkyl_transfer": ("[M+H]+",),
    # "methyl_benzoate_co2_transfer": ("[M+H]+",),
    # "anisoyl_fluoride_co_f_transfer": ("[M]+",),
    # "short_haloalkyl_ammonium_c3_transfer": ("[M+H]+", "[M]+"),
    # "n_fluoro_sulfonamide_f_transfer": ("[M-H]-",),
    "n_phenylbenzamide_co_smiles": ("[M-H]-",),
    "short_haloalkyl_ammonium_c2_transfer": ("[M+H]+", "[M]+"),
    "n_acyl_aryl_sulfonamide_carbonyl_o_transfer": ("[M-H]-",),
    "diacylhydrazine_carboxylate_o_transfer": ("[M-H]-",),
    "primary_aryl_sulfonamide_so2": ("[M-H]-",),
    "primary_aryl_sulfonamide_o_transfer": ("[M+H]+",),
    "phenylalanine_carboxyl_o_transfer": ("[M+H]+",),
    "hydroxamic_acid_nh_excision": ("[M+H]+",),
    "pfas_sulfonate_f_transfer": ("[M-H]-",),
    "substituted_aryl_sulfonamide_o_transfer": ("[M+H]+",),
}
_MASSTER_FRAGMENTATION_MODES: dict[str, tuple[str, ...]] = {
    # DISABLED 2026-09-29 with its rule; no registered rule has a mode guard now.
    # "anisoyl_fluoride_co_f_transfer": ("EI",),
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


def _connected_piece_count(
    atoms: Sequence[int],
    endpoints: tuple[tuple[int, ...], ...],
) -> int:
    """Return the number of connected pieces of the induced precursor subgraph on ``atoms``."""
    parent = {atom: atom for atom in atoms}

    def find(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    for left, right in zip(*endpoints, strict=True):
        if left in parent and right in parent:
            parent[find(left)] = find(right)
    return len({find(atom) for atom in atoms})


def _is_connected(atoms: Sequence[int], endpoints: tuple[tuple[int, ...], ...]) -> bool:
    """Return whether the induced precursor subgraph on ``atoms`` is connected."""
    return _connected_piece_count(atoms, endpoints) == 1


def formed_bond_count(atoms: Sequence[int], endpoints: tuple[tuple[int, ...], ...]) -> int:
    """Return the new bonds a template side needs to be one connected ion.

    A side with ``k`` disconnected pieces in the precursor needs at least ``k - 1`` new
    bonds, which the SMARTS assembly forms between its pieces. Each one consumes a valence
    slot on both of its atoms. An ordinary (connected) side forms none.

    Args:
        atoms: Molecule-local atom indices of the side.
        endpoints: Molecule-local bond endpoints, as two parallel sequences.

    Returns:
        The number of pieces minus one.
    """
    return _connected_piece_count(atoms, endpoints) - 1


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
    mechanism; SMARTS rules therefore use one-based indices here. A template whose
    sides are all connected repeats the ordinary partition of its own cut, so it is
    dropped: every returned template has at least one disconnected side.
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

    endpoints = tuple(tuple(int(atom) for atom in column) for column in edges.T)
    unique: dict[
        tuple[tuple[int, ...], tuple[tuple[int, ...], ...], int],
        MassterTemplate,
    ] = {}
    for template in templates:
        if all(_is_connected(component, endpoints) for component in template.components):
            continue
        key = (
            template.broken_bond_indices,
            template.components,
            template.rule_index,
        )
        unique.setdefault(key, template)
    return tuple(unique.values())
