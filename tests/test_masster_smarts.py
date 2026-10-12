"""SMARTS-template support for MASSter candidates."""

from __future__ import annotations

from collections import Counter

import pytest
import torch as th
from rdkit import Chem

from fragnnet.fragmentation.smarts_prepass import FRAG_RULES, NL_RULES, NeutralLossRule
from fragnnet.models.masster.path_selection.candidate_builder import build_masster_candidates
from fragnnet.models.masster.path_selection.smarts_templates import (
    MASSTER_FRAG_RULE_NAMES,
    MASSTER_FRAG_RULES,
    MASSTER_NL_RULE_NAMES,
    MASSTER_NL_RULES,
    MASSTER_NUM_RULES,
    MASSTER_RULE_VOCABULARY,
    MassterTemplate,
    _is_connected,
    build_masster_templates,
    filter_masster_templates,
)
from fragnnet.utils import mol_utils

_CUT_COUNT_WEIGHTS = (0.0, 0.0, 1.0)
_NEW_RULE_EXAMPLES = (
    # DISABLED 2026-09-29 with their rules in path_selection/smarts_templates.py.
    # ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccncc1"),
    ("n_phenylbenzamide_co_smiles", "O=C(Nc1ccccc1)c1ccccc1"),
    # ("methyl_benzoate_co2_transfer", "COC(=O)c1ccccc1"),
    # ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)F)cc1"),
    ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCBr"),
    # ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCCl"),
    # ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccccc1)NF"),
)
# Rules DISABLED 2026-09-29, each with a structure its SMARTS would match.
_DISABLED_RULE_EXAMPLES = (
    ("diarylvinyl_c2h2_loss", "C(=Cc1ccccc1)c1ccccc1"),
    ("diaryl_ketone_co_loss", "O=C(c1ccccc1)c1ccccc1"),
    ("aminobenzothiazole_hcn_loss", "Nc1nc2ccccc2s1"),
    ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccncc1"),
    ("methyl_benzoate_co2_transfer", "COC(=O)c1ccccc1"),
    ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)F)cc1"),
    ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCCl"),
    ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccccc1)NF"),
)


def _templates_for_rule(smiles: str, rule_name: str):
    mol = Chem.MolFromSmiles(smiles)
    return tuple(
        template
        for template in build_masster_templates(mol)
        if template.rule_name == rule_name
    )


def _component_element_counts(mol: Chem.Mol, component: tuple[int, ...]) -> Counter:
    return Counter(mol.GetAtomWithIdx(index).GetSymbol() for index in component)


def test_masster_registry_counts_and_literature_rule_order() -> None:
    # 15 registered rules; the vocabulary keeps all 23 indices, disabled rules included.
    assert len(MASSTER_FRAG_RULE_NAMES) == 2
    assert len(MASSTER_NL_RULE_NAMES) == 13
    assert MASSTER_NUM_RULES == 23
    assert MASSTER_FRAG_RULE_NAMES[0] == "n_acyl_aryl_sulfonamide_carbonyl_o_transfer"
    assert MASSTER_NL_RULE_NAMES[4:6] == tuple(rule_name for rule_name, _ in _NEW_RULE_EXAMPLES)


@pytest.mark.parametrize(("rule_name", "smiles"), _DISABLED_RULE_EXAMPLES)
def test_disabled_rules_keep_their_vocabulary_slot_and_build_no_templates(
    rule_name: str, smiles: str
) -> None:
    assert rule_name in MASSTER_RULE_VOCABULARY
    assert rule_name not in (*MASSTER_FRAG_RULE_NAMES, *MASSTER_NL_RULE_NAMES)
    assert not _templates_for_rule(smiles, rule_name)


def test_all_masster_smarts_compile() -> None:
    for rule in (*MASSTER_FRAG_RULES, *MASSTER_NL_RULES):
        assert Chem.MolFromSmarts(rule.reactant_smarts) is not None, rule.name


@pytest.mark.parametrize(
    ("rule_name", "smiles"),
    [
        ("n_phenylbenzamide_co_smiles", "O=C(Nc1ccccc1)c1ccccc1"),
        ("n_phenylbenzamide_co_smiles", "O=C(Nc1ccc(F)cc1)c1ccccc1"),
        # DISABLED 2026-09-29 with their rules in path_selection/smarts_templates.py.
        # ("methyl_benzoate_co2_transfer", "COC(=O)c1ccccc1"),
        # ("methyl_benzoate_co2_transfer", "COC(=O)c1ccc(Cl)cc1"),
        # ("anisoyl_fluoride_co_f_transfer", "COc1ccccc1C(=O)F"),
        # ("anisoyl_fluoride_co_f_transfer", "COc1cccc(C(=O)F)c1"),
        # ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)F)cc1"),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCCl"),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCBr"),
        ("short_haloalkyl_ammonium_c2_transfer", "C[N+](C)(C)CCI"),
        # ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCCl"),
        # ("short_haloalkyl_ammonium_c3_transfer", "C[N+](C)(C)CCCBr"),
        # ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccccn1"),
        # ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1cccnc1"),
        # ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccncc1"),
        # ("pyridyl_carbamate_alkyl_transfer", "CCN(CC)C(=O)Oc1ccncc1"),
        # (
        #     "pyridyl_carbamate_alkyl_transfer",
        #     "CCCCN(CCCC)C(=O)Oc1ccncc1",
        # ),
        # ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccccc1)NF"),
        # ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccc(F)cc1)NF"),
    ],
)
def test_new_rules_match_supported_precursor_structures(
    rule_name: str,
    smiles: str,
) -> None:
    assert _templates_for_rule(smiles, rule_name)


@pytest.mark.parametrize(
    ("rule_name", "smiles"),
    [
        ("n_phenylbenzamide_co_smiles", "O=C(N(C)c1ccccc1)c1ccccc1"),
        ("n_phenylbenzamide_co_smiles", "O=C(c1ccccc1)c1ccccc1"),
        # DISABLED 2026-09-29 with their rules in path_selection/smarts_templates.py.
        # ("methyl_benzoate_co2_transfer", "CCOC(=O)c1ccccc1"),
        # ("methyl_benzoate_co2_transfer", "COC(C)=O"),
        # ("anisoyl_fluoride_co_f_transfer", "O=C(F)c1ccccc1"),
        # ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)Cl)cc1"),
        # ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)OC)cc1"),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCF"),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCCCBr"),
        ("short_haloalkyl_ammonium_c2_transfer", "CC(=O)NCCBr"),
        # ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCBr"),
        # ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCCBr"),
        # ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCF"),
        # ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccccc1"),
        # ("pyridyl_carbamate_alkyl_transfer", "CCCN(CCC)C(=O)Oc1ccncc1"),
        # ("pyridyl_carbamate_alkyl_transfer", "CC(=O)Oc1ccncc1"),
        # ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccccc1)N"),
        # ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(C)NF"),
        # ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(F)c1ccccc1"),
    ],
)
def test_new_rules_reject_close_negative_structures(
    rule_name: str,
    smiles: str,
) -> None:
    assert not _templates_for_rule(smiles, rule_name)


@pytest.mark.parametrize(
    ("rule_name", "smiles", "expected_component"),
    [
        (
            "n_phenylbenzamide_co_smiles",
            "O=C(Nc1ccccc1)c1ccccc1",
            Counter({"C": 1, "O": 1}),
        ),
        # DISABLED 2026-09-29 with their rules in path_selection/smarts_templates.py.
        # (
        #     "methyl_benzoate_co2_transfer",
        #     "COC(=O)c1ccccc1",
        #     Counter({"O": 2, "C": 1}),
        # ),
        # (
        #     "anisoyl_fluoride_co_f_transfer",
        #     "COc1ccc(C(=O)F)cc1",
        #     Counter({"C": 1, "O": 1}),
        # ),
        (
            "short_haloalkyl_ammonium_c2_transfer",
            "CN(C)CCBr",
            Counter({"C": 2}),
        ),
        # (
        #     "short_haloalkyl_ammonium_c3_transfer",
        #     "CN(C)CCCCl",
        #     Counter({"C": 3}),
        # ),
        # (
        #     "n_fluoro_sulfonamide_f_transfer",
        #     "O=S(=O)(c1ccccc1)NF",
        #     Counter({"O": 2, "S": 1, "F": 1}),
        # ),
    ],
)
def test_new_neutral_loss_rules_build_the_intended_atom_component(
    rule_name: str,
    smiles: str,
    expected_component: Counter,
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    template = _templates_for_rule(smiles, rule_name)[0]
    component_counts = [
        _component_element_counts(mol, component) for component in template.components
    ]

    assert expected_component in component_counts


# DISABLED 2026-09-29 with pyridyl_carbamate_alkyl_transfer in path_selection/smarts_templates.py.
# @pytest.mark.parametrize(
#     ("smiles", "expected_transferred_component"),
#     [
#         ("CN(C)C(=O)Oc1ccncc1", Counter({"C": 6, "N": 1, "O": 1})),
#         ("CCN(CC)C(=O)Oc1ccncc1", Counter({"C": 7, "N": 1, "O": 1})),
#         (
#             "CCCCN(CCCC)C(=O)Oc1ccncc1",
#             Counter({"C": 9, "N": 1, "O": 1}),
#         ),
#     ],
# )
# def test_pyridyl_carbamate_rule_retains_the_complete_selected_alkyl_group(
#     smiles: str,
#     expected_transferred_component: Counter,
# ) -> None:
#     mol = Chem.MolFromSmiles(smiles)
#     templates = _templates_for_rule(smiles, "pyridyl_carbamate_alkyl_transfer")
#
#     assert any(
#         expected_transferred_component
#         in [_component_element_counts(mol, component) for component in template.components]
#         for template in templates
#     )


@pytest.mark.parametrize(
    ("rule_name", "smiles", "precursor_type", "fragmentation_mode"),
    [
        ("n_phenylbenzamide_co_smiles", "O=C(Nc1ccccc1)c1ccccc1", "[M-H]-", None),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCBr", "[M+H]+", None),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCBr", "[M]+", None),
        # DISABLED 2026-09-29 with their rules in path_selection/smarts_templates.py.
        # ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccncc1", "[M+H]+", None),
        # ("methyl_benzoate_co2_transfer", "COC(=O)c1ccccc1", "[M+H]+", None),
        # ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)F)cc1", "[M]+", "EI"),
        # ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCCl", "[M]+", None),
        # ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccccc1)NF", "[M-H]-", None),
    ],
)
def test_new_rules_keep_only_literature_supported_ionization_conditions(
    rule_name: str,
    smiles: str,
    precursor_type: str,
    fragmentation_mode: str | None,
) -> None:
    templates = _templates_for_rule(smiles, rule_name)
    supported = filter_masster_templates(
        templates,
        precursor_type=precursor_type,
        fragmentation_mode=fragmentation_mode,
    )
    wrong_precursor = filter_masster_templates(
        templates,
        precursor_type="[M-H]-" if precursor_type != "[M-H]-" else "[M+H]+",
        fragmentation_mode=fragmentation_mode,
    )

    assert supported
    assert not wrong_precursor


def test_ei_only_template_is_rejected_when_fragmentation_mode_is_unknown_or_cid() -> None:
    # The EI-only anisoyl rule was DISABLED 2026-09-29, leaving no registered rule with a
    # fragmentation-mode guard, so a synthetic template exercises the filter.
    templates = (
        MassterTemplate(
            broken_bond_indices=(0,),
            components=((0,), (1,)),
            rule_index=1,
            rule_name="ei_only",
            allowed_precursor_types=("[M]+",),
            allowed_fragmentation_modes=("EI",),
        ),
    )

    assert filter_masster_templates(templates, precursor_type="[M]+", fragmentation_mode="EI")
    for fragmentation_mode in (None, "CID", "HCD"):
        assert not filter_masster_templates(
            templates,
            precursor_type="[M]+",
            fragmentation_mode=fragmentation_mode,
        )


def test_existing_rules_remain_unrestricted_pending_separate_mode_audit() -> None:
    templates = _templates_for_rule(
        "c1ccccc1S(=O)c1ccccc1",
        "sulfoxide_so_loss",
    )

    assert filter_masster_templates(
        templates,
        precursor_type="[M+H]+",
        fragmentation_mode=None,
    )


@pytest.mark.parametrize(("rule_name", "smiles"), _NEW_RULE_EXAMPLES)
def test_new_rules_attach_to_candidate_masks_as_distinct_mechanisms(
    rule_name: str,
    smiles: str,
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    templates = build_masster_templates(mol)
    rule_template = next(template for template in templates if template.rule_name == rule_name)
    endpoints = th.tensor(
        [
            [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
            [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
        ],
        dtype=th.long,
    )

    candidates = build_masster_candidates(
        th.full((mol.GetNumBonds(),), -8.0),
        chemical_bond_index=endpoints,
        atom_batch_indices=th.zeros(mol.GetNumAtoms(), dtype=th.long),
        atom_ptr=th.tensor([0, mol.GetNumAtoms()], dtype=th.long),
        max_masks=1,
        include_single_bond_cuts=False,
        smarts_templates=(templates,),
    )

    mechanism_index = candidates.mechanism_rule_indices.tolist().index(rule_template.rule_index)
    mask_index = candidates.mechanism_mask_indices[mechanism_index].item()
    assert candidates.mask_cut_counts[mask_index].item() == len(rule_template.broken_bond_indices)
    assert (candidates.component_mechanism_indices == mechanism_index).sum().item() == 2


def test_literature_rule_indices_follow_the_shared_rules() -> None:
    # The 2026-09-27 drop of the ring rules and aryl_sulfonate_so3, and the gated rules added
    # the same day, renumbered these; no SMARTS checkpoint existed to reassign.
    # The rules DISABLED on 2026-09-29 keep their slots (8, 12, 13, 15, 16), so these stay.
    expected_indices = {
        "n_phenylbenzamide_co_smiles": 11,
        "short_haloalkyl_ammonium_c2_transfer": 14,
    }

    for rule_name, smiles in _NEW_RULE_EXAMPLES:
        assert _templates_for_rule(smiles, rule_name)[0].rule_index == expected_indices[rule_name]


def test_new_rule_templates_partition_every_precursor_atom_once() -> None:
    for rule_name, smiles in _NEW_RULE_EXAMPLES:
        mol = Chem.MolFromSmiles(smiles)
        for template in _templates_for_rule(smiles, rule_name):
            flattened = [atom for component in template.components for atom in component]
            assert sorted(flattened) == list(range(mol.GetNumAtoms()))
            assert len(flattened) == len(set(flattened))


def test_new_rules_add_partitions_not_produced_by_ordinary_bond_deletion() -> None:
    for rule_name, smiles in _NEW_RULE_EXAMPLES:
        mol = Chem.MolFromSmiles(smiles)
        for template in _templates_for_rule(smiles, rule_name):
            fragmented = Chem.FragmentOnBonds(
                mol,
                list(template.broken_bond_indices),
                addDummies=False,
            )
            ordinary_components = tuple(
                sorted(tuple(sorted(component)) for component in Chem.GetMolFrags(fragmented))
            )

            assert template.components != ordinary_components


def test_masster_templates_preserve_atom_partitions_and_cut_bonds() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1S(=O)c1ccccc1")

    templates = build_masster_templates(mol)

    sulfoxide = next(
        template for template in templates if template.rule_name == "sulfoxide_so_loss"
    )
    assert sulfoxide.broken_bond_indices == (5, 7)
    assert sorted(atom for component in sulfoxide.components for atom in component) == list(
        range(mol.GetNumAtoms())
    )
    assert set(sulfoxide.components[0]).isdisjoint(sulfoxide.components[1])


def test_masster_owns_a_curated_rule_registry() -> None:
    assert "crf12_0" not in MASSTER_FRAG_RULE_NAMES
    assert "crf12_1" not in MASSTER_FRAG_RULE_NAMES
    assert "crf13_2" not in MASSTER_FRAG_RULE_NAMES
    assert "morphinan_ring_cleavage" not in MASSTER_FRAG_RULE_NAMES
    assert "sulfoxide_so_loss" in MASSTER_NL_RULE_NAMES
    assert all(rule.merge_outer_map_nums for rule in MASSTER_FRAG_RULES)


def test_masster_drops_the_ring_rules_and_the_single_cut_sulfonate() -> None:
    # Shared ring-cut rules stay in FRAG_RULES for FraGNNet's prepass, not in MASSter.
    shared_ring_rules = {
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
    }
    assert shared_ring_rules <= {rule.name for rule in FRAG_RULES}
    assert shared_ring_rules.isdisjoint(MASSTER_FRAG_RULE_NAMES)
    assert MASSTER_FRAG_RULE_NAMES == (
        "n_acyl_aryl_sulfonamide_carbonyl_o_transfer",
        "diacylhydrazine_carboxylate_o_transfer",
    )
    assert "aryl_sulfonate_so3" in {rule.name for rule in NL_RULES}
    assert "aryl_sulfonate_so3" not in MASSTER_NL_RULE_NAMES
    assert not build_masster_templates(Chem.MolFromSmiles("O=c1c2ccccc2oc2ccccc12"))


@pytest.mark.parametrize(
    ("smiles", "expected_rules"),
    [
        # Acyclic sulfonamide: the product Ar...NH-Het is disconnected, so it is kept, and so
        # is the O-transfer product Ar...O of the secondary sulfonamide.
        ("Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1", ["aryl_sulfonamide_so2", "substituted_aryl_sulfonamide_o_transfer"]),
        # Saccharin: Ar and N stay joined through the C(=O) ring, so SO2 extrusion only
        # repeats the ordinary partition of the same two cuts.
        ("O=C1NS(=O)(=O)c2ccccc21", []),
    ],
)
def test_templates_repeating_the_ordinary_partition_are_dropped(
    smiles: str, expected_rules: list[str]
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    templates = build_masster_templates(mol)
    assert [template.rule_name for template in templates] == expected_rules
    for template in templates:
        fragmented = Chem.FragmentOnBonds(
            mol, list(template.broken_bond_indices), addDummies=False
        )
        ordinary = {tuple(sorted(piece)) for piece in Chem.GetMolFrags(fragmented)}
        assert any(component not in ordinary for component in template.components)


def test_masster_candidates_attach_template_as_mechanism_of_same_mask() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1S(=O)c1ccccc1")
    templates = build_masster_templates(mol)
    endpoints = th.tensor(
        [
            [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
            [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
        ],
        dtype=th.long,
    )

    candidates = build_masster_candidates(
        th.full((mol.GetNumBonds(),), -8.0),
        chemical_bond_index=endpoints,
        atom_batch_indices=th.zeros(mol.GetNumAtoms(), dtype=th.long),
        atom_ptr=th.tensor([0, mol.GetNumAtoms()], dtype=th.long),
        max_masks=1,
        include_single_bond_cuts=False,
        smarts_templates=(templates,),
    )

    sulfoxide_rule_index = next(
        template.rule_index for template in templates if template.rule_name == "sulfoxide_so_loss"
    )
    mechanism_index = candidates.mechanism_rule_indices.tolist().index(sulfoxide_rule_index)
    mask_index = candidates.mechanism_mask_indices[mechanism_index].item()
    assert candidates.mask_cut_counts[mask_index].item() == 2
    assert (candidates.component_mechanism_indices == mechanism_index).sum().item() == 2


def test_no_templates_leave_one_ordinary_mechanism_per_mask() -> None:
    candidates = build_masster_candidates(
        th.tensor([-1.0]),
        chemical_bond_index=th.tensor([[0], [1]], dtype=th.long),
        atom_batch_indices=th.zeros(2, dtype=th.long),
        atom_ptr=th.tensor([0, 2], dtype=th.long),
        max_masks=1,
        include_single_bond_cuts=False,
    )

    assert th.equal(
        candidates.mechanism_mask_indices,
        th.arange(candidates.mask_log_probabilities.numel()),
    )
    assert candidates.mechanism_rule_indices.tolist() == [0]
    assert th.equal(
        candidates.component_mechanism_indices,
        candidates.component_mask_indices,
    )


def _batched_candidates(smiles_list: list[str], **kwargs):
    mols = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    templates = tuple(build_masster_templates(mol) for mol in mols)
    begin: list[int] = []
    end: list[int] = []
    batch: list[int] = []
    ptr = [0]
    offset = 0
    for molecule_index, mol in enumerate(mols):
        for bond in mol.GetBonds():
            begin.append(bond.GetBeginAtomIdx() + offset)
            end.append(bond.GetEndAtomIdx() + offset)
        batch.extend([molecule_index] * mol.GetNumAtoms())
        offset += mol.GetNumAtoms()
        ptr.append(offset)
    candidates = build_masster_candidates(
        th.full((len(begin),), -4.0),
        chemical_bond_index=th.tensor([begin, end], dtype=th.long),
        atom_batch_indices=th.tensor(batch, dtype=th.long),
        atom_ptr=th.tensor(ptr, dtype=th.long),
        include_single_bond_cuts=True,
        smarts_templates=templates,
        **kwargs,
    )
    return candidates, templates


def test_template_representatives_stay_in_range_for_later_batch_members() -> None:
    candidates, _ = _batched_candidates(
        ["O=C(Nc1ccccc1)c1ccccc1", "c1ccccc1S(=O)c1ccccc1", "CN(C)CCBr"],
        max_masks=16,
        max_unique_components=16,
        max_enumerated_masks=64,
    )
    representatives = candidates.unique_representative_component_indices

    assert representatives.numel() == candidates.num_unique_components
    assert bool((representatives < candidates.num_components).all())
    assert th.equal(
        candidates.component_unique_indices[representatives],
        th.arange(candidates.num_unique_components),
    )


def test_templates_fire_on_every_batch_member_without_shifting_masks() -> None:
    candidates, templates = _batched_candidates(
        ["O=C(Nc1ccccc1)c1ccccc1", "c1ccccc1S(=O)c1ccccc1", "CN(C)CCBr"],
        max_masks=16,
    )
    expected_rules = {template.rule_index for molecule in templates for template in molecule}
    mechanism_batch_indices = candidates.mask_batch_indices[candidates.mechanism_mask_indices]
    smarts_mechanisms = candidates.mechanism_rule_indices > 0

    assert expected_rules <= set(candidates.mechanism_rule_indices.tolist())
    assert set(mechanism_batch_indices[smarts_mechanisms].tolist()) == {0, 1, 2}
    for molecule_index in range(3):
        mask_probabilities = candidates.mask_log_probabilities[
            candidates.mask_batch_indices == molecule_index
        ].exp()
        assert mask_probabilities.sum().item() <= 1.0 + 1e-5


def test_grouped_smarts_lookup_matches_distinct_view_assembly_and_gradients() -> None:
    chemical_bond_index = th.tensor(
        [[0, 1, 2, 4, 5, 6], [1, 2, 3, 5, 6, 7]],
        dtype=th.long,
    )
    atom_batch_indices = th.tensor([0] * 4 + [1] * 4, dtype=th.long)
    atom_ptr = th.tensor([0, 4, 8], dtype=th.long)
    templates = (
        (
            MassterTemplate(
                broken_bond_indices=(0, 2),
                components=((0, 3), (1, 2)),
                rule_index=3,
                rule_name="view_a",
            ),
        ),
        (
            MassterTemplate(
                broken_bond_indices=(0, 2),
                components=((0, 2), (1, 3)),
                rule_index=7,
                rule_name="view_b",
            ),
        ),
    )
    kwargs = {
        "chemical_bond_index": chemical_bond_index,
        "atom_batch_indices": atom_batch_indices,
        "atom_ptr": atom_ptr,
        "max_masks": 4,
        "max_unique_components": 4,
        "max_enumerated_masks": 16,
        "include_single_bond_cuts": True,
        "smarts_templates": templates,
    }
    repeated_logits = th.tensor(
        [0.7, -0.3, 0.1, 0.7, -0.3, 0.1],
        requires_grad=True,
    )
    grouped_logits = repeated_logits.detach().clone().requires_grad_(True)

    repeated = build_masster_candidates(repeated_logits, **kwargs)
    grouped = build_masster_candidates(
        grouped_logits,
        **kwargs,
        candidate_group_ids=th.tensor([41, 41], dtype=th.long),
    )

    for name, expected in vars(repeated).items():
        if name == "timings":
            continue
        actual = getattr(grouped, name)
        if isinstance(expected, th.Tensor):
            assert th.equal(actual, expected), name
        else:
            assert actual == expected, name
    assert grouped.mechanism_rule_indices.tolist().count(3) == 1
    assert grouped.mechanism_rule_indices.tolist().count(7) == 1

    weights = th.linspace(0.25, 1.75, repeated.mask_log_probabilities.numel())
    (repeated.mask_log_probabilities * weights).sum().backward()
    (grouped.mask_log_probabilities * weights).sum().backward()
    assert th.equal(grouped_logits.grad, repeated_logits.grad)


def test_mechanism_head_scores_only_masks_with_a_template_alternative() -> None:
    from test_masster_model import _build_batch, _build_model

    from fragnnet.models.masster.bond_fragment_heads import pool_boundary_bond_embeddings
    from fragnnet.models.masster.masster_model import _segment_logsumexp

    smiles = ("Cc1ccc(S(=O)(=O)Nc2ccccn2)cc1", "CCO", "Oc1ccc(S(=O)(=O)O)cc1")
    batch = _build_batch(smiles)
    model = _build_model(batch, max_masks=8, unique_fragment_mode=True, use_smarts_templates=True)
    model.eval()
    candidates = build_masster_candidates(
        th.linspace(-2.0, 2.0, batch.chemical_bond_index.shape[1]),
        chemical_bond_index=batch.chemical_bond_index,
        atom_batch_indices=batch.batch,
        atom_ptr=batch.ptr,
        max_masks=8,
        include_single_bond_cuts=True,
        max_unique_components=8,
        max_enumerated_masks=32,
        smarts_templates=tuple(build_masster_templates(Chem.MolFromSmiles(s)) for s in smiles),
    )
    generator = th.Generator().manual_seed(3)
    context_dim = model.mechanism_head.rule_embedding.embedding_dim
    bond_dim = model.mechanism_head.scorer[0].in_features - 2 * context_dim
    atom_embeddings = th.randn(batch.num_nodes, bond_dim, generator=generator)
    context = th.randn(len(smiles), context_dim, generator=generator)

    actual = model._mechanism_log_probabilities(
        candidates, atom_embeddings, batch.chemical_bond_index, context
    )

    # Brute force: score every mechanism, then normalize within each mask.
    bond_embeddings = (
        atom_embeddings[batch.chemical_bond_index[0]] + atom_embeddings[batch.chemical_bond_index[1]]
    ) / 2.0
    num_masks = candidates.mask_log_probabilities.numel()
    logits = model.mechanism_head(
        pool_boundary_bond_embeddings(
            bond_embeddings,
            candidates.mechanism_boundary_bond_indices,
            candidates.boundary_mechanism_indices,
            num_components=candidates.num_mechanisms,
        ),
        candidates.mechanism_rule_indices,
        context[candidates.mask_batch_indices[candidates.mechanism_mask_indices]],
    )
    expected = logits - _segment_logsumexp(logits, candidates.mechanism_mask_indices, num_masks)[
        candidates.mechanism_mask_indices
    ]
    assert candidates.num_mechanisms > num_masks
    th.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-6)
    single = th.bincount(candidates.mechanism_mask_indices, minlength=num_masks) == 1
    assert th.equal(actual[single[candidates.mechanism_mask_indices]], th.zeros(int(single.sum())))


# Rules admitted by the 2026-09-27 per-class gate: (rule, standardized positive, admitted
# precursor type, product heavy-atom composition).
_GATED_RULE_EXAMPLES = (
    ("primary_aryl_sulfonamide_so2", "Nc1ccc(S(N)(=O)=O)cc1", "[M-H]-", {"C": 6, "N": 2}),
    ("tertiary_aryl_sulfonamide_so2", "CN(C)S(=O)(=O)c1ccccc1", "[M+H]+", {"C": 8, "N": 1}),
    ("primary_aryl_sulfonamide_o_transfer", "Nc1ccc(S(N)(=O)=O)cc1", "[M+H]+", {"C": 6, "N": 1, "O": 1}),
    # The whole N side leaves with the sulfinyl: CH3N-S=O, and (CH3)2N-S=O with the bromoaryl kept.
    ("substituted_aryl_sulfonamide_o_transfer", "CNS(=O)(=O)c1ccccc1", "[M+H]+", {"C": 6, "O": 1}),
    ("substituted_aryl_sulfonamide_o_transfer", "CN(C)S(=O)(=O)c1ccc(Br)cc1", "[M+H]+", {"C": 6, "O": 1, "Br": 1}),
    ("phenylalanine_carboxyl_o_transfer", "NC(Cc1ccccc1)C(=O)O", "[M+H]+", {"C": 7, "O": 1}),
    ("hydroxamic_acid_nh_excision", "ONC(=O)c1ccccc1", "[M+H]+", {"C": 7, "O": 2}),
    (
        "pfas_sulfonate_f_transfer",
        "O=S(=O)(O)C(F)(F)C(F)(F)C(F)(F)C(F)(F)F",
        "[M-H]-",
        {"S": 1, "O": 3, "F": 1},
    ),
    ("n_acyl_aryl_sulfonamide_carbonyl_o_transfer", "CC(=O)NS(=O)(=O)c1ccccc1", "[M-H]-", {"C": 6, "O": 1}),
    (
        "diacylhydrazine_carboxylate_o_transfer",
        "CCc1ccc(cc1)C(=O)NN(C(C)(C)C)C(=O)c1cc(C)cc(C)c1",
        "[M-H]-",
        {"C": 9, "O": 2},
    ),
)


@pytest.mark.parametrize(("rule_name", "smiles", "precursor_type", "product"), _GATED_RULE_EXAMPLES)
def test_gated_rules_build_their_rearranged_product_on_standardized_molecules(
    rule_name: str, smiles: str, precursor_type: str, product: dict[str, int]
) -> None:
    # Standardized input is what training sees: e.g. hydroxamic acids become ON=C(O)R.
    mol = mol_utils.mol_from_smiles(smiles)
    templates = [t for t in build_masster_templates(mol) if t.rule_name == rule_name]
    assert len(templates) == 1
    endpoints = tuple(zip(*[(b.GetBeginAtomIdx(), b.GetEndAtomIdx()) for b in mol.GetBonds()]))
    disconnected = [c for c in templates[0].components if not _is_connected(c, endpoints)]
    assert [dict(_component_element_counts(mol, c)) for c in disconnected] == [product]
    assert filter_masster_templates(tuple(templates), precursor_type=precursor_type, fragmentation_mode=None)
    other = "[M+H]+" if precursor_type == "[M-H]-" else "[M-H]-"
    if rule_name != "tertiary_aryl_sulfonamide_so2":
        assert not filter_masster_templates(
            tuple(templates), precursor_type=other, fragmentation_mode=None
        )


@pytest.mark.parametrize(
    ("rule_name", "smiles"),
    [
        # Secondary sulfonamide: the shared aryl_sulfonamide_so2 covers it, not these.
        ("primary_aryl_sulfonamide_so2", "CNS(=O)(=O)c1ccccc1"),
        ("primary_aryl_sulfonamide_o_transfer", "CNS(=O)(=O)c1ccccc1"),
        ("tertiary_aryl_sulfonamide_so2", "CNS(=O)(=O)c1ccccc1"),
        # Primary sulfonamide: the primary O-transfer rule covers it, not the substituted one.
        ("substituted_aryl_sulfonamide_o_transfer", "NS(=O)(=O)c1ccccc1"),
        # Sultam: the N side is fused to the aryl ring, so there is no separate leaving group.
        ("substituted_aryl_sulfonamide_o_transfer", "O=S1(=O)NCc2ccccc21"),
        # N-sulfonyl amidine: standardization writes the sulfonyl N as an imine (NX2).
        ("substituted_aryl_sulfonamide_o_transfer", "CC(=O)Nc1ccc(S(=O)(=O)NC2=NCCS2)cc1"),
        # Alkyl sulfonamide: no aryl to receive N or O.
        ("primary_aryl_sulfonamide_so2", "CS(N)(=O)=O"),
        # Phenylalanine amide and N-acetyl phenylalanine: no free carboxyl or no primary amine.
        ("phenylalanine_carboxyl_o_transfer", "NC(Cc1ccccc1)C(N)=O"),
        ("phenylalanine_carboxyl_o_transfer", "CC(=O)NC(Cc1ccccc1)C(=O)O"),
        # Plain amide and N-methoxy amide: no N-hydroxyl.
        ("hydroxamic_acid_nh_excision", "NC(=O)c1ccccc1"),
        ("hydroxamic_acid_nh_excision", "CONC(=O)c1ccccc1"),
        # Methanesulfonic acid: no perfluorinated alpha carbon.
        ("pfas_sulfonate_f_transfer", "CS(=O)(=O)O"),
        # Benzamide: an amide without the sulfonyl.
        ("n_acyl_aryl_sulfonamide_carbonyl_o_transfer", "CC(=O)Nc1ccccc1"),
        # Diacylhydrazine without the tert-butyl nitrogen.
        ("diacylhydrazine_carboxylate_o_transfer", "O=C(NNC(=O)c1ccccc1)c1ccccc1"),
    ],
)
def test_gated_rules_reject_close_negative_structures(rule_name: str, smiles: str) -> None:
    mol = mol_utils.mol_from_smiles(smiles)
    assert rule_name not in {t.rule_name for t in build_masster_templates(mol)}


def test_neutral_loss_side_map_nums_must_be_neutral_loss_atoms() -> None:
    with pytest.raises(ValueError, match="subset of nl_map_nums"):
        NeutralLossRule("bad", "[c:1]-[S:2]", nl_map_nums=[2], nl_mass=None, nl_side_map_nums=[1])
