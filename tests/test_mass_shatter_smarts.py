"""SMARTS-template support for MASShatter candidates."""

from __future__ import annotations

from collections import Counter

import pytest
import torch as th
from rdkit import Chem

from fragnnet.frag.mass_shatter_candidates import build_mass_shatter_candidates
from fragnnet.frag.mass_shatter_templates import (
    MASS_SHATTER_FRAG_RULE_NAMES,
    MASS_SHATTER_FRAG_RULES,
    MASS_SHATTER_NL_RULE_NAMES,
    MASS_SHATTER_NL_RULES,
    MASS_SHATTER_NUM_RULES,
    build_mass_shatter_templates,
    filter_mass_shatter_templates,
)

_CUT_COUNT_WEIGHTS = (0.0, 0.0, 1.0)
_NEW_RULE_EXAMPLES = (
    ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccncc1"),
    ("n_phenylbenzamide_co_smiles", "O=C(Nc1ccccc1)c1ccccc1"),
    ("methyl_benzoate_co2_transfer", "COC(=O)c1ccccc1"),
    ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)F)cc1"),
    ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCBr"),
    ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCCl"),
    ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccccc1)NF"),
)


def _templates_for_rule(smiles: str, rule_name: str):
    mol = Chem.MolFromSmiles(smiles)
    return tuple(
        template
        for template in build_mass_shatter_templates(mol)
        if template.rule_name == rule_name
    )


def _component_element_counts(mol: Chem.Mol, component: tuple[int, ...]) -> Counter:
    return Counter(mol.GetAtomWithIdx(index).GetSymbol() for index in component)


def test_mass_shatter_registry_appends_seven_literature_templates() -> None:
    assert len(MASS_SHATTER_FRAG_RULE_NAMES) == 12
    assert len(MASS_SHATTER_NL_RULE_NAMES) == 14
    assert MASS_SHATTER_NUM_RULES == 26
    assert MASS_SHATTER_FRAG_RULE_NAMES[-1] == "pyridyl_carbamate_alkyl_transfer"
    assert MASS_SHATTER_NL_RULE_NAMES[-6:] == tuple(
        rule_name for rule_name, _ in _NEW_RULE_EXAMPLES[1:]
    )


def test_all_mass_shatter_smarts_compile() -> None:
    for rule in (*MASS_SHATTER_FRAG_RULES, *MASS_SHATTER_NL_RULES):
        assert Chem.MolFromSmarts(rule.reactant_smarts) is not None, rule.name


@pytest.mark.parametrize(
    ("rule_name", "smiles"),
    [
        ("n_phenylbenzamide_co_smiles", "O=C(Nc1ccccc1)c1ccccc1"),
        ("n_phenylbenzamide_co_smiles", "O=C(Nc1ccc(F)cc1)c1ccccc1"),
        ("methyl_benzoate_co2_transfer", "COC(=O)c1ccccc1"),
        ("methyl_benzoate_co2_transfer", "COC(=O)c1ccc(Cl)cc1"),
        ("anisoyl_fluoride_co_f_transfer", "COc1ccccc1C(=O)F"),
        ("anisoyl_fluoride_co_f_transfer", "COc1cccc(C(=O)F)c1"),
        ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)F)cc1"),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCCl"),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCBr"),
        ("short_haloalkyl_ammonium_c2_transfer", "C[N+](C)(C)CCI"),
        ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCCl"),
        ("short_haloalkyl_ammonium_c3_transfer", "C[N+](C)(C)CCCBr"),
        ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccccn1"),
        ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1cccnc1"),
        ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccncc1"),
        ("pyridyl_carbamate_alkyl_transfer", "CCN(CC)C(=O)Oc1ccncc1"),
        (
            "pyridyl_carbamate_alkyl_transfer",
            "CCCCN(CCCC)C(=O)Oc1ccncc1",
        ),
        ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccccc1)NF"),
        ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccc(F)cc1)NF"),
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
        ("methyl_benzoate_co2_transfer", "CCOC(=O)c1ccccc1"),
        ("methyl_benzoate_co2_transfer", "COC(C)=O"),
        ("anisoyl_fluoride_co_f_transfer", "O=C(F)c1ccccc1"),
        ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)Cl)cc1"),
        ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)OC)cc1"),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCF"),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCCCBr"),
        ("short_haloalkyl_ammonium_c2_transfer", "CC(=O)NCCBr"),
        ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCBr"),
        ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCCBr"),
        ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCF"),
        ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccccc1"),
        ("pyridyl_carbamate_alkyl_transfer", "CCCN(CCC)C(=O)Oc1ccncc1"),
        ("pyridyl_carbamate_alkyl_transfer", "CC(=O)Oc1ccncc1"),
        ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccccc1)N"),
        ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(C)NF"),
        ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(F)c1ccccc1"),
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
        (
            "methyl_benzoate_co2_transfer",
            "COC(=O)c1ccccc1",
            Counter({"O": 2, "C": 1}),
        ),
        (
            "anisoyl_fluoride_co_f_transfer",
            "COc1ccc(C(=O)F)cc1",
            Counter({"C": 1, "O": 1}),
        ),
        (
            "short_haloalkyl_ammonium_c2_transfer",
            "CN(C)CCBr",
            Counter({"C": 2}),
        ),
        (
            "short_haloalkyl_ammonium_c3_transfer",
            "CN(C)CCCCl",
            Counter({"C": 3}),
        ),
        (
            "n_fluoro_sulfonamide_f_transfer",
            "O=S(=O)(c1ccccc1)NF",
            Counter({"O": 2, "S": 1, "F": 1}),
        ),
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


@pytest.mark.parametrize(
    ("smiles", "expected_transferred_component"),
    [
        ("CN(C)C(=O)Oc1ccncc1", Counter({"C": 6, "N": 1, "O": 1})),
        ("CCN(CC)C(=O)Oc1ccncc1", Counter({"C": 7, "N": 1, "O": 1})),
        (
            "CCCCN(CCCC)C(=O)Oc1ccncc1",
            Counter({"C": 9, "N": 1, "O": 1}),
        ),
    ],
)
def test_pyridyl_carbamate_rule_retains_the_complete_selected_alkyl_group(
    smiles: str,
    expected_transferred_component: Counter,
) -> None:
    mol = Chem.MolFromSmiles(smiles)
    templates = _templates_for_rule(smiles, "pyridyl_carbamate_alkyl_transfer")

    assert any(
        expected_transferred_component
        in [_component_element_counts(mol, component) for component in template.components]
        for template in templates
    )


@pytest.mark.parametrize(
    ("rule_name", "smiles", "precursor_type", "fragmentation_mode"),
    [
        ("pyridyl_carbamate_alkyl_transfer", "CN(C)C(=O)Oc1ccncc1", "[M+H]+", None),
        ("n_phenylbenzamide_co_smiles", "O=C(Nc1ccccc1)c1ccccc1", "[M-H]-", None),
        ("methyl_benzoate_co2_transfer", "COC(=O)c1ccccc1", "[M+H]+", None),
        ("anisoyl_fluoride_co_f_transfer", "COc1ccc(C(=O)F)cc1", "[M]+", "EI"),
        ("short_haloalkyl_ammonium_c2_transfer", "CN(C)CCBr", "[M+H]+", None),
        ("short_haloalkyl_ammonium_c3_transfer", "CN(C)CCCCl", "[M]+", None),
        ("n_fluoro_sulfonamide_f_transfer", "O=S(=O)(c1ccccc1)NF", "[M-H]-", None),
    ],
)
def test_new_rules_keep_only_literature_supported_ionization_conditions(
    rule_name: str,
    smiles: str,
    precursor_type: str,
    fragmentation_mode: str | None,
) -> None:
    templates = _templates_for_rule(smiles, rule_name)
    supported = filter_mass_shatter_templates(
        templates,
        precursor_type=precursor_type,
        fragmentation_mode=fragmentation_mode,
    )
    wrong_precursor = filter_mass_shatter_templates(
        templates,
        precursor_type="[M-H]-" if precursor_type != "[M-H]-" else "[M+H]+",
        fragmentation_mode=fragmentation_mode,
    )

    assert supported
    assert not wrong_precursor


def test_ei_anisoyl_rule_is_rejected_when_fragmentation_mode_is_unknown_or_cid() -> None:
    templates = _templates_for_rule(
        "COc1ccc(C(=O)F)cc1",
        "anisoyl_fluoride_co_f_transfer",
    )

    for fragmentation_mode in (None, "CID", "HCD"):
        assert not filter_mass_shatter_templates(
            templates,
            precursor_type="[M]+",
            fragmentation_mode=fragmentation_mode,
        )


def test_existing_rules_remain_unrestricted_pending_separate_mode_audit() -> None:
    templates = _templates_for_rule(
        "c1ccccc1S(=O)c1ccccc1",
        "sulfoxide_so_loss",
    )

    assert filter_mass_shatter_templates(
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
    templates = build_mass_shatter_templates(mol)
    rule_template = next(template for template in templates if template.rule_name == rule_name)
    endpoints = th.tensor(
        [
            [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
            [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
        ],
        dtype=th.long,
    )

    candidates = build_mass_shatter_candidates(
        th.full((mol.GetNumBonds(),), -8.0),
        chemical_bond_index=endpoints,
        atom_batch_indices=th.zeros(mol.GetNumAtoms(), dtype=th.long),
        atom_ptr=th.tensor([0, mol.GetNumAtoms()], dtype=th.long),
        max_masks=1,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=False,
        smarts_templates=(templates,),
    )

    mechanism_index = candidates.mechanism_rule_indices.tolist().index(rule_template.rule_index)
    mask_index = candidates.mechanism_mask_indices[mechanism_index].item()
    assert candidates.mask_cut_counts[mask_index].item() == len(rule_template.broken_bond_indices)
    assert (candidates.component_mechanism_indices == mechanism_index).sum().item() == 2


def test_new_rule_indices_append_without_reassigning_existing_rules() -> None:
    expected_indices = {
        "pyridyl_carbamate_alkyl_transfer": 20,
        "n_phenylbenzamide_co_smiles": 21,
        "methyl_benzoate_co2_transfer": 22,
        "anisoyl_fluoride_co_f_transfer": 23,
        "short_haloalkyl_ammonium_c2_transfer": 24,
        "short_haloalkyl_ammonium_c3_transfer": 25,
        "n_fluoro_sulfonamide_f_transfer": 26,
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


def test_mass_shatter_templates_preserve_atom_partitions_and_cut_bonds() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1S(=O)c1ccccc1")

    templates = build_mass_shatter_templates(mol)

    sulfoxide = next(
        template for template in templates if template.rule_name == "sulfoxide_so_loss"
    )
    assert sulfoxide.broken_bond_indices == (5, 7)
    assert sorted(atom for component in sulfoxide.components for atom in component) == list(
        range(mol.GetNumAtoms())
    )
    assert set(sulfoxide.components[0]).isdisjoint(sulfoxide.components[1])


def test_mass_shatter_owns_a_curated_rule_registry() -> None:
    assert "crf12_0" not in MASS_SHATTER_FRAG_RULE_NAMES
    assert "crf12_1" not in MASS_SHATTER_FRAG_RULE_NAMES
    assert "crf13_2" not in MASS_SHATTER_FRAG_RULE_NAMES
    assert "morphinan_ring_cleavage" not in MASS_SHATTER_FRAG_RULE_NAMES
    assert "sulfoxide_so_loss" in MASS_SHATTER_NL_RULE_NAMES
    assert all(rule.merge_outer_map_nums for rule in MASS_SHATTER_FRAG_RULES)


def test_mass_shatter_candidates_attach_template_as_mechanism_of_same_mask() -> None:
    mol = Chem.MolFromSmiles("c1ccccc1S(=O)c1ccccc1")
    templates = build_mass_shatter_templates(mol)
    endpoints = th.tensor(
        [
            [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
            [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
        ],
        dtype=th.long,
    )

    candidates = build_mass_shatter_candidates(
        th.full((mol.GetNumBonds(),), -8.0),
        chemical_bond_index=endpoints,
        atom_batch_indices=th.zeros(mol.GetNumAtoms(), dtype=th.long),
        atom_ptr=th.tensor([0, mol.GetNumAtoms()], dtype=th.long),
        max_masks=1,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=_CUT_COUNT_WEIGHTS,
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
    candidates = build_mass_shatter_candidates(
        th.tensor([-1.0]),
        chemical_bond_index=th.tensor([[0], [1]], dtype=th.long),
        atom_batch_indices=th.zeros(2, dtype=th.long),
        atom_ptr=th.tensor([0, 2], dtype=th.long),
        max_masks=1,
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=_CUT_COUNT_WEIGHTS,
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
    templates = tuple(build_mass_shatter_templates(mol) for mol in mols)
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
    candidates = build_mass_shatter_candidates(
        th.full((len(begin),), -4.0),
        chemical_bond_index=th.tensor([begin, end], dtype=th.long),
        atom_batch_indices=th.tensor(batch, dtype=th.long),
        atom_ptr=th.tensor(ptr, dtype=th.long),
        explore_masks=0,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=_CUT_COUNT_WEIGHTS,
        include_single_bond_cuts=True,
        smarts_templates=templates,
        **kwargs,
    )
    return candidates, templates


def test_template_representatives_stay_in_range_for_later_batch_members() -> None:
    candidates, _ = _batched_candidates(
        ["O=C(Nc1ccccc1)c1ccccc1", "COC(=O)c1ccc(Cl)cc1", "CN(C)CCBr"],
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
        ["O=C(Nc1ccccc1)c1ccccc1", "COC(=O)c1ccc(Cl)cc1", "CN(C)CCBr"],
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
