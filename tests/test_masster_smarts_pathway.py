"""SMARTS rules as part of MASSter's fragmentation pathway.

Plan: docs/plans/masster/masster_smarts_pathway_plan.md. Phase 0 (fork_rng construction,
formed-bond valence, stable rule vocabulary, rule provenance) and arm ``p1`` (the
rule-pathway token).
"""

from __future__ import annotations

import zlib
from types import SimpleNamespace

import pytest
import torch as th
from rdkit import Chem

from fragnnet import runner
from fragnnet.frag.masster_candidates import MassterStaticGraph, build_masster_candidates
from fragnnet.frag.masster_templates import (
    _MASSTER_RULE_INDEX,
    MASSTER_NUM_RULES,
    MASSTER_RULE_NAME_HASHES,
    MASSTER_RULE_NAMES,
    MASSTER_RULE_VOCABULARY,
    MassterTemplate,
    build_masster_templates,
    formed_bond_count,
)
from fragnnet.model.masster_model import MassterModel
from fragnnet.pl_model.masster_pl import MassterPL
from tests.test_masster_model import _build_batch, _build_model

# N-phenyl benzenesulfonamide: aryl_sulfonamide_so2 (rule 1, both polarities) loses SO2 and
# joins the two aryl-side pieces, so its product side is two precursor pieces. As a secondary
# arylsulfonamide it also matches substituted_aryl_sulfonamide_o_transfer ([M+H]+).
_SULFONAMIDE = "O=S(=O)(Nc1ccccc1)c1ccccc1"
_SO2_ATOMS = (0, 1, 2)
_PRODUCT_ATOMS = tuple(range(3, 16))
_UNIQUE_KWARGS = {
    "unique_fragment_mode": True,
    "max_masks": 6,
    "max_enumerated_masks": 16,
    "use_smarts_templates": True,
}


def _endpoints(mol: Chem.Mol) -> tuple[tuple[int, ...], tuple[int, ...]]:
    return (
        tuple(bond.GetBeginAtomIdx() for bond in mol.GetBonds()),
        tuple(bond.GetEndAtomIdx() for bond in mol.GetBonds()),
    )


def _static_graph(batch, smiles: tuple[str, ...]) -> MassterStaticGraph:
    return MassterStaticGraph(
        chemical_bond_index=batch.chemical_bond_index,
        atom_batch_indices=batch.batch,
        atom_ptr=batch.ptr,
        smarts_templates=tuple(
            build_masster_templates(Chem.MolFromSmiles(molecule_smiles))
            for molecule_smiles in smiles
        ),
    )


def _forward(model: MassterModel, batch, smiles: tuple[str, ...], **kwargs):
    return model(
        mol_pyg=batch,
        spec_prec_type=th.zeros(len(smiles), dtype=th.long),
        spec_nce=th.full((len(smiles),), 30.0),
        static_graph=_static_graph(batch, smiles),
        **kwargs,
    )


# 1. Stable rule vocabulary (Phase 0.3).


def test_rule_indices_follow_the_append_only_vocabulary() -> None:
    assert set(MASSTER_RULE_NAMES) <= set(MASSTER_RULE_VOCABULARY)
    assert MASSTER_NUM_RULES == len(MASSTER_RULE_VOCABULARY) == 23
    assert MASSTER_RULE_VOCABULARY[0] == "aryl_sulfonamide_so2"
    assert MASSTER_RULE_VOCABULARY[-1] == "substituted_aryl_sulfonamide_o_transfer"
    assert {name: index - 1 for name, index in _MASSTER_RULE_INDEX.items()} == {
        name: position for position, name in enumerate(MASSTER_RULE_VOCABULARY)
    }
    expected_hashes = tuple(zlib.crc32(name.encode("utf-8")) for name in MASSTER_RULE_VOCABULARY)
    assert expected_hashes == MASSTER_RULE_NAME_HASHES


def test_smarts_model_records_its_rule_vocabulary() -> None:
    batch = _build_batch((_SULFONAMIDE,))
    with_smarts = _build_model(batch, use_smarts_templates=True)
    without_smarts = _build_model(batch)

    assert th.equal(
        with_smarts.smarts_rule_name_hashes,
        th.tensor(MASSTER_RULE_NAME_HASHES, dtype=th.long),
    )
    assert "smarts_rule_name_hashes" not in without_smarts.state_dict()


def _truncated_state_dict(model: MassterModel, num_rules: int) -> dict[str, th.Tensor]:
    """A state dict as a checkpoint trained on the first ``num_rules`` rules would save it."""
    state = {key: value.clone() for key, value in model.state_dict().items()}
    state["smarts_rule_name_hashes"] = state["smarts_rule_name_hashes"][:num_rules]
    embedding_key = "mechanism_head.rule_embedding.weight"
    state[embedding_key] = state[embedding_key][: 1 + num_rules]
    residual_key = "intensity_head.rule_pathway_residual.weight"
    if residual_key in state:
        state[residual_key] = state[residual_key][:, :num_rules]
    return state


def test_checkpoint_with_fewer_rules_loads_and_ignores_appended_rules() -> None:
    batch = _build_batch((_SULFONAMIDE,))
    source = _build_model(batch, rule_pathway_token=True, **_UNIQUE_KWARGS)
    with th.no_grad():
        source.mechanism_head.rule_embedding.weight.normal_()
        source.intensity_head.rule_pathway_residual.weight.normal_()
    state = _truncated_state_dict(source, 5)
    target = _build_model(batch, rule_pathway_token=True, **_UNIQUE_KWARGS)

    target.load_state_dict(state)

    assert target.smarts_rule_name_hashes.tolist() == list(MASSTER_RULE_NAME_HASHES[:5])
    embedding = target.mechanism_head.rule_embedding.weight
    assert embedding.shape == (1 + MASSTER_NUM_RULES, embedding.shape[1])
    assert th.equal(embedding[:6], source.mechanism_head.rule_embedding.weight[:6])
    assert th.count_nonzero(embedding[6:]) == 0
    residual = target.intensity_head.rule_pathway_residual.weight
    assert th.equal(residual[:, :5], source.intensity_head.rule_pathway_residual.weight[:, :5])
    assert th.count_nonzero(residual[:, 5:]) == 0

    def template(rule_index: int) -> MassterTemplate:
        return MassterTemplate((0,), ((0,), (1,)), rule_index, "rule")

    eligible = target._eligible_smarts_templates(
        ((template(5), template(6), template(22)),),
        precursor_type_indices=th.tensor([0]),
        fragmentation_mode_indices=None,
    )
    assert [kept.rule_index for kept in eligible[0]] == [5]

    # The shorter list survives a second save, so a fine-tuned copy still ignores rule 6+.
    reloaded = _build_model(batch, rule_pathway_token=True, **_UNIQUE_KWARGS)
    reloaded.load_state_dict(target.state_dict())
    assert reloaded.smarts_rule_name_hashes.numel() == 5


def test_checkpoint_with_reordered_rules_is_rejected() -> None:
    batch = _build_batch((_SULFONAMIDE,))
    model = _build_model(batch, use_smarts_templates=True)
    state = model.state_dict()
    state["smarts_rule_name_hashes"] = state["smarts_rule_name_hashes"].flip(0)

    with pytest.raises(ValueError, match="append-only"):
        _build_model(batch, use_smarts_templates=True).load_state_dict(state)


def test_smarts_checkpoint_without_rule_list_is_rejected() -> None:
    batch = _build_batch((_SULFONAMIDE,))
    state = _build_model(batch, use_smarts_templates=True).state_dict()
    del state["smarts_rule_name_hashes"]

    with pytest.raises(ValueError, match="predates the stable rule vocabulary"):
        _build_model(batch, use_smarts_templates=True).load_state_dict(state)


# 2. The mechanism head no longer shifts other initializations (Phase 0.1).


def test_smarts_mechanism_head_leaves_other_parameters_unchanged() -> None:
    batch = _build_batch((_SULFONAMIDE,))
    with_smarts = _build_model(batch, use_smarts_templates=True).state_dict()
    without_smarts = _build_model(batch).state_dict()

    assert set(with_smarts) - set(without_smarts) == {
        "smarts_rule_name_hashes",
        "mechanism_head.rule_embedding.weight",
        "mechanism_head.scorer.0.weight",
        "mechanism_head.scorer.0.bias",
        "mechanism_head.scorer.3.weight",
        "mechanism_head.scorer.3.bias",
    }
    for key, value in without_smarts.items():
        assert th.equal(value, with_smarts[key]), key


# 3. Formed bonds take back valence slots (Phase 0.2).


def test_formed_bond_count_is_pieces_minus_one() -> None:
    chain = ((0, 1, 2, 3), (1, 2, 3, 4))
    assert formed_bond_count((0, 1, 2), chain) == 0
    assert formed_bond_count((0, 4), chain) == 1
    assert formed_bond_count((0, 2, 4), chain) == 2

    mol = Chem.MolFromSmiles(_SULFONAMIDE)
    (template,) = [
        t for t in build_masster_templates(mol) if t.rule_name == "aryl_sulfonamide_so2"
    ]
    assert template.components == (_SO2_ATOMS, _PRODUCT_ATOMS)
    assert [formed_bond_count(side, _endpoints(mol)) for side in template.components] == [0, 1]


@pytest.mark.parametrize("backend", ["cpu", "cuda"])
def test_candidates_carry_formed_bonds_on_assembled_sides_only(backend: str) -> None:
    if backend == "cuda" and not th.cuda.is_available():
        pytest.skip("CUDA required")
    device = th.device(backend)
    mol = Chem.MolFromSmiles(_SULFONAMIDE)
    templates = tuple(
        t for t in build_masster_templates(mol) if t.rule_name == "aryl_sulfonamide_so2"
    )

    candidates = build_masster_candidates(
        th.full((mol.GetNumBonds(),), -8.0, device=device),
        chemical_bond_index=th.tensor(_endpoints(mol), dtype=th.long, device=device),
        atom_batch_indices=th.zeros(mol.GetNumAtoms(), dtype=th.long, device=device),
        atom_ptr=th.tensor([0, mol.GetNumAtoms()], dtype=th.long, device=device),
        max_masks=1,
        max_unique_components=1,
        max_enumerated_masks=4,
        include_single_bond_cuts=False,
        smarts_templates=(templates,),
        candidate_backend=backend,
    )

    component_rules = candidates.mechanism_rule_indices[candidates.component_mechanism_indices]
    component_atoms = [
        tuple(candidates.atom_indices[candidates.atom_component_indices == row].tolist())
        for row in range(candidates.num_components)
    ]
    template_sides = [
        atoms for atoms, rule in zip(component_atoms, component_rules.tolist()) if rule == 1
    ]
    assert template_sides == [_SO2_ATOMS, _PRODUCT_ATOMS]
    # Only the SO2-free product, the one disconnected side, forms a bond.
    assert candidates.component_formed_bond_counts.tolist() == [
        int(atoms == _PRODUCT_ATOMS) for atoms in component_atoms
    ]


def test_valence_bound_takes_back_two_slots_per_formed_bond() -> None:
    # Pentane's two terminal methyls as one assembled side: two cut bonds, one formed bond,
    # no unsaturation. The cut bonds alone would allow +2 H; the formed bond uses both slots.
    batch = _build_batch(("CCCCC",))
    model = _build_model(
        batch,
        precursor_types=("[M+H]+",),
        include_cmf=False,
        max_h_transfer=4,
        h_transfer_bound="valence",
    )
    atoms = th.tensor([0, 4])
    arguments = (batch, atoms, th.zeros_like(atoms), th.tensor([0]), 1, th.tensor([0]))
    shifts = list(range(-model.max_h_transfer, model.max_h_transfer + 1))

    def max_gain(formed: th.Tensor | None) -> int:
        _, valid = model._fragment_masses_and_valid_channels(*arguments, th.tensor([2]), formed)
        return max(shift for shift in shifts if valid[0, 0, shifts.index(shift)].item())

    assert max_gain(None) == 2
    assert max_gain(th.tensor([0])) == 2
    assert max_gain(th.tensor([1])) == 0


# 4. Rule provenance (Phase 0.4).


def test_fragment_rule_indicators_mark_each_rule_once_per_fragment() -> None:
    candidates = SimpleNamespace(
        mechanism_rule_indices=th.tensor([0, 0, 3, 1]),
        component_mechanism_indices=th.tensor([0, 1, 2, 2, 3, 2, 3]),
    )
    fragments = th.tensor([0, 1, 2, 1, 0, 2, 0])

    indicators = MassterModel._fragment_rule_indicators(candidates, fragments, 3, th.float32)

    expected = th.zeros(3, MASSTER_NUM_RULES)
    expected[0, 0] = 1.0  # rule 1, from two rows
    expected[1, 2] = 1.0  # rule 3
    expected[2, 2] = 1.0  # rule 3, from two rows
    assert th.equal(indicators, expected)


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_annotation_reports_the_rule_behind_each_fragment(unique_fragment_mode: bool) -> None:
    smiles = (_SULFONAMIDE,)
    batch = _build_batch(smiles)
    model = _build_model(
        batch,
        unique_fragment_mode=unique_fragment_mode,
        max_masks=6 if unique_fragment_mode else 4,
        max_enumerated_masks=16,
        use_smarts_templates=True,
    ).eval()

    output = _forward(model, batch, smiles, return_annotation=True)

    rule_fragments = output["fragment_rule_fragment_idxs"].tolist()
    rule_indices = output["fragment_rule_idxs"].tolist()
    o_transfer = MASSTER_RULE_VOCABULARY.index("substituted_aryl_sulfonamide_o_transfer") + 1
    assert set(rule_indices) <= {1, o_transfer} and 1 in rule_indices
    memberships = output["fragment_atom_fragment_idxs"]
    rule_atom_sets = {
        tuple(sorted(output["fragment_atom_indices"][memberships == fragment].tolist()))
        for fragment, rule in zip(rule_fragments, rule_indices)
        if rule == 1
    }
    # Both template sides carry the rule, the lost SO2 as well as the assembled product.
    assert rule_atom_sets == {_SO2_ATOMS, _PRODUCT_ATOMS}


def test_annotation_provenance_is_empty_without_smarts() -> None:
    smiles = (_SULFONAMIDE,)
    batch = _build_batch(smiles)
    model = _build_model(batch, unique_fragment_mode=True, max_masks=4).eval()

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([30.0]),
        return_annotation=True,
    )

    assert output["fragment_rule_fragment_idxs"].numel() == 0
    assert output["fragment_rule_idxs"].numel() == 0


# 5. Arm p1: rule provenance in the intensity token.


def test_rule_pathway_token_requires_smarts_and_unique_fragments() -> None:
    batch = _build_batch((_SULFONAMIDE,))
    with pytest.raises(ValueError, match="rule_pathway_token requires"):
        _build_model(batch, unique_fragment_mode=True, rule_pathway_token=True)
    with pytest.raises(ValueError, match="rule_pathway_token requires"):
        _build_model(batch, use_smarts_templates=True, rule_pathway_token=True)


def test_rule_pathway_token_starts_as_the_templates_model_and_trains() -> None:
    smiles = (_SULFONAMIDE,)
    batch = _build_batch(smiles)
    control = _build_model(batch, **_UNIQUE_KWARGS).eval()
    arm = _build_model(batch, rule_pathway_token=True, **_UNIQUE_KWARGS).eval()

    control_state, arm_state = control.state_dict(), arm.state_dict()
    assert set(arm_state) - set(control_state) == {"intensity_head.rule_pathway_residual.weight"}
    for key, value in control_state.items():
        assert th.equal(value, arm_state[key]), key
    residual = arm.intensity_head.rule_pathway_residual
    assert residual.weight.shape == (16, MASSTER_NUM_RULES)
    assert residual.bias is None
    assert th.count_nonzero(residual.weight) == 0

    expected = _forward(control, batch, smiles)
    actual = _forward(arm, batch, smiles)
    for name in ("pred_mzs", "pred_batch_idxs", "pred_logprobs"):
        assert th.equal(expected[name], actual[name]), name

    arm.train()
    (-_forward(arm, batch, smiles)["pred_logprobs"].mean()).backward()
    gradient = residual.weight.grad
    # Only the columns of the matched rules see a candidate: rule 1 (SO2 extrusion) and the
    # substituted O-transfer, which this secondary sulfonamide also matches.
    matched = [0, MASSTER_RULE_VOCABULARY.index("substituted_aryl_sulfonamide_o_transfer")]
    assert th.count_nonzero(gradient[:, 0]) > 0
    unmatched = [c for c in range(gradient.shape[1]) if c not in matched]
    assert th.count_nonzero(gradient[:, unmatched]) == 0


def test_rule_pathway_token_pl_threads_hparam() -> None:
    config = runner.load_config("config", None, None, model_type="masster")
    config.update(
        model_type="masster",
        compile=False,
        sparse_cosine_similarity=True,
        oos_tolerance_multiple=1,
        gaussian_renormalize=True,
        mol_hidden_size=16,
        mol_num_layers=1,
        masster_context_dim=8,
        masster_metadata_embedding_dim=4,
        masster_head_hidden_dim=16,
        masster_intensity_transformer_dim=16,
        masster_intensity_transformer_num_layers=2,
        masster_intensity_transformer_feedforward_dim=32,
        masster_intensity_transformer_num_inducing_points=4,
        masster_unique_fragment_mode=True,
        masster_selected_mask_limit=2,
        masster_unique_fragment_target=2,
        masster_ranked_mask_budget=2,
        masster_max_h_transfer=1,
        masster_smarts_templates=True,
        masster_rule_pathway_token=True,
    )

    model = MassterPL(**config).model

    assert model.rule_pathway_token is True
    assert model.intensity_head.rule_pathway_dim == MASSTER_NUM_RULES
