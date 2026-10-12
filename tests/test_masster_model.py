"""End-to-end tests for the first trainable MASSter model slice."""

import itertools
import math
from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest
import torch as th
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcExactMolWt
from torch_geometric.data import Batch

import fragnnet.model.masster_model as masster_model_module
from fragnnet.frag import compute_frags
from fragnnet.frag.masster_candidates import MassterStaticGraph
from fragnnet.frag.masster_templates import MassterTemplate, build_masster_templates
from fragnnet.fragmentation.hydrogen_cap import compute_cc_h_cap
from fragnnet.model.bond_fragment_heads import BondFocusedBreakHead
from fragnnet.model.masster_model import (
    MassterModel,
    _absolute_binary_channel_keep_mask,
    _cmf_duplicate_crf_shift,
    _fixed_top_k_channel_keep_mask,
    _fragment_release_log_probabilities,
    _ion_mode_logits_from_ion_h_support,
    _missing_valid_fallback_mask,
    _relative_channel_keep_mask,
    _relative_fragment_keep_mask,
    _relative_joint_channel_keep_mask,
    _top_k_fragment_keep_mask,
)
from fragnnet.pl_model.masster_pl import _proposal_coverage_loss
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.formula_utils import H_MASS, formula_to_peak_mzs
from fragnnet.utils.frag_utils import extract_mol_info


def _build_batch(smiles: tuple[str, ...] = ("CCO", "CO")) -> Batch:
    return Batch.from_data_list(
        [
            get_mol_graph(
                Chem.MolFromSmiles(molecule_smiles),
                ["a_onehot", "a_degree"],
                ["b_degree"],
                0,
                True,
            )
            for molecule_smiles in smiles
        ]
    )


def _build_model(
    batch: Batch,
    *,
    bond_num_layers: int = 0,
    proposal_num_mixtures: int = 1,
    max_masks: int = 2,
    unique_fragment_mode: bool = False,
    max_enumerated_masks: int | None = None,
    min_mask_relative_probability: float = 0.0,
    min_fragment_relative_probability: float = 0.0,
    fragment_selection: str = "mask_order",
    selected_fragment_target: int | None = None,
    use_mask_residuals: bool = True,
    use_base_probability_feature: bool = True,
    return_base_spectrum: bool = False,
    detach_base_ion_h: bool = False,
    proposal_selection_only: bool = False,
    separate_proposal_encoder: bool = False,
    share_proposal_across_ev: bool = False,
    separate_ion_mode_proposal_head: bool = False,
    ion_mode_proposal_binary_logits: bool = False,
    min_ion_mode_relative_probability: float = 0.0,
    separate_ion_h_support_head: bool = False,
    ion_h_support_binary_logits: bool = False,
    gate_independent_intensity_with_support: bool = False,
    min_ion_h_relative_probability: float = 0.0,
    min_ion_h_absolute_probability: float = 0.0,
    apply_channel_selection_in_training: bool = False,
    intensity_transformer_attention: str = "induced",
    intensity_mz_positional_encoding: bool = False,
    intensity_nl_positional_encoding: bool = False,
    intensity_mass_window: float = 0.0,
    intensity_output: str = "probability",
    precursor_gnn_type: str = "GINE",
    ce_insert_type: str = "id",
    ce_insert_size: int = 6,
    use_smarts_templates: bool = False,
    include_cmf: bool = True,
    max_h_transfer: int = 1,
    h_transfer_bound: str = "none",
    precursor_types: tuple[str, ...] = ("[M+H]+", "[M-H]-"),
    fragmentation_modes: tuple[str, ...] = (),
    context_in_gnn: bool = False,
    encoder_context_token: bool = False,
    candidate_backend: str = "cpu",
    proposal_method: str = "cut_masks",
    boundary_bond_encoder: bool = False,
    candidate_features: Sequence[str] = ("c_cut_count", "c_size"),
    candidate_atom_dropout: float = 0.0,
    candidate_boundary_dropout: float = 0.0,
    candidate_boundary_block: bool = True,
    boundary_local_layers: int = 0,
    boundary_encoder_dropout: float = 0.0,
    fragment_connectivity_layers: int = 0,
    fragment_connectivity_hidden_dim: int = 32,
    fragment_connectivity_messages: bool = True,
    intensity_input_dropout: float = 0.0,
    intensity_set_message_substitution: float = 0.0,
    intensity_set_key_dropout: float = 0.0,
    memorization_sink_dim: int = 0,
    memorization_sink_num_molecules: int = 0,
    memorization_sink_dropout: float = 0.0,
    global_message_substitution: float = 0.0,
    global_message_noise_std: float = 0.0,
    global_message_swap: float = 0.0,
    global_message_swap_unit: str = "atom",
    encoder_atom_mask: float = 0.0,
    encoder_bond_dropout: float = 0.0,
    light_side_token: bool = False,
    intensity_candidate_graph: bool = False,
    rule_pathway_token: bool = False,
    **model_kwargs: object,
) -> MassterModel:
    th.manual_seed(903)
    return MassterModel(
        node_feature_dim=batch.x.shape[1],
        edge_feature_dim=batch.edge_attr.shape[1],
        precursor_hidden_dim=16,
        precursor_num_layers=2,
        precursor_num_heads=4,
        precursor_gnn_type=precursor_gnn_type,
        precursor_normalization="layer",
        precursor_dropout=0.0,
        context_dim=8,
        context_in_gnn=context_in_gnn,
        encoder_context_token=encoder_context_token,
        metadata_embedding_dim=4,
        head_hidden_dim=12,
        head_dropout=0.0,
        bond_num_layers=bond_num_layers,
        proposal_num_mixtures=proposal_num_mixtures,
        intensity_transformer_dim=16,
        intensity_transformer_num_layers=2,
        intensity_transformer_num_heads=4,
        intensity_transformer_feedforward_dim=32,
        intensity_transformer_num_inducing_points=4,
        intensity_transformer_attention=intensity_transformer_attention,
        intensity_mz_positional_encoding=intensity_mz_positional_encoding,
        intensity_nl_positional_encoding=intensity_nl_positional_encoding,
        intensity_mass_window=intensity_mass_window,
        intensity_transformer_dropout=0.0,
        intensity_output=intensity_output,
        use_mask_residuals=use_mask_residuals,
        use_base_probability_feature=use_base_probability_feature,
        return_base_spectrum=return_base_spectrum,
        detach_base_ion_h=detach_base_ion_h,
        proposal_selection_only=proposal_selection_only,
        separate_proposal_encoder=separate_proposal_encoder,
        share_proposal_across_ev=share_proposal_across_ev,
        separate_ion_mode_proposal_head=separate_ion_mode_proposal_head,
        ion_mode_proposal_binary_logits=ion_mode_proposal_binary_logits,
        min_ion_mode_relative_probability=min_ion_mode_relative_probability,
        separate_ion_h_support_head=separate_ion_h_support_head,
        ion_h_support_binary_logits=ion_h_support_binary_logits,
        gate_independent_intensity_with_support=gate_independent_intensity_with_support,
        min_ion_h_relative_probability=min_ion_h_relative_probability,
        min_ion_h_absolute_probability=min_ion_h_absolute_probability,
        apply_channel_selection_in_training=apply_channel_selection_in_training,
        precursor_types=precursor_types,
        instrument_types=[],
        fragmentation_modes=fragmentation_modes,
        int_embedder="abs-sines",
        ce_insert_type=ce_insert_type,
        ce_insert_size=ce_insert_size,
        use_nce=True,
        use_ace=False,
        nce_max=200.0,
        nce_mean=30.0,
        nce_std=10.0,
        ace_max=100.0,
        ace_mean=20.0,
        ace_std=15.0,
        max_masks=max_masks,
        unique_fragment_mode=unique_fragment_mode,
        max_enumerated_masks=(max_masks if max_enumerated_masks is None else max_enumerated_masks),
        min_mask_relative_probability=min_mask_relative_probability,
        min_fragment_relative_probability=min_fragment_relative_probability,
        fragment_selection=fragment_selection,
        selected_fragment_target=selected_fragment_target,
        include_single_bond_cuts=True,
        max_h_transfer=max_h_transfer,
        h_transfer_bound=h_transfer_bound,
        include_cmf=include_cmf,
        charge_pooling="logsumexp",
        use_smarts_templates=use_smarts_templates,
        candidate_backend=candidate_backend,
        proposal_method=proposal_method,
        boundary_bond_encoder=boundary_bond_encoder,
        candidate_features=candidate_features,
        candidate_atom_dropout=candidate_atom_dropout,
        candidate_boundary_dropout=candidate_boundary_dropout,
        candidate_boundary_block=candidate_boundary_block,
        boundary_local_layers=boundary_local_layers,
        boundary_encoder_dropout=boundary_encoder_dropout,
        fragment_connectivity_layers=fragment_connectivity_layers,
        fragment_connectivity_hidden_dim=fragment_connectivity_hidden_dim,
        fragment_connectivity_messages=fragment_connectivity_messages,
        intensity_input_dropout=intensity_input_dropout,
        intensity_set_message_substitution=intensity_set_message_substitution,
        intensity_set_key_dropout=intensity_set_key_dropout,
        memorization_sink_dim=memorization_sink_dim,
        memorization_sink_num_molecules=memorization_sink_num_molecules,
        memorization_sink_dropout=memorization_sink_dropout,
        global_message_substitution=global_message_substitution,
        global_message_noise_std=global_message_noise_std,
        global_message_swap=global_message_swap,
        global_message_swap_unit=global_message_swap_unit,
        encoder_atom_mask=encoder_atom_mask,
        encoder_bond_dropout=encoder_bond_dropout,
        light_side_token=light_side_token,
        intensity_candidate_graph=intensity_candidate_graph,
        rule_pathway_token=rule_pathway_token,
        **model_kwargs,
    )


def test_masster_selects_bond_focused_head() -> None:
    batch = _build_batch()
    model = _build_model(batch, bond_num_layers=2)

    assert isinstance(model.bond_break_head, BondFocusedBreakHead)


@pytest.mark.parametrize(
    ("smiles", "atom_indices", "formula", "adduct"),
    [
        ("C[N+](C)(C)C", [0, 1, 2, 3, 4], "C4H12N", "[M]+"),
        ("C[O-]", [0, 1], "CH3O", "[M-H]-"),
        ("C[N+](=O)[O-]", [3], "O", "[M+H]+"),
        ("CCO", [0, 1, 2], "C2H6O", "[M+H]+"),
    ],
)
def test_fragment_masses_match_elemental_formula(
    smiles: str,
    atom_indices: list[int],
    formula: str,
    adduct: str,
) -> None:
    batch = _build_batch((smiles,))
    model = _build_model(
        batch,
        precursor_types=(adduct,),
        include_cmf=False,
        max_h_transfer=0,
    )
    masses, valid = model._fragment_masses_and_valid_channels(
        batch,
        th.tensor(atom_indices),
        th.zeros(len(atom_indices), dtype=th.long),
        th.tensor([0]),
        1,
        th.tensor([0]),
    )
    expected = formula_to_peak_mzs(formula, adduct, isotopes=False)[0]
    assert valid.item()
    assert masses.item() == pytest.approx(expected, abs=1e-9, rel=0)


@pytest.mark.parametrize("separate_encoder", [False, True])
@pytest.mark.parametrize("shared_proposal", [False, True])
def test_smarts_proposal_gradients_respect_encoder_isolation(
    separate_encoder: bool,
    shared_proposal: bool,
) -> None:
    smiles = "c1ccccc1S(=O)c1ccccc1"
    batch = _build_batch((smiles, smiles))
    model = _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=8,
        use_smarts_templates=True,
        proposal_selection_only=True,
        separate_proposal_encoder=separate_encoder,
        use_base_probability_feature=False,
        share_proposal_across_ev=shared_proposal,
    )
    templates = build_masster_templates(Chem.MolFromSmiles(smiles))
    static_graph = MassterStaticGraph(
        chemical_bond_index=batch.chemical_bond_index,
        atom_batch_indices=batch.batch,
        atom_ptr=batch.ptr,
        smarts_templates=(templates, templates),
    )
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([10.0, 40.0]),
        "static_graph": static_graph,
        "proposal_group_ids": th.tensor([7, 7]) if shared_proposal else None,
    }
    output = model(**inputs)
    if shared_proposal:
        scores = output["proposal_fragment_logprobs"]
        indices = output["proposal_fragment_batch_idxs"]
        assert th.allclose(scores[indices == 0], scores[indices == 1])
    (-output["proposal_fragment_logprobs"].mean()).backward()
    assert _parameter_gradient_sum(model.precursor_encoder) == 0.0
    assert _parameter_gradient_sum(model.context_encoder) == 0.0
    assert _parameter_gradient_sum(model.intensity_head) == 0.0
    assert _parameter_gradient_sum(model.mechanism_head) > 0.0
    assert _parameter_gradient_sum(model.bond_break_head) > 0.0
    if separate_encoder:
        assert _parameter_gradient_sum(model.proposal_encoder) > 0.0

    model.zero_grad(set_to_none=True)
    output = model(**inputs)
    (-output["pred_logprobs"].mean()).backward()
    assert _parameter_gradient_sum(model.mechanism_head) == 0.0
    assert _parameter_gradient_sum(model.bond_break_head) == 0.0
    assert _parameter_gradient_sum(model.precursor_encoder) > 0.0
    if separate_encoder:
        assert _parameter_gradient_sum(model.proposal_encoder) == 0.0


@pytest.mark.parametrize("with_topology", [False, True])
def test_smarts_model_rejects_missing_templates(with_topology: bool) -> None:
    batch = _build_batch(("CCO",))
    model = _build_model(batch, use_smarts_templates=True)
    static_graph = (
        MassterStaticGraph(
            chemical_bond_index=batch.chemical_bond_index,
            atom_batch_indices=batch.batch,
            atom_ptr=batch.ptr,
        )
        if with_topology else None
    )
    with pytest.raises(ValueError, match="require static_graph.smarts_templates"):
        model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0]),
            spec_nce=th.tensor([25.0]),
            static_graph=static_graph,
        )


def test_relative_fragment_keep_mask_is_batched_and_map_relative() -> None:
    keep = _relative_fragment_keep_mask(
        th.log(th.tensor([0.6, 0.3, 0.1, 0.8, 0.2])),
        th.tensor([0, 0, 0, 1, 1]),
        batch_size=2,
        min_relative_probability=0.25,
    )

    assert keep.tolist() == [True, True, False, True, True]


def test_relative_channel_keep_mask_is_per_fragment() -> None:
    logits = th.tensor([[[0.0, -1.0], [2.0, 1.0]], [[4.0, 3.0], [0.0, 0.0]]])
    valid = th.tensor([[[True, True], [True, True]], [[True, True], [False, False]]])

    keep = _relative_channel_keep_mask(logits, valid, 0.25)

    assert keep.tolist() == [
        [[False, False], [True, True]],
        [[True, True], [False, False]],
    ]
    assert th.equal(_relative_channel_keep_mask(logits, valid, 0.0), valid)


def test_relative_channel_keep_mask_uses_probabilities_for_binary_logits() -> None:
    logits = th.tensor([[[10.0, 5.0]]])
    valid = th.ones_like(logits, dtype=th.bool)

    probability_keep = _relative_channel_keep_mask(
        logits,
        valid,
        0.01,
        binary_logits=True,
    )
    odds_keep = _relative_channel_keep_mask(logits, valid, 0.01)

    assert probability_keep.tolist() == [[[True, True]]]
    assert odds_keep.tolist() == [[[True, False]]]


def test_absolute_binary_channel_keep_mask_can_remove_a_whole_fragment() -> None:
    probabilities = th.tensor(
        [
            [[0.001, 0.002]],
            [[0.2, 0.05]],
            [[0.001, 0.003]],
        ]
    )
    valid = th.ones_like(probabilities, dtype=th.bool)

    keep = _absolute_binary_channel_keep_mask(
        th.logit(probabilities),
        valid,
        fragment_batch_indices=th.tensor([0, 0, 1]),
        batch_size=2,
        min_probability=0.1,
    )

    assert keep.tolist() == [
        [[False, False]],
        [[True, False]],
        [[False, True]],
    ]


def test_ion_mode_fallback_aggregates_binary_channel_probabilities() -> None:
    channel_probabilities = th.tensor([[[0.5, 0.5], [0.2, 0.9]]])
    valid_channels = th.tensor([[[True, True], [True, False]]])

    mode_logits = _ion_mode_logits_from_ion_h_support(
        th.logit(channel_probabilities),
        valid_channels,
        binary_logits=True,
    )

    th.testing.assert_close(mode_logits.sigmoid(), th.tensor([[0.75, 0.2]]))
    assert _relative_channel_keep_mask(
        mode_logits,
        valid_channels.any(dim=2),
        0.25,
        binary_logits=True,
    ).tolist() == [[True, True]]


def test_ion_mode_fallback_preserves_categorical_marginalization() -> None:
    channel_probabilities = th.tensor([[[0.4, 0.1], [0.3, 0.2]]])
    valid_channels = th.ones_like(channel_probabilities, dtype=th.bool)

    mode_log_probabilities = _ion_mode_logits_from_ion_h_support(
        channel_probabilities.log(),
        valid_channels,
        binary_logits=False,
    )

    th.testing.assert_close(mode_log_probabilities.exp(), th.tensor([[0.5, 0.5]]))


def test_channel_selection_uses_explicit_semantics_not_head_presence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_binary_flags: list[bool] = []

    def record_semantics(
        logits: th.Tensor,
        valid_mask: th.Tensor,
        min_relative_probability: float,
        *,
        binary_logits: bool = False,
    ) -> th.Tensor:
        observed_binary_flags.append(binary_logits)
        return _relative_channel_keep_mask(
            logits,
            valid_mask,
            min_relative_probability,
            binary_logits=binary_logits,
        )

    monkeypatch.setattr(
        masster_model_module,
        "_relative_channel_keep_mask",
        record_semantics,
    )
    batch = _build_batch()
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        separate_ion_h_support_head=True,
        ion_h_support_binary_logits=False,
    ).eval()

    with th.no_grad():
        model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0, 1]),
            spec_nce=th.tensor([25.0, 35.0]),
        )

    assert observed_binary_flags == [False, False]


def test_fixed_top_k_channel_keep_mask_is_stable_and_per_fragment() -> None:
    logits = th.tensor([[[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]], [[3.0, 2.0, 1.0], [0.0, 0.0, 0.0]]])
    valid = th.tensor(
        [
            [[True, False, True], [True, False, False]],
            [[True, True, True], [False, False, False]],
        ]
    )

    keep = _fixed_top_k_channel_keep_mask(logits, valid, top_k=2)

    assert keep.tolist() == [
        [[True, False, True], [False, False, False]],
        [[True, True, False], [False, False, False]],
    ]
    assert th.equal(_fixed_top_k_channel_keep_mask(logits, valid, top_k=10), valid)


def test_relative_joint_channel_keep_mask_is_per_spectrum_with_fragment_fallback() -> None:
    weights = th.tensor(
        [
            [[1.0, 0.5, 0.01]],
            [[0.02, 0.01, 0.001]],
            [[0.2, 0.01, 0.001]],
        ]
    )
    valid = th.ones_like(weights, dtype=th.bool)

    keep = _relative_joint_channel_keep_mask(
        weights.log(),
        valid,
        fragment_batch_indices=th.tensor([0, 0, 1]),
        batch_size=2,
        min_relative_weight=0.1,
    )

    assert keep.tolist() == [
        [[True, True, False]],
        [[True, False, False]],
        [[True, False, False]],
    ]
    assert th.equal(
        _relative_joint_channel_keep_mask(
            weights.log(),
            valid,
            fragment_batch_indices=th.tensor([0, 0, 1]),
            batch_size=2,
            min_relative_weight=0.0,
        ),
        valid,
    )


@pytest.mark.parametrize("top_k", [0, -1, True, 1.5])
def test_fixed_top_k_channel_keep_mask_rejects_invalid_k(top_k: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        _fixed_top_k_channel_keep_mask(
            th.ones((1, 1, 2)),
            th.ones((1, 1, 2), dtype=th.bool),
            top_k,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    ("precursor_type", "duplicate_shift"),
    [("[M+H]+", -1), ("[M-H]-", 1), ("[M+Na]+", None), ("[M+Cl]-", None)],
)
def test_cmf_duplicate_crf_shift_is_adduct_aware(
    precursor_type: str,
    duplicate_shift: int | None,
) -> None:
    assert _cmf_duplicate_crf_shift(precursor_type, 4) == duplicate_shift


def test_cmf_validity_excludes_only_reachable_duplicates() -> None:
    batch = _build_batch()
    model = _build_model(
        batch,
        precursor_types=("[M+H]+", "[M-H]-", "[M+Na]+", "[M+Cl]-"),
    )

    assert dict(zip(model.precursor_types, model.cmf_precursor_valid.tolist())) == {
        "[M+Cl]-": True,
        "[M+H]+": False,
        "[M+Na]+": True,
        "[M-H]-": False,
    }


@pytest.mark.parametrize("precursor_type", ("[M+Na]+", "[M+Cl]-"))
def test_non_degenerate_cmf_uses_the_full_hydrogen_transfer_ladder(
    precursor_type: str,
) -> None:
    batch = _build_batch(("CCO",))
    model = _build_model(
        batch,
        precursor_types=(precursor_type,),
        max_h_transfer=1,
    )
    # The C-C component of ethanol, whose complementary neutral (OH) still holds one
    # hydrogen, so the +1 rung is reachable. The intact molecule would not qualify: its
    # complement is empty, so it can only lose hydrogens.
    atom_indices = th.tensor([0, 1])
    ion_mzs, valid = model._fragment_masses_and_valid_channels(
        batch,
        atom_indices,
        th.zeros_like(atom_indices),
        th.tensor([0]),
        1,
        th.tensor([0]),
    )

    cmf_mzs = ion_mzs[0, 1]
    assert valid[0, 1].tolist() == [True, True, True]
    assert th.allclose(
        cmf_mzs[1:] - cmf_mzs[:-1],
        th.full((2,), H_MASS, dtype=cmf_mzs.dtype),
    )


def test_signed_adduct_deltas_track_removed_elements() -> None:
    batch = _build_batch(("CC",))
    model = _build_model(batch, precursor_types=("[M-H-CO2]-",))

    assert model.required_atomic_numbers.tolist() == [1, 6, 8]
    assert model.crf_formula_deltas.tolist() == [[-1, -1, -2]]


@pytest.mark.parametrize(
    ("smiles", "atom_index", "adduct", "mode", "expected"),
    [
        ("CC(=O)C", 1, "[M-H]-", 0, False),
        ("CCO", 0, "[M-H2O+H]+", 0, False),
        ("CCO", 2, "[M-H2O+H]+", 0, False),  # Empty formula has negative m/z.
        ("CO", 1, "[M-H]-", 0, True),
        ("CC(=O)C", 1, "[M+Cl]-", 1, False),
    ],
)
def test_ion_channels_require_available_atoms_and_positive_mass(
    smiles: str,
    atom_index: int,
    adduct: str,
    mode: int,
    expected: bool,
) -> None:
    batch = _build_batch((smiles,))
    model = _build_model(batch, precursor_types=(adduct,))
    _, valid = model._fragment_masses_and_valid_channels(
        batch,
        th.tensor([atom_index]),
        th.tensor([0]),
        th.tensor([0]),
        1,
        th.tensor([0]),
    )

    assert valid[0, mode, model.max_h_transfer].item() is expected


def test_loading_legacy_chemistry_buffers_rebuilds_signed_deltas() -> None:
    batch = _build_batch(("CCO",))
    model = _build_model(batch, precursor_types=("[M-H-CO2]-",))
    legacy_state = model.state_dict()
    legacy_state["required_atomic_numbers"] = th.tensor([1])
    legacy_state["crf_formula_deltas"] = th.tensor([[1]])
    legacy_state["cmf_formula_deltas"] = th.tensor([[0]])
    restored = _build_model(batch, precursor_types=("[M-H-CO2]-",))

    restored.load_state_dict(legacy_state, strict=True)

    assert restored.required_atomic_numbers.tolist() == [1, 6, 8]
    assert restored.crf_formula_deltas.tolist() == [[-1, -1, -2]]
    assert restored.cmf_formula_deltas.tolist() == [[0, 0, 0]]
    for name, parameter in restored.named_parameters():
        th.testing.assert_close(parameter, legacy_state[name])


@pytest.mark.parametrize(
    ("unique_fragment_mode", "intensity_output"),
    [(False, "probability"), (True, "probability"), (True, "independent")],
)
def test_fragments_without_valid_channels_are_removed_and_mass_renormalized(
    unique_fragment_mode: bool,
    intensity_output: str,
) -> None:
    batch = _build_batch(("CC(=O)C", "CO"))
    model = _build_model(
        batch,
        precursor_types=("[M-H]-",),
        include_cmf=False,
        max_h_transfer=0,
        unique_fragment_mode=unique_fragment_mode,
        max_masks=8,
        return_base_spectrum=True,
        intensity_output=intensity_output,
        proposal_selection_only=(intensity_output == "independent"),
        use_base_probability_feature=(intensity_output != "independent"),
    ).eval()
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([25.0, 25.0]),
    )

    assert output["ion_h_chemical_valid_channels"].flatten(1).any(dim=1).all()
    assert (output["pred_mzs"] > 0).all()
    # The bare carbonyl C and O fragments cannot lose H.
    for impossible_mz in (10.992723548, 14.987638168):
        assert not th.isclose(
            output["pred_mzs"],
            output["pred_mzs"].new_tensor(impossible_mz),
            atol=1e-6,
            rtol=0.0,
        ).any()
    probability_keys = ["base_pred_logprobs"]
    if intensity_output == "probability":
        probability_keys.append("pred_logprobs")
        objective = -output["pred_logprobs"].mean()
    else:
        assert th.isfinite(output["pred_intensities"]).all()
        assert (output["pred_intensities"] > 0).all()
        objective = output["pred_intensities"].mean()
    for key in probability_keys:
        totals = th.zeros(2).index_add(0, output["pred_batch_idxs"], output[key].exp())
        th.testing.assert_close(totals, th.ones(2))
    objective.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert th.isfinite(parameter.grad).all()


@pytest.mark.parametrize("mixed_batch", [False, True])
@pytest.mark.parametrize("independent", [False, True])
@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not th.cuda.is_available(), reason="CUDA not available"),
        ),
    ],
)
def test_no_chemically_valid_fragments_return_empty_output(
    mixed_batch: bool,
    device: str,
    independent: bool,
) -> None:
    smiles = ("FC(F)(F)F", "CCO") if mixed_batch else ("FC(F)(F)F",)
    batch = _build_batch(smiles).to(device)
    model = (
        _build_model(
            batch,
            unique_fragment_mode=True,
            precursor_types=("[M-H-H2O]-",),
            intensity_output="independent" if independent else "probability",
            proposal_selection_only=independent,
            use_base_probability_feature=not independent,
            separate_ion_h_support_head=independent,
            ion_h_support_binary_logits=independent,
            min_ion_h_absolute_probability=0.5 if independent else 0.0,
        )
        .to(device)
        .eval()
    )

    with th.no_grad():
        output = model(
            mol_pyg=batch,
            spec_prec_type=th.zeros(len(smiles), dtype=th.long, device=device),
            spec_nce=th.full((len(smiles),), 25.0, device=device),
        )
    assert (output["pred_batch_idxs"] != 0).all()
    assert bool(output["pred_mzs"].numel()) is mixed_batch
    values = output["pred_intensities" if independent else "pred_logprobs"]
    assert th.isfinite(values).all()
    assert th.isfinite(output["retained_joint_channel_intensity_fraction"]).all()
    # A device-side assertion would also poison this subsequent CUDA operation.
    assert th.ones(1, device=device).sum().item() == 1.0


def test_masster_embeds_each_ce_before_pooling_merged_spectra() -> None:
    batch = _build_batch()
    model = _build_model(batch, ce_insert_type="embed", ce_insert_size=6).eval()
    encoder = model.context_encoder
    nce = th.tensor([20.0, 40.0, 60.0])
    nce_batch_indices = th.tensor([0, 0, 1])

    assert encoder.nce_embedding is not None
    assert encoder.projector[0].in_features == 4 + 6
    embedded = encoder.nce_embedding(nce.round().long().unsqueeze(1))
    expected_ce = th.stack(((embedded[0] + embedded[1]) / 2.0, embedded[2]))
    expected = encoder.projector(
        th.cat((encoder.precursor_embedding(th.tensor([0, 1])), expected_ce), dim=1)
    )
    actual = encoder(
        batch_size=2,
        precursor_types=th.tensor([0, 1]),
        instrument_types=None,
        fragmentation_modes=None,
        nce=nce,
        nce_batch_indices=nce_batch_indices,
        ace=None,
        ace_batch_indices=None,
    )

    assert th.allclose(actual, expected)


def test_masster_spectrum_loss_reaches_ce_embedding() -> None:
    batch = _build_batch()
    model = _build_model(batch, ce_insert_type="embed")
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([25.0, 35.0]),
    )

    (-output["pred_logprobs"].mean()).backward()

    assert model.context_encoder.nce_embedding is not None
    projection = model.context_encoder.nce_embedding[1].weight
    assert projection.grad is not None
    assert th.isfinite(projection.grad).all()
    assert projection.grad.abs().sum() > 0


def test_masster_independent_output_returns_normal_space_intensities() -> None:
    batch = _build_batch()
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        intensity_output="independent",
    )
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([25.0, 35.0]),
    )

    assert "pred_logprobs" not in output
    assert "pred_intensities" in output
    assert th.isfinite(output["pred_intensities"]).all()
    assert (output["pred_intensities"] >= 0).all()
    assert not th.allclose(
        output["pred_intensities"].sum().reshape(1),
        th.ones(1),
    )

    output["pred_intensities"].sum().backward()
    assert model.intensity_head.ion_hydrogen_head.weight.grad is not None


def test_independent_intensity_is_soft_gated_by_binary_support() -> None:
    batch = _build_batch()
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        separate_ion_h_support_head=True,
        ion_h_support_binary_logits=True,
        gate_independent_intensity_with_support=True,
        intensity_output="independent",
    ).train()
    support_head = model.intensity_head.ion_h_support_head
    assert support_head is not None
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([25.0, 35.0]),
    }

    with th.no_grad():
        support_head.weight.zero_()
        support_head.bias.fill_(th.logit(th.tensor(0.25)))
        low_support = model(**inputs)
        support_head.bias.fill_(th.logit(th.tensor(0.75)))
    high_support = model(**inputs)

    assert th.equal(high_support["pred_mzs"], low_support["pred_mzs"])
    th.testing.assert_close(
        high_support["pred_intensities"],
        3.0 * low_support["pred_intensities"],
    )
    high_support["pred_intensities"].sum().backward()
    assert support_head.bias.grad is not None
    assert support_head.bias.grad.abs().sum() > 0


def test_independent_intensity_gate_is_opt_in() -> None:
    batch = _build_batch()
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        separate_ion_h_support_head=True,
        ion_h_support_binary_logits=True,
        intensity_output="independent",
    ).train()
    support_head = model.intensity_head.ion_h_support_head
    assert support_head is not None
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([25.0, 35.0]),
    }

    with th.no_grad():
        support_head.weight.zero_()
        support_head.bias.fill_(th.logit(th.tensor(0.25)))
        low_support = model(**inputs)
        support_head.bias.fill_(th.logit(th.tensor(0.75)))
        high_support = model(**inputs)

    th.testing.assert_close(high_support["pred_intensities"], low_support["pred_intensities"])


def test_full_candidate_attention_requires_unique_fragment_mode() -> None:
    """The legacy pathway-token mode emits one token per occurrence, so the cost is unbounded."""
    batch = _build_batch()
    with pytest.raises(ValueError, match="full candidate attention requires unique_fragment_mode"):
        _build_model(
            batch,
            unique_fragment_mode=False,
            intensity_transformer_attention="full",
        )
    # Unique-fragment mode bounds the candidate count, so the same arm constructs there.
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        max_enumerated_masks=8,
        intensity_transformer_attention="full",
    )
    assert model.intensity_head.attention_type == "full"
    assert model.intensity_head.inducing_tokens is None


def test_independent_intensity_gate_requires_independent_output_and_support_head() -> None:
    batch = _build_batch()
    with pytest.raises(ValueError, match="gating independent intensity by support requires"):
        _build_model(
            batch,
            unique_fragment_mode=True,
            use_base_probability_feature=False,
            proposal_selection_only=True,
            gate_independent_intensity_with_support=True,
            intensity_output="probability",
        )


def test_independent_intensity_gate_requires_binary_support_logits() -> None:
    batch = _build_batch()
    with pytest.raises(ValueError, match="requires binary support logits"):
        _build_model(
            batch,
            unique_fragment_mode=True,
            use_base_probability_feature=False,
            proposal_selection_only=True,
            separate_ion_h_support_head=True,
            ion_h_support_binary_logits=False,
            gate_independent_intensity_with_support=True,
            intensity_output="independent",
        )


def test_absolute_support_floor_is_evaluation_only_and_keeps_each_spectrum_nonempty() -> None:
    batch = _build_batch()
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        separate_ion_h_support_head=True,
        ion_h_support_binary_logits=True,
        min_ion_h_absolute_probability=1.0,
        intensity_output="independent",
    )
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([25.0, 35.0]),
    }

    with th.no_grad():
        training_output = model.train()(**inputs)
        model.eval()
        model.set_evaluation_joint_channel_relative_threshold(0.1)
        evaluation_output = model(**inputs)

    assert evaluation_output["pred_mzs"].numel() < training_output["pred_mzs"].numel()
    assert th.bincount(evaluation_output["pred_batch_idxs"], minlength=2).gt(0).all()


def test_masster_shares_one_ce_independent_proposal_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fragnnet.frag import masster_core

    batch = _build_batch(("CCCO",) * 5)
    model = _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=12,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        separate_proposal_encoder=True,
        share_proposal_across_ev=True,
    ).eval()
    precursor_batch_sizes = []
    proposal_batch_sizes = []
    topology_call_count = 0

    original_build_component_arrays = masster_core.build_component_arrays_raw

    def record_topology_call(*args, **kwargs):
        nonlocal topology_call_count
        topology_call_count += 1
        return original_build_component_arrays(*args, **kwargs)

    monkeypatch.setattr(
        masster_core,
        "build_component_arrays_raw",
        record_topology_call,
    )

    def record_precursor_batch(_module, args, _output):
        precursor_batch_sizes.append(int(args[1].max().item()) + 1)

    def record_proposal_batch(_module, args, _output):
        proposal_batch_sizes.append(int(args[1].max().item()) + 1)

    assert model.proposal_encoder is not None
    precursor_hook = model.precursor_encoder.register_forward_hook(record_precursor_batch)
    proposal_hook = model.proposal_encoder.register_forward_hook(record_proposal_batch)
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.zeros(5, dtype=th.long),
        spec_nce=th.tensor([10.0, 20.0, 30.0, 40.0, 50.0]),
        proposal_group_ids=th.full((5,), 17),
    )
    precursor_hook.remove()
    proposal_hook.remove()

    bond_count = batch.chemical_bond_index.shape[1] // 5
    assert precursor_batch_sizes == [1]
    assert proposal_batch_sizes == [1]
    assert topology_call_count == 1
    first_bond_logits = output["bond_break_logits"][:bond_count]
    for spectrum_index in range(1, 5):
        start = spectrum_index * bond_count
        assert th.allclose(
            first_bond_logits,
            output["bond_break_logits"][start : start + bond_count],
        )
    assert output["cross_ev_fragment_batch_idxs"].unique().tolist() == list(range(5))


def test_shared_proposal_ion_mode_head_is_collision_energy_blind() -> None:
    batch = _build_batch(("CCCO",) * 2)
    model = _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=12,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        share_proposal_across_ev=True,
        separate_ion_mode_proposal_head=True,
        ion_mode_proposal_binary_logits=True,
        precursor_types=("[M+Na]+",),
    ).eval()

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.zeros(2, dtype=th.long),
        spec_nce=th.tensor([10.0, 60.0]),
        proposal_group_ids=th.full((2,), 3),
    )

    # One proposal per group: both energies get the same ion-mode proposal.
    fragment_spectra = output["ion_h_fragment_batch_idxs"]
    first = output["ion_mode_proposal_logits"][fragment_spectra == 0]
    second = output["ion_mode_proposal_logits"][fragment_spectra == 1]
    assert first.numel() > 0
    # Rows at different batch positions differ by matmul rounding only (~1e-8); reading the
    # CE-conditioned context instead moved these logits by 0.007-0.03.
    assert th.allclose(first, second, rtol=0.0, atol=1e-6)


def test_masster_rejects_mismatched_graphs_in_shared_group() -> None:
    batch = _build_batch(("CCCC", "CCCO"))
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        max_enumerated_masks=12,
        share_proposal_across_ev=True,
    ).eval()

    with pytest.raises(ValueError, match="identical atom features"):
        model(
            mol_pyg=batch,
            spec_prec_type=th.zeros(2, dtype=th.long),
            spec_nce=th.tensor([10.0, 20.0]),
            proposal_group_ids=th.zeros(2, dtype=th.long),
        )


def test_selection_only_proposal_loss_reaches_bernoulli_mixture() -> None:
    batch = _build_batch(("CCCO",))
    model = _build_model(
        batch,
        max_masks=5,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        proposal_num_mixtures=2,
    )
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([25.0]),
    )

    assert output["bond_break_logits"].shape[1] == 2
    assert th.allclose(
        output["proposal_mixture_log_weights"].exp().sum(dim=1),
        th.ones(1),
        atol=1e-6,
    )
    (-output["proposal_fragment_logprobs"][0]).backward()

    mixture_head = model.proposal_mixture_head
    assert mixture_head is not None
    mixture_output = mixture_head.head[-1]
    assert isinstance(mixture_output, th.nn.Linear)
    assert mixture_output.weight.grad is not None
    assert th.isfinite(mixture_output.weight.grad).all()
    assert mixture_output.weight.grad.abs().sum() > 0
    bond_output = model.bond_break_head.head[-1]
    assert isinstance(bond_output, th.nn.Linear)
    assert bond_output.weight.grad is not None
    assert th.isfinite(bond_output.weight.grad).all()
    assert bond_output.weight.grad.abs().sum() > 0


def test_masster_forward_conserves_selected_probability_per_molecule() -> None:
    batch = _build_batch()
    model = _build_model(batch)

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([20.0, 40.0]),
    )

    assert not hasattr(model, "mol_pool")
    assert output["pred_mzs"].dtype == th.float64
    assert output["pred_mzs"].shape == output["pred_logprobs"].shape
    assert output["pred_batch_idxs"].unique(sorted=True).tolist() == [0, 1]
    assert th.isfinite(output["pred_mzs"]).all()
    assert th.isfinite(output["pred_logprobs"]).all()
    assert output["retained_mask_probability"].shape == (2,)
    assert output["selected_mask_count"].tolist() == [4, 2]
    assert output["component_pathway_count"].tolist() == [8, 3]
    assert output["components_per_mask"].tolist() == pytest.approx([2.0, 1.5])
    assert th.all(output["unique_components_per_mask"] <= output["components_per_mask"])
    cut_fraction_sum = sum(
        output[f"mask_cut_fraction_{bucket}"] for bucket in ("0", "1", "2", "3", "4", "5", "6plus")
    )
    assert cut_fraction_sum.tolist() == pytest.approx([1.0, 1.0])
    assert th.allclose(
        output["mask_cut_fraction_4plus"],
        output["mask_cut_fraction_4"]
        + output["mask_cut_fraction_5"]
        + output["mask_cut_fraction_6plus"],
    )
    assert output["candidate_mzs_by_cut"].shape == output["candidate_cut_counts"].shape
    assert th.equal(output["candidate_batch_idxs_by_cut"], output["pred_batch_idxs"])
    assert output["candidate_total_batch_ms"].shape == (2,)
    assert output["candidate_total_molecule_ms"].min() >= 0.0
    protonated_ethanol_mz = formula_to_peak_mzs(
        "C2H6O",
        "[M+H]+",
        isotopes=False,
    )[0]
    ethanol_predictions = output["pred_mzs"][output["pred_batch_idxs"] == 0]
    assert th.isclose(
        ethanol_predictions,
        th.tensor(protonated_ethanol_mz, dtype=th.float64),
        atol=1e-9,
        rtol=0.0,
    ).any()

    predicted_mass = th.zeros(2).index_add(
        0,
        output["pred_batch_idxs"],
        output["pred_logprobs"].exp(),
    )
    assert predicted_mass.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)
    assert "pred_oos_logprobs" not in output


@pytest.mark.skipif(
    not th.cuda.is_available() or not th.cuda.is_bf16_supported(),
    reason="CUDA BF16 not available",
)
def test_masster_supports_cuda_bfloat16_autocast() -> None:
    batch = _build_batch(("C1CC1", "CCCO")).to("cuda")
    model = _build_model(
        batch,
        bond_num_layers=2,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=8,
    ).cuda()

    with th.autocast(device_type="cuda", dtype=th.bfloat16):
        output = model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0, 0], device="cuda"),
            spec_nce=th.tensor([20.0, 25.0], device="cuda"),
        )
        loss = -output["pred_logprobs"].mean()
    loss.backward()

    assert th.isfinite(output["pred_logprobs"]).all()
    assert th.isfinite(output["unique_fragment_effective_count"]).all()
    assert th.isfinite(output["retained_mask_probability"]).all()
    assert model.charge_head.head[0].weight.grad is not None
    assert th.isfinite(model.charge_head.head[0].weight.grad).all()


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_masster_smarts_mechanisms_preserve_probability_and_train_head(
    unique_fragment_mode: bool,
) -> None:
    smiles = "c1ccccc1S(=O)c1ccccc1"
    batch = _build_batch((smiles,))
    model = _build_model(
        batch,
        max_masks=4 if unique_fragment_mode else 1,
        unique_fragment_mode=unique_fragment_mode,
        max_enumerated_masks=8,
        use_smarts_templates=True,
    )
    static_graph = MassterStaticGraph(
        chemical_bond_index=batch.chemical_bond_index,
        atom_batch_indices=batch.batch,
        atom_ptr=batch.ptr,
        smarts_templates=((build_masster_templates(Chem.MolFromSmiles(smiles))),),
    )

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([25.0]),
        static_graph=static_graph,
    )

    total_probability = output["pred_logprobs"].exp().sum()
    assert total_probability.item() == pytest.approx(1.0, abs=2e-6)
    (-output["pred_logprobs"].mean()).backward()
    assert model.mechanism_head is not None
    assert model.mechanism_head.rule_embedding.weight.grad is not None


# The EI-only anisoyl rule was DISABLED 2026-09-29, leaving no registered rule with a
# fragmentation-mode guard, so the tests below use a synthetic EI-only template.
_EI_ONLY_TEMPLATE = MassterTemplate(
    broken_bond_indices=(0,),
    components=((0,), (1, 2)),
    rule_index=1,
    rule_name="ei_only",
    allowed_precursor_types=("[M]+",),
    allowed_fragmentation_modes=("EI",),
)


def test_masster_filters_templates_by_adduct_and_fragmentation_mode() -> None:
    smiles = ("CN(C)CCBr", "CCO")
    batch = _build_batch(smiles)
    model = _build_model(
        batch,
        use_smarts_templates=True,
        precursor_types=("[M+H]+", "[M]+"),
        fragmentation_modes=("CID", "EI"),
    )
    templates = (build_masster_templates(Chem.MolFromSmiles(smiles[0])), (_EI_ONLY_TEMPLATE,))

    eligible = model._eligible_smarts_templates(
        templates,
        precursor_type_indices=th.tensor([0, 1]),
        fragmentation_mode_indices=th.tensor([0, 1]),
    )
    wrong_mode = model._eligible_smarts_templates(
        templates,
        precursor_type_indices=th.tensor([0, 1]),
        fragmentation_mode_indices=th.tensor([0, 0]),
    )

    assert "short_haloalkyl_ammonium_c2_transfer" in {
        template.rule_name for template in eligible[0]
    }
    assert "ei_only" in {template.rule_name for template in eligible[1]}
    assert "ei_only" not in {template.rule_name for template in wrong_mode[1]}


def test_masster_excludes_ei_only_rule_without_mode_metadata() -> None:
    batch = _build_batch(("CCO",))
    model = _build_model(
        batch,
        use_smarts_templates=True,
        precursor_types=("[M]+",),
    )

    eligible = model._eligible_smarts_templates(
        ((_EI_ONLY_TEMPLATE,),),
        precursor_type_indices=th.tensor([0]),
        fragmentation_mode_indices=None,
    )

    assert "ei_only" not in {template.rule_name for template in eligible[0]}


@pytest.mark.parametrize("precursor_gnn_type", ["Graphormer", "GINE_GRAPHORMER"])
def test_masster_runs_with_graphormer_precursor_encoder(
    precursor_gnn_type: str,
) -> None:
    batch = _build_batch()
    model = _build_model(batch, precursor_gnn_type=precursor_gnn_type)

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([20.0, 40.0]),
    )

    assert output["pred_mzs"].shape == output["pred_logprobs"].shape
    assert th.isfinite(output["pred_logprobs"]).all()


@pytest.mark.parametrize("unique_fragment_mode", [False, True])
def test_masster_returns_normalized_pre_residual_spectrum(
    unique_fragment_mode: bool,
) -> None:
    batch = _build_batch()
    model = _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=unique_fragment_mode,
        max_enumerated_masks=8,
        use_base_probability_feature=False,
        return_base_spectrum=True,
    )

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([20.0, 40.0]),
    )

    base_mass = th.zeros(2).index_add(
        0,
        output["pred_batch_idxs"],
        output["base_pred_logprobs"].exp(),
    )
    assert not model.intensity_head.use_base_probability_feature
    assert output["base_pred_logprobs"].shape == output["pred_logprobs"].shape
    assert base_mass.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)


def _parameter_gradient_sum(module: th.nn.Module) -> float:
    return sum(
        parameter.grad.abs().sum().item()
        for parameter in module.parameters()
        if parameter.grad is not None
    )


def test_detached_base_ion_h_preserves_values_and_isolates_intensity_gradients() -> None:
    batch = _build_batch()
    attached_model = _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=8,
        use_base_probability_feature=False,
        return_base_spectrum=True,
        detach_base_ion_h=False,
    )
    detached_model = _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=8,
        use_base_probability_feature=False,
        return_base_spectrum=True,
        detach_base_ion_h=True,
    )
    model_inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([20.0, 40.0]),
    }

    attached_output = attached_model(**model_inputs)
    detached_output = detached_model(**model_inputs)

    assert th.equal(attached_output["base_pred_logprobs"], detached_output["base_pred_logprobs"])
    assert th.equal(attached_output["pred_logprobs"], detached_output["pred_logprobs"])

    (-attached_output["base_pred_logprobs"].mean()).backward()
    (-detached_output["base_pred_logprobs"].mean()).backward()

    assert _parameter_gradient_sum(attached_model.intensity_head) > 0.0
    assert _parameter_gradient_sum(detached_model.intensity_head) == 0.0
    assert _parameter_gradient_sum(detached_model.bond_break_head) > 0.0
    assert _parameter_gradient_sum(detached_model.charge_head) > 0.0


def test_selection_only_intensity_ignores_proposal_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _build_batch(("CCCO",))
    model = _build_model(
        batch,
        max_masks=5,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_base_probability_feature=False,
        proposal_selection_only=True,
    ).eval()
    model_inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0]),
        "spec_nce": th.tensor([25.0]),
    }
    with th.no_grad():
        baseline = model(**model_inputs)

    build_candidates = masster_model_module.build_masster_candidates

    def lower_selected_mask_probabilities(*args: object, **kwargs: object) -> object:
        candidates = build_candidates(*args, **kwargs)
        offsets = th.linspace(
            -1.0,
            -0.2,
            candidates.mask_log_probabilities.numel(),
            device=candidates.mask_log_probabilities.device,
        )
        return replace(
            candidates,
            mask_log_probabilities=candidates.mask_log_probabilities + offsets,
        )

    monkeypatch.setattr(
        masster_model_module,
        "build_masster_candidates",
        lower_selected_mask_probabilities,
    )
    with th.no_grad():
        shifted = model(**model_inputs)

    baseline_conditional = baseline["pred_logprobs"] - th.logsumexp(
        baseline["pred_logprobs"], dim=0
    )
    shifted_conditional = shifted["pred_logprobs"] - th.logsumexp(shifted["pred_logprobs"], dim=0)
    assert th.equal(baseline["pred_mzs"], shifted["pred_mzs"])
    assert th.allclose(baseline_conditional, shifted_conditional, atol=1e-6, rtol=0.0)
    assert not th.allclose(
        baseline["retained_mask_probability"],
        shifted["retained_mask_probability"],
    )
    assert shifted["pred_logprobs"].exp().sum().item() == pytest.approx(1.0, abs=2e-6)


def test_selection_only_primary_spectrum_does_not_update_proposal_heads() -> None:
    batch = _build_batch(("CCCO",))
    model = _build_model(
        batch,
        max_masks=5,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_base_probability_feature=False,
        proposal_selection_only=True,
    )
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([25.0]),
    )

    (-output["pred_logprobs"].mean()).backward()

    assert _parameter_gradient_sum(model.bond_break_head) == 0.0
    assert _parameter_gradient_sum(model.charge_head) == 0.0
    assert _parameter_gradient_sum(model.precursor_encoder) > 0.0
    assert _parameter_gradient_sum(model.context_encoder) > 0.0
    assert _parameter_gradient_sum(model.intensity_head) > 0.0
    assert (
        output["proposal_candidate_mzs"].shape == output["proposal_candidate_fragment_idxs"].shape
    )
    assert (
        output["proposal_fragment_logprobs"].shape == output["proposal_fragment_batch_idxs"].shape
    )


def test_selection_only_spectrum_has_no_oos_gate() -> None:
    batch = _build_batch(("CCCO",))
    model = _build_model(
        batch,
        max_masks=5,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_base_probability_feature=False,
        proposal_selection_only=True,
    )
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([25.0]),
    )

    assert "pred_oos_logprobs" not in output
    assert output["pred_logprobs"].exp().sum().item() == pytest.approx(1.0, abs=2e-6)
    assert model.intensity_head.oos_head is None

    (-output["pred_logprobs"].mean()).backward()

    assert _parameter_gradient_sum(model.bond_break_head) == 0.0
    assert _parameter_gradient_sum(model.charge_head) == 0.0


@pytest.mark.parametrize("context_in_gnn", [False, True])
def test_separate_proposal_encoder_isolates_both_objectives(context_in_gnn: bool) -> None:
    batch = _build_batch(("CCCO",))
    model_kwargs = {
        "max_masks": 5,
        "unique_fragment_mode": True,
        "max_enumerated_masks": 16,
        "use_base_probability_feature": False,
        "proposal_selection_only": True,
        "separate_proposal_encoder": True,
        "context_in_gnn": context_in_gnn,
    }
    model_inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0]),
        "spec_nce": th.tensor([25.0]),
    }

    primary_model = _build_model(batch, **model_kwargs)
    primary_output = primary_model(**model_inputs)
    (-primary_output["pred_logprobs"].mean()).backward()

    assert primary_model.proposal_encoder is not None
    assert _parameter_gradient_sum(primary_model.precursor_encoder) > 0.0
    assert _parameter_gradient_sum(primary_model.proposal_encoder) == 0.0
    assert _parameter_gradient_sum(primary_model.bond_break_head) == 0.0
    assert _parameter_gradient_sum(primary_model.charge_head) == 0.0

    proposal_model = _build_model(batch, **model_kwargs)
    proposal_output = proposal_model(**model_inputs)
    fragmented_candidate_index = proposal_output["candidate_cut_counts"].argmax().reshape(1)
    proposal_loss = _proposal_coverage_loss(
        true_mzs=proposal_output["proposal_candidate_mzs"][fragmented_candidate_index].detach(),
        true_logprobs=th.zeros(1),
        true_batch_idxs=th.zeros(1, dtype=th.long),
        candidate_mzs=proposal_output["proposal_candidate_mzs"],
        candidate_fragment_idxs=proposal_output["proposal_candidate_fragment_idxs"],
        fragment_logprobs=proposal_output["proposal_fragment_logprobs"],
        fragment_batch_idxs=proposal_output["proposal_fragment_batch_idxs"],
        batch_size=1,
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )
    proposal_loss.sum().backward()

    assert proposal_model.proposal_encoder is not None
    assert _parameter_gradient_sum(proposal_model.proposal_encoder) > 0.0
    assert _parameter_gradient_sum(proposal_model.bond_break_head) > 0.0
    assert _parameter_gradient_sum(proposal_model.charge_head) > 0.0
    assert _parameter_gradient_sum(proposal_model.precursor_encoder) == 0.0
    assert _parameter_gradient_sum(proposal_model.context_encoder) == 0.0
    assert _parameter_gradient_sum(proposal_model.intensity_head) == 0.0


def test_selection_only_proposal_loss_updates_only_proposal_heads() -> None:
    batch = _build_batch(("CCCO",))
    model = _build_model(
        batch,
        max_masks=5,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_base_probability_feature=False,
        proposal_selection_only=True,
    )
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([25.0]),
    )
    fragmented_candidate_index = output["candidate_cut_counts"].argmax().reshape(1)
    true_mz = output["proposal_candidate_mzs"][fragmented_candidate_index].detach()
    proposal_loss = _proposal_coverage_loss(
        true_mzs=true_mz,
        true_logprobs=th.zeros(1),
        true_batch_idxs=th.zeros(1, dtype=th.long),
        candidate_mzs=output["proposal_candidate_mzs"],
        candidate_fragment_idxs=output["proposal_candidate_fragment_idxs"],
        fragment_logprobs=output["proposal_fragment_logprobs"],
        fragment_batch_idxs=output["proposal_fragment_batch_idxs"],
        batch_size=1,
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )

    proposal_loss.sum().backward()

    assert _parameter_gradient_sum(model.bond_break_head) > 0.0
    assert _parameter_gradient_sum(model.charge_head) > 0.0
    assert _parameter_gradient_sum(model.precursor_encoder) == 0.0
    assert _parameter_gradient_sum(model.context_encoder) == 0.0
    assert _parameter_gradient_sum(model.intensity_head) == 0.0


def test_unique_fragment_mode_marginalizes_paths_and_uses_boundary_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _build_batch(("C1CC1",))
    model = _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=8,
        use_mask_residuals=False,
    ).eval()
    transformer_token_counts: list[int] = []
    original_intensity_forward = model.intensity_head.forward

    def capture_token_count(*args: object, **kwargs: object) -> tuple[th.Tensor, ...]:
        fragment_embeddings = args[0]
        assert isinstance(fragment_embeddings, th.Tensor)
        transformer_token_counts.append(fragment_embeddings.shape[0])
        return original_intensity_forward(*args, **kwargs)

    monkeypatch.setattr(model.intensity_head, "forward", capture_token_count)
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([20.0]),
    )

    unique_count = int(output["unique_component_count"].sum().item())
    pathway_count = int(output["component_pathway_count"].sum().item())
    assert transformer_token_counts == [unique_count]
    assert unique_count < pathway_count
    # A cyclopropane atom set has at most two precursor boundary bonds even when
    # its parent pathway contains all three cuts.
    assert output["candidate_cut_counts"].max().item() == 2

    predicted_mass = output["pred_logprobs"].exp().sum()
    assert predicted_mass.item() == pytest.approx(1.0, abs=2e-6)


@pytest.mark.parametrize("intensity_transformer_attention", ["induced", "full"])
def test_unique_fragment_spectrum_loss_reaches_bond_charge_and_ion_heads(
    intensity_transformer_attention: str,
) -> None:
    batch = _build_batch(("CCCO",))
    model = _build_model(
        batch,
        max_masks=5,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_mask_residuals=False,
        intensity_transformer_attention=intensity_transformer_attention,
    )
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([25.0]),
    )

    (-output["pred_logprobs"].mean()).backward()

    parameters = [
        model.precursor_encoder.input_project.weight,
        model.bond_break_head.head[0].weight,
        model.charge_head.head[0].weight,
        model.intensity_head.input_projection[0].weight,
        model.intensity_head.ion_hydrogen_head.weight,
        model.context_encoder.projector[0].weight,
    ]
    for parameter in parameters:
        assert parameter.grad is not None
        assert th.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


@pytest.mark.parametrize("intensity_transformer_attention", ["induced", "full"])
def test_mz_positional_encoding_feeds_candidate_masses_to_the_head(
    intensity_transformer_attention: str,
) -> None:
    """The encoding reads each candidate's mass, adds no state, and moves the prediction."""
    batch = _build_batch(("CCOC", "CCO"))
    kwargs = {
        "max_masks": 4,
        "unique_fragment_mode": True,
        "max_enumerated_masks": 8,
        "intensity_transformer_attention": intensity_transformer_attention,
    }
    control = _build_model(batch, **kwargs).eval()
    arm = _build_model(batch, **kwargs, intensity_mz_positional_encoding=True).eval()
    control_state, arm_state = control.state_dict(), arm.state_dict()
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([10.0, 40.0]),
    }

    assert arm.intensity_head.uses_token_masses
    assert not control.intensity_head.uses_token_masses
    assert list(arm_state) == list(control_state)
    assert all(th.equal(arm_state[key], control_state[key]) for key in control_state)
    arm_output, control_output = arm(**inputs), control(**inputs)
    assert arm_output["pred_logprobs"].shape == control_output["pred_logprobs"].shape
    assert not th.allclose(arm_output["pred_logprobs"], control_output["pred_logprobs"])


@pytest.mark.parametrize("unique_fragment_mode", [True, False])
def test_encoder_context_token_makes_atom_embeddings_depend_on_nce(
    unique_fragment_mode: bool,
) -> None:
    """The encoder attends to each spectrum's context; the rest of the model is unchanged."""
    batch = _build_batch(("CCOC", "CCO"))
    kwargs = {
        "max_masks": 4,
        "unique_fragment_mode": unique_fragment_mode,
        "max_enumerated_masks": 8,
        "precursor_gnn_type": "GINE_GSA",
    }
    control = _build_model(batch, **kwargs).eval()
    arm = _build_model(batch, **kwargs, encoder_context_token=True).eval()
    control_state, arm_state = control.state_dict(), arm.state_dict()

    new_keys = set(arm_state) - set(control_state)
    assert new_keys and all(
        key.startswith("precursor_encoder.gnn.gsa_context_") for key in new_keys
    )
    assert all(th.equal(arm_state[key], control_state[key]) for key in control_state)

    def atom_embeddings(model: MassterModel, nce: list[float]) -> th.Tensor:
        captured = []
        hook = model.precursor_encoder.register_forward_hook(
            lambda _module, _inputs, output: captured.append(output)
        )
        model(mol_pyg=batch, spec_prec_type=th.tensor([0, 0]), spec_nce=th.tensor(nce))
        hook.remove()
        return captured[0]

    low, high = [10.0, 40.0], [10.0, 90.0]
    first_atoms = batch.batch == 0
    arm_low, arm_high = atom_embeddings(arm, low), atom_embeddings(arm, high)
    assert th.allclose(arm_low[first_atoms], arm_high[first_atoms], atol=1e-6)
    assert not th.allclose(arm_low[~first_atoms], arm_high[~first_atoms], atol=1e-5)
    control_low, control_high = atom_embeddings(control, low), atom_embeddings(control, high)
    assert th.equal(control_low, control_high)


def test_encoder_context_token_requires_global_attention() -> None:
    batch = _build_batch(("CCO",))
    with pytest.raises(ValueError, match="requires precursor_gnn_type='GINE_GSA'"):
        _build_model(batch, encoder_context_token=True)


@pytest.mark.parametrize(
    ("intensity_transformer_attention", "unique_fragment_mode"),
    [("full", True), ("induced", True), ("induced", False)],
    ids=["full_unique", "induced_unique", "induced_pathway_tokens"],
)
def test_nl_positional_encoding_feeds_precursor_masses_to_the_head(
    intensity_transformer_attention: str,
    unique_fragment_mode: bool,
) -> None:
    """Both head call sites pass each spectrum's intact neutral mass; no state is added."""
    smiles = ("CCOC", "CCO")
    batch = _build_batch(smiles)
    kwargs = {
        "max_masks": 4,
        "unique_fragment_mode": unique_fragment_mode,
        "max_enumerated_masks": 8,
        "intensity_transformer_attention": intensity_transformer_attention,
    }
    control = _build_model(batch, **kwargs).eval()
    arm = _build_model(batch, **kwargs, intensity_nl_positional_encoding=True).eval()
    control_state, arm_state = control.state_dict(), arm.state_dict()
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([10.0, 40.0]),
    }
    received = []
    head_forward = arm.intensity_head.forward

    def spy(*args, **head_kwargs):
        received.append(
            (
                head_kwargs["precursor_neutral_masses"],
                head_kwargs["component_neutral_masses"],
                args[5],
            )
        )
        return head_forward(*args, **head_kwargs)

    arm.intensity_head.forward = spy
    arm_output, control_output = arm(**inputs), control(**inputs)

    assert arm.intensity_head.uses_token_masses
    assert list(arm_state) == list(control_state)
    assert all(th.equal(arm_state[key], control_state[key]) for key in control_state)
    ((precursors, components, component_batch),) = received
    expected = th.tensor(
        [CalcExactMolWt(Chem.MolFromSmiles(smile)) for smile in smiles], dtype=th.float64
    )
    assert th.allclose(precursors.double(), expected, atol=1e-3)
    # No candidate outweighs its own precursor, and the precursor itself is a candidate.
    losses = precursors[component_batch] - components
    assert (losses > -1e-4).all()
    for spectrum in range(len(smiles)):
        assert losses[component_batch == spectrum].abs().min() < 1e-4
    assert arm_output["pred_logprobs"].shape == control_output["pred_logprobs"].shape
    assert not th.allclose(arm_output["pred_logprobs"], control_output["pred_logprobs"])


@pytest.mark.parametrize(
    ("intensity_transformer_attention", "intensity_mass_window"),
    [("full", 0.5), ("full", 1000.0), ("none", 0.0)],
    ids=["narrow_window", "window_covers_set", "no_attention"],
)
def test_restricted_candidate_attention_runs_through_the_model(
    intensity_transformer_attention: str,
    intensity_mass_window: float,
) -> None:
    """Mass-local and no-attention allocators build, run, and train on real candidate sets."""
    batch = _build_batch(("CCOC", "CCO"))
    kwargs = {"max_masks": 4, "unique_fragment_mode": True, "max_enumerated_masks": 8}
    full = _build_model(batch, **kwargs, intensity_transformer_attention="full").eval()
    arm = _build_model(
        batch,
        **kwargs,
        intensity_transformer_attention=intensity_transformer_attention,
        intensity_mass_window=intensity_mass_window,
    )
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([10.0, 40.0]),
    }

    assert arm.intensity_head.uses_token_masses == (intensity_mass_window > 0.0)
    arm_output = arm.eval()(**inputs)
    full_output = full(**inputs)
    assert arm_output["pred_logprobs"].shape == full_output["pred_logprobs"].shape
    assert th.isfinite(arm_output["pred_logprobs"]).all()
    if intensity_mass_window > 100.0:
        # Every candidate of these small molecules lies within 1000 Da of every other.
        assert th.allclose(arm_output["pred_logprobs"], full_output["pred_logprobs"], atol=1e-6)
    else:
        assert not th.allclose(arm_output["pred_logprobs"], full_output["pred_logprobs"])
    arm.train()(**inputs)["pred_logprobs"].logsumexp(0).sum().backward()
    gradients = [p.grad for p in arm.intensity_head.parameters() if p.grad is not None]
    assert gradients and all(th.isfinite(gradient).all() for gradient in gradients)


def test_unique_fragment_mode_keeps_batched_marginals_separate() -> None:
    batch = _build_batch(("C1CC1", "CCCO"))
    model = _build_model(
        batch,
        max_masks=4,
        unique_fragment_mode=True,
        max_enumerated_masks=8,
        use_mask_residuals=False,
    ).eval()

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([20.0, 25.0]),
        spec_nce_batch_idxs=th.tensor([0, 1]),
    )

    predicted_mass = th.zeros(2).index_add(
        0,
        output["pred_batch_idxs"],
        output["pred_logprobs"].exp(),
    )
    assert predicted_mass.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)
    assert (
        (output["productive_mask_fraction"] > 0.0) & (output["productive_mask_fraction"] <= 1.0)
    ).all()
    assert (
        (output["duplicate_component_pathway_fraction"] >= 0.0)
        & (output["duplicate_component_pathway_fraction"] < 1.0)
    ).all()
    assert (output["latent_pathways_per_unique_fragment"] >= 1.0).all()
    assert (
        (output["unique_fragment_target_fill_fraction"] > 0.0)
        & (output["unique_fragment_target_fill_fraction"] <= 1.0)
    ).all()
    assert (
        (output["unique_fragment_target_reached"] == 0.0)
        | (output["unique_fragment_target_reached"] == 1.0)
    ).all()
    assert (
        (output["unique_fragment_effective_count"] >= 1.0)
        & (output["unique_fragment_effective_count"] <= output["unique_component_count"] + 1e-5)
    ).all()
    assert (
        (output["unique_fragment_top1_mass_fraction"] > 0.0)
        & (output["unique_fragment_top1_mass_fraction"] <= 1.0)
    ).all()
    assert (
        (output["duplicate_pathway_probability_fraction"] >= 0.0)
        & (output["duplicate_pathway_probability_fraction"] <= 1.0)
    ).all()


def test_fragment_probability_cutoff_filters_before_intensity_scoring() -> None:
    batch = _build_batch(("C1CC1", "CCCO"))
    model = _build_model(
        batch,
        max_masks=6,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        min_fragment_relative_probability=1.0,
        use_base_probability_feature=False,
        proposal_selection_only=True,
    ).eval()

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([20.0, 25.0]),
    )

    before = output["fragment_count_before_cutoff"]
    after = output["fragment_count_after_cutoff"]
    assert (before > 0).all()
    assert ((after > 0) & (after <= before)).all()
    assert (after < before).any()
    assert th.allclose(
        output["fragment_cutoff_pruned_fraction"],
        1.0 - after / before,
    )
    assert (
        (output["retained_fragment_probability"] > 0.0)
        & (output["retained_fragment_probability"] <= 1.0)
    ).all()

    proposal_mass = th.zeros(2).index_add(
        0,
        output["proposal_fragment_batch_idxs"],
        output["proposal_fragment_logprobs"].exp(),
    )
    predicted_mass = th.zeros(2).index_add(
        0,
        output["pred_batch_idxs"],
        output["pred_logprobs"].exp(),
    )
    assert proposal_mass.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)
    assert predicted_mass.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)


def _connected_components(
    num_atoms: int,
    bonds: list[tuple[int, int]],
    cut: Sequence[int],
) -> set[frozenset[int]]:
    parent = list(range(num_atoms))

    def find(atom: int) -> int:
        while parent[atom] != atom:
            parent[atom] = parent[parent[atom]]
            atom = parent[atom]
        return atom

    for (begin, end), is_cut in zip(bonds, cut):
        if not is_cut:
            parent[find(begin)] = find(end)
    groups: dict[int, set[int]] = {}
    for atom in range(num_atoms):
        groups.setdefault(find(atom), set()).add(atom)
    return {frozenset(group) for group in groups.values()}


def _brute_force_fragment_probabilities(
    smiles: str,
    logits: th.Tensor,
) -> tuple[int, list[tuple[int, int]], dict[frozenset[int], float], dict[frozenset[int], float]]:
    """Enumerate every cut mask: each atom set's marginal and its intact-release probability."""
    molecule = Chem.MolFromSmiles(smiles)
    bonds = [(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) for bond in molecule.GetBonds()]
    cut_probabilities = th.sigmoid(logits.double()).tolist()
    marginals: dict[frozenset[int], float] = {}
    releases: dict[frozenset[int], float] = {}
    for cut in itertools.product((0, 1), repeat=len(bonds)):
        mask_probability = math.prod(
            p if is_cut else 1.0 - p for p, is_cut in zip(cut_probabilities, cut)
        )
        for component in _connected_components(molecule.GetNumAtoms(), bonds, cut):
            marginals[component] = marginals.get(component, 0.0) + mask_probability
            interior_intact = not any(
                is_cut
                for (begin, end), is_cut in zip(bonds, cut)
                if begin in component and end in component
            )
            if interior_intact:
                releases[component] = releases.get(component, 0.0) + mask_probability
    return molecule.GetNumAtoms(), bonds, marginals, releases


@pytest.mark.parametrize("smiles", ["CC(C)CO", "C1CC1CO"])
def test_release_probability_is_the_intact_event_and_bounds_the_atom_set_marginal(
    smiles: str,
) -> None:
    generator = th.Generator().manual_seed(17)
    num_bonds = Chem.MolFromSmiles(smiles).GetNumBonds()
    logits = 1.5 * th.randn(num_bonds, generator=generator)
    num_atoms, bonds, marginals, releases = _brute_force_fragment_probabilities(smiles, logits)
    atom_sets = sorted(marginals, key=sorted)

    atom_indices, atom_fragments, boundary_bonds, boundary_fragments = [], [], [], []
    interior_counts = []
    for fragment_index, atom_set in enumerate(atom_sets):
        for atom in sorted(atom_set):
            atom_indices.append(atom)
            atom_fragments.append(fragment_index)
        interior_counts.append(sum(begin in atom_set and end in atom_set for begin, end in bonds))
        for bond_index, (begin, end) in enumerate(bonds):
            if (begin in atom_set) != (end in atom_set):
                boundary_bonds.append(bond_index)
                boundary_fragments.append(fragment_index)
    closed_form = _fragment_release_log_probabilities(
        logits,
        th.tensor(bonds, dtype=th.long).T,
        num_atoms,
        th.tensor(atom_indices, dtype=th.long),
        th.tensor(atom_fragments, dtype=th.long),
        th.tensor(boundary_bonds, dtype=th.long),
        th.tensor(boundary_fragments, dtype=th.long),
        len(atom_sets),
    ).double()

    release = th.tensor([releases[atom_set] for atom_set in atom_sets], dtype=th.float64)
    marginal = th.tensor([marginals[atom_set] for atom_set in atom_sets], dtype=th.float64)
    # A connected atom set contains a ring exactly when it has as many interior bonds as atoms.
    cyclic = th.tensor(
        [count >= len(atom_set) for count, atom_set in zip(interior_counts, atom_sets)]
    )
    assert th.allclose(closed_form.exp(), release, rtol=1e-5, atol=1e-8)
    assert th.allclose(closed_form.exp()[~cyclic], marginal[~cyclic], rtol=1e-5, atol=1e-8)
    # A ring bond inside the fragment can break without changing its atom set.
    assert (closed_form.exp()[cyclic] < marginal[cyclic] - 1e-9).all()
    assert bool(cyclic.any()) == ("1" in smiles)


def test_top_k_fragment_keep_mask_keeps_forced_then_best_scores() -> None:
    scores = th.tensor([0.1, 0.5, -th.inf, 0.3, 0.9, 0.2, 0.2, 0.7, 0.4, 0.0, 5.0, 1.0, 9.0, 3.0])
    batch_indices = th.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3])
    forced = th.tensor(
        [True, False, False, False, False]
        + [False, False, False, False]
        + [True, True, True, False]
        + [False]
    )

    keep = _top_k_fragment_keep_mask(scores, batch_indices, forced, batch_size=4, k=3)

    # Molecule 0: the forced fragment plus the two best others; -inf never wins a slot.
    # Molecule 1: the 0.2 tie breaks by pool order. Molecule 2: three forced fill the budget,
    # so even the best-scoring free fragment is dropped. Molecule 3: fewer fragments than k.
    assert keep.tolist() == [
        True, True, False, False, True,
        True, False, True, True,
        True, True, True, False,
        True,
    ]  # fmt: skip
    assert _top_k_fragment_keep_mask(scores, batch_indices, forced, batch_size=4, k=10).all()
    with pytest.raises(ValueError, match="k must be positive"):
        _top_k_fragment_keep_mask(scores, batch_indices, forced, batch_size=4, k=0)


def test_marginal_selection_keeps_forced_fragments_and_scores_the_whole_pool() -> None:
    batch = _build_batch(("CCCCO", "CCCO"))
    model = _build_model(
        batch,
        max_masks=15,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_base_probability_feature=False,
        proposal_selection_only=True,
        fragment_selection="marginal",
        selected_fragment_target=11,
    ).eval()

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([20.0, 25.0]),
    )

    # CCCCO has 15 contiguous atom sets, nine of them forced (the intact molecule and both
    # sides of its four single cuts), so two slots go by release score. CCCO has only 10.
    assert output["fragment_count_before_cutoff"].tolist() == [15.0, 10.0]
    assert output["fragment_count_after_cutoff"].tolist() == [11.0, 10.0]
    pool_batch = output["proposal_fragment_batch_idxs"]
    assert th.bincount(pool_batch, minlength=2).tolist() == [15, 10]
    pool_mass = th.zeros(2).index_add(0, pool_batch, output["proposal_fragment_logprobs"].exp())
    assert pool_mass.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)
    assert th.isin(output["candidate_mzs_by_cut"], output["pool_candidate_mzs"]).all()
    assert output["pool_candidate_mzs"].numel() > output["candidate_mzs_by_cut"].numel()
    predicted_mass = th.zeros(2).index_add(
        0,
        output["pred_batch_idxs"],
        output["pred_logprobs"].exp(),
    )
    assert predicted_mass.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)


def test_marginal_selection_budget_holds_when_release_scores_tie() -> None:
    """Symmetric bonds give bit-identical release scores; K must still bound the kept set."""
    batch = _build_batch(("c1ccccc1",))
    kept_counts = []
    for selected_fragment_target in range(1, 11):
        model = _build_model(
            batch,
            max_masks=10,
            unique_fragment_mode=True,
            max_enumerated_masks=20,
            use_base_probability_feature=False,
            proposal_selection_only=True,
            fragment_selection="marginal",
            selected_fragment_target=selected_fragment_target,
        ).eval()
        with th.no_grad():
            output = model(mol_pyg=batch, spec_prec_type=th.tensor([0]), spec_nce=th.tensor([25.0]))
        assert output["fragment_count_before_cutoff"].tolist() == [10.0]
        kept_counts.append(int(output["fragment_count_after_cutoff"].item()))

    pool = output["proposal_fragment_logprobs"]
    # The premise: several pool fragments share the top release score exactly.
    assert int((pool == pool.max()).sum()) > 1
    # Only the intact ring is forced (a single ring cut releases nothing), so K alone decides.
    assert kept_counts == list(range(1, 11))


def test_missing_valid_fallback_adds_one_fragment_only_where_none_is_kept() -> None:
    scores = th.tensor([0.0, 2.0, 2.0, 2.0, 1.0, 5.0, 3.0, 3.0, 4.0])
    valid = th.tensor([False, True, True, True, True, True, False, False, False])
    keep = th.tensor([True, False, False, False, False, True, True, False, False])
    batch_indices = th.tensor([0, 0, 0, 0, 1, 1, 2, 2, 2])

    fallback = _missing_valid_fallback_mask(scores, valid, keep, batch_indices, batch_size=3)

    # Molecule 0 keeps only an invalid fragment: exactly one of its three tied best valid
    # fragments is added, the first in pool order. Molecule 1 already keeps a valid fragment.
    # Molecule 2 has no valid fragment to add.
    assert fallback.tolist() == [False, True, False, False, False, False, False, False, False]


def test_marginal_selection_trains_the_proposal_on_fragments_outside_the_budget() -> None:
    """Same kept set; only marginal mode lets an excluded fragment's peak reach the proposal."""
    batch = _build_batch(("CCCCO",))
    shared = {
        "unique_fragment_mode": True,
        "max_enumerated_masks": 16,
        "use_base_probability_feature": False,
        "proposal_selection_only": True,
        # One channel per fragment keeps every atom set at a distinct mass.
        "max_h_transfer": 0,
        "include_cmf": False,
    }
    models = {
        "marginal": _build_model(
            batch,
            max_masks=15,
            fragment_selection="marginal",
            selected_fragment_target=9,
            **shared,
        ),
        "mask_order": _build_model(batch, max_masks=9, **shared),
    }
    outputs = {
        name: model(mol_pyg=batch, spec_prec_type=th.tensor([0]), spec_nce=th.tensor([25.0]))
        for name, model in models.items()
    }
    kept = outputs["marginal"]["candidate_mzs_by_cut"]
    # Both keep exactly the intact molecule and the eight single-cut fragments.
    assert th.equal(kept.sort().values, outputs["mask_order"]["candidate_mzs_by_cut"].sort().values)
    pool = outputs["marginal"]["proposal_candidate_mzs"]
    outside = (pool[:, None] - kept[None, :]).abs().min(dim=1).values > 1e-5 * pool.clamp(min=200.0)
    assert outside.any()
    true_mz = pool[outside][:1].detach()

    results = {}
    for name, model in models.items():
        output = outputs[name]
        loss = _proposal_coverage_loss(
            true_mzs=true_mz,
            true_logprobs=th.zeros(1),
            true_batch_idxs=th.zeros(1, dtype=th.long),
            candidate_mzs=output["proposal_candidate_mzs"],
            candidate_fragment_idxs=output["proposal_candidate_fragment_idxs"],
            fragment_logprobs=output["proposal_fragment_logprobs"],
            fragment_batch_idxs=output["proposal_fragment_batch_idxs"],
            batch_size=1,
            tolerance=1e-5,
            relative=True,
            tolerance_min_mz=200.0,
        )
        loss.sum().backward()
        results[name] = (loss.item(), _parameter_gradient_sum(model.bond_break_head))

    assert results["marginal"][0] > 0.0
    assert results["marginal"][1] > 0.0
    assert results["mask_order"] == (0.0, 0.0)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"fragment_selection": "marginal"}, "integer selected_fragment_target"),
        (
            {"fragment_selection": "marginal", "selected_fragment_target": 0},
            "integer selected_fragment_target",
        ),
        (
            {"fragment_selection": "marginal", "selected_fragment_target": 7},
            "integer selected_fragment_target",
        ),
        (
            {"fragment_selection": "marginal", "selected_fragment_target": True},
            "integer selected_fragment_target",
        ),
        ({"selected_fragment_target": 3}, "only read by fragment_selection='marginal'"),
        ({"fragment_selection": "ranked"}, "unknown fragment_selection"),
        (
            {
                "fragment_selection": "marginal",
                "selected_fragment_target": 3,
                "proposal_selection_only": False,
            },
            "requires unique mode, proposal_selection_only",
        ),
        (
            {
                "fragment_selection": "marginal",
                "selected_fragment_target": 3,
                "min_fragment_relative_probability": 0.5,
            },
            "requires unique mode, proposal_selection_only",
        ),
        (
            {
                "fragment_selection": "marginal",
                "selected_fragment_target": 3,
                "proposal_num_mixtures": 2,
            },
            "requires unique mode, proposal_selection_only",
        ),
    ],
)
def test_marginal_fragment_selection_rejects_invalid_configurations(
    overrides: dict[str, object],
    message: str,
) -> None:
    config: dict[str, object] = {
        "max_masks": 6,
        "unique_fragment_mode": True,
        "max_enumerated_masks": 16,
        "use_base_probability_feature": False,
        "proposal_selection_only": True,
    }
    config.update(overrides)
    with pytest.raises(ValueError, match=message):
        _build_model(_build_batch(("CCCO",)), **config)


def test_mask_residual_reallocates_only_retained_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _build_batch()
    model = _build_model(batch).eval()
    forward_kwargs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([20.0, 40.0]),
    }
    original_intensity_forward = model.intensity_head.forward

    with th.no_grad():
        baseline = model(**forward_kwargs)

    def shifted_intensity_forward(*args: object, **kwargs: object) -> tuple[th.Tensor, ...]:
        component_residuals, mask_residuals, ion_h_logits = original_intensity_forward(
            *args,
            **kwargs,
        )
        mask_residuals = th.linspace(
            -1.0,
            1.0,
            mask_residuals.shape[0],
            device=mask_residuals.device,
        )
        return component_residuals, mask_residuals, ion_h_logits

    monkeypatch.setattr(model.intensity_head, "forward", shifted_intensity_forward)
    with th.no_grad():
        shifted = model(**forward_kwargs)

    assert not th.allclose(baseline["pred_logprobs"], shifted["pred_logprobs"])
    assert th.allclose(
        baseline["retained_mask_probability"],
        shifted["retained_mask_probability"],
    )
    shifted_mass = th.zeros(2).index_add(
        0,
        shifted["pred_batch_idxs"],
        shifted["pred_logprobs"].exp(),
    )
    assert shifted_mass.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)


def test_disabled_mask_residual_does_not_reallocate_probability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _build_batch()
    model = _build_model(batch, use_mask_residuals=False).eval()
    forward_kwargs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([20.0, 40.0]),
    }
    original_intensity_forward = model.intensity_head.forward

    with th.no_grad():
        baseline = model(**forward_kwargs)

    def shifted_intensity_forward(*args: object, **kwargs: object) -> tuple[th.Tensor, ...]:
        component_residuals, mask_residuals, ion_h_logits = original_intensity_forward(
            *args,
            **kwargs,
        )
        return (
            component_residuals,
            th.linspace(
                -1.0,
                1.0,
                mask_residuals.shape[0],
                device=mask_residuals.device,
            ),
            ion_h_logits,
        )

    monkeypatch.setattr(model.intensity_head, "forward", shifted_intensity_forward)
    with th.no_grad():
        shifted = model(**forward_kwargs)

    assert th.allclose(baseline["pred_logprobs"], shifted["pred_logprobs"])


def test_masster_spectrum_loss_reaches_encoder_and_all_heads() -> None:
    batch = _build_batch()
    model = _build_model(batch)
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([25.0, 35.0]),
    )

    loss = -output["pred_logprobs"].mean()
    loss.backward()

    parameters = [
        model.precursor_encoder.input_project.weight,
        model.bond_break_head.head[0].weight,
        model.charge_head.head[0].weight,
        model.intensity_head.input_projection[0].weight,
        model.intensity_head.layers[0].latent_attention.in_proj_weight,
        model.intensity_head.component_residual_head.weight,
        model.intensity_head.mask_residual_head.weight,
        model.intensity_head.ion_hydrogen_head.weight,
        model.context_encoder.projector[0].weight,
    ]
    for parameter in parameters:
        assert parameter.grad is not None
        assert th.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_masster_spectrum_loss_reaches_bond_focused_layers() -> None:
    batch = _build_batch()
    model = _build_model(batch, bond_num_layers=2)
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([25.0, 35.0]),
    )

    (-output["pred_logprobs"].mean()).backward()

    parameters = [
        model.bond_break_head.bond_project[0].weight,
        model.bond_break_head.atom_updates[0][0].weight,
        model.bond_break_head.bond_updates[0][0].weight,
    ]
    for parameter in parameters:
        assert parameter.grad is not None
        assert th.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_masster_requires_regenerated_graph_chemistry_fields() -> None:
    batch = _build_batch()
    model = _build_model(batch)
    del batch.atom_hydrogen_counts

    with pytest.raises(ValueError, match="regenerate molecular preprocessing caches"):
        model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0, 1]),
            spec_nce=th.tensor([25.0, 35.0]),
        )


def test_channel_selection_leaves_full_candidate_consumers_unchanged() -> None:
    batch = _build_batch()
    dense_model = _build_model(
        batch,
        unique_fragment_mode=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
    ).eval()
    sparse_model = _build_model(
        batch,
        unique_fragment_mode=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        min_ion_h_relative_probability=1.0,
    ).eval()
    sparse_model.load_state_dict(dense_model.state_dict())
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([25.0, 35.0]),
    }

    with th.no_grad():
        dense = dense_model(**inputs)
        sparse = sparse_model(**inputs)

    assert sparse["pred_mzs"].numel() < dense["pred_mzs"].numel()
    assert th.equal(sparse["candidate_mzs_by_cut"], dense["candidate_mzs_by_cut"])
    assert th.equal(sparse["proposal_candidate_mzs"], dense["proposal_candidate_mzs"])
    assert th.equal(sparse["cross_ev_candidate_mzs"], dense["cross_ev_candidate_mzs"])
    assert th.equal(
        sparse["ion_h_chemical_valid_channels"],
        dense["ion_h_chemical_valid_channels"],
    )
    assert sparse["ion_h_dense_log_weights"].shape == sparse["ion_h_dense_mzs"].shape
    assert th.isfinite(
        sparse["ion_h_dense_log_weights"][sparse["ion_h_chemical_valid_channels"]]
    ).all()
    removed_outputs = {
        "full_candidate_ion_mode_idxs",
        "full_candidate_h_shift_idxs",
        "proposal_candidate_ion_mode_idxs",
        "proposal_candidate_h_shift_idxs",
        "ion_h_spectral_output_channels",
    }
    assert removed_outputs.isdisjoint(sparse)


def test_evaluation_top_k_keeps_one_channel_per_fragment_and_is_reversible() -> None:
    batch = _build_batch()
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        separate_ion_h_support_head=True,
        ion_h_support_binary_logits=True,
    ).eval()
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([25.0, 35.0]),
    }

    with th.no_grad():
        dense = model(**inputs)
        model.set_evaluation_ion_h_top_k(1)
        top_one = model(**inputs)
        model.set_evaluation_ion_h_top_k(0)
        restored = model(**inputs)

    assert th.equal(top_one["retained_ion_h_count"], top_one["fragment_count_after_cutoff"])
    assert top_one["pred_mzs"].numel() < dense["pred_mzs"].numel()
    assert th.equal(top_one["candidate_mzs_by_cut"], dense["candidate_mzs_by_cut"])
    assert th.equal(top_one["proposal_candidate_mzs"], dense["proposal_candidate_mzs"])
    assert th.equal(top_one["cross_ev_candidate_mzs"], dense["cross_ev_candidate_mzs"])
    assert th.equal(restored["pred_mzs"], dense["pred_mzs"])
    assert th.equal(restored["pred_logprobs"], dense["pred_logprobs"])


def test_evaluation_top_k_is_ignored_during_training() -> None:
    batch = _build_batch()
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        separate_ion_h_support_head=True,
        ion_h_support_binary_logits=True,
    ).train()
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([25.0, 35.0]),
    }

    with th.no_grad():
        dense = model(**inputs)
        model.set_evaluation_ion_h_top_k(1)
        top_k_configured = model(**inputs)

    assert th.equal(top_k_configured["pred_mzs"], dense["pred_mzs"])
    assert th.equal(top_k_configured["pred_logprobs"], dense["pred_logprobs"])


def test_evaluation_joint_threshold_is_reversible_and_ignored_during_training() -> None:
    batch = _build_batch()
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        separate_ion_h_support_head=True,
        ion_h_support_binary_logits=True,
    ).eval()
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([25.0, 35.0]),
    }
    with th.no_grad():
        model.intensity_head.ion_hydrogen_head.weight.zero_()
        model.intensity_head.ion_hydrogen_head.bias.copy_(
            th.arange(model.intensity_head.ion_hydrogen_head.bias.numel())
        )
        dense = model(**inputs)
        model.set_evaluation_joint_channel_relative_threshold(1.0)
        thresholded = model(**inputs)
        model.set_evaluation_joint_channel_relative_threshold(0.0)
        restored = model(**inputs)
        model.train()
        model.set_evaluation_joint_channel_relative_threshold(1.0)
        training_output = model(**inputs)

    assert th.equal(
        thresholded["retained_ion_h_count"],
        thresholded["fragment_count_after_cutoff"],
    )
    assert thresholded["pred_mzs"].numel() < dense["pred_mzs"].numel()
    assert dense["retained_joint_channel_intensity_fraction"].tolist() == pytest.approx(
        [1.0, 1.0]
    )
    assert th.all(thresholded["retained_joint_channel_intensity_fraction"] < 1.0)
    assert th.equal(restored["pred_mzs"], dense["pred_mzs"])
    assert th.equal(training_output["pred_mzs"], dense["pred_mzs"])


@pytest.mark.parametrize("top_k", [-1, True, 1.5])
def test_evaluation_top_k_rejects_invalid_values(top_k: object) -> None:
    model = _build_model(_build_batch())
    with pytest.raises(ValueError, match="nonnegative integer"):
        model.set_evaluation_ion_h_top_k(top_k)  # type: ignore[arg-type]


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan"), True, "0.1"])
def test_evaluation_joint_threshold_rejects_invalid_values(threshold: object) -> None:
    model = _build_model(_build_batch())
    with pytest.raises(ValueError, match=r"finite and in \[0, 1\]"):
        model.set_evaluation_joint_channel_relative_threshold(threshold)  # type: ignore[arg-type]


def test_retained_channel_probability_uses_dense_intensity_distribution() -> None:
    batch = _build_batch(("CCO",))
    model = _build_model(
        batch,
        unique_fragment_mode=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        separate_ion_h_support_head=True,
        ion_h_support_binary_logits=True,
        min_ion_h_relative_probability=1.0,
        precursor_types=("[M+H]+",),
    ).eval()
    support_head = model.intensity_head.ion_h_support_head
    assert support_head is not None
    with th.no_grad():
        model.intensity_head.ion_hydrogen_head.weight.zero_()
        model.intensity_head.ion_hydrogen_head.bias.fill_(-10.0)
        # Channel 1 is the zero-shift CRF rung, valid for every component. Channel 2 (+1 H)
        # is unreachable for the intact molecule, so pinning probability there would leak
        # mass onto other channels instead of testing the support cutoff.
        model.intensity_head.ion_hydrogen_head.bias[1] = 10.0
        support_head.weight.zero_()
        support_head.bias.fill_(-10.0)
        support_head.bias[0] = 10.0

        output = model(
            mol_pyg=batch,
            spec_prec_type=th.tensor([0]),
            spec_nce=th.tensor([25.0]),
        )

    assert output["ion_h_cutoff_pruned_fraction"].item() > 0.0
    assert output["retained_ion_h_probability"].item() < 1e-6


@pytest.mark.parametrize(
    ("bound", "atom_indices", "cut_count", "expected_max_gain"),
    [
        # Legacy: hydrogen may be gained with no source at all.
        ("none", [0, 1, 2], 0, 4),
        ("none", [2], 1, 4),
        # The complementary neutral must have the hydrogens to give. The intact molecule has
        # no complement, so it can only lose them -- this is the channel that put the
        # survivor ion at prec_mz + 4H and crashed the binned metrics.
        ("complement", [0, 1, 2], 0, 0),
        ("complement", [0, 1], 1, 1),  # complement OH holds one
        ("complement", [2], 1, 4),  # complement C2H5 holds five, the ladder caps at four
        # FraGNNet parity: the fragment must also have somewhere to put them. A lone O from
        # one severed bond becomes OH, never OH3, however much hydrogen the complement has.
        ("valence", [0, 1, 2], 0, 0),
        ("valence", [2], 1, 1),
    ],
)
def test_hydrogen_transfer_bounds(
    bound: str,
    atom_indices: list[int],
    cut_count: int,
    expected_max_gain: int,
) -> None:
    """``h_transfer_bound`` selects how much of FraGNNet's hydrogen cap MASSter applies.

    FraGNNet has always bounded a fragment's post-transfer hydrogen count by
    ``min(compute_cc_h_cap(...), num_hs_prior + max_h_transfer, num_hs)``
    (``frag_utils``). MASSter only required nonnegative element counts, which bounds
    losses but not gains. ``complement`` adds the ``num_hs`` term, ``valence`` adds the
    free-valence cap as well, and ``none`` keeps the original unbounded ladder because those
    unreachable channels also serve as negatives in the candidate softmax.
    """
    batch = _build_batch(("CCO",))
    model = _build_model(
        batch,
        precursor_types=("[M+H]+",),
        include_cmf=False,
        max_h_transfer=4,
        h_transfer_bound=bound,
    )
    indices = th.tensor(atom_indices)
    ion_mzs, valid = model._fragment_masses_and_valid_channels(
        batch,
        indices,
        th.zeros_like(indices),
        th.tensor([0]),
        1,
        th.tensor([0]),
        th.tensor([cut_count]),
    )

    shifts = list(range(-model.max_h_transfer, model.max_h_transfer + 1))
    for offset, shift in enumerate(shifts):
        if shift > 0:
            assert valid[0, 0, offset].item() == (shift <= expected_max_gain), (
                f"+{shift} H under bound {bound!r}"
            )
    assert valid[0, 0, shifts.index(0)].item(), "the untransferred channel is always valid"

    if bound != "none":
        # Nothing survives above the precursor ion itself.
        all_atoms = th.arange(batch.x.shape[0])
        precursor_mzs, _ = model._fragment_masses_and_valid_channels(
            batch,
            all_atoms,
            th.zeros_like(all_atoms),
            th.tensor([0]),
            1,
            th.tensor([0]),
            th.tensor([0]),
        )
        precursor_mz = precursor_mzs[0, 0, model.max_h_transfer].item()
        assert ion_mzs[0, 0][valid[0, 0]].max().item() <= precursor_mz + 1e-9


def test_valence_bound_requires_cut_counts() -> None:
    """The valence bound cannot be applied without knowing what was severed."""
    batch = _build_batch(("CCO",))
    model = _build_model(batch, precursor_types=("[M+H]+",), h_transfer_bound="valence")
    indices = th.arange(batch.x.shape[0])
    with pytest.raises(ValueError, match="fragment_cut_counts"):
        model._fragment_masses_and_valid_channels(
            batch, indices, th.zeros_like(indices), th.tensor([0]), 1, th.tensor([0])
        )


def test_h_transfer_bound_rejects_unknown_values() -> None:
    batch = _build_batch(("CCO",))
    with pytest.raises(ValueError, match="h_transfer_bound"):
        _build_model(batch, h_transfer_bound="complememt")


@pytest.mark.parametrize(
    "smiles",
    [
        "CCO",
        "CC(=O)Oc1ccccc1C(=O)O",
        "CN1C=NC2=C1C(=O)N(C)C(=O)N2C",
        "CC#N",
        "C1CCCCC1",
        "C[N+](C)(C)CC(=O)[O-]",
    ],
)
def test_valence_bound_matches_fragnnet_reference(smiles: str) -> None:
    """MASSter's tensor bound must equal FraGNNet's own per-fragment computation.

    FraGNNet derives the cap one fragment at a time, in numpy and Cython
    (``compute_cc_h_cap`` over ``update_bonds``), which cannot run inside a batched GPU
    forward: it would need a host sync and a Python loop over every fragment in the batch.
    MASSter instead pre-aggregates the same arithmetic per atom
    (``feat_utils._atom_free_valence_slots``) and adds one slot per severed bond. That
    identity is what this test pins, using FraGNNet's functions as the oracle.
    """
    mol = Chem.MolFromSmiles(smiles)
    mol_d = extract_mol_info(mol)
    ve_arr = mol_d["ve_arr"]
    sbond_arr = mol_d["sbond_arr"]
    hs_arr = mol_d["hs_arr"]

    batch = _build_batch((smiles,))
    model = _build_model(
        batch,
        precursor_types=("[M+H]+",),
        include_cmf=False,
        max_h_transfer=4,
        h_transfer_bound="valence",
    )

    num_atoms = mol.GetNumAtoms()
    fragments = [list(range(num_atoms))] + [[i] for i in range(num_atoms)]
    if num_atoms > 2:
        fragments.append(list(range(num_atoms - 1)))

    for atom_ids in fragments:
        cc = np.array(atom_ids, dtype=np.int32)
        fragment_sbond, _ = compute_frags.update_bonds(
            cc,
            sbond_arr.astype(np.int32),
            mol_d["bond_mask_arr"].astype(np.uint8),
            mol_d["bonds"].astype(np.int32),
            mol_d["atoms_to_bonds"],
        )
        # FraGNNet's own cap, then its two extra terms from compute_cc_h_options.
        cap = compute_cc_h_cap(cc, ve_arr.copy(), fragment_sbond, 0)
        hydrogens_prior = int(hs_arr[cc].sum())
        reference_gain = min(
            int(cap) - hydrogens_prior,
            model.max_h_transfer,
            int(mol_d["num_hs"]) - hydrogens_prior,
        )
        # Bonds severed at this fragment's boundary, from FraGNNet's own arrays.
        cut_count = int(sbond_arr[cc].sum() - fragment_sbond[cc].sum())

        indices = th.tensor(atom_ids)
        _, valid = model._fragment_masses_and_valid_channels(
            batch,
            indices,
            th.zeros_like(indices),
            th.tensor([0]),
            1,
            th.tensor([0]),
            th.tensor([cut_count]),
        )
        shifts = list(range(-model.max_h_transfer, model.max_h_transfer + 1))
        gains = [s for s in shifts if s > 0 and valid[0, 0, shifts.index(s)].item()]
        model_gain = max(gains) if gains else 0
        assert model_gain == max(reference_gain, 0), (
            f"{smiles} fragment {atom_ids}: MASSter allows +{model_gain} H, "
            f"FraGNNet allows +{reference_gain}"
        )
