"""End-to-end tests for the first trainable MASShatter model slice."""

from dataclasses import replace

import pytest
import torch as th
from rdkit import Chem
from torch_geometric.data import Batch

import fragnnet.model.mass_shatter_model as mass_shatter_model_module
from fragnnet.frag.mass_shatter_candidates import MassShatterStaticGraph
from fragnnet.frag.mass_shatter_templates import build_mass_shatter_templates
from fragnnet.model.bond_fragment_heads import BondFocusedBreakHead
from fragnnet.model.mass_shatter_model import (
    MassShatterModel,
    _absolute_binary_channel_keep_mask,
    _cmf_duplicate_crf_shift,
    _fixed_top_k_channel_keep_mask,
    _ion_mode_logits_from_ion_h_support,
    _relative_channel_keep_mask,
    _relative_fragment_keep_mask,
    _relative_joint_channel_keep_mask,
)
from fragnnet.pl_model.mass_shatter_pl import _proposal_coverage_loss
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.formula_utils import formula_to_peak_mzs


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
    intensity_atom_decoder_num_layers: int = 0,
    intensity_output: str = "probability",
    precursor_gnn_type: str = "GINE",
    ce_insert_type: str = "id",
    ce_insert_size: int = 6,
    use_smarts_templates: bool = False,
    include_cmf: bool = True,
    max_h_transfer: int = 1,
    precursor_types: tuple[str, ...] = ("[M+H]+", "[M-H]-"),
    fragmentation_modes: tuple[str, ...] = (),
) -> MassShatterModel:
    th.manual_seed(903)
    return MassShatterModel(
        node_feature_dim=batch.x.shape[1],
        edge_feature_dim=batch.edge_attr.shape[1],
        precursor_hidden_dim=16,
        precursor_num_layers=2,
        precursor_num_heads=4,
        precursor_gnn_type=precursor_gnn_type,
        precursor_normalization="layer",
        precursor_dropout=0.0,
        context_dim=8,
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
        intensity_atom_decoder_num_layers=intensity_atom_decoder_num_layers,
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
        include_single_bond_cuts=True,
        max_h_transfer=max_h_transfer,
        include_cmf=include_cmf,
        charge_pooling="logsumexp",
        use_smarts_templates=use_smarts_templates,
    )


def test_mass_shatter_selects_bond_focused_head() -> None:
    batch = _build_batch()
    model = _build_model(batch, bond_num_layers=2)

    assert isinstance(model.bond_break_head, BondFocusedBreakHead)


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
        mass_shatter_model_module,
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


def test_mass_shatter_embeds_each_ce_before_pooling_merged_spectra() -> None:
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


def test_mass_shatter_spectrum_loss_reaches_ce_embedding() -> None:
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


def test_mass_shatter_independent_output_returns_normal_space_intensities() -> None:
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


def test_mass_shatter_shares_one_ce_independent_proposal_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fragnnet.frag import mass_shatter_core

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

    original_build_component_arrays = mass_shatter_core.build_component_arrays_raw

    def record_topology_call(*args, **kwargs):
        nonlocal topology_call_count
        topology_call_count += 1
        return original_build_component_arrays(*args, **kwargs)

    monkeypatch.setattr(
        mass_shatter_core,
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


def test_mass_shatter_rejects_mismatched_graphs_in_shared_group() -> None:
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


def test_mass_shatter_spectrum_loss_reaches_masked_atom_decoder() -> None:
    batch = _build_batch()
    model = _build_model(batch, intensity_atom_decoder_num_layers=1)
    gate = model.intensity_head.atom_decoder_gate
    assert gate is not None
    with th.no_grad():
        gate.fill_(1.0)
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([25.0, 35.0]),
    )

    (-output["pred_logprobs"].mean()).backward()

    decoder = model.intensity_head.atom_decoder
    assert decoder is not None
    cross_attention = decoder.layers[0].multihead_attn.in_proj_weight
    assert cross_attention.grad is not None
    assert th.isfinite(cross_attention.grad).all()
    assert cross_attention.grad.abs().sum() > 0


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


def test_mass_shatter_forward_conserves_selected_probability_per_molecule() -> None:
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
def test_mass_shatter_supports_cuda_bfloat16_autocast() -> None:
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
def test_mass_shatter_smarts_mechanisms_preserve_probability_and_train_head(
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
    static_graph = MassShatterStaticGraph(
        chemical_bond_index=batch.chemical_bond_index,
        atom_batch_indices=batch.batch,
        atom_ptr=batch.ptr,
        smarts_templates=((build_mass_shatter_templates(Chem.MolFromSmiles(smiles))),),
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


def test_mass_shatter_filters_templates_by_adduct_and_fragmentation_mode() -> None:
    smiles = ("COC(=O)c1ccccc1", "COc1ccc(C(=O)F)cc1")
    batch = _build_batch(smiles)
    model = _build_model(
        batch,
        use_smarts_templates=True,
        precursor_types=("[M+H]+", "[M]+"),
        fragmentation_modes=("CID", "EI"),
    )
    templates = tuple(
        build_mass_shatter_templates(Chem.MolFromSmiles(molecule_smiles))
        for molecule_smiles in smiles
    )

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

    assert "methyl_benzoate_co2_transfer" in {template.rule_name for template in eligible[0]}
    assert "anisoyl_fluoride_co_f_transfer" in {template.rule_name for template in eligible[1]}
    assert "anisoyl_fluoride_co_f_transfer" not in {
        template.rule_name for template in wrong_mode[1]
    }


def test_mass_shatter_excludes_ei_only_rule_without_mode_metadata() -> None:
    smiles = "COc1ccc(C(=O)F)cc1"
    batch = _build_batch((smiles,))
    model = _build_model(
        batch,
        use_smarts_templates=True,
        precursor_types=("[M]+",),
    )
    templates = (build_mass_shatter_templates(Chem.MolFromSmiles(smiles)),)

    eligible = model._eligible_smarts_templates(
        templates,
        precursor_type_indices=th.tensor([0]),
        fragmentation_mode_indices=None,
    )

    assert "anisoyl_fluoride_co_f_transfer" not in {template.rule_name for template in eligible[0]}


@pytest.mark.parametrize("precursor_gnn_type", ["Graphormer", "GINE_GRAPHORMER"])
def test_mass_shatter_runs_with_graphormer_precursor_encoder(
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
def test_mass_shatter_returns_normalized_pre_residual_spectrum(
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

    build_candidates = mass_shatter_model_module.build_mass_shatter_candidates

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
        mass_shatter_model_module,
        "build_mass_shatter_candidates",
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


def test_separate_proposal_encoder_isolates_both_objectives() -> None:
    batch = _build_batch(("CCCO",))
    model_kwargs = {
        "max_masks": 5,
        "unique_fragment_mode": True,
        "max_enumerated_masks": 16,
        "use_base_probability_feature": False,
        "proposal_selection_only": True,
        "separate_proposal_encoder": True,
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


def test_unique_fragment_spectrum_loss_reaches_bond_charge_and_ion_heads() -> None:
    batch = _build_batch(("CCCO",))
    model = _build_model(
        batch,
        max_masks=5,
        unique_fragment_mode=True,
        max_enumerated_masks=16,
        use_mask_residuals=False,
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


def test_mass_shatter_spectrum_loss_reaches_encoder_and_all_heads() -> None:
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


def test_mass_shatter_spectrum_loss_reaches_bond_focused_layers() -> None:
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


def test_mass_shatter_requires_regenerated_graph_chemistry_fields() -> None:
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
        model.intensity_head.ion_hydrogen_head.bias[2] = 10.0
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
