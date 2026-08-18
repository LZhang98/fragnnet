"""End-to-end tests for the first trainable MASShatter model slice."""

import pytest
import torch as th
from rdkit import Chem
from torch_geometric.data import Batch

import fragnnet.model.mass_shatter_model as mass_shatter_model_module
from fragnnet.model.bond_fragment_heads import BondFocusedBreakHead
from fragnnet.model.mass_shatter_model import MassShatterModel
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
    explore_masks: int = 0,
    max_masks: int = 2,
    use_mask_residuals: bool = True,
    exploration_cut_count_weights: tuple[float, ...] = (
        0.0,
        0.0,
        0.35,
        0.30,
        0.20,
        0.10,
        0.05,
    ),
) -> MassShatterModel:
    th.manual_seed(903)
    return MassShatterModel(
        node_feature_dim=batch.x.shape[1],
        edge_feature_dim=batch.edge_attr.shape[1],
        precursor_hidden_dim=16,
        precursor_num_layers=2,
        precursor_gnn_type="GINE",
        precursor_normalization="layer",
        precursor_dropout=0.0,
        context_dim=8,
        metadata_embedding_dim=4,
        head_hidden_dim=12,
        head_dropout=0.0,
        bond_num_layers=bond_num_layers,
        intensity_transformer_dim=16,
        intensity_transformer_num_layers=2,
        intensity_transformer_num_heads=4,
        intensity_transformer_feedforward_dim=32,
        intensity_transformer_num_inducing_points=4,
        intensity_transformer_dropout=0.0,
        use_mask_residuals=use_mask_residuals,
        precursor_types=["[M+H]+", "[M-H]-"],
        instrument_types=[],
        fragmentation_modes=[],
        use_nce=True,
        use_ace=False,
        nce_mean=30.0,
        nce_std=10.0,
        ace_mean=20.0,
        ace_std=15.0,
        max_masks=max_masks,
        min_mask_probability=0.0,
        explore_masks=explore_masks,
        exploration_mode="bounded_additive",
        exploration_temperature=1.0,
        exploration_cut_count_weights=exploration_cut_count_weights,
        include_single_bond_cuts=True,
        max_h_transfer=1,
        include_cmf=True,
        charge_pooling="logsumexp",
    )


def test_mass_shatter_selects_bond_focused_head() -> None:
    batch = _build_batch()
    model = _build_model(batch, bond_num_layers=2)

    assert isinstance(model.bond_break_head, BondFocusedBreakHead)


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
    assert output["exploration_mask_count"].tolist() == [0.0, 0.0]
    assert output["exploration_mean_cut_count"].tolist() == [0.0, 0.0]
    assert output["component_pathway_count"].tolist() == [8, 3]
    assert output["components_per_mask"].tolist() == pytest.approx([2.0, 1.5])
    assert th.all(output["unique_components_per_mask"] <= output["components_per_mask"])
    cut_fraction_sum = sum(
        output[f"mask_cut_fraction_{bucket}"]
        for bucket in ("0", "1", "2", "3", "4plus")
    )
    assert cut_fraction_sum.tolist() == pytest.approx([1.0, 1.0])
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
    total_mass = predicted_mass + output["pred_oos_logprobs"].exp()
    assert total_mass.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)
    assert output["retained_mask_probability"].tolist() == pytest.approx(
        predicted_mass.tolist(), abs=2e-6
    )


def test_mass_shatter_forward_conserves_probability_with_bounded_exploration() -> None:
    batch = _build_batch(("CCCCC",))
    model = _build_model(
        batch,
        max_masks=1,
        explore_masks=2,
        exploration_cut_count_weights=(0.0, 0.0, 1.0),
    )
    model.train()

    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0]),
        spec_nce=th.tensor([20.0]),
    )

    assert output["exploration_mask_count"].tolist() == [2.0]
    assert output["exploration_mean_cut_count"].tolist() == [2.0]
    predicted_mass = output["pred_logprobs"].exp().sum()
    total_mass = predicted_mass + output["pred_oos_logprobs"].exp().sum()
    assert total_mass.item() == pytest.approx(1.0, abs=2e-6)


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
    shifted_total = shifted_mass + shifted["pred_oos_logprobs"].exp()
    assert shifted_total.tolist() == pytest.approx([1.0, 1.0], abs=2e-6)


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


def test_mass_shatter_uses_mixed_masks_only_during_training(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch = _build_batch()
    model = _build_model(batch, explore_masks=1)
    observed_explore_masks: list[int] = []
    build_candidates = mass_shatter_model_module.build_mass_shatter_candidates

    def capture_explore_masks(*args: object, **kwargs: object) -> object:
        observed_explore_masks.append(int(kwargs["explore_masks"]))
        return build_candidates(*args, **kwargs)

    monkeypatch.setattr(
        mass_shatter_model_module,
        "build_mass_shatter_candidates",
        capture_explore_masks,
    )
    forward_kwargs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 1]),
        "spec_nce": th.tensor([20.0, 40.0]),
    }

    model.train()
    model(**forward_kwargs)
    model.eval()
    with th.no_grad():
        model(**forward_kwargs)

    assert observed_explore_masks == [1, 0]


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
