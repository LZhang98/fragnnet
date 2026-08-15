"""End-to-end tests for the first trainable MASShatter model slice."""

import pytest
import torch as th
from rdkit import Chem
from torch_geometric.data import Batch

from fragnnet.model.mass_shatter_model import MassShatterModel
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.formula_utils import formula_to_peak_mzs


def _build_batch() -> Batch:
    return Batch.from_data_list(
        [
            get_mol_graph(
                Chem.MolFromSmiles(smiles),
                ["a_onehot", "a_degree"],
                ["b_degree"],
                0,
                True,
            )
            for smiles in ["CCO", "CO"]
        ]
    )


def _build_model(batch: Batch) -> MassShatterModel:
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
        precursor_types=["[M+H]+", "[M-H]-"],
        instrument_types=[],
        fragmentation_modes=[],
        use_nce=True,
        use_ace=False,
        nce_mean=30.0,
        nce_std=10.0,
        ace_mean=20.0,
        ace_std=15.0,
        max_masks=2,
        include_single_bond_cuts=True,
        max_h_transfer=1,
        include_cmf=True,
        charge_pooling="logsumexp",
    )


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
        model.ion_hydrogen_head.head[0].weight,
        model.context_encoder.projector[0].weight,
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
