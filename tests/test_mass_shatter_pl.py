"""Integration tests for MASShatter's SpectrumPL training wrapper."""

import math
from pathlib import Path

import numpy as np
import pytest
import torch as th
import yaml
from rdkit import Chem
from torch_geometric.data import Batch, Data

from fragnnet.dataset import SpecMolDataset
from fragnnet.model.bond_fragment_heads import BondFocusedBreakHead
from fragnnet.model.loss import get_sparse_cross_entropy_fn
from fragnnet.model.nn_blocks import GINEGraphormer
from fragnnet.pl_model import MassShatterPL, SpectrumPL
from fragnnet.pl_model.mass_shatter_pl import (
    _candidate_coverage_by_cut_count,
    _cross_ev_fragment_delta_loss,
    _proposal_coverage_loss,
)
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls


def _build_pl(**config_overrides: object) -> tuple[MassShatterPL, dict]:
    template_path = Path(__file__).parents[1] / "config" / "template.yml"
    with template_path.open() as template_file:
        config = yaml.safe_load(template_file)
    config.update(
        model_type="mass_shatter",
        compile=False,
        sparse_cosine_similarity=True,
        oos_tolerance_multiple=1,
        gaussian_renormalize=True,
        mass_shatter_selected_mask_limit=2,
        mass_shatter_unique_fragment_target=2,
        mass_shatter_ranked_mask_budget=2,
        mass_shatter_max_h_transfer=1,
    )
    config.update(config_overrides)
    return MassShatterPL(**config), config


def _loss_diagnostics() -> dict[str, th.Tensor]:
    return {
        "retained_mask_probability": th.tensor([0.8]),
        "selected_mask_count": th.tensor([2]),
        "exploration_mask_count": th.tensor([0]),
        "exploration_mean_cut_count": th.tensor([0.0]),
        "component_pathway_count": th.tensor([4]),
        "mean_mask_cut_count": th.tensor([2.0]),
        "proposal_single_cut_fraction": th.tensor([0.25]),
        "proposal_mixture_entropy": th.tensor([0.6]),
        "proposal_mixture_effective_count": th.tensor([1.8]),
        "proposal_mixture_top1_weight": th.tensor([0.7]),
        "unique_component_count": th.tensor([2]),
        "components_per_mask": th.tensor([2.0]),
        "unique_components_per_mask": th.tensor([1.0]),
        "latent_pathways_per_unique_fragment": th.tensor([2.0]),
        "unique_fragment_target_fill_fraction": th.tensor([1.0]),
        "unique_fragment_target_reached": th.tensor([1.0]),
        "unique_fragment_effective_count": th.tensor([1.8]),
        "unique_fragment_top1_mass_fraction": th.tensor([0.6]),
        "duplicate_pathway_probability_fraction": th.tensor([0.2]),
        "productive_mask_fraction": th.tensor([1.0]),
        "duplicate_component_pathway_fraction": th.tensor([0.5]),
        "mask_cut_fraction_0": th.tensor([0.0]),
        "mask_cut_fraction_1": th.tensor([0.5]),
        "mask_cut_fraction_2": th.tensor([0.5]),
        "mask_cut_fraction_3": th.tensor([0.0]),
        "mask_cut_fraction_4plus": th.tensor([0.0]),
        "candidate_input_transfer_batch_ms": th.tensor([1.0]),
        "candidate_mask_selection_batch_ms": th.tensor([2.0]),
        "candidate_topology_batch_ms": th.tensor([3.0]),
        "candidate_assembly_batch_ms": th.tensor([4.0]),
        "candidate_output_transfer_batch_ms": th.tensor([5.0]),
        "candidate_total_batch_ms": th.tensor([15.0]),
        "candidate_total_molecule_ms": th.tensor([15.0]),
    }


def test_mass_shatter_registration_uses_spectrum_only_dataset() -> None:
    assert get_model_cls("mass_shatter") is MassShatterPL
    assert get_dataset_cls("mass_shatter") is SpecMolDataset
    assert issubclass(MassShatterPL, SpectrumPL)


def test_mass_shatter_pl_builds_configured_bond_focused_head() -> None:
    model, _ = _build_pl(mass_shatter_bond_num_layers=2)

    assert isinstance(model.model.bond_break_head, BondFocusedBreakHead)


def test_mass_shatter_pl_builds_gine_graphormer_backbone() -> None:
    model, _ = _build_pl(
        mol_gnn_type="GINE_GRAPHORMER",
        mol_normalization="layer",
        mol_hidden_size=16,
        mol_num_layers=2,
        mol_num_heads=4,
        mass_shatter_bond_num_layers=2,
    )

    assert isinstance(model.model.precursor_encoder.gnn, GINEGraphormer)


def test_mass_shatter_pl_disables_mask_residuals_from_config() -> None:
    model, _ = _build_pl(mass_shatter_use_mask_residuals=False)

    assert not model.model.use_mask_residuals


def test_mass_shatter_pl_uses_selected_mask_limit_in_non_unique_mode() -> None:
    model, _ = _build_pl(
        mass_shatter_selected_mask_limit=3,
        mass_shatter_unique_fragment_target=7,
        mass_shatter_ranked_mask_budget=9,
    )

    assert model.model.max_masks == 3


def test_mass_shatter_pl_uses_map_relative_mask_probability() -> None:
    model, _ = _build_pl(mass_shatter_min_mask_relative_probability=0.05)

    assert model.model.min_mask_relative_probability == pytest.approx(0.05)


def test_mass_shatter_pl_builds_variant_b_from_config() -> None:
    model, _ = _build_pl(
        mass_shatter_proposal_anchored=True,
        mass_shatter_log_base_ce=True,
        mass_shatter_base_ce_weight=0.2,
    )

    assert not model.model.intensity_head.use_base_probability_feature
    assert model.model.return_base_spectrum
    assert "base_spec_ce" in model.loss_names


def test_mass_shatter_pl_detaches_base_ion_h_from_config() -> None:
    model, _ = _build_pl(
        mass_shatter_log_base_ce=True,
        mass_shatter_detach_base_ion_h=True,
    )

    assert model.model.detach_base_ion_h


def test_mass_shatter_pl_builds_proposal_selection_only_from_config() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
    )

    assert model.model.proposal_selection_only
    assert "proposal_loss" in model.loss_names


def test_mass_shatter_ignores_legacy_oos_config() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_learned_oos=True,
        mass_shatter_learned_oos_initial_probability=0.2,
    )

    assert not model.hparams.predict_oos
    assert not model.hparams.oos_loss
    assert not model.hparams.mass_shatter_learned_oos
    assert "true_oos_prob" not in model.hparams.auxiliary_scores
    assert "true_oos_e" not in model.hparams.auxiliary_scores
    assert model.model.intensity_head.oos_head is None
    assert "oos_ce" not in model.loss_names


def test_mass_shatter_pl_builds_separate_proposal_encoder_from_config() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_separate_proposal_encoder=True,
    )

    assert model.model.separate_proposal_encoder
    assert model.model.proposal_encoder is not None
    assert model.model.proposal_encoder is not model.model.precursor_encoder


def test_mass_shatter_eval_forward_omits_cross_ev_training_context(monkeypatch) -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_share_proposal_across_ev=True,
        cross_ev_fragment_loss_weight=0.1,
    )
    monkeypatch.setattr(
        model.model,
        "forward",
        lambda **_kwargs: {
            "pred_mzs": th.tensor([100.0]),
            "pred_logprobs": th.tensor([0.0]),
            "pred_batch_idxs": th.tensor([0]),
            "cross_ev_candidate_mzs": th.tensor([100.0]),
            "cross_ev_candidate_fragment_idxs": th.tensor([0]),
            "cross_ev_fragment_logprobs": th.tensor([0.0]),
            "cross_ev_fragment_batch_idxs": th.tensor([0]),
        },
    )
    model.eval()

    output = model.forward(
        mol_pyg=None,
        spec_prec_type=th.tensor([0]),
        batch_size=1,
        group_id=th.tensor([7]),
    )

    assert "cross_ev_enabled" not in output
    assert "cross_ev_group_ids" not in output
    assert "cross_ev_ce_values" not in output
    assert "cross_ev_candidate_mzs" not in output
    assert "cross_ev_candidate_fragment_idxs" not in output
    assert "cross_ev_fragment_logprobs" not in output
    assert "cross_ev_fragment_batch_idxs" not in output


def test_proposal_selection_only_requires_its_own_supervision() -> None:
    with pytest.raises(ValueError, match="requires a positive"):
        _build_pl(
            mass_shatter_unique_fragment_mode=True,
            mass_shatter_proposal_anchored=True,
            mass_shatter_proposal_selection_only=True,
            mass_shatter_proposal_loss_weight=0.0,
        )


def test_proposal_anchored_requires_base_ce_supervision() -> None:
    with pytest.raises(ValueError, match="requires a positive"):
        _build_pl(
            mass_shatter_proposal_anchored=True,
            mass_shatter_base_ce_weight=0.0,
        )


def test_proposal_coverage_loss_counts_each_fragment_once_per_peak() -> None:
    fragment_logprobs = th.log(th.tensor([0.4, 0.3], requires_grad=True))
    coverage_loss = _proposal_coverage_loss(
        true_mzs=th.tensor([100.0]),
        true_logprobs=th.tensor([0.0]),
        true_batch_idxs=th.tensor([0]),
        candidate_mzs=th.tensor([100.0, 100.0, 200.0]),
        candidate_fragment_idxs=th.tensor([0, 0, 1]),
        fragment_logprobs=fragment_logprobs,
        fragment_batch_idxs=th.tensor([0, 0]),
        batch_size=1,
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )

    assert coverage_loss.tolist() == pytest.approx([-math.log(0.4)])


def test_cross_ev_fragment_delta_loss_tracks_observed_change() -> None:
    fragment_probabilities = th.tensor([0.7, 0.3, 0.4, 0.6], requires_grad=True)
    loss, eligible = _cross_ev_fragment_delta_loss(
        true_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        true_logprobs=th.log(th.tensor([0.8, 0.2, 0.3, 0.7])),
        true_batch_idxs=th.tensor([0, 0, 1, 1]),
        candidate_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        candidate_fragment_idxs=th.tensor([0, 1, 2, 3]),
        fragment_logprobs=fragment_probabilities.log(),
        fragment_batch_idxs=th.tensor([0, 0, 1, 1]),
        group_ids=th.tensor([9, 9]),
        ce_values=th.tensor([20.0, 40.0]),
        batch_size=2,
        tolerance=1.0e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )

    expected_pair_loss = ((th.tensor([-0.3, 0.3]) - th.tensor([-0.5, 0.5])) ** 2).mean()
    assert eligible.tolist() == [1.0, 1.0]
    assert th.allclose(loss, th.full((2,), 0.5 * expected_pair_loss))
    loss.sum().backward()
    assert fragment_probabilities.grad is not None
    assert fragment_probabilities.grad.abs().sum() > 0


def test_cross_ev_fragment_delta_loss_normalizes_proposal_coverage() -> None:
    loss, eligible = _cross_ev_fragment_delta_loss(
        true_mzs=th.tensor([100.0, 900.0, 100.0, 200.0]),
        true_logprobs=th.log(th.tensor([0.8, 0.2, 0.3, 0.7])),
        true_batch_idxs=th.tensor([0, 0, 1, 1]),
        candidate_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        candidate_fragment_idxs=th.tensor([0, 1, 2, 3]),
        fragment_logprobs=th.log(th.tensor([0.999, 0.001, 0.299, 0.701])),
        fragment_batch_idxs=th.tensor([0, 0, 1, 1]),
        group_ids=th.tensor([9, 9]),
        ce_values=th.tensor([20.0, 40.0]),
        batch_size=2,
        tolerance=1.0e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )

    assert eligible.tolist() == [1.0, 1.0]
    assert th.allclose(loss, th.zeros(2), atol=1.0e-10)


def test_cross_ev_fragment_delta_loss_skips_missing_coverage_endpoint() -> None:
    loss, eligible = _cross_ev_fragment_delta_loss(
        true_mzs=th.tensor([900.0, 100.0]),
        true_logprobs=th.zeros(2),
        true_batch_idxs=th.tensor([0, 1]),
        candidate_mzs=th.tensor([100.0, 100.0]),
        candidate_fragment_idxs=th.tensor([0, 1]),
        fragment_logprobs=th.zeros(2),
        fragment_batch_idxs=th.tensor([0, 1]),
        group_ids=th.tensor([9, 9]),
        ce_values=th.tensor([20.0, 40.0]),
        batch_size=2,
        tolerance=1.0e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )

    assert eligible.tolist() == [0.0, 0.0]
    assert loss.tolist() == [0.0, 0.0]


def test_selection_only_loss_combines_spectrum_and_proposal_objectives() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
    )
    proposal_fragment_probabilities = th.tensor([0.4, 0.3], requires_grad=True)
    proposal_fragment_logprobs = th.log(proposal_fragment_probabilities)
    actual = model.loss_fn(
        true_mzs=th.tensor([100.0]),
        true_logprobs=th.tensor([0.0]),
        true_batch_idxs=th.tensor([0]),
        pred_mzs=th.tensor([100.0, 200.0]),
        pred_logprobs=th.log(th.tensor([0.8, 0.1])),
        pred_batch_idxs=th.tensor([0, 0]),
        proposal_candidate_mzs=th.tensor([100.0, 100.0, 200.0]),
        proposal_candidate_fragment_idxs=th.tensor([0, 0, 1]),
        proposal_fragment_logprobs=proposal_fragment_logprobs,
        proposal_fragment_batch_idxs=th.tensor([0, 0]),
        batch_size=1,
        **_loss_diagnostics(),
    )

    assert actual["proposal_loss"].tolist() == pytest.approx([-math.log(0.4)])
    assert actual["loss"].tolist() == pytest.approx(
        (actual["primary_loss"] + 0.1 * actual["proposal_loss"]).tolist()
    )
    actual["loss"].sum().backward()
    assert proposal_fragment_probabilities.grad is not None
    assert proposal_fragment_probabilities.grad.abs().sum() > 0.0


def test_mass_shatter_pl_builds_unique_fragment_mode_from_config() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_target=4,
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_ranked_mask_budget=16,
    )

    assert model.model.unique_fragment_mode
    assert model.model.max_masks == 4
    assert model.model.max_enumerated_masks == 16


def test_mass_shatter_transfer_retains_static_graph_on_cpu() -> None:
    model, _ = _build_pl()
    mol_pyg = Batch.from_data_list(
        [
            Data(
                x=th.ones((2, 1)),
                chemical_bond_index=th.tensor([[0], [1]], dtype=th.long),
            )
        ]
    )

    transferred = model.transfer_batch_to_device(
        {"mol_pyg": mol_pyg},
        th.device("cpu"),
        0,
    )
    static_graph = transferred["mass_shatter_static_graph"]

    assert static_graph.chemical_bond_index.device.type == "cpu"
    assert static_graph.atom_batch_indices.tolist() == [0, 0]
    assert static_graph.atom_ptr.tolist() == [0, 2]


def test_mass_shatter_transfer_precomputes_smarts_templates_when_enabled() -> None:
    model, _ = _build_pl(mass_shatter_smarts_templates=True)
    smiles = "c1ccccc1S(=O)c1ccccc1"
    mol = Chem.MolFromSmiles(smiles)
    chemical_bond_index = th.tensor(
        [
            [bond.GetBeginAtomIdx() for bond in mol.GetBonds()],
            [bond.GetEndAtomIdx() for bond in mol.GetBonds()],
        ],
        dtype=th.long,
    )
    mol_pyg = Batch.from_data_list(
        [
            Data(
                x=th.ones((mol.GetNumAtoms(), 1)),
                chemical_bond_index=chemical_bond_index,
            )
        ]
    )

    transferred = model.transfer_batch_to_device(
        {"mol_pyg": mol_pyg, "mol_smiles": np.asarray([smiles])},
        th.device("cpu"),
        0,
    )

    templates = transferred["mass_shatter_static_graph"].smarts_templates
    assert templates is not None
    assert any(template.rule_name == "sulfoxide_so_loss" for template in templates[0])


def test_mass_shatter_primary_loss_uses_candidate_only_sparse_cross_entropy() -> None:
    model, config = _build_pl(
        mass_shatter_log_base_ce=True,
        mass_shatter_base_ce_weight=0.2,
        oos_loss=True,
    )
    true_mzs = th.tensor([100.0, 150.0])
    true_logprobs = th.log(th.tensor([0.6, 0.4]))
    true_batch_indices = th.tensor([0, 0], dtype=th.long)
    pred_mzs = th.tensor([100.0, 180.0])
    pred_logprobs = th.log(th.tensor([0.5, 0.3]))
    base_pred_logprobs = th.log(th.tensor([0.3, 0.5])).requires_grad_()
    pred_batch_indices = th.tensor([0, 0], dtype=th.long)
    actual = model.loss_fn(
        true_mzs=true_mzs,
        true_logprobs=true_logprobs,
        true_batch_idxs=true_batch_indices,
        pred_mzs=pred_mzs,
        pred_logprobs=pred_logprobs,
        base_pred_logprobs=base_pred_logprobs,
        pred_batch_idxs=pred_batch_indices,
        retained_mask_probability=th.tensor([0.8]),
        selected_mask_count=th.tensor([2]),
        exploration_mask_count=th.tensor([1]),
        exploration_mean_cut_count=th.tensor([3.0]),
        component_pathway_count=th.tensor([4]),
        mean_mask_cut_count=th.tensor([2.0]),
        proposal_single_cut_fraction=th.tensor([0.25]),
        proposal_mixture_entropy=th.tensor([0.6]),
        proposal_mixture_effective_count=th.tensor([1.8]),
        proposal_mixture_top1_weight=th.tensor([0.7]),
        unique_component_count=th.tensor([3]),
        components_per_mask=th.tensor([2.0]),
        unique_components_per_mask=th.tensor([1.5]),
        latent_pathways_per_unique_fragment=th.tensor([4.0 / 3.0]),
        unique_fragment_target_fill_fraction=th.tensor([0.75]),
        unique_fragment_target_reached=th.tensor([0.0]),
        unique_fragment_effective_count=th.tensor([2.5]),
        unique_fragment_top1_mass_fraction=th.tensor([0.6]),
        duplicate_pathway_probability_fraction=th.tensor([0.2]),
        productive_mask_fraction=th.tensor([0.5]),
        duplicate_component_pathway_fraction=th.tensor([0.25]),
        mask_cut_fraction_0=th.tensor([0.25]),
        mask_cut_fraction_1=th.tensor([0.25]),
        mask_cut_fraction_2=th.tensor([0.5]),
        mask_cut_fraction_3=th.tensor([0.0]),
        mask_cut_fraction_4plus=th.tensor([0.0]),
        candidate_input_transfer_batch_ms=th.tensor([1.0]),
        candidate_mask_selection_batch_ms=th.tensor([2.0]),
        candidate_topology_batch_ms=th.tensor([3.0]),
        candidate_assembly_batch_ms=th.tensor([4.0]),
        candidate_output_transfer_batch_ms=th.tensor([5.0]),
        candidate_total_batch_ms=th.tensor([15.0]),
        candidate_total_molecule_ms=th.tensor([15.0]),
        batch_size=1,
    )
    sparse_ce_fn = get_sparse_cross_entropy_fn(
        dist=config["output_distribution"],
        vectorized=config["loss_vectorized"],
        tolerance=config["loss_tolerance_rel"],
        relative=True,
        tolerance_min_mz=config["loss_tolerance_min_mz"],
        oos_tolerance_multiple=config["oos_tolerance_multiple"],
        gaussian_renormalize=config["gaussian_renormalize"],
        pm_tolerance_multiple=config["pm_tolerance_multiple"],
        loss_batch_size=config["loss_batch_size"],
    )
    expected_spec_ce, _, expected_unmatched, _ = sparse_ce_fn(
        true_mzs,
        true_logprobs,
        true_batch_indices,
        pred_mzs,
        pred_logprobs,
        pred_batch_indices,
        th.zeros(1),
    )
    expected_base_spec_ce, _, _, _ = sparse_ce_fn(
        true_mzs,
        true_logprobs,
        true_batch_indices,
        pred_mzs,
        base_pred_logprobs,
        pred_batch_indices,
        th.zeros(1),
    )

    assert th.allclose(actual["spec_ce"], expected_spec_ce)
    assert th.allclose(actual["base_spec_ce"], expected_base_spec_ce)
    assert "oos_ce" not in actual
    assert th.allclose(
        actual["loss"],
        expected_spec_ce + 0.2 * expected_base_spec_ce,
    )
    actual["loss"].sum().backward()
    assert base_pred_logprobs.grad is not None
    assert base_pred_logprobs.grad.abs().sum() > 0.0
    assert th.allclose(actual["candidate_wrecall"], 1.0 - expected_unmatched.exp())
    assert actual["candidate_wrecall"].tolist() == pytest.approx([0.6])
    assert actual["retained_mask_probability"].tolist() == pytest.approx([0.8])
    assert actual["selected_mask_count"].tolist() == [2.0]
    assert actual["exploration_mask_count"].tolist() == [1.0]
    assert actual["exploration_mean_cut_count"].tolist() == [3.0]
    assert actual["component_pathway_count"].tolist() == [4.0]
    assert actual["mean_mask_cut_count"].tolist() == [2.0]
    assert actual["proposal_single_cut_fraction"].tolist() == [0.25]
    assert actual["proposal_mixture_entropy"].tolist() == pytest.approx([0.6])
    assert actual["proposal_mixture_effective_count"].tolist() == pytest.approx([1.8])
    assert actual["proposal_mixture_top1_weight"].tolist() == pytest.approx([0.7])
    assert actual["unique_component_count"].tolist() == [3.0]
    assert actual["components_per_mask"].tolist() == [2.0]
    assert actual["unique_components_per_mask"].tolist() == [1.5]
    assert actual["latent_pathways_per_unique_fragment"].tolist() == pytest.approx([4.0 / 3.0])
    assert actual["unique_fragment_target_fill_fraction"].tolist() == [0.75]
    assert actual["unique_fragment_target_reached"].tolist() == [0.0]
    assert actual["unique_fragment_effective_count"].tolist() == [2.5]
    assert actual["unique_fragment_top1_mass_fraction"].tolist() == pytest.approx([0.6])
    assert actual["duplicate_pathway_probability_fraction"].tolist() == pytest.approx([0.2])
    assert actual["productive_mask_fraction"].tolist() == [0.5]
    assert actual["duplicate_component_pathway_fraction"].tolist() == [0.25]
    assert actual["candidate_total_batch_ms"].tolist() == [15.0]


def test_weighted_recall_and_precision_ignore_ragged_padding() -> None:
    model, _ = _build_pl(auxiliary_scores=["wrecall", "wprecision"])

    actual = model.metric_fn(
        true_mzs=th.tensor([100.0, 200.0, 400.0]),
        true_ints=th.tensor([0.6, 0.4, 1.0]),
        true_batch_idxs=th.tensor([0, 0, 1]),
        pred_mzs=th.tensor([100.0, 400.0, 500.0]),
        pred_ints=th.tensor([0.7, 0.8, 0.2]),
        pred_batch_idxs=th.tensor([0, 1, 1]),
        candidate_mzs_by_cut=th.tensor([100.0, 400.0, 500.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 1, 1]),
        candidate_cut_counts=th.tensor([1, 2, 2]),
        true_prec_mzs=th.tensor([600.0, 600.0]),
        split="val",
    )

    assert th.isfinite(actual["wrecall"]).all()
    assert th.isfinite(actual["wprecision"]).all()
    assert actual["wrecall"].tolist() == pytest.approx([0.6, 1.0])
    assert actual["wprecision"].tolist() == pytest.approx([0.7, 0.8])
    assert actual["candidate_recall_cut_1"].tolist() == pytest.approx([0.5, 0.0])
    assert actual["candidate_wrecall_cut_1"].tolist() == pytest.approx([0.6, 0.0])
    assert actual["candidate_precision_cut_1"].tolist() == pytest.approx([1.0, 0.0])
    assert actual["candidate_recall_cut_2"].tolist() == pytest.approx([0.0, 1.0])
    assert actual["candidate_wrecall_cut_2"].tolist() == pytest.approx([0.0, 1.0])
    assert actual["candidate_precision_cut_2"].tolist() == pytest.approx([0.0, 0.5])


def test_candidate_coverage_by_cut_count_uses_per_instrument_tolerance() -> None:
    actual = _candidate_coverage_by_cut_count(
        true_mzs=th.tensor([100.0, 100.0]),
        true_ints=th.tensor([1.0, 1.0]),
        true_batch_idxs=th.tensor([0, 1]),
        candidate_mzs=th.tensor([100.01, 100.01]),
        candidate_batch_idxs=th.tensor([0, 1]),
        candidate_cut_counts=th.tensor([3, 3]),
        batch_size=2,
        tolerance=1.0e-5,
        relative=True,
        tolerance_min_mz=200.0,
        tol_per_sample=th.tensor([1.0e-5, 2.0e-5]),
        min_mz_per_sample=th.tensor([200.0, 750.0]),
    )

    assert actual["candidate_recall_cut_3"].tolist() == pytest.approx([0.0, 1.0])
    assert actual["candidate_wrecall_cut_3"].tolist() == pytest.approx([0.0, 1.0])
    assert actual["candidate_precision_cut_3"].tolist() == pytest.approx([0.0, 1.0])
