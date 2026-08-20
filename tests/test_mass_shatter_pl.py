"""Integration tests for MASShatter's SpectrumPL training wrapper."""

from pathlib import Path

import pytest
import torch as th
import yaml
from torch_geometric.data import Batch, Data

from fragnnet.dataset import SpecMolDataset
from fragnnet.model.bond_fragment_heads import BondFocusedBreakHead
from fragnnet.model.loss import get_sparse_cross_entropy_fn
from fragnnet.pl_model import MassShatterPL, SpectrumPL
from fragnnet.pl_model.mass_shatter_pl import _candidate_coverage_by_cut_count
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
        mass_shatter_explore_masks=1,
        mass_shatter_max_h_transfer=1,
    )
    config.update(config_overrides)
    return MassShatterPL(**config), config


def test_mass_shatter_registration_uses_spectrum_only_dataset() -> None:
    assert get_model_cls("mass_shatter") is MassShatterPL
    assert get_dataset_cls("mass_shatter") is SpecMolDataset
    assert issubclass(MassShatterPL, SpectrumPL)


def test_mass_shatter_pl_builds_configured_bond_focused_head() -> None:
    model, _ = _build_pl(mass_shatter_bond_num_layers=2)

    assert isinstance(model.model.bond_break_head, BondFocusedBreakHead)


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


def test_mass_shatter_pl_builds_variant_b_from_config() -> None:
    model, _ = _build_pl(
        mass_shatter_proposal_anchored=True,
        mass_shatter_log_base_ce=True,
        mass_shatter_base_ce_weight=0.2,
    )

    assert not model.model.intensity_head.use_base_probability_feature
    assert model.model.return_base_spectrum
    assert "base_spec_ce" in model.loss_names


def test_proposal_anchored_requires_base_ce_supervision() -> None:
    with pytest.raises(ValueError, match="requires a positive"):
        _build_pl(
            mass_shatter_proposal_anchored=True,
            mass_shatter_base_ce_weight=0.0,
        )


def test_mass_shatter_pl_builds_unique_fragment_mode_from_config() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_target=4,
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_ranked_mask_budget=16,
        mass_shatter_explore_masks=0,
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


def test_mass_shatter_primary_loss_matches_fragnnet_sparse_cross_entropy() -> None:
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
    pred_oos_logprobs = th.log(th.tensor([0.2]))

    actual = model.loss_fn(
        true_mzs=true_mzs,
        true_logprobs=true_logprobs,
        true_batch_idxs=true_batch_indices,
        pred_mzs=pred_mzs,
        pred_logprobs=pred_logprobs,
        base_pred_logprobs=base_pred_logprobs,
        pred_batch_idxs=pred_batch_indices,
        pred_oos_logprobs=pred_oos_logprobs,
        retained_mask_probability=th.tensor([0.8]),
        selected_mask_count=th.tensor([2]),
        exploration_mask_count=th.tensor([1]),
        exploration_mean_cut_count=th.tensor([3.0]),
        component_pathway_count=th.tensor([4]),
        mean_mask_cut_count=th.tensor([2.0]),
        proposal_single_cut_fraction=th.tensor([0.25]),
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
    expected_ios, expected_oos, expected_true_oos, _ = sparse_ce_fn(
        true_mzs,
        true_logprobs,
        true_batch_indices,
        pred_mzs,
        pred_logprobs,
        pred_batch_indices,
        pred_oos_logprobs,
    )
    expected_base_ios, expected_base_oos, _, _ = sparse_ce_fn(
        true_mzs,
        true_logprobs,
        true_batch_indices,
        pred_mzs,
        base_pred_logprobs,
        pred_batch_indices,
        pred_oos_logprobs,
    )

    assert th.allclose(actual["ios_ce"], expected_ios)
    assert th.allclose(actual["oos_ce"], expected_oos)
    assert th.allclose(actual["spec_ce"], expected_ios + expected_oos)
    assert th.allclose(actual["base_ios_ce"], expected_base_ios)
    assert th.allclose(actual["base_oos_ce"], expected_base_oos)
    assert th.allclose(actual["base_spec_ce"], expected_base_ios + expected_base_oos)
    assert th.allclose(
        actual["loss"],
        expected_ios + expected_oos + 0.2 * (expected_base_ios + expected_base_oos),
    )
    actual["loss"].sum().backward()
    assert base_pred_logprobs.grad is not None
    assert base_pred_logprobs.grad.abs().sum() > 0.0
    assert th.allclose(actual["candidate_wrecall"], 1.0 - expected_true_oos.exp())
    assert actual["candidate_wrecall"].tolist() == pytest.approx([0.6])
    assert actual["retained_mask_probability"].tolist() == pytest.approx([0.8])
    assert actual["selected_mask_count"].tolist() == [2.0]
    assert actual["exploration_mask_count"].tolist() == [1.0]
    assert actual["exploration_mean_cut_count"].tolist() == [3.0]
    assert actual["component_pathway_count"].tolist() == [4.0]
    assert actual["mean_mask_cut_count"].tolist() == [2.0]
    assert actual["proposal_single_cut_fraction"].tolist() == [0.25]
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


def test_exploration_cut_count_mean_excludes_molecules_without_samples() -> None:
    model, _ = _build_pl()
    model.metric_names = {"exploration_mean_cut_count"}
    model.auxiliary_metric_names = set()
    model._reset_running_metrics("train")
    model._accumulate_running_metrics(
        {
            "exploration_mean_cut_count": th.tensor([0.0, 2.0, 3.0]),
            "exploration_mask_count": th.tensor([0.0, 1.0, 2.0]),
            "spec_per_group": th.ones(3),
            "spec_per_mol": th.ones(3),
            "group_per_mol": th.ones(3),
        },
        "train",
    )

    mean_metrics, _ = model._reduce_running_metrics("train")

    assert mean_metrics["exploration_mean_cut_count"].item() == pytest.approx(8.0 / 3.0)
