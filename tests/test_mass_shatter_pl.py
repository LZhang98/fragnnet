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
        mass_shatter_max_masks=2,
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
    model, config = _build_pl()
    true_mzs = th.tensor([100.0, 150.0])
    true_logprobs = th.log(th.tensor([0.6, 0.4]))
    true_batch_indices = th.tensor([0, 0], dtype=th.long)
    pred_mzs = th.tensor([100.0, 180.0])
    pred_logprobs = th.log(th.tensor([0.5, 0.3]))
    pred_batch_indices = th.tensor([0, 0], dtype=th.long)
    pred_oos_logprobs = th.log(th.tensor([0.2]))

    actual = model.loss_fn(
        true_mzs=true_mzs,
        true_logprobs=true_logprobs,
        true_batch_idxs=true_batch_indices,
        pred_mzs=pred_mzs,
        pred_logprobs=pred_logprobs,
        pred_batch_idxs=pred_batch_indices,
        pred_oos_logprobs=pred_oos_logprobs,
            retained_mask_probability=th.tensor([0.8]),
            selected_mask_count=th.tensor([2]),
            exploration_mask_count=th.tensor([1]),
            exploration_mean_cut_count=th.tensor([3.0]),
            component_pathway_count=th.tensor([4]),
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

    assert th.allclose(actual["ios_ce"], expected_ios)
    assert th.allclose(actual["oos_ce"], expected_oos)
    assert th.allclose(actual["spec_ce"], expected_ios + expected_oos)
    assert th.allclose(actual["candidate_wrecall"], 1.0 - expected_true_oos.exp())
    assert actual["candidate_wrecall"].tolist() == pytest.approx([0.6])
    assert actual["retained_mask_probability"].tolist() == pytest.approx([0.8])
    assert actual["selected_mask_count"].tolist() == [2.0]
    assert actual["exploration_mask_count"].tolist() == [1.0]
    assert actual["exploration_mean_cut_count"].tolist() == [3.0]
    assert actual["component_pathway_count"].tolist() == [4.0]
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
        true_prec_mzs=th.tensor([600.0, 600.0]),
        split="val",
    )

    assert th.isfinite(actual["wrecall"]).all()
    assert th.isfinite(actual["wprecision"]).all()
    assert actual["wrecall"].tolist() == pytest.approx([0.6, 1.0])
    assert actual["wprecision"].tolist() == pytest.approx([0.7, 0.8])


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
