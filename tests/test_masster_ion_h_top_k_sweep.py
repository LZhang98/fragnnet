"""Tests for checkpoint-only MASSter ion/H top-k sweeps."""

import pytest
import torch as th
from lightning import LightningModule
from rdkit import Chem
from torch_geometric.data import Batch

from fragnnet.models.masster.candidates.smarts_templates import build_masster_templates
from fragnnet.models.masster.masster_pl import MassterPL
from fragnnet.utils.feat_utils import get_mol_graph
from scripts.eval.masster_ion_h_top_k_sweep import (
    build_selections,
    build_support_selections,
    evaluate_selection,
    metric_names,
    validate_fragment_relative_thresholds,
    validate_joint_relative_thresholds,
    validate_top_ks,
)
from scripts.eval.masster_mask_budget_sweep import evaluate_budget


@pytest.fixture
def smarts_sweep_model() -> MassterPL:
    # Exercise the real transfer hook without constructing unused training modules.
    model = MassterPL.__new__(MassterPL)
    LightningModule.__init__(model)
    model.save_hyperparameters({"masster_smarts_templates": True})
    return model


@pytest.mark.parametrize("evaluate", [evaluate_selection, evaluate_budget])
def test_evaluate_selection_prepares_smarts_templates(
    smarts_sweep_model: MassterPL,
    monkeypatch: pytest.MonkeyPatch,
    evaluate,
) -> None:
    smiles = "c1ccccc1S(=O)c1ccccc1"
    molecule = Chem.MolFromSmiles(smiles)
    batch = {
        "mol_pyg": Batch.from_data_list(
            [
                get_mol_graph(molecule, ["a_onehot", "a_degree"], ["b_degree"], 0, True),
            ]
        ),
        "mol_smiles": [smiles],
        "mol_graph_binary": [molecule.ToBinary()],
    }
    expected_templates = build_masster_templates(molecule)
    assert expected_templates

    def common_step(transferred: dict, split: str, log: bool) -> dict:
        static_graph = transferred["masster_static_graph"]
        assert static_graph.chemical_bond_index.device.type == "cpu"
        assert static_graph.smarts_templates == (expected_templates,)
        assert split == "val"
        assert log is False
        return {
            "templates": th.tensor([len(static_graph.smarts_templates[0])]),
            "spec_per_group": th.ones(1),
            "spec_per_mol": th.ones(1),
            "group_per_mol": th.ones(1),
        }

    monkeypatch.setattr(smarts_sweep_model, "_common_step", common_step)
    result = evaluate(
        smarts_sweep_model,
        [batch],
        th.device("cpu"),
        "val",
        ("templates",),
        "none",
        -1,
    )

    assert result["templates"] == len(expected_templates)
    assert result["num_spectra"] == 1


@pytest.mark.parametrize("evaluate", [evaluate_selection, evaluate_budget])
@pytest.mark.parametrize("error_type", [ValueError, RuntimeError])
def test_sweep_skips_invalid_batch_and_continues(
    evaluate,
    caplog: pytest.LogCaptureFixture,
    error_type,
) -> None:
    class Model:
        def transfer_batch_to_device(self, batch, device, dataloader_idx):
            return batch

        def _common_step(self, batch, split, log):
            if batch == "invalid":
                raise error_type("invalid spectrum")
            return {
                "score": th.tensor([0.75]),
                "spec_per_group": th.ones(1),
                "spec_per_mol": th.ones(1),
                "group_per_mol": th.ones(1),
            }

    if error_type is RuntimeError:
        with pytest.raises(RuntimeError, match="invalid spectrum"):
            evaluate(Model(), ["invalid"], th.device("cpu"), "val", ("score",), "none", -1)
        return
    result = evaluate(
        Model(), ["invalid", "valid"], th.device("cpu"), "val", ("score",), "none", -1
    )

    assert result["num_spectra"] == 1
    assert result["num_skipped_batches"] == 1
    assert result["score"] == 0.75
    assert "Skipping invalid evaluation batch 0" in caplog.text


def test_validate_top_ks_keeps_dense_reference_and_requested_order() -> None:
    assert validate_top_ks([0, 1, 2, 3]) == [0, 1, 2, 3]


@pytest.mark.parametrize("top_ks", [[], [-1, 0], [0, 1, 1], [1, 2]])
def test_validate_top_ks_rejects_invalid_values(top_ks: list[int]) -> None:
    with pytest.raises(ValueError):
        validate_top_ks(top_ks)


def test_validate_fragment_relative_thresholds_keeps_requested_order() -> None:
    assert validate_fragment_relative_thresholds([0.0, 0.001, 0.01]) == [0.0, 0.001, 0.01]


@pytest.mark.parametrize("thresholds", [[], [-0.1, 0.0], [0.0, 1.1], [0.0, 0.1, 0.1], [0.1]])
def test_validate_fragment_relative_thresholds_rejects_invalid_values(
    thresholds: list[float],
) -> None:
    with pytest.raises(ValueError):
        validate_fragment_relative_thresholds(thresholds)


def test_validate_joint_relative_thresholds_keeps_requested_order() -> None:
    assert validate_joint_relative_thresholds([0.001, 0.01, 0.1]) == [
        0.001,
        0.01,
        0.1,
    ]


@pytest.mark.parametrize("thresholds", [[0.0], [-0.1], [1.1], [0.1, 0.1]])
def test_validate_joint_relative_thresholds_rejects_invalid_values(
    thresholds: list[float],
) -> None:
    with pytest.raises(ValueError):
        validate_joint_relative_thresholds(thresholds)


def test_metric_names_only_requests_configured_auxiliary_scores() -> None:
    names = metric_names(
        {
            "auxiliary_scores": ["cos_sim", "precision", "recall"],
            "eval_mz_bin_res": [0.01],
        }
    )

    assert "precision" in names
    assert "recall" in names
    assert "cos_sim_0.01" in names
    assert "wprecision" not in names
    assert "wrecall" not in names
    assert "fragment_count_before_cutoff" in names
    assert "fragment_count_after_cutoff" in names
    assert "retained_fragment_probability" in names
    assert "fragment_cutoff_pruned_fraction" in names


def test_build_selections_crosses_fragment_thresholds_with_top_ks() -> None:
    selections = build_selections(
        fragment_relative_thresholds=[0.0, 0.01],
        top_ks=[0, 4],
        joint_relative_thresholds=[],
    )

    assert selections == [
        ("dense", 0.0, 0, 0.0),
        ("top_k_4", 0.0, 4, 0.0),
        ("fragment_relative_0.01", 0.01, 0, 0.0),
        ("fragment_relative_0.01_top_k_4", 0.01, 4, 0.0),
    ]


def test_support_threshold_rules_are_separate_from_intensity_and_top_k() -> None:
    assert build_support_selections([0.0, 0.001], [0.2, 0.5]) == [
        ("support_absolute_0.2", 0.0, 0, 0.2),
        ("support_absolute_0.5", 0.0, 0, 0.5),
        ("fragment_relative_0.001_support_absolute_0.2", 0.001, 0, 0.2),
        ("fragment_relative_0.001_support_absolute_0.5", 0.001, 0, 0.5),
    ]


@pytest.mark.parametrize("thresholds", [[0.0], [float("nan")], [1.1], [0.5, 0.5]])
def test_support_threshold_rules_reject_invalid_floors(thresholds: list[float]) -> None:
    with pytest.raises(ValueError):
        build_support_selections([0.0], thresholds)
