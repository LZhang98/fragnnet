"""Integration tests for MASShatter's SpectrumPL training wrapper."""

import math
import runpy
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch as th
import yaml
from rdkit import Chem
from torch_geometric.data import Batch, Data

import fragnnet.runner as runner
from fragnnet.dataset import SpecMolDataset, SpecMolFragDataset
from fragnnet.frag.mass_shatter_candidates import MassShatterStaticGraph
from fragnnet.frag.mass_shatter_templates import build_mass_shatter_templates
from fragnnet.model.bond_fragment_heads import BondFocusedBreakHead
from fragnnet.model.loss import get_sparse_cross_entropy_fn
from fragnnet.model.nn_blocks import GINEGraphormer
from fragnnet.pl_model import MassShatterPL, SpectrumPL
from fragnnet.pl_model.mass_shatter_pl import (
    _balanced_binary_support_loss,
    _candidate_coverage_by_cut_count,
    _candidate_weighted_recall,
    _cross_ev_fragment_delta_loss,
    _ion_h_required_k_metrics,
    _ion_mode_support_loss,
    _observational_channel_targets,
    _proposal_coverage_loss,
    _shared_group_proposal_coverage_loss,
)
from fragnnet.utils import script_utils
from fragnnet.utils.feat_utils import get_mol_graph
from fragnnet.utils.pl_utils import get_pl_hparams
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls
from fragnnet.utils.spec_utils import calculate_match_mzs


def _build_pl(**config_overrides: object) -> tuple[MassShatterPL, dict]:
    template_path = Path(__file__).parents[1] / "config" / "models_template"
    # Resolve through load_config so the mass_shatter overlay is applied, as a real run does.
    config = runner.load_config(str(template_path), None, None, model_type="mass_shatter")
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


def test_attention_topology_migrates_legacy_checkpoints_and_requires_a_fresh_key() -> None:
    """Old checkpoints predate the key; fresh configs must declare it explicitly."""
    legacy, config = _build_pl()
    del config["mass_shatter_intensity_transformer_attention"]
    checkpoint = {"hyper_parameters": dict(config)}
    migrated = get_pl_hparams(checkpoint)

    assert migrated["mass_shatter_intensity_transformer_attention"] == "induced"
    assert "mass_shatter_intensity_transformer_attention" not in checkpoint["hyper_parameters"]
    restored = MassShatterPL(**migrated)
    restored.load_state_dict(legacy.state_dict(), strict=True)
    assert restored.model.intensity_head.attention_type == "induced"
    assert restored.model.intensity_head.inducing_tokens is not None

    # An explicit key survives migration untouched, and a missing one must not default.
    migrated["mass_shatter_intensity_transformer_attention"] = "full"
    assert (
        get_pl_hparams({"hyper_parameters": migrated})[
            "mass_shatter_intensity_transformer_attention"
        ]
        == "full"
    )
    with pytest.raises(AttributeError, match="mass_shatter_intensity_transformer_attention"):
        MassShatterPL(**config)


def test_attention_topology_rejects_unknown_values() -> None:
    # The head validates during model construction, which precedes the loss-setup check.
    with pytest.raises(ValueError, match="unknown attention_type: inducing"):
        _build_pl(mass_shatter_intensity_transformer_attention="inducing")


def test_nonzero_atom_decoder_layers_fail_at_setup() -> None:
    """The removed atom decoder must not be silently ignored by an old config."""
    with pytest.raises(ValueError, match="atom_decoder_num_layers must be 0"):
        _build_pl(mass_shatter_intensity_atom_decoder_num_layers=4)


def test_merged_independent_distance_configs_enable_requested_model() -> None:
    config_dir = (
        Path(__file__).parents[1] / "config" / "massshatter" / "merged" / "channel_selection"
    )
    config_paths = sorted(config_dir.glob("nist20_merged_independent_*_hattr_*.yml"))

    assert len(config_paths) == 8
    for config_path in config_paths:
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file)
        assert config["mass_shatter_peak_matching"] == "hungarian"
        assert config["mass_shatter_min_ion_h_relative_probability"] == pytest.approx(0.01)
        assert config["mass_shatter_min_ion_h_absolute_probability"] == pytest.approx(0.01)
        assert config["mass_shatter_ion_h_support_binary_logits"] is True
        assert config["mass_shatter_apply_channel_selection_in_training"] is False
        assert config["mass_shatter_channel_support_use_intensity_prior"] is False
        assert config["mass_shatter_proposal_num_mixtures"] == 1
        # The atom decoder was removed on 2026-09-19; these configs no longer set the key
        # and inherit the template default of 0.
        assert "mass_shatter_intensity_atom_decoder_num_layers" not in config
        assert config["mass_shatter_proposal_ints_transform"] == "none"
        if config["loss_type"] == "kl_distance":
            assert config["mass_shatter_kl_epsilon"] == pytest.approx(1.0e-4)


def test_proposal_warmup_configs_enable_proposal_only_epochs_0_through_15() -> None:
    config_dir = Path(__file__).parents[1] / "config" / "massshatter" / "merged" / "proposal_warmup"
    config_paths = sorted(config_dir.glob("*proposal_warmup_e0_15.yml"))

    assert len(config_paths) == 3
    for config_path in config_paths:
        with config_path.open() as config_file:
            config = yaml.safe_load(config_file)
        assert config["mass_shatter_proposal_only_until_epoch"] == 15
        assert config["mass_shatter_separate_proposal_encoder"] is False
        assert "shared_proposal_encoder" in config["wandb_tags"]
        assert "proposal_warmup_epochs_0_15" in config["wandb_tags"]
        assert config["wandb_name"].endswith("_shared_proposal_warmup_e0_15")


def test_baseline_directory_contains_probability_ce_control() -> None:
    baseline_path = (
        Path(__file__).parents[1]
        / "config"
        / "massshatter"
        / "merged"
        / "baseline"
        / "nist20_merged_probability_ce_baseline.yml"
    )
    with baseline_path.open() as config_file:
        baseline_config = yaml.safe_load(config_file)
    assert baseline_config["mass_shatter_intensity_output"] == "probability"
    assert baseline_config["mass_shatter_proposal_num_mixtures"] == 1
    assert baseline_config["mass_shatter_gate_independent_intensity_with_support"] is False
    assert baseline_config["mass_shatter_separate_proposal_encoder"] is True
    assert baseline_config["mass_shatter_separate_ion_h_support_head"] is False
    assert baseline_config["mass_shatter_ion_h_support_binary_logits"] is False
    assert baseline_config["mass_shatter_min_ion_h_relative_probability"] == 0.0
    assert baseline_config["mass_shatter_min_ion_h_absolute_probability"] == 0.0
    assert baseline_config["mass_shatter_ion_h_support_loss_weight"] == 0.0
    assert baseline_config["mass_shatter_channel_diagnostics"] is False
    assert baseline_config["mass_shatter_proposal_only_until_epoch"] is None
    assert baseline_config["loss_type"] == "cross_entropy"
    assert baseline_config["wandb_name"].endswith("_baseline")


@pytest.mark.parametrize("proposal_only_until_epoch", [True, -1, 1.5])
def test_proposal_only_epoch_rejects_invalid_values(proposal_only_until_epoch: object) -> None:
    with pytest.raises(
        ValueError,
        match="mass_shatter_proposal_only_until_epoch must be null or a nonnegative integer",
    ):
        _build_pl(mass_shatter_proposal_only_until_epoch=proposal_only_until_epoch)


@pytest.mark.parametrize(
    ("config_overrides", "message"),
    [
        (
            {"mass_shatter_proposal_selection_only": False},
            "mass_shatter_proposal_selection_only=True",
        ),
        (
            {
                "mass_shatter_proposal_selection_only": True,
                "mass_shatter_separate_proposal_encoder": True,
                "mass_shatter_proposal_loss_weight": 0.0,
            },
            "positive mass_shatter_proposal_loss_weight",
        ),
    ],
)
def test_proposal_only_schedule_rejects_incompatible_model_configs(
    config_overrides: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_pl(
            mass_shatter_proposal_only_until_epoch=2,
            **config_overrides,
        )


def test_proposal_warmup_supports_smarts_templates() -> None:
    pl, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_separate_proposal_encoder=False,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_proposal_only_until_epoch=2,
        mass_shatter_smarts_templates=True,
    )

    assert pl.model.mechanism_head is not None
    assert pl.model.mechanism_head in pl._proposal_modules()

    pl._apply_staged_training(epoch=1)
    assert pl._proposal_only_training
    assert all(parameter.requires_grad for parameter in pl.model.mechanism_head.parameters())
    assert all(not parameter.requires_grad for parameter in pl.model.intensity_head.parameters())

    pl._apply_staged_training(epoch=3)
    assert not pl._proposal_only_training
    assert all(parameter.requires_grad for parameter in pl.model.mechanism_head.parameters())
    assert all(parameter.requires_grad for parameter in pl.model.intensity_head.parameters())


@pytest.mark.parametrize("separate_encoder", [False, True])
def test_smarts_proposal_warmup_isolates_gradients_across_boundary(
    separate_encoder: bool,
) -> None:
    smiles = "c1ccccc1S(=O)c1ccccc1"
    pl, config = _build_pl(
        mass_shatter_unique_fragment_target=4,
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_ranked_mask_budget=8,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_separate_proposal_encoder=separate_encoder,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_proposal_only_until_epoch=1,
        mass_shatter_smarts_templates=True,
        mass_shatter_proposal_anchored=True,
    )
    params = config["mol_params"]
    mol = Chem.MolFromSmiles(smiles)
    graph = get_mol_graph(
        mol,
        params["pyg_node_feats"],
        params["pyg_edge_feats"],
        params["pyg_pe_embed_k"],
        True,
    )
    batch = Batch.from_data_list([graph, graph])
    templates = build_mass_shatter_templates(mol)
    static_graph = MassShatterStaticGraph(
        chemical_bond_index=batch.chemical_bond_index,
        atom_batch_indices=batch.batch,
        atom_ptr=batch.ptr,
        smarts_templates=(templates, templates),
    )
    inputs = {
        "mol_pyg": batch,
        "spec_prec_type": th.tensor([0, 0]),
        "spec_inst_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([10.0, 40.0]),
        "static_graph": static_graph,
    }

    def gradient_sum(module: th.nn.Module) -> float:
        return sum(
            parameter.grad.abs().sum().item()
            for parameter in module.parameters()
            if parameter.grad is not None
        )

    pl._apply_staged_training(epoch=1)
    output = pl.model(**inputs)
    assert th.isfinite(output["proposal_fragment_logprobs"]).all()
    (-output["proposal_fragment_logprobs"].mean()).backward()
    assert gradient_sum(pl.model.mechanism_head) > 0.0
    assert gradient_sum(pl.model.intensity_head) == 0.0

    pl.model.zero_grad(set_to_none=True)
    pl._apply_staged_training(epoch=3)
    output = pl.model(**inputs)
    assert th.isfinite(output["pred_logprobs"]).all()
    (-output["proposal_fragment_logprobs"].mean() - output["pred_logprobs"].mean()).backward()
    assert gradient_sum(pl.model.mechanism_head) > 0.0
    assert gradient_sum(pl.model.intensity_head) > 0.0
    if separate_encoder:
        assert gradient_sum(pl.model.proposal_encoder) > 0.0
        assert gradient_sum(pl.model.precursor_encoder) > 0.0


def test_proposal_warmup_enables_intensity_branch_after_boundary() -> None:
    pl, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_separate_proposal_encoder=False,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_proposal_only_until_epoch=15,
        mass_shatter_separate_ion_h_support_head=True,
        mass_shatter_ion_h_support_binary_logits=True,
        mass_shatter_ion_h_support_loss_weight=0.1,
    )

    pl._apply_staged_training(epoch=15)
    assert pl._proposal_only_training
    assert pl.model.proposal_input_gradients_enabled
    assert all(
        parameter.requires_grad
        for module in pl._proposal_modules()
        for parameter in module.parameters()
    )
    assert all(parameter.requires_grad for parameter in pl.model.precursor_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in pl.model.intensity_head.parameters())
    assert all(
        not parameter.requires_grad
        for parameter in pl.model.intensity_head.ion_h_support_head.parameters()
    )

    proposal_parameter = next(pl.model.precursor_encoder.parameters())
    pl._apply_staged_training(epoch=16)
    assert not pl._proposal_only_training
    assert pl.model.proposal_input_gradients_enabled
    assert all(
        parameter.requires_grad
        for module in pl._proposal_modules()
        for parameter in module.parameters()
    )
    assert proposal_parameter.requires_grad
    assert all(parameter.requires_grad for parameter in pl.model.precursor_encoder.parameters())
    assert all(parameter.requires_grad for parameter in pl.model.intensity_head.parameters())
    assert all(
        parameter.requires_grad
        for parameter in pl.model.intensity_head.ion_h_support_head.parameters()
    )


def test_proposal_warmup_enables_joint_loss_after_boundary() -> None:
    template_path = Path(__file__).parents[1] / "config" / "models_template"
    spec_params = runner.load_config(
        str(template_path), None, None, model_type="mass_shatter"
    )["spec_params"]
    spec_params["prec_types"] = ["[M+Na]+"]
    pl, _ = _build_pl(
        spec_params=spec_params,
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_separate_proposal_encoder=False,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_proposal_only_until_epoch=15,
        mass_shatter_separate_ion_h_support_head=True,
        mass_shatter_ion_h_support_binary_logits=True,
        mass_shatter_ion_h_support_loss_weight=0.1,
        mass_shatter_separate_ion_mode_proposal_head=True,
        mass_shatter_ion_mode_proposal_binary_logits=True,
        mass_shatter_ion_mode_proposal_loss_weight=0.2,
    )

    def calculate_loss(
        pred_probabilities: th.Tensor,
        proposal_probabilities: th.Tensor,
        support_logits: th.Tensor,
        ion_mode_logits: th.Tensor,
    ) -> dict[str, th.Tensor]:
        return pl.loss_fn(
            true_mzs=th.tensor([100.0]),
            true_logprobs=th.tensor([0.0]),
            true_batch_idxs=th.tensor([0]),
            pred_mzs=th.tensor([100.0, 200.0]),
            pred_logprobs=pred_probabilities.log(),
            pred_batch_idxs=th.tensor([0, 0]),
            candidate_mzs_by_cut=th.tensor([100.0, 200.0]),
            candidate_batch_idxs_by_cut=th.tensor([0, 0]),
            proposal_candidate_mzs=th.tensor([100.0, 200.0]),
            proposal_candidate_fragment_idxs=th.tensor([0, 1]),
            proposal_fragment_logprobs=proposal_probabilities.log(),
            proposal_fragment_batch_idxs=th.tensor([0, 0]),
            ion_h_dense_mzs=th.tensor([[[100.0], [100.0]], [[200.0], [200.0]]]),
            ion_h_dense_log_weights=th.zeros(2, 2, 1),
            ion_h_chemical_valid_channels=th.ones(2, 2, 1, dtype=th.bool),
            ion_h_fragment_batch_idxs=th.tensor([0, 0]),
            ion_h_support_logits=support_logits,
            ion_mode_proposal_logits=ion_mode_logits,
            batch_size=1,
            **_loss_diagnostics(),
        )

    pl._apply_staged_training(epoch=15)
    pred_probabilities = th.tensor([0.8, 0.2], requires_grad=True)
    proposal_probabilities = th.tensor([0.7, 0.3], requires_grad=True)
    support_logits = th.zeros(2, 2, 1, requires_grad=True)
    ion_mode_logits = th.zeros(2, 2, requires_grad=True)
    proposal_phase = calculate_loss(
        pred_probabilities,
        proposal_probabilities,
        support_logits,
        ion_mode_logits,
    )
    assert proposal_phase["loss"].tolist() == pytest.approx(
        (
            0.1 * proposal_phase["proposal_loss"] + 0.2 * proposal_phase["ion_mode_proposal_loss"]
        ).tolist()
    )
    proposal_phase["loss"].sum().backward()
    assert pred_probabilities.grad is None
    assert proposal_probabilities.grad is not None
    assert support_logits.grad is None
    assert ion_mode_logits.grad is not None

    pl.eval()
    pred_probabilities = th.tensor([0.8, 0.2], requires_grad=True)
    proposal_probabilities = th.tensor([0.7, 0.3], requires_grad=True)
    support_logits = th.zeros(2, 2, 1, requires_grad=True)
    ion_mode_logits = th.zeros(2, 2, requires_grad=True)
    validation_phase = calculate_loss(
        pred_probabilities,
        proposal_probabilities,
        support_logits,
        ion_mode_logits,
    )
    expected_validation_loss = (
        validation_phase["primary_loss"]
        + 0.1 * validation_phase["proposal_loss"]
        + 0.1 * validation_phase["ion_h_support_loss"]
        + 0.2 * validation_phase["ion_mode_proposal_loss"]
    )
    assert validation_phase["loss"].tolist() == pytest.approx(expected_validation_loss.tolist())

    pl.train()
    pl._apply_staged_training(epoch=16)
    pred_probabilities = th.tensor([0.8, 0.2], requires_grad=True)
    proposal_probabilities = th.tensor([0.7, 0.3], requires_grad=True)
    support_logits = th.zeros(2, 2, 1, requires_grad=True)
    ion_mode_logits = th.zeros(2, 2, requires_grad=True)
    intensity_phase = calculate_loss(
        pred_probabilities,
        proposal_probabilities,
        support_logits,
        ion_mode_logits,
    )
    expected_joint_loss = (
        intensity_phase["primary_loss"]
        + 0.1 * intensity_phase["proposal_loss"]
        + 0.1 * intensity_phase["ion_h_support_loss"]
        + 0.2 * intensity_phase["ion_mode_proposal_loss"]
    )
    assert intensity_phase["loss"].tolist() == pytest.approx(expected_joint_loss.tolist())
    intensity_phase["loss"].sum().backward()
    assert pred_probabilities.grad is not None
    assert proposal_probabilities.grad is not None
    assert support_logits.grad is not None
    assert ion_mode_logits.grad is not None


@pytest.mark.parametrize("precision", [16, "16", "16-mixed"])
def test_kl_distance_rejects_fp16_mixed_precision(precision: int | str) -> None:
    with pytest.raises(ValueError, match="does not support FP16 mixed precision"):
        _build_pl(loss_type="kl_distance", precision=precision)


@pytest.mark.parametrize(
    ("intensity_output", "loss_type"),
    [
        ("probability", "cross_entropy"),
        ("probability", "cosine_distance"),
        ("independent", "cosine_distance"),
    ],
)
@pytest.mark.parametrize("pair_weight", [0.0, 0.1, 10.0])
def test_precursor_pair_supervision_is_explicitly_unsupported(
    intensity_output: str, loss_type: str, pair_weight: float
) -> None:
    config = {
        "mass_shatter_unique_fragment_mode": True,
        "mass_shatter_proposal_anchored": True,
        "mass_shatter_proposal_selection_only": True,
        "mass_shatter_proposal_loss_weight": 1.0,
        "mass_shatter_intensity_output": intensity_output,
        "loss_type": loss_type,
        "precursor_pair_loss_weight": pair_weight,
    }
    if pair_weight == 0.0:
        model, _ = _build_pl(**config)
        assert callable(model.loss_fn)
    else:
        with pytest.raises(
            ValueError, match="MASShatter does not support precursor_pair_loss_weight"
        ):
            _build_pl(**config)


def _loss_diagnostics() -> dict[str, th.Tensor]:
    return {
        "retained_mask_probability": th.tensor([0.8]),
        "selected_mask_count": th.tensor([2]),
        "component_pathway_count": th.tensor([4]),
        "mean_mask_cut_count": th.tensor([2.0]),
        "proposal_single_cut_fraction": th.tensor([0.25]),
        "proposal_mixture_entropy": th.tensor([0.6]),
        "proposal_mixture_effective_count": th.tensor([1.8]),
        "proposal_mixture_top1_weight": th.tensor([0.7]),
        "unique_component_count": th.tensor([2]),
        "fragment_count_before_cutoff": th.tensor([2]),
        "fragment_count_after_cutoff": th.tensor([2]),
        "retained_fragment_probability": th.tensor([1.0]),
        "fragment_cutoff_pruned_fraction": th.tensor([0.0]),
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
        "mask_cut_fraction_4": th.tensor([0.0]),
        "mask_cut_fraction_5": th.tensor([0.0]),
        "mask_cut_fraction_6plus": th.tensor([0.0]),
        "mask_cut_fraction_4plus": th.tensor([0.0]),
        "candidate_input_transfer_batch_ms": th.tensor([1.0]),
        "candidate_mask_selection_batch_ms": th.tensor([2.0]),
        "candidate_topology_assembly_batch_ms": th.tensor([7.0]),
        "candidate_output_transfer_batch_ms": th.tensor([5.0]),
        "candidate_total_batch_ms": th.tensor([15.0]),
        "candidate_total_molecule_ms": th.tensor([15.0]),
        "valid_ion_h_count": th.tensor([8.0]),
        "retained_ion_h_count": th.tensor([6.0]),
        "retained_ion_h_probability": th.tensor([0.9]),
        "retained_joint_channel_intensity_fraction": th.tensor([0.85]),
        "ion_h_cutoff_pruned_fraction": th.tensor([0.25]),
        "valid_ion_mode_count": th.tensor([3.0]),
        "retained_ion_mode_count": th.tensor([2.0]),
        "retained_ion_mode_probability": th.tensor([0.95]),
        "ion_mode_cutoff_pruned_fraction": th.tensor([1.0 / 3.0]),
    }


def test_observational_channel_targets_use_gaussian_posterior() -> None:
    channel_mzs = th.tensor([[[100.0, 101.0], [200.0, 201.0]]])
    valid = th.tensor([[[True, True], [True, False]]])

    targets = _observational_channel_targets(
        true_mzs=th.tensor([101.0, 200.0]),
        true_batch_idxs=th.tensor([0, 0]),
        channel_mzs=channel_mzs,
        channel_log_weights=th.zeros_like(channel_mzs),
        chemical_valid_channels=valid,
        fragment_batch_idxs=th.tensor([0]),
        batch_size=1,
        tolerance=0.01,
        relative=False,
        tolerance_min_mz=0.0,
        attribution="gaussian_posterior",
    )

    assert targets.tolist() == [[[0.0, 1.0], [1.0, 0.0]]]


def test_gaussian_posterior_splits_ambiguous_peak_by_channel_prior() -> None:
    channel_mzs = th.tensor([[[99.99, 100.01]]], dtype=th.float64)
    channel_weights = th.tensor([[[0.8, 0.2]]], requires_grad=True)

    targets = _observational_channel_targets(
        true_mzs=th.tensor([100.0]),
        true_batch_idxs=th.tensor([0]),
        channel_mzs=channel_mzs,
        channel_log_weights=channel_weights.log(),
        chemical_valid_channels=th.ones_like(channel_mzs, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0]),
        batch_size=1,
        tolerance=0.02,
        relative=False,
        tolerance_min_mz=0.0,
        attribution="gaussian_posterior",
    )

    assert targets.flatten().tolist() == pytest.approx([0.8, 0.2])
    assert targets.sum().item() == pytest.approx(1.0)
    assert not targets.requires_grad


def test_gaussian_posterior_can_ignore_predicted_intensity_prior() -> None:
    channel_mzs = th.tensor([[[99.99, 100.01]]], dtype=th.float64)
    channel_weights = th.tensor([[[0.8, 0.2]]], requires_grad=True)

    targets = _observational_channel_targets(
        true_mzs=th.tensor([100.0]),
        true_batch_idxs=th.tensor([0]),
        channel_mzs=channel_mzs,
        channel_log_weights=channel_weights.log(),
        chemical_valid_channels=th.ones_like(channel_mzs, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0]),
        batch_size=1,
        tolerance=0.02,
        relative=False,
        tolerance_min_mz=0.0,
        attribution="gaussian_posterior",
        use_intensity_prior=False,
    )

    assert targets.flatten().tolist() == pytest.approx([0.5, 0.5])
    assert not targets.requires_grad


def test_hungarian_channel_attribution_is_hard_one_to_one() -> None:
    channel_mzs = th.tensor([[[100.0, 100.004, 101.0]]])

    targets = _observational_channel_targets(
        true_mzs=th.tensor([100.0, 101.0]),
        true_batch_idxs=th.tensor([0, 0]),
        channel_mzs=channel_mzs,
        channel_log_weights=th.zeros_like(channel_mzs),
        chemical_valid_channels=th.ones_like(channel_mzs, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0]),
        batch_size=1,
        tolerance=0.01,
        relative=False,
        tolerance_min_mz=0.0,
        attribution="hungarian",
    )

    assert targets.tolist() == [[[1.0, 0.0, 1.0]]]


def test_hungarian_channel_attribution_drops_infeasible_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_shapes: list[th.Size] = []

    def assignment(cost: th.Tensor, *, maximize: bool) -> tuple[th.Tensor, th.Tensor]:
        observed_shapes.append(cost.shape)
        assert not maximize
        return th.tensor([0]), th.tensor([0])

    monkeypatch.setattr(
        "fragnnet.model.candidate_support.linear_sum_assignment",
        assignment,
    )
    targets = _observational_channel_targets(
        true_mzs=th.tensor([100.0]),
        true_batch_idxs=th.tensor([0]),
        channel_mzs=th.tensor([[[100.0, 200.0, 300.0]]]),
        channel_log_weights=th.zeros(1, 1, 3),
        chemical_valid_channels=th.ones(1, 1, 3, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0]),
        batch_size=1,
        tolerance=0.01,
        relative=False,
        tolerance_min_mz=0.0,
        attribution="hungarian",
        use_intensity_prior=False,
    )

    assert observed_shapes == [th.Size([1, 1])]
    assert targets.tolist() == [[[1.0, 0.0, 0.0]]]


def test_channel_targets_never_attribute_across_spectra() -> None:
    # Both spectra carry the same channel mass; each peak may only claim its own spectrum's
    # channel, which the batched implementation enforces through the (batch, m/z) sort key.
    channel_mzs = th.tensor([[[100.0]], [[100.0]]])

    targets = _observational_channel_targets(
        true_mzs=th.tensor([100.0, 500.0]),
        true_batch_idxs=th.tensor([0, 1]),
        channel_mzs=channel_mzs,
        channel_log_weights=th.zeros_like(channel_mzs),
        chemical_valid_channels=th.ones_like(channel_mzs, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0, 1]),
        batch_size=2,
        tolerance=0.01,
        relative=False,
        tolerance_min_mz=0.0,
        attribution="gaussian_posterior",
    )

    assert targets.tolist() == [[[1.0]], [[0.0]]]


@pytest.mark.parametrize("attribution", ["gaussian_posterior", "hungarian"])
def test_channel_targets_are_zero_when_no_peak_is_within_tolerance(attribution: str) -> None:
    channel_mzs = th.tensor([[[100.0, 300.0]]])

    targets = _observational_channel_targets(
        true_mzs=th.tensor([200.0]),
        true_batch_idxs=th.tensor([0]),
        channel_mzs=channel_mzs,
        channel_log_weights=th.zeros_like(channel_mzs),
        chemical_valid_channels=th.ones_like(channel_mzs, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0]),
        batch_size=1,
        tolerance=0.01,
        relative=False,
        tolerance_min_mz=0.0,
        attribution=attribution,
    )

    assert targets.tolist() == [[[0.0, 0.0]]]


def test_channel_targets_respect_per_spectrum_relative_tolerance() -> None:
    # 5 mDa off at m/z 500: inside a 20 ppm window, outside a 1 ppm one. The two spectra
    # share a batch, so a correct implementation must apply each spectrum's own tolerance.
    channel_mzs = th.tensor([[[500.005]], [[500.005]]])

    targets = _observational_channel_targets(
        true_mzs=th.tensor([500.0, 500.0]),
        true_batch_idxs=th.tensor([0, 1]),
        channel_mzs=channel_mzs,
        channel_log_weights=th.zeros_like(channel_mzs),
        chemical_valid_channels=th.ones_like(channel_mzs, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0, 1]),
        batch_size=2,
        tolerance=1e-5,
        relative=True,
        tolerance_min_mz=200.0,
        attribution="gaussian_posterior",
        tol_per_sample=th.tensor([2e-5, 1e-6]),
        min_mz_per_sample=th.tensor([200.0, 200.0]),
    )

    assert targets.tolist() == [[[1.0]], [[0.0]]]


def test_channel_targets_take_the_strongest_peak_per_channel() -> None:
    # One channel matched by two peaks keeps the larger responsibility: the near-exact peak
    # splits its mass with the decoy channel, the offset peak does not.
    channel_mzs = th.tensor([[[100.0, 100.002]]], dtype=th.float64)

    targets = _observational_channel_targets(
        true_mzs=th.tensor([100.0, 100.008], dtype=th.float64),
        true_batch_idxs=th.tensor([0, 0]),
        channel_mzs=channel_mzs,
        channel_log_weights=th.zeros_like(channel_mzs),
        chemical_valid_channels=th.ones_like(channel_mzs, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0]),
        batch_size=1,
        tolerance=0.01,
        relative=False,
        tolerance_min_mz=0.0,
        attribution="gaussian_posterior",
    )

    # Channel 0 is claimed most strongly by the peak at 100.0 (responsibility 0.505),
    # channel 1 by the peak at 100.008 (0.535); each channel keeps its own maximum.
    assert targets.flatten().tolist() == pytest.approx([0.50500, 0.53495], abs=1e-5)


def test_balanced_support_loss_handles_single_class_spectra() -> None:
    logits = th.zeros(2, 1, 2, requires_grad=True)
    targets = th.tensor([[[True, True]], [[True, False]]])
    loss = _balanced_binary_support_loss(
        logits,
        targets,
        th.ones_like(targets),
        th.tensor([0, 1]),
        batch_size=2,
    )
    loss.sum().backward()

    assert loss.tolist() == pytest.approx([math.log(2.0), math.log(2.0)])
    assert logits.grad is not None


def test_balanced_support_loss_rejects_spectrum_with_no_valid_states() -> None:
    logits = th.zeros(2, 1, 2)
    targets = th.zeros(2, 1, 2, dtype=th.bool)
    valid_mask = th.tensor([[[True, True]], [[False, False]]])
    with pytest.raises(ValueError, match="valid support state"):
        _balanced_binary_support_loss(
            logits,
            targets,
            valid_mask,
            th.tensor([0, 1]),
            batch_size=2,
        )


def test_balanced_support_loss_allows_empty_batches() -> None:
    logits = th.zeros(2, 1, 2, requires_grad=True)
    targets = th.zeros(2, 1, 2, dtype=th.bool)
    valid_mask = th.tensor([[[True, True]], [[False, False]]])
    loss = _balanced_binary_support_loss(
        logits,
        targets,
        valid_mask,
        th.tensor([0, 1]),
        batch_size=2,
        allow_empty_batches=True,
    )
    loss.sum().backward()

    assert loss.tolist() == pytest.approx([math.log(2.0), 0.0])
    assert logits.grad is not None


def test_ion_mode_support_loss_skips_single_mode_fragments() -> None:
    logits = th.zeros(3, 2, requires_grad=True)
    loss = _ion_mode_support_loss(
        logits,
        th.tensor([[True, False], [True, False], [False, True]]),
        th.tensor([[True, False], [True, True], [True, True]]),
        th.tensor([0, 1, 1]),
        batch_size=2,
    )

    loss.sum().backward()

    assert loss.tolist() == pytest.approx([0.0, math.log(2.0)])
    assert logits.grad is not None
    assert logits.grad[0].abs().sum().item() == 0.0
    assert logits.grad[1:].abs().sum().item() > 0.0


def test_ion_h_required_k_metrics_separate_supported_fragments() -> None:
    targets = th.tensor(
        [
            [[0.7, 0.2, 0.1, 0.0]],
            [[0.0, 0.0, 0.0, 0.0]],
            [[0.6, 0.4, 0.0, 0.0]],
        ]
    )
    logits = th.tensor(
        [
            [[4.0, 3.0, 2.0, 1.0]],
            [[4.0, 3.0, 2.0, 1.0]],
            [[3.0, 4.0, 2.0, 1.0]],
        ]
    )

    metrics = _ion_h_required_k_metrics(
        channel_targets=targets,
        support_logits=logits,
        chemical_valid_channels=th.ones_like(targets, dtype=th.bool),
        fragment_batch_idxs=th.tensor([0, 0, 1]),
        batch_size=2,
    )

    assert metrics["valid_ion_h_per_fragment_mean"].tolist() == [4.0, 4.0]
    assert metrics["attributed_ion_h_per_fragment_mean"].tolist() == [1.5, 2.0]
    assert metrics["ion_h_supported_fragment_fraction"].tolist() == [0.5, 1.0]
    assert metrics["required_ion_h_top_k_95_mean"].tolist() == [3.0, 2.0]
    assert metrics["required_ion_h_top_k_95_p90"].tolist() == [3.0, 2.0]
    assert metrics["required_ion_h_top_k_99_max"].tolist() == [3.0, 2.0]


def test_mass_shatter_registration_uses_spectrum_only_dataset() -> None:
    assert get_model_cls("mass_shatter") is MassShatterPL
    assert get_dataset_cls("mass_shatter") is SpecMolDataset
    assert issubclass(MassShatterPL, SpectrumPL)


def test_candidate_metric_wrapper_does_not_repeat_loss_wrecall() -> None:
    model, _ = _build_pl(
        auxiliary_scores=[],
        mass_shatter_candidate_coverage_metrics=True,
        mass_shatter_candidate_cut_diagnostics=True,
    )
    mzs = th.tensor([100.0])
    intensities = th.tensor([1.0])
    batch_indices = th.tensor([0])

    metrics = model.metric_fn(
        true_mzs=mzs,
        true_ints=intensities,
        true_logprobs=intensities.log(),
        true_batch_idxs=batch_indices,
        pred_mzs=mzs,
        pred_ints=intensities,
        pred_logprobs=intensities.log(),
        pred_batch_idxs=batch_indices,
        true_prec_mzs=th.tensor([200.0]),
        split="val",
        candidate_mzs_by_cut=mzs,
        candidate_batch_idxs_by_cut=batch_indices,
        candidate_cut_counts=th.tensor([1]),
    )

    assert "candidate_wrecall" not in metrics


@pytest.mark.parametrize("matching", ["closest", "hungarian"])
@pytest.mark.parametrize(
    "loss_type",
    [
        "cosine_distance",
        "jss_distance",
        "kl_distance",
    ],
)
def test_independent_peak_list_loss_avoids_gaussian_ce(matching: str, loss_type: str) -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=1.0,
        mass_shatter_intensity_output="independent",
        mass_shatter_peak_matching=matching,
        mass_shatter_intensity_entropy_match_weight=0.1,
        loss_type=loss_type,
    )
    true_mzs = th.tensor([100.0, 150.0])
    true_intensities = th.tensor([0.75, 0.25])
    pred_mzs = th.tensor([99.996, 100.004, 150.0])
    pred_intensities = th.tensor([0.25, 0.5, 0.25], requires_grad=True)
    true_batch_indices = th.zeros(2, dtype=th.long)
    pred_batch_indices = th.zeros(3, dtype=th.long)

    result = model.loss_fn(
        true_mzs=true_mzs,
        true_logprobs=true_intensities.log(),
        true_intensities=true_intensities,
        true_batch_idxs=true_batch_indices,
        pred_mzs=pred_mzs,
        pred_intensities=pred_intensities,
        pred_batch_idxs=pred_batch_indices,
        candidate_mzs_by_cut=pred_mzs,
        candidate_batch_idxs_by_cut=pred_batch_indices,
        candidate_cut_counts=th.tensor([1, 1, 2]),
        proposal_candidate_mzs=pred_mzs,
        proposal_candidate_fragment_idxs=th.tensor([0, 0, 1]),
        proposal_fragment_logprobs=th.tensor([0.5, 0.5]).log(),
        proposal_fragment_batch_idxs=th.tensor([0, 0]),
        batch_size=1,
        **_loss_diagnostics(),
    )
    result["loss"].sum().backward()

    assert "spec_ce" not in result
    assert "spec_peak_distance" in result
    assert "spec_entropy_match" in result
    assert pred_intensities.grad is not None
    assert th.isfinite(pred_intensities.grad).all()


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


def test_mass_shatter_pl_uses_map_relative_fragment_probability() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_min_fragment_relative_probability=0.05,
    )

    assert model.model.min_fragment_relative_probability == pytest.approx(0.05)


def test_fragment_probability_cutoff_requires_unique_fragment_mode() -> None:
    with pytest.raises(ValueError, match="requires unique_fragment_mode"):
        _build_pl(mass_shatter_min_fragment_relative_probability=0.05)


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
    assert "fragment_precision" not in model.loss_names


def test_mass_shatter_pl_builds_separate_ion_h_support_head() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_separate_ion_h_support_head=True,
        mass_shatter_ion_h_support_binary_logits=True,
        mass_shatter_ion_h_support_loss_weight=0.2,
        mass_shatter_min_ion_h_relative_probability=0.1,
    )

    assert model.model.intensity_head.ion_h_support_head is not None
    assert model.model.ion_h_support_binary_logits
    assert "ion_h_support_loss" in model.loss_names
    # The support loss builds the channel targets but no longer logs their summaries.
    assert "fragment_precision" not in model.loss_names


def test_channel_support_metrics_can_be_requested_beside_the_support_loss() -> None:
    model, config = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_separate_ion_h_support_head=True,
        mass_shatter_ion_h_support_binary_logits=True,
        mass_shatter_ion_h_support_loss_weight=0.2,
        mass_shatter_channel_support_metrics=True,
    )

    assert config["mass_shatter_channel_support_metrics"] is True
    assert "fragment_precision" in model.loss_names
    assert "required_ion_h_top_k_95_mean" in model.loss_names


def test_channel_support_metrics_are_off_by_default() -> None:
    _, config = _build_pl()

    assert config["mass_shatter_channel_support_metrics"] is False


def test_binary_logit_semantics_must_match_bce_support_loss() -> None:
    with pytest.raises(ValueError, match="binary-logit semantics"):
        _build_pl(
            mass_shatter_unique_fragment_mode=True,
            mass_shatter_proposal_anchored=True,
            mass_shatter_proposal_selection_only=True,
            mass_shatter_proposal_loss_weight=0.1,
            mass_shatter_separate_ion_h_support_head=True,
            mass_shatter_ion_h_support_binary_logits=False,
            mass_shatter_ion_h_support_loss_weight=0.2,
        )


def test_absolute_support_floor_builds_independent_hurdle_model() -> None:
    model, _ = _build_pl(
        loss_type="jss_distance",
        mass_shatter_intensity_output="independent",
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_separate_ion_h_support_head=True,
        mass_shatter_ion_h_support_binary_logits=True,
        mass_shatter_ion_h_support_loss_weight=0.1,
        mass_shatter_min_ion_h_absolute_probability=0.01,
    )

    assert model.model.intensity_output == "independent"
    assert model.model.min_ion_h_absolute_probability == pytest.approx(0.01)


def test_absolute_support_floor_rejects_probability_output() -> None:
    with pytest.raises(ValueError, match="requires independent intensity"):
        _build_pl(
            mass_shatter_unique_fragment_mode=True,
            mass_shatter_proposal_anchored=True,
            mass_shatter_proposal_selection_only=True,
            mass_shatter_proposal_loss_weight=0.1,
            mass_shatter_separate_ion_h_support_head=True,
            mass_shatter_ion_h_support_binary_logits=True,
            mass_shatter_ion_h_support_loss_weight=0.1,
            mass_shatter_min_ion_h_absolute_probability=0.01,
        )


def test_channel_support_attribution_defaults_to_gaussian_posterior() -> None:
    model, _ = _build_pl()

    assert model.hparams.mass_shatter_channel_support_attribution == "gaussian_posterior"


def test_channel_support_attribution_rejects_unknown_method() -> None:
    with pytest.raises(ValueError, match="channel_support_attribution"):
        _build_pl(mass_shatter_channel_support_attribution="all_matches")


def test_channel_diagnostics_require_selection_only_mode() -> None:
    with pytest.raises(ValueError, match="requires unique-fragment"):
        _build_pl(mass_shatter_channel_diagnostics=True)


def test_training_channel_selection_rejects_unsupervised_arm() -> None:
    with pytest.raises(ValueError, match="requires a supervised support head"):
        _build_pl(
            mass_shatter_min_ion_h_relative_probability=0.1,
            mass_shatter_apply_channel_selection_in_training=True,
        )


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


def test_barcode_target_requires_proposal_supervision() -> None:
    with pytest.raises(ValueError, match="requires a positive"):
        _build_pl(mass_shatter_proposal_barcode_target=True)


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


def test_proposal_coverage_loss_can_use_barcode_target_weights() -> None:
    fragment_probabilities = th.tensor([0.8, 0.2])
    common_kwargs = {
        "true_mzs": th.tensor([100.0, 200.0]),
        "true_logprobs": th.log(th.tensor([0.9, 0.1])),
        "true_batch_idxs": th.tensor([0, 0]),
        "candidate_mzs": th.tensor([100.0, 200.0]),
        "candidate_fragment_idxs": th.tensor([0, 1]),
        "fragment_logprobs": fragment_probabilities.log(),
        "fragment_batch_idxs": th.tensor([0, 0]),
        "batch_size": 1,
        "tolerance": 1.0e-5,
        "relative": True,
        "tolerance_min_mz": 200.0,
    }

    intensity_weighted = _proposal_coverage_loss(**common_kwargs)
    barcode_weighted = _proposal_coverage_loss(
        **common_kwargs,
        barcode_target=True,
    )

    assert intensity_weighted.tolist() == pytest.approx(
        [-0.9 * math.log(0.8) - 0.1 * math.log(0.2)]
    )
    assert barcode_weighted.tolist() == pytest.approx([-0.5 * math.log(0.8) - 0.5 * math.log(0.2)])


def _reference_proposal_coverage_loss(
    *,
    true_mzs: th.Tensor,
    true_logprobs: th.Tensor,
    true_batch_idxs: th.Tensor,
    candidate_mzs: th.Tensor,
    candidate_fragment_idxs: th.Tensor,
    fragment_logprobs: th.Tensor,
    fragment_batch_idxs: th.Tensor,
    batch_size: int,
    tolerance: float,
    relative: bool,
    tolerance_min_mz: float,
    tol_per_sample: th.Tensor | None = None,
    min_mz_per_sample: th.Tensor | None = None,
    barcode_target: bool = False,
) -> th.Tensor:
    """Per-spectrum reference for the batched proposal coverage loss.

    This is the loop the production implementation replaced: it builds one dense
    peak-by-channel match grid per spectrum. It exists only to pin the batched result.
    """
    coverage_loss = fragment_logprobs.new_zeros(batch_size)
    for batch_index in range(batch_size):
        true_mask = true_batch_idxs == batch_index
        batch_true_mzs = true_mzs[true_mask]
        batch_fragment_indices = th.where(fragment_batch_idxs == batch_index)[0]
        batch_true_probabilities = (
            true_logprobs.new_full(batch_true_mzs.shape, 1.0 / batch_true_mzs.numel())
            if barcode_target
            else true_logprobs[true_mask].exp()
        )
        first_fragment_index = batch_fragment_indices[0]
        candidate_mask = fragment_batch_idxs[candidate_fragment_idxs] == batch_index
        batch_candidate_mzs = candidate_mzs[candidate_mask]
        local_fragment_indices = candidate_fragment_idxs[candidate_mask] - first_fragment_index
        if tol_per_sample is None:
            channel_match = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=tolerance,
                relative=relative,
                tolerance_min_mz=tolerance_min_mz,
            )
        else:
            per_peak_tolerance = tol_per_sample[batch_index] * th.maximum(
                batch_true_mzs,
                min_mz_per_sample[batch_index],
            )
            channel_match = calculate_match_mzs(
                batch_true_mzs,
                batch_candidate_mzs,
                tolerance=0.0,
                relative=False,
                tol_per_true=per_peak_tolerance,
            )
        fragment_match_counts = fragment_logprobs.new_zeros(
            (batch_true_mzs.numel(), batch_fragment_indices.numel())
        )
        fragment_match_counts.scatter_add_(
            1,
            local_fragment_indices.reshape(1, -1).expand(batch_true_mzs.numel(), -1),
            channel_match.to(fragment_match_counts),
        )
        fragment_match = fragment_match_counts > 0
        peak_log_coverage = th.logsumexp(
            fragment_logprobs[batch_fragment_indices]
            .reshape(1, -1)
            .masked_fill(~fragment_match, -th.inf),
            dim=1,
        )
        coverage_loss[batch_index] = th.where(
            fragment_match.any(dim=1),
            -batch_true_probabilities * peak_log_coverage,
            th.zeros_like(peak_log_coverage),
        ).sum()
    return coverage_loss


def _random_proposal_batch(
    generator: th.Generator,
    batch_size: int = 6,
) -> tuple[dict[str, th.Tensor | int], th.Tensor]:
    """Build a ragged proposal batch whose peaks are partly covered by several channels."""
    true_mzs = []
    true_logprobs = []
    true_batch_idxs = []
    candidate_mzs = []
    candidate_fragment_idxs = []
    fragment_batch_idxs = []
    num_fragments = 0
    for batch_index in range(batch_size):
        num_peaks = int(th.randint(1, 8, (1,), generator=generator))
        peak_mzs = th.rand(num_peaks, generator=generator) * 400.0 + 50.0
        true_mzs.append(peak_mzs)
        true_logprobs.append(th.log_softmax(th.rand(num_peaks, generator=generator), dim=0))
        true_batch_idxs.append(th.full((num_peaks,), batch_index, dtype=th.long))
        spectrum_fragments = int(th.randint(1, 6, (1,), generator=generator))
        for fragment in range(num_fragments, num_fragments + spectrum_fragments):
            # Two channels per fragment: one on an observed peak (so duplicate matches of
            # the same fragment occur), one at an arbitrary mass.
            claimed = peak_mzs[int(th.randint(0, num_peaks, (1,), generator=generator))]
            candidate_mzs.append(th.stack([claimed, th.rand(1, generator=generator)[0] * 450.0]))
            candidate_fragment_idxs.append(th.full((2,), fragment, dtype=th.long))
        fragment_batch_idxs.append(th.full((spectrum_fragments,), batch_index, dtype=th.long))
        num_fragments += spectrum_fragments
    fragment_logits = th.rand(num_fragments, generator=generator, requires_grad=True)
    fragment_batch_idxs = th.cat(fragment_batch_idxs)
    fragment_logprobs = th.cat(
        [
            th.log_softmax(fragment_logits[fragment_batch_idxs == batch_index], dim=0)
            for batch_index in range(batch_size)
        ]
    )
    inputs = {
        "true_mzs": th.cat(true_mzs),
        "true_logprobs": th.cat(true_logprobs),
        "true_batch_idxs": th.cat(true_batch_idxs),
        "candidate_mzs": th.cat(candidate_mzs),
        "candidate_fragment_idxs": th.cat(candidate_fragment_idxs),
        "fragment_logprobs": fragment_logprobs,
        "fragment_batch_idxs": fragment_batch_idxs,
        "batch_size": batch_size,
    }
    return inputs, fragment_logits


@pytest.mark.parametrize("barcode_target", [False, True])
@pytest.mark.parametrize("per_sample_tolerance", [False, True])
def test_proposal_coverage_loss_matches_per_spectrum_reference(
    barcode_target: bool,
    per_sample_tolerance: bool,
) -> None:
    generator = th.Generator().manual_seed(20260918)
    inputs, fragment_logits = _random_proposal_batch(generator)
    batch_size = inputs["batch_size"]
    tolerance_kwargs = {
        "tolerance": 1.0e-5,
        "relative": True,
        "tolerance_min_mz": 200.0,
        "barcode_target": barcode_target,
    }
    if per_sample_tolerance:
        tolerance_kwargs["tol_per_sample"] = th.linspace(1.0e-6, 4.0e-5, batch_size)
        tolerance_kwargs["min_mz_per_sample"] = th.full((batch_size,), 200.0)

    batched = _proposal_coverage_loss(**inputs, **tolerance_kwargs)
    reference = _reference_proposal_coverage_loss(**inputs, **tolerance_kwargs)
    batched_grad = th.autograd.grad(batched.sum(), fragment_logits, retain_graph=True)[0]
    reference_grad = th.autograd.grad(reference.sum(), fragment_logits)[0]

    assert batched.shape == (batch_size,)
    assert th.allclose(batched, reference, atol=1e-6)
    assert th.allclose(batched_grad, reference_grad, atol=1e-6)
    assert batched_grad.abs().sum() > 0.0


def test_proposal_coverage_loss_never_matches_across_spectra() -> None:
    # Both spectra propose the same channel mass; only spectrum 0 observes it, so the
    # second spectrum stays uncovered and contributes nothing.
    loss = _proposal_coverage_loss(
        true_mzs=th.tensor([100.0, 400.0]),
        true_logprobs=th.zeros(2),
        true_batch_idxs=th.tensor([0, 1]),
        candidate_mzs=th.tensor([100.0, 100.0]),
        candidate_fragment_idxs=th.tensor([0, 1]),
        fragment_logprobs=th.log(th.tensor([0.5, 0.5])),
        fragment_batch_idxs=th.tensor([0, 1]),
        batch_size=2,
        tolerance=1.0e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )

    assert loss.tolist() == pytest.approx([-math.log(0.5), 0.0])


def test_proposal_coverage_loss_stays_differentiable_without_any_match() -> None:
    fragment_logprobs = th.zeros(1, requires_grad=True)
    loss = _proposal_coverage_loss(
        true_mzs=th.tensor([100.0]),
        true_logprobs=th.zeros(1),
        true_batch_idxs=th.tensor([0]),
        candidate_mzs=th.tensor([300.0]),
        candidate_fragment_idxs=th.tensor([0]),
        fragment_logprobs=fragment_logprobs,
        fragment_batch_idxs=th.tensor([0]),
        batch_size=1,
        tolerance=1.0e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )

    assert loss.tolist() == [0.0]
    loss.sum().backward()
    assert fragment_logprobs.grad.tolist() == [0.0]


def test_proposal_coverage_loss_requires_peaks_and_fragments() -> None:
    common_kwargs = {
        "candidate_mzs": th.tensor([100.0]),
        "candidate_fragment_idxs": th.tensor([0]),
        "fragment_logprobs": th.zeros(1),
        "batch_size": 2,
        "tolerance": 1.0e-5,
        "relative": True,
        "tolerance_min_mz": 200.0,
    }
    with pytest.raises(ValueError, match="spectrum 1 has no true peaks"):
        _proposal_coverage_loss(
            **common_kwargs,
            true_mzs=th.tensor([100.0]),
            true_logprobs=th.zeros(1),
            true_batch_idxs=th.tensor([0]),
            fragment_batch_idxs=th.tensor([0]),
        )
    with pytest.raises(ValueError, match="spectrum 0 has no proposed fragments"):
        _proposal_coverage_loss(
            **common_kwargs,
            true_mzs=th.tensor([100.0, 100.0]),
            true_logprobs=th.zeros(2),
            true_batch_idxs=th.tensor([0, 1]),
            fragment_batch_idxs=th.tensor([1]),
        )


def test_shared_group_proposal_loss_uses_merged_group_target() -> None:
    shared_fragment_probabilities = th.tensor([0.6, 0.4], requires_grad=True)
    loss = _shared_group_proposal_coverage_loss(
        true_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        true_logprobs=th.log(th.tensor([0.8, 0.2, 0.3, 0.7])),
        true_batch_idxs=th.tensor([0, 0, 1, 1]),
        candidate_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        candidate_fragment_idxs=th.tensor([0, 1, 2, 3]),
        fragment_logprobs=shared_fragment_probabilities.log().repeat(2),
        fragment_batch_idxs=th.tensor([0, 0, 1, 1]),
        group_ids=th.tensor([9, 9]),
        batch_size=2,
        tolerance=1.0e-5,
        relative=True,
        tolerance_min_mz=200.0,
    )

    expected = -0.55 * math.log(0.6) - 0.45 * math.log(0.4)
    assert loss.tolist() == pytest.approx([expected, expected])
    loss.mean().backward()
    assert shared_fragment_probabilities.grad is not None
    assert shared_fragment_probabilities.grad.abs().sum() > 0.0


def test_shared_group_proposal_loss_can_use_barcode_target() -> None:
    loss = _shared_group_proposal_coverage_loss(
        true_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        true_logprobs=th.log(th.tensor([0.8, 0.2, 0.3, 0.7])),
        true_batch_idxs=th.tensor([0, 0, 1, 1]),
        candidate_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        candidate_fragment_idxs=th.tensor([0, 1, 2, 3]),
        fragment_logprobs=th.log(th.tensor([0.6, 0.4, 0.6, 0.4])),
        fragment_batch_idxs=th.tensor([0, 0, 1, 1]),
        group_ids=th.tensor([9, 9]),
        batch_size=2,
        tolerance=1.0e-5,
        relative=True,
        tolerance_min_mz=200.0,
        barcode_target=True,
    )

    expected = -0.5 * math.log(0.6) - 0.5 * math.log(0.4)
    assert loss.tolist() == pytest.approx([expected, expected])


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


def test_selection_only_loss_skips_unrequested_channel_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
    )
    proposal_fragment_probabilities = th.tensor([0.4, 0.3], requires_grad=True)
    proposal_fragment_logprobs = th.log(proposal_fragment_probabilities)

    model.eval()

    def unexpected_channel_target(*args: object, **kwargs: object) -> th.Tensor:
        raise AssertionError("dense channel matching must stay off unless requested")

    monkeypatch.setattr(
        "fragnnet.pl_model.mass_shatter_pl._observational_channel_targets",
        unexpected_channel_target,
    )
    actual = model.loss_fn(
        true_mzs=th.tensor([100.0]),
        true_logprobs=th.tensor([0.0]),
        true_batch_idxs=th.tensor([0]),
        pred_mzs=th.tensor([100.0, 200.0]),
        pred_logprobs=th.log(th.tensor([0.8, 0.1])),
        pred_batch_idxs=th.tensor([0, 0]),
        candidate_mzs_by_cut=th.tensor([100.0, 200.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 0]),
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


def test_proposal_target_can_use_a_different_intensity_transform() -> None:
    model, _ = _build_pl(
        ints_transform="sqrt",
        mass_shatter_proposal_ints_transform="none",
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
    )
    transformed_true_probabilities = th.tensor([0.75, 0.25])
    proposal_fragment_probabilities = th.tensor([0.8, 0.2])

    actual = model.loss_fn(
        true_mzs=th.tensor([100.0, 200.0]),
        true_logprobs=transformed_true_probabilities.log(),
        true_intensities=transformed_true_probabilities,
        true_batch_idxs=th.tensor([0, 0]),
        pred_mzs=th.tensor([100.0, 200.0]),
        pred_logprobs=transformed_true_probabilities.log(),
        pred_batch_idxs=th.tensor([0, 0]),
        candidate_mzs_by_cut=th.tensor([100.0, 200.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 0]),
        proposal_candidate_mzs=th.tensor([100.0, 200.0]),
        proposal_candidate_fragment_idxs=th.tensor([0, 1]),
        proposal_fragment_logprobs=proposal_fragment_probabilities.log(),
        proposal_fragment_batch_idxs=th.tensor([0, 0]),
        batch_size=1,
        **_loss_diagnostics(),
    )

    raw_true_probabilities = th.tensor([0.9, 0.1])
    expected = -(raw_true_probabilities * proposal_fragment_probabilities.log()).sum()
    assert actual["proposal_loss"].item() == pytest.approx(expected.item())


def test_requested_channel_diagnostics_are_computed_during_training() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_channel_diagnostics=True,
    )
    proposal_fragment_logprobs = th.log(th.tensor([0.6, 0.4]))

    actual = model.loss_fn(
        true_mzs=th.tensor([100.0]),
        true_logprobs=th.tensor([0.0]),
        true_batch_idxs=th.tensor([0]),
        pred_mzs=th.tensor([100.0, 200.0]),
        pred_logprobs=th.log(th.tensor([0.8, 0.1])),
        pred_batch_idxs=th.tensor([0, 0]),
        candidate_mzs_by_cut=th.tensor([100.0, 200.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 0]),
        proposal_candidate_mzs=th.tensor([100.0, 200.0]),
        proposal_candidate_fragment_idxs=th.tensor([0, 1]),
        proposal_fragment_logprobs=proposal_fragment_logprobs,
        proposal_fragment_batch_idxs=th.tensor([0, 0]),
        ion_h_dense_mzs=th.tensor([[[100.0]], [[200.0]]]),
        ion_h_dense_log_weights=th.log(th.tensor([[[0.6]], [[0.4]]])),
        ion_h_chemical_valid_channels=th.ones(2, 1, 1, dtype=th.bool),
        ion_h_support_logits=th.zeros(2, 1, 1),
        ion_h_fragment_batch_idxs=th.tensor([0, 0]),
        batch_size=1,
        **_loss_diagnostics(),
    )

    assert model.training
    assert actual["fragment_precision"].tolist() == pytest.approx([0.5])
    assert actual["fragment_weighted_precision"].tolist() == pytest.approx([0.6])
    assert actual["ion_mode_support_positive_rate"].tolist() == pytest.approx([0.5])
    assert actual["ion_h_support_positive_rate"].tolist() == pytest.approx([0.5])
    assert actual["valid_ion_h_per_fragment_mean"].tolist() == pytest.approx([1.0])
    assert actual["attributed_ion_h_per_fragment_mean"].tolist() == pytest.approx([0.5])
    assert actual["ion_h_supported_fragment_fraction"].tolist() == pytest.approx([0.5])
    assert actual["required_ion_h_top_k_99_p90"].tolist() == pytest.approx([1.0])


@pytest.mark.parametrize("channels,base_ce", [(False, True), (True, False), (False, False)])
def test_monitoring_switches_preserve_training_loss_and_gradients(
    channels: bool, base_ce: bool
) -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_channel_diagnostics=True,
        mass_shatter_log_base_ce=True,
    )
    results = []
    for enable_channels, enable_base_ce in [(True, True), (channels, base_ce)]:
        model.hparams.mass_shatter_channel_diagnostics = enable_channels
        model.hparams.mass_shatter_log_base_ce = enable_base_ce
        model._setup_loss_fn()
        logits = th.tensor([0.8, 0.2], requires_grad=True)
        proposal_logits = th.tensor([0.6, 0.4], requires_grad=True)
        actual = model.loss_fn(
            true_mzs=th.tensor([100.0]),
            true_logprobs=th.tensor([0.0]),
            true_batch_idxs=th.tensor([0]),
            pred_mzs=th.tensor([100.0, 200.0]),
            pred_logprobs=logits.log_softmax(0),
            pred_batch_idxs=th.tensor([0, 0]),
            base_pred_logprobs=proposal_logits.log_softmax(0),
            candidate_mzs_by_cut=th.tensor([100.0, 200.0]),
            candidate_batch_idxs_by_cut=th.tensor([0, 0]),
            proposal_candidate_mzs=th.tensor([100.0, 200.0]),
            proposal_candidate_fragment_idxs=th.tensor([0, 1]),
            proposal_fragment_logprobs=proposal_logits.log_softmax(0),
            proposal_fragment_batch_idxs=th.tensor([0, 0]),
            ion_h_dense_mzs=th.tensor([[[100.0]], [[200.0]]]),
            ion_h_dense_log_weights=th.log(th.tensor([[[0.6]], [[0.4]]])),
            ion_h_chemical_valid_channels=th.ones(2, 1, 1, dtype=th.bool),
            ion_h_support_logits=th.zeros(2, 1, 1),
            ion_h_fragment_batch_idxs=th.tensor([0, 0]),
            batch_size=1,
            **_loss_diagnostics(),
        )
        actual["loss"].sum().backward()
        results.append((actual["loss"].detach(), logits.grad, proposal_logits.grad))
    for baseline, reduced in zip(results[0], results[1], strict=True):
        th.testing.assert_close(baseline, reduced, rtol=0, atol=0)


def test_selection_only_loss_uses_barcode_proposal_target() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_proposal_barcode_target=True,
    )
    actual = model.loss_fn(
        true_mzs=th.tensor([100.0, 200.0]),
        true_logprobs=th.log(th.tensor([0.9, 0.1])),
        true_batch_idxs=th.tensor([0, 0]),
        pred_mzs=th.tensor([100.0, 200.0]),
        pred_logprobs=th.log(th.tensor([0.8, 0.2])),
        pred_batch_idxs=th.tensor([0, 0]),
        candidate_mzs_by_cut=th.tensor([100.0, 200.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 0]),
        proposal_candidate_mzs=th.tensor([100.0, 200.0]),
        proposal_candidate_fragment_idxs=th.tensor([0, 1]),
        proposal_fragment_logprobs=th.log(th.tensor([0.8, 0.2])),
        proposal_fragment_batch_idxs=th.tensor([0, 0]),
        batch_size=1,
        **_loss_diagnostics(),
    )

    expected = -0.5 * math.log(0.8) - 0.5 * math.log(0.2)
    assert actual["proposal_loss"].tolist() == pytest.approx([expected])


def test_shared_selection_only_loss_routes_through_merged_group_target() -> None:
    model, _ = _build_pl(
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_loss_weight=0.1,
        mass_shatter_share_proposal_across_ev=True,
    )
    shared_fragment_probabilities = th.tensor([0.6, 0.4], requires_grad=True)
    actual = model.loss_fn(
        true_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        true_logprobs=th.log(th.tensor([0.8, 0.2, 0.3, 0.7])),
        true_batch_idxs=th.tensor([0, 0, 1, 1]),
        pred_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        pred_logprobs=th.log(th.tensor([0.5, 0.5, 0.5, 0.5])),
        pred_batch_idxs=th.tensor([0, 0, 1, 1]),
        candidate_mzs_by_cut=th.tensor([100.0, 200.0, 100.0, 200.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 0, 1, 1]),
        proposal_candidate_mzs=th.tensor([100.0, 200.0, 100.0, 200.0]),
        proposal_candidate_fragment_idxs=th.tensor([0, 1, 2, 3]),
        proposal_fragment_logprobs=shared_fragment_probabilities.log().repeat(2),
        proposal_fragment_batch_idxs=th.tensor([0, 0, 1, 1]),
        proposal_group_ids=th.tensor([9, 9]),
        batch_size=2,
        **_loss_diagnostics(),
    )

    expected = -0.55 * math.log(0.6) - 0.45 * math.log(0.4)
    assert actual["proposal_loss"].tolist() == pytest.approx([expected, expected])


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


def test_smarts_standalone_prediction_matches_lightning_transfer() -> None:
    model, config = _build_pl(
        mass_shatter_smarts_templates=True,
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_unique_fragment_target=4,
        mass_shatter_ranked_mask_budget=8,
    )
    smiles = "c1ccccc1S(=O)c1ccccc1"
    params = config["mol_params"]
    graph = get_mol_graph(
        Chem.MolFromSmiles(smiles),
        params["pyg_node_feats"],
        params["pyg_edge_feats"],
        params["pyg_pe_embed_k"],
        True,
    )
    batch = {
        "mol_pyg": Batch.from_data_list([graph]),
        "mol_smiles": np.asarray([smiles]),
        "spec_prec_type": th.tensor([model.model.precursor_types.index("[M+H]+")]),
        "spec_nce": th.tensor([25.0]),
        "spec_inst_type": th.tensor([0]),
    }
    model.eval()
    with th.no_grad():
        standalone = model.predict_step(**batch)
        transferred = model.transfer_batch_to_device(dict(batch), th.device("cpu"), 0)
        lightning = model.predict_step(**transferred)
    assert lightning["unique_component_count"].item() >= 9
    for key in ("pred_mzs", "pred_logprobs", "candidate_mzs_by_cut", "unique_component_count"):
        th.testing.assert_close(standalone[key], lightning[key])


@pytest.mark.parametrize("output_mode", ["probability", "independent"])
def test_prediction_only_pickle_preserves_processed_intensities(output_mode: str) -> None:
    model, config = _build_pl(
        mass_shatter_intensity_output=output_mode,
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_loss_weight=0.1,
        loss_type="cosine_distance",
        ints_transform="sqrt",
    )
    params = config["mol_params"]
    graph = get_mol_graph(
        Chem.MolFromSmiles("CCO"),
        params["pyg_node_feats"],
        params["pyg_edge_feats"],
        params["pyg_pe_embed_k"],
        True,
    )
    batches = [
        {
            "mol_pyg": Batch.from_data_list([graph.clone()]),
            "spec_prec_type": th.tensor([0]),
            "spec_inst_type": th.tensor([0]),
            "spec_nce": th.tensor([energy]),
        }
        for energy in (25.0, 35.0)
    ]
    model.eval()
    with th.no_grad():
        expected = [model.inference_step(batch) for batch in batches]
    output = script_utils.run_inference_pkl(
        model,
        batches,
        th.device("cpu"),
        evaluate=False,
        output_set={"pred_mzs", "pred_ints"},
    )
    # `inference_step` has already untransformed and L1-normalized these; the exporter
    # only rescales the base peak to 1000. A second inverse sqrt transform would change
    # the ratios, which the rescale cannot hide.
    expected_batch_idxs = th.cat([x["pred_batch_idxs"] + index for index, x in enumerate(expected)])
    th.testing.assert_close(
        output["pred_ints"],
        script_utils.batched_mf1000_normalize(
            th.cat([x["pred_ints"] for x in expected]), expected_batch_idxs
        ),
    )
    th.testing.assert_close(output["pred_mzs"], th.cat([x["pred_mzs"] for x in expected]))
    th.testing.assert_close(output["pred_batch_idxs"], expected_batch_idxs)


@pytest.mark.parametrize("output_mode", ["probability", "independent"])
@pytest.mark.parametrize("existing_dag_directory", [False, True])
def test_mass_shatter_cli_exports_without_dags(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    output_mode: str,
    existing_dag_directory: bool,
) -> None:
    model, config = _build_pl(
        mass_shatter_intensity_output=output_mode,
        mass_shatter_unique_fragment_mode=True,
        mass_shatter_proposal_selection_only=True,
        mass_shatter_proposal_anchored=True,
        mass_shatter_proposal_loss_weight=0.1,
        loss_type="cosine_distance",
        ints_transform="sqrt",
        auxiliary_scores=[],
    )
    smiles = ["CCO", "CO"]
    params = config["mol_params"]
    graphs = [
        get_mol_graph(
            Chem.MolFromSmiles(s),
            params["pyg_node_feats"],
            params["pyg_edge_feats"],
            params["pyg_pe_embed_k"],
            True,
        )
        for s in smiles
    ]
    batch = {
        "mol_pyg": Batch.from_data_list(graphs),
        "mol_smiles": smiles,
        "mol_id": th.tensor([11, 22]),
        "spec_unique_id": th.tensor([101, 102]),
        "spec_prec_type": th.tensor([0, 0]),
        "spec_inst_type": th.tensor([0, 0]),
        "spec_nce": th.tensor([25.0, 35.0]),
        "batch_size": th.tensor(2),
    }
    model.eval()
    with th.no_grad():
        prediction = model.predict_step(**batch)
    weights = (
        prediction["pred_intensities"]
        if output_mode == "independent"
        else prediction["pred_logprobs"].exp()
    )
    mol_path, spec_path = tmp_path / "mol.pkl", tmp_path / "spec.pkl"
    pd.DataFrame({"mol_id": [11, 22]}).to_pickle(mol_path)
    pd.DataFrame({"mol_id": [11, 22], "spec_id": [101, 102]}).to_pickle(spec_path)
    dag_dir = tmp_path / "unused_dags"
    if existing_dag_directory:
        dag_dir.mkdir()
    config["frag_dp"] = str(dag_dir)

    def dataset_init(self, **kwargs):
        assert type(self) is SpecMolDataset
        assert kwargs["mol_fp"]["mol_id"].tolist() == [11, 22]
        assert kwargs["spec_fp"]["spec_id"].tolist() == [101, 102]
        self.spec_df = kwargs["spec_fp"]

    # Isolate file/config loading and dataset preprocessing; execute the actual CLI,
    # model prediction, intensity conversion, tabular iterator, and chunk writer.
    monkeypatch.setattr(SpecMolDataset, "__init__", dataset_init)
    monkeypatch.setattr(runner, "init_dataloader", lambda *_args: [batch])
    monkeypatch.setattr(script_utils, "get_ckpt_path", lambda **_kwargs: "unused.ckpt")
    monkeypatch.setattr(script_utils, "get_config_from_ckpt", lambda **_kwargs: config)
    monkeypatch.setattr(script_utils, "init_model_from_ckpt", lambda *_args: model)
    monkeypatch.setattr(script_utils, "setup_device", lambda *_args: th.device("cpu"))
    output_dir = tmp_path / "predictions"
    # `existing_dag_directory` also varies the chunk size so both the periodic and the
    # final chunk writer run. Parquet is no longer an option, so both arms write pkl.
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference.py",
            "--mol_fp",
            str(mol_path),
            "--spec_fp",
            str(spec_path),
            "--save_dp",
            str(output_dir),
            "--save_format",
            "pkl",
            "--min_intensity",
            "0",
            "--save_chunk_size",
            "1" if existing_dag_directory else "10000",
        ],
    )
    runpy.run_path(str(Path(__file__).parents[1] / "scripts/run_inference.py"), run_name="__main__")
    saved = pd.read_pickle(output_dir / "predictions_chunk_0.pkl")
    assert saved["spec_id"].tolist() == [101, 102]
    for index in range(2):
        mask = prediction["pred_batch_idxs"] == index
        th.testing.assert_close(
            th.as_tensor(saved.iloc[index]["pred_mzs"], dtype=prediction["pred_mzs"].dtype),
            prediction["pred_mzs"][mask],
        )
        # sqrt-transformed scores return to a per-spectrum base-peak scale of 1000.
        expected = (weights[mask] / weights[mask].max()).square() * 1000.0
        th.testing.assert_close(
            th.as_tensor(saved.iloc[index]["pred_ints"], dtype=expected.dtype),
            expected,
            rtol=1e-5,
            atol=1e-4,
        )


@pytest.mark.parametrize("model_type", ["frag_gnn", "fragnnet"])
def test_inference_cli_preserves_dag_model_filtering(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    model_type: str,
) -> None:
    _, config = _build_pl()
    config["model_type"] = model_type
    mol_path, spec_path = tmp_path / "mol.pkl", tmp_path / "spec.pkl"
    pd.DataFrame({"mol_id": [11, 22]}).to_pickle(mol_path)
    pd.DataFrame({"mol_id": [11, 22], "spec_id": [101, 102]}).to_pickle(spec_path)
    dag_dir = tmp_path / "dags"
    dag_dir.mkdir()
    (dag_dir / "11.hdf5").touch()

    class DatasetChecked(Exception):
        pass

    def dataset_init(self, **kwargs):
        assert type(self) is SpecMolFragDataset
        assert kwargs["mol_fp"]["mol_id"].tolist() == [11]
        assert kwargs["spec_fp"]["spec_id"].tolist() == [101]
        raise DatasetChecked

    monkeypatch.setattr(SpecMolFragDataset, "__init__", dataset_init)
    monkeypatch.setattr(script_utils, "get_ckpt_path", lambda **_kwargs: "unused.ckpt")
    monkeypatch.setattr(script_utils, "get_config_from_ckpt", lambda **_kwargs: config)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_inference.py",
            "--mol_fp",
            str(mol_path),
            "--spec_fp",
            str(spec_path),
            "--save_dp",
            str(tmp_path / "predictions"),
            "--frag_dp",
            str(dag_dir),
        ],
    )
    with pytest.raises(DatasetChecked):
        runpy.run_path(
            str(Path(__file__).parents[1] / "scripts/run_inference.py"),
            run_name="__main__",
        )


def test_mass_shatter_primary_loss_uses_candidate_only_sparse_cross_entropy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, config = _build_pl(
        mass_shatter_log_base_ce=True,
        mass_shatter_base_ce_weight=0.2,
        oos_loss=True,
        mass_shatter_candidate_coverage_metrics=True,
        mass_shatter_candidate_timing_metrics=True,
    )
    true_mzs = th.tensor([100.0, 150.0])
    true_logprobs = th.log(th.tensor([0.6, 0.4]))
    true_batch_indices = th.tensor([0, 0], dtype=th.long)
    pred_mzs = th.tensor([100.0, 180.0])
    pred_logprobs = th.log(th.tensor([0.5, 0.3]))
    base_pred_logprobs = th.log(th.tensor([0.3, 0.5])).requires_grad_()
    pred_batch_indices = th.tensor([0, 0], dtype=th.long)

    def unexpected_bucket_matching(*args: object, **kwargs: object) -> dict[str, th.Tensor]:
        raise AssertionError("CE candidate_wrecall must not compute cut-bucket metrics")

    monkeypatch.setattr(
        "fragnnet.pl_model.mass_shatter_pl._candidate_coverage_by_cut_count",
        unexpected_bucket_matching,
    )
    actual = model.loss_fn(
        true_mzs=true_mzs,
        true_logprobs=true_logprobs,
        true_batch_idxs=true_batch_indices,
        pred_mzs=pred_mzs,
        pred_logprobs=pred_logprobs,
        base_pred_logprobs=base_pred_logprobs,
        pred_batch_idxs=pred_batch_indices,
        candidate_mzs_by_cut=th.tensor([100.0, 150.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 0]),
        retained_mask_probability=th.tensor([0.8]),
        selected_mask_count=th.tensor([2]),
        component_pathway_count=th.tensor([4]),
        mean_mask_cut_count=th.tensor([2.0]),
        proposal_single_cut_fraction=th.tensor([0.25]),
        proposal_mixture_entropy=th.tensor([0.6]),
        proposal_mixture_effective_count=th.tensor([1.8]),
        proposal_mixture_top1_weight=th.tensor([0.7]),
        unique_component_count=th.tensor([3]),
        fragment_count_before_cutoff=th.tensor([3]),
        fragment_count_after_cutoff=th.tensor([2]),
        retained_fragment_probability=th.tensor([0.9]),
        fragment_cutoff_pruned_fraction=th.tensor([1.0 / 3.0]),
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
        mask_cut_fraction_4=th.tensor([0.0]),
        mask_cut_fraction_5=th.tensor([0.0]),
        mask_cut_fraction_6plus=th.tensor([0.0]),
        mask_cut_fraction_4plus=th.tensor([0.0]),
        candidate_input_transfer_batch_ms=th.tensor([1.0]),
        candidate_mask_selection_batch_ms=th.tensor([2.0]),
        candidate_topology_assembly_batch_ms=th.tensor([7.0]),
        candidate_output_transfer_batch_ms=th.tensor([5.0]),
        candidate_total_batch_ms=th.tensor([15.0]),
        candidate_total_molecule_ms=th.tensor([15.0]),
        valid_ion_h_count=th.tensor([8.0]),
        retained_ion_h_count=th.tensor([6.0]),
        retained_ion_h_probability=th.tensor([0.9]),
        retained_joint_channel_intensity_fraction=th.tensor([0.85]),
        ion_h_cutoff_pruned_fraction=th.tensor([0.25]),
        valid_ion_mode_count=th.tensor([3.0]),
        retained_ion_mode_count=th.tensor([2.0]),
        retained_ion_mode_probability=th.tensor([0.95]),
        ion_mode_cutoff_pruned_fraction=th.tensor([1.0 / 3.0]),
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
    expected_spec_ce, _, _, _ = sparse_ce_fn(
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
    assert actual["candidate_wrecall"].tolist() == pytest.approx([1.0])
    assert actual["retained_mask_probability"].tolist() == pytest.approx([0.8])
    assert actual["selected_mask_count"].tolist() == [2.0]
    assert actual["component_pathway_count"].tolist() == [4.0]
    assert actual["mean_mask_cut_count"].tolist() == [2.0]
    assert actual["proposal_single_cut_fraction"].tolist() == [0.25]
    assert actual["proposal_mixture_entropy"].tolist() == pytest.approx([0.6])
    assert actual["proposal_mixture_effective_count"].tolist() == pytest.approx([1.8])
    assert actual["proposal_mixture_top1_weight"].tolist() == pytest.approx([0.7])
    assert actual["unique_component_count"].tolist() == [3.0]
    assert actual["fragment_count_before_cutoff"].tolist() == [3.0]
    assert actual["fragment_count_after_cutoff"].tolist() == [2.0]
    assert actual["retained_fragment_probability"].tolist() == pytest.approx([0.9])
    assert actual["fragment_cutoff_pruned_fraction"].tolist() == pytest.approx([1.0 / 3.0])
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
    model, _ = _build_pl(
        auxiliary_scores=["wrecall", "wprecision"],
        mass_shatter_candidate_coverage_metrics=True,
        mass_shatter_candidate_cut_diagnostics=True,
    )

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


def _looped_candidate_weighted_recall(**kwargs) -> th.Tensor:
    """Per-spectrum reference for the batched matcher, written straight from its contract."""
    batch_size = kwargs["batch_size"]
    true_ints = kwargs["true_ints"]
    recall = true_ints.new_zeros(batch_size)
    for batch_index in range(batch_size):
        peak_mask = kwargs["true_batch_idxs"] == batch_index
        batch_mzs = kwargs["true_mzs"][peak_mask]
        batch_ints = true_ints[peak_mask]
        candidate_mask = kwargs["candidate_batch_idxs"] == batch_index
        batch_candidates = kwargs["candidate_mzs"][candidate_mask]
        tol_per_sample = kwargs.get("tol_per_sample")
        if tol_per_sample is None:
            match = calculate_match_mzs(
                batch_mzs,
                batch_candidates,
                tolerance=kwargs["tolerance"],
                relative=kwargs["relative"],
                tolerance_min_mz=kwargs["tolerance_min_mz"],
            )
        else:
            match = calculate_match_mzs(
                batch_mzs,
                batch_candidates,
                tolerance=0.0,
                relative=False,
                tol_per_true=tol_per_sample[batch_index]
                * th.maximum(batch_mzs, kwargs["min_mz_per_sample"][batch_index]),
            )
        recall[batch_index] = (batch_ints * match.any(dim=1)).sum() / batch_ints.sum()
    return recall


def test_candidate_weighted_recall_matches_the_per_spectrum_reference() -> None:
    """The batched matcher runs every training step, so it must equal the loop it replaced."""
    generator = th.Generator().manual_seed(11)
    peak_counts = th.randint(1, 20, (6,), generator=generator)
    candidate_counts = th.randint(0, 80, (6,), generator=generator)
    # Spectrum 4 gets no candidates at all: its coverage must be 0, not a division by zero.
    candidate_counts[4] = 0
    true_batch_idxs = th.repeat_interleave(th.arange(6), peak_counts)
    candidate_batch_idxs = th.repeat_interleave(th.arange(6), candidate_counts)
    # A coarse shared m/z grid makes a large share of the peaks genuinely matchable.
    true_mzs = th.randint(5000, 50000, (int(peak_counts.sum()),), generator=generator) / 100
    candidate_mzs = (
        th.randint(5000, 50000, (int(candidate_counts.sum()),), generator=generator) / 100
    )
    shared = {
        "true_mzs": true_mzs.float(),
        "true_ints": th.rand(int(peak_counts.sum()), generator=generator),
        "true_batch_idxs": true_batch_idxs,
        "candidate_mzs": candidate_mzs.float(),
        "candidate_batch_idxs": candidate_batch_idxs,
        "batch_size": 6,
    }
    cases = [
        {"tolerance": 1e-5, "relative": True, "tolerance_min_mz": 200.0},
        {"tolerance": 0.01, "relative": False, "tolerance_min_mz": 200.0},
        {
            "tolerance": 1e-5,
            "relative": True,
            "tolerance_min_mz": 200.0,
            "tol_per_sample": th.full((6,), 2e-5),
            "min_mz_per_sample": th.full((6,), 750.0),
        },
    ]
    for case in cases:
        actual = _candidate_weighted_recall(**shared, **case)
        expected = _looped_candidate_weighted_recall(**shared, **case)
        # Index-add reduction reassociates the float sums; the values are otherwise the same.
        assert th.allclose(actual, expected, rtol=0, atol=1e-6)
        assert actual.shape == (6,)
        assert ((actual >= 0.0) & (actual <= 1.0)).all()
    assert _candidate_weighted_recall(**shared, **cases[0])[4].item() == 0.0


def test_candidate_weighted_recall_rejects_a_spectrum_without_peaks() -> None:
    with pytest.raises(ValueError, match="spectrum 1 has no true peaks"):
        _candidate_weighted_recall(
            true_mzs=th.tensor([100.0, 200.0]),
            true_ints=th.tensor([1.0, 1.0]),
            true_batch_idxs=th.tensor([0, 2]),
            candidate_mzs=th.tensor([100.0]),
            candidate_batch_idxs=th.tensor([0]),
            batch_size=3,
            tolerance=0.01,
            relative=False,
            tolerance_min_mz=200.0,
        )


def test_candidate_coverage_metrics_are_off_by_default() -> None:
    model, config = _build_pl()

    assert config["mass_shatter_candidate_coverage_metrics"] is False
    assert config["mass_shatter_candidate_cut_diagnostics"] is False
    assert "candidate_wrecall" not in model.loss_names
    # The candidate-generation timings keep their names; only coverage is dropped.
    coverage_names = {"candidate_wrecall", "candidate_recall"}
    assert not coverage_names & set(model.metric_names)
    assert not any(name.startswith("candidate_wrecall_cut_") for name in model.metric_names)
    # The metric function is left untouched, so it never asks for the candidate m/z values.
    assert "candidate_mzs_by_cut" not in model.extra_metric_args
    assert "candidate_batch_idxs_by_cut" not in model.extra_metric_args


def test_loss_fn_omits_candidate_wrecall_when_coverage_is_disabled() -> None:
    model, _ = _build_pl()

    actual = model.loss_fn(
        true_mzs=th.tensor([100.0]),
        true_logprobs=th.tensor([0.0]),
        true_batch_idxs=th.tensor([0]),
        pred_mzs=th.tensor([100.0, 200.0]),
        pred_logprobs=th.log(th.tensor([0.8, 0.1])),
        pred_batch_idxs=th.tensor([0, 0]),
        candidate_mzs_by_cut=th.tensor([100.0, 200.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 0]),
        batch_size=1,
        **_loss_diagnostics(),
    )

    assert "candidate_wrecall" not in actual
    # SpectrumPL logs exactly loss_names, so every one of them must still be produced.
    assert set(model.loss_names) <= set(actual)


def test_candidate_timing_metrics_are_off_by_default() -> None:
    model, config = _build_pl()

    assert config["mass_shatter_candidate_timing_metrics"] is False
    timing_names = {
        "candidate_input_transfer_batch_ms",
        "candidate_mask_selection_batch_ms",
        "candidate_topology_assembly_batch_ms",
        "candidate_output_transfer_batch_ms",
        "candidate_total_batch_ms",
        "candidate_total_molecule_ms",
    }
    assert not timing_names & set(model.metric_names)

    actual = model.loss_fn(
        true_mzs=th.tensor([100.0]),
        true_logprobs=th.tensor([0.0]),
        true_batch_idxs=th.tensor([0]),
        pred_mzs=th.tensor([100.0, 200.0]),
        pred_logprobs=th.log(th.tensor([0.8, 0.1])),
        pred_batch_idxs=th.tensor([0, 0]),
        candidate_mzs_by_cut=th.tensor([100.0, 200.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 0]),
        batch_size=1,
        **_loss_diagnostics(),
    )

    # The model still reports the timings; they are simply not logged.
    assert not timing_names & set(actual)
    assert set(model.loss_names) <= set(actual)


def test_candidate_timing_metrics_are_logged_when_enabled() -> None:
    model, _ = _build_pl(mass_shatter_candidate_timing_metrics=True)

    assert "candidate_total_batch_ms" in model.loss_names
    assert "candidate_total_molecule_ms" in model.metric_names


def test_cut_diagnostics_require_candidate_coverage_metrics() -> None:
    with pytest.raises(ValueError, match="mass_shatter_candidate_coverage_metrics"):
        _build_pl(mass_shatter_candidate_cut_diagnostics=True)


def test_cut_buckets_stay_off_when_only_the_coverage_metrics_are_enabled() -> None:
    model, config = _build_pl(
        auxiliary_scores=["wrecall"],
        mass_shatter_candidate_coverage_metrics=True,
    )

    assert config["mass_shatter_candidate_cut_diagnostics"] is False
    assert "candidate_cut_counts" not in model.extra_metric_args
    assert not any(name.startswith("candidate_wrecall_cut_") for name in model.metric_names)

    actual = model.metric_fn(
        true_mzs=th.tensor([100.0, 200.0, 400.0]),
        true_ints=th.tensor([0.6, 0.4, 1.0]),
        true_batch_idxs=th.tensor([0, 0, 1]),
        pred_mzs=th.tensor([100.0, 400.0, 500.0]),
        pred_ints=th.tensor([0.7, 0.8, 0.2]),
        pred_batch_idxs=th.tensor([0, 1, 1]),
        candidate_mzs_by_cut=th.tensor([100.0, 400.0, 500.0]),
        candidate_batch_idxs_by_cut=th.tensor([0, 1, 1]),
        true_prec_mzs=th.tensor([600.0, 600.0]),
        split="val",
    )

    # The overall coverage stays; only the per-cut-count breakdown is dropped.
    assert actual["candidate_recall"].tolist() == pytest.approx([0.5, 1.0])
    assert not any(key.startswith("candidate_recall_cut_") for key in actual)
    assert not any(key.startswith("candidate_wrecall_cut_") for key in actual)
    assert not any(key.startswith("candidate_precision_cut_") for key in actual)


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


def test_candidate_coverage_by_cut_count_reports_deep_cut_buckets() -> None:
    actual = _candidate_coverage_by_cut_count(
        true_mzs=th.tensor([100.0, 200.0, 300.0, 400.0]),
        true_ints=th.tensor([0.1, 0.2, 0.3, 0.4]),
        true_batch_idxs=th.zeros(4, dtype=th.long),
        candidate_mzs=th.tensor([100.0, 200.0, 300.0, 400.0]),
        candidate_batch_idxs=th.zeros(4, dtype=th.long),
        candidate_cut_counts=th.tensor([4, 5, 6, 7]),
        batch_size=1,
        tolerance=0.01,
        relative=False,
        tolerance_min_mz=0.0,
    )

    assert actual["candidate_recall_cut_4"].item() == pytest.approx(0.25)
    assert actual["candidate_wrecall_cut_4"].item() == pytest.approx(0.1)
    assert actual["candidate_precision_cut_4"].item() == pytest.approx(1.0)
    assert actual["candidate_recall_cut_5"].item() == pytest.approx(0.25)
    assert actual["candidate_wrecall_cut_5"].item() == pytest.approx(0.2)
    assert actual["candidate_precision_cut_5"].item() == pytest.approx(1.0)
    assert actual["candidate_recall_cut_6plus"].item() == pytest.approx(0.5)
    assert actual["candidate_wrecall_cut_6plus"].item() == pytest.approx(0.7)
    assert actual["candidate_precision_cut_6plus"].item() == pytest.approx(1.0)
    assert actual["candidate_recall_cut_4plus"].item() == pytest.approx(1.0)
    assert actual["candidate_wrecall_cut_4plus"].item() == pytest.approx(1.0)
