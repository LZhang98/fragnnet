"""Integration tests for MASShatter's SpectrumPL training wrapper."""

from pathlib import Path

import pytest
import torch as th
import yaml

from fragnnet.dataset import SpecMolDataset
from fragnnet.model.loss import get_sparse_cross_entropy_fn
from fragnnet.pl_model import MassShatterPL, SpectrumPL
from fragnnet.utils.setup_utils import get_dataset_cls, get_model_cls


def _build_pl() -> tuple[MassShatterPL, dict]:
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
        mass_shatter_max_h_transfer=1,
    )
    return MassShatterPL(**config), config


def test_mass_shatter_registration_uses_spectrum_only_dataset() -> None:
    assert get_model_cls("mass_shatter") is MassShatterPL
    assert get_dataset_cls("mass_shatter") is SpecMolDataset
    assert issubclass(MassShatterPL, SpectrumPL)


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
        component_pathway_count=th.tensor([4]),
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
    assert actual["component_pathway_count"].tolist() == [4.0]
