"""Tests for NeimsModel's collision-energy input."""

import pytest
import torch as th

from fragnnet.model.neims_model import NeimsModel


def _neims(use_nce: bool, use_ace: bool) -> NeimsModel:
    return NeimsModel(
        mol_fingerprint_morgan=True,
        mol_fingerprint_rdkit=False,
        mol_fingerprint_maccs=False,
        morgan_radius=2,
        morgan_nbits=64,
        rdkit_nbits=64,
        mlp_hidden_size=16,
        mlp_dropout=0.0,
        mlp_num_layers=1,
        mlp_use_residuals=True,
        neims_bottleneck_factor=0.5,
        mz_max=300.0,
        mz_bin_res=1.0,
        ff_prec_mz_offset=0,
        ff_bidirectional=True,
        ff_output_map_size=-1,
        ff_output_activation="relu",
        int_embedder="abs-sines",
        ce_insert_type="qet",
        ce_insert_location="mlp",
        ce_insert_merge=True,
        ce_insert_size=8,
        use_nce=use_nce,
        nce_mean=68.3,
        nce_std=59.1,
        nce_max=300.0,
        use_ace=use_ace,
        ace_max=300.0,
        ace_mean=37.8,
        ace_std=27.8,
        prec_insert_location="none",
        prec_insert_size=8,
        prec_types=["[M+H]+"],
        inst_insert_location="mlp",
        inst_insert_size=8,
        inst_types=["ORB", "QTOF"],
        log_min=1e-9,
    )


def _batch(n: int = 3) -> dict:
    return {
        "mol_fingerprint": (th.arange(n * 64).reshape(n, 64) % 3 == 0).float(),
        "spec_prec_mz": th.linspace(100.0, 250.0, n),
        "spec_ce": th.linspace(20.0, 60.0, n),
        "spec_ce_batch_idxs": th.arange(n),
        "spec_ace": th.linspace(10.0, 40.0, n),
        "spec_ace_batch_idxs": th.arange(n),
        "spec_inst_type": th.arange(n) % 2,
    }


@pytest.mark.parametrize("use_nce, use_ace", [(True, False), (False, True), (True, True)])
def test_every_ce_channel_combination_predicts(use_nce, use_ace):
    """ACE reaches the embedding; before 2026-09-27 any ACE config failed its assert."""
    model = _neims(use_nce, use_ace).eval()
    assert model.ce_mlp_input_dim == 8 * (int(use_nce) + int(use_ace))
    out = model(**_batch())
    assert out["pred_specs"].shape == (3, 300)
    th.testing.assert_close(out["pred_specs"].sum(dim=1), th.ones(3))


def test_the_ace_reading_changes_the_prediction():
    model = _neims(use_nce=False, use_ace=True).eval()
    batch = _batch()
    base = model(**batch)["pred_specs"]
    shifted = model(**{**batch, "spec_ace": batch["spec_ace"] + 30.0})["pred_specs"]
    assert not th.allclose(base, shifted)
