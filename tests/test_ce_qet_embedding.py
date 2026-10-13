"""Collision energy on the breakdown-curve axis: smooth, alias-free, missing-aware, legacy intact.

``ce_insert_type: embed`` looks the rounded CE up in the ``int_embedder`` table. The default
abs-sines table gives x and 2**k - x identical rows (NCE 76 and 180 at nce_max 200, ACE 30 and
98 eV at ace_max 100) and neighbouring energies unrelated rows. ``ce_insert_type: qet`` embeds the
raw CE as a standardized 1 / (CE + 200) plus a missing-CE flag through a learned linear layer, in
the shared CE model (FraGNNet and the other models) and in MASSter's context encoder. Existing
``embed`` checkpoints store the abs-sines table, so it must keep its shape. ``qet_mlp`` replaces the
linear layer with Linear -> SiLU -> Linear; ``qet`` keeps its checkpoint keys.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as th

from fragnnet.layers.int_featurizer import FourierFeaturizerAbsoluteSines
from fragnnet.layers.metadata_embedding import QET_ENERGY_OFFSET, QETEnergyEncoder
from tests.test_ce_multi_location import _build
from tests.test_masster_model import _build_batch, _build_model
from tests.test_merged_ce_context import _context_encoder

NCE_MAX = 200
NCE_MEAN, NCE_STD = 60.0, 40.0


def _masster_nce(ce_insert_type: str) -> tuple[th.nn.Module, Callable[[th.Tensor], th.Tensor]]:
    """MASSter's NCE embedding module and its per-reading embedding function."""
    encoder = _context_encoder(
        ce_insert_type=ce_insert_type, nce_max=NCE_MAX, nce_mean=NCE_MEAN, nce_std=NCE_STD
    )
    assert encoder.nce_embedding is not None

    def embed(values: th.Tensor) -> th.Tensor:
        return encoder._embed_ce_values(
            values, None, values.numel(), encoder.nce_max, encoder.nce_embedding
        )

    return encoder.nce_embedding, embed


def _shared_nce(ce_insert_type: str) -> tuple[th.nn.Module, Callable[[th.Tensor], th.Tensor]]:
    """The shared CE model's NCE embedding module and embedding function, as FraGNNet builds them."""
    model = _build(
        ce_insert_type=ce_insert_type,
        int_embedder="abs-sines",
        nce_max=float(NCE_MAX),
        nce_mean=NCE_MEAN,
        nce_std=NCE_STD,
    )

    def embed(values: th.Tensor) -> th.Tensor:
        return model.ce_embedder(model.ce_transform(values))

    return model.ce_embedder, embed


PATHS = {"masster": _masster_nce, "shared": _shared_nce}


def _z_reader(mean: float = NCE_MEAN, std: float = NCE_STD) -> QETEnergyEncoder:
    """An encoder whose output is exactly z: weight (1, 0), bias 0."""
    encoder = QETEnergyEncoder(mean, std, out_dim=1)
    with th.no_grad():
        encoder.linear.weight.copy_(th.tensor([[1.0, 0.0]]))
        encoder.linear.bias.zero_()
    return encoder


@pytest.mark.parametrize("path", PATHS)
def test_qet_uses_the_energy_encoder_and_embed_keeps_the_abs_sines_table(path: str) -> None:
    qet_module, _ = PATHS[path]("qet")
    legacy_module, _ = PATHS[path]("embed")

    assert isinstance(qet_module, QETEnergyEncoder)
    assert tuple(qet_module.linear.weight.shape) == (qet_module.linear.out_features, 2)
    # Checkpoints trained with "embed" store this exact table; its shape must not change.
    assert isinstance(legacy_module[0], FourierFeaturizerAbsoluteSines)
    assert tuple(legacy_module[0].int_to_feat_matrix.shape) == (NCE_MAX + 1, 8)


@pytest.mark.parametrize("path", PATHS)
def test_qet_separates_the_abs_sines_alias_pair(path: str) -> None:
    pair = th.tensor([76.0, 180.0])
    _, qet_embed = PATHS[path]("qet")
    _, legacy_embed = PATHS[path]("embed")

    qet = qet_embed(pair)
    legacy = legacy_embed(pair)

    assert not th.allclose(qet[0], qet[1], atol=1e-3)
    # The defect "qet" exists for: 256 - 76 = 180, so abs-sines cannot tell them apart.
    th.testing.assert_close(legacy[0], legacy[1], atol=1e-4, rtol=0.0)


def test_qet_axis_is_monotone_smooth_and_unclamped() -> None:
    reader = _z_reader()
    energies = th.arange(0.0, 301.0)

    z = reader(energies).squeeze(1)

    assert (z[1:] > z[:-1]).all()
    # Standardized near the mean: z(mean) = 0 and dz/dCE = 1 / std there.
    assert z[int(NCE_MEAN)].item() == pytest.approx(0.0, abs=1e-6)
    slope = (z[int(NCE_MEAN) + 1] - z[int(NCE_MEAN) - 1]).item() / 2.0
    assert slope == pytest.approx(1.0 / NCE_STD, rel=1e-3)
    # The axis is the breakdown curve's 1 / (CE + offset), up to an affine map.
    beta = 1.0 / (energies + QET_ENERGY_OFFSET)
    th.testing.assert_close(z, (beta[int(NCE_MEAN)] - beta) * (NCE_MEAN + 200.0) ** 2 / NCE_STD)
    # No clamp at nce_max: energies past the table keep rising.
    assert z[250] > z[200]


def test_qet_embeds_an_unseen_energy_between_its_neighbours() -> None:
    encoder = QETEnergyEncoder(NCE_MEAN, NCE_STD, out_dim=8)
    e165, e170, e175 = encoder(th.tensor([165.0, 170.0, 175.0]))
    z165, z170, z175 = _z_reader()(th.tensor([165.0, 170.0, 175.0])).squeeze(1)

    # Affine in z: 170 sits on the segment between 165 and 175.
    t = (z170 - z165) / (z175 - z165)
    th.testing.assert_close(e170, e165 + t * (e175 - e165))
    assert 0.0 < t.item() < 1.0


@pytest.mark.parametrize("path", PATHS)
def test_qet_missing_ce_uses_its_own_flag(path: str) -> None:
    module, embed = PATHS[path]("qet")
    missing, at_mean, zero = embed(th.tensor([-1.0, NCE_MEAN, 0.0]))

    # Missing reads z = 0 and flag 1, so it differs from the mean energy (z = 0, flag 0) only by
    # the learned flag column, and from CE 0 as well.
    th.testing.assert_close(missing - at_mean, module.linear.weight[:, 1])
    assert not th.allclose(missing, zero)


def test_masster_qet_pools_stepped_readings_per_spectrum() -> None:
    encoder = _context_encoder(ce_insert_type="qet", nce_mean=NCE_MEAN, nce_std=NCE_STD)
    values = th.tensor([20.0, 40.0, 60.0])
    per_reading = encoder.nce_embedding(values)

    pooled = encoder._embed_ce_values(
        values, th.tensor([0, 0, 1]), 2, encoder.nce_max, encoder.nce_embedding
    )

    th.testing.assert_close(
        pooled, th.stack(((per_reading[0] + per_reading[1]) / 2, per_reading[2]))
    )


def test_qet_separates_the_ace_alias_pair() -> None:
    kwargs = {"use_nce": False, "use_ace": True, "ace_max": 100, "ace_mean": 20, "ace_std": 15}
    encoder = _context_encoder(ce_insert_type="qet", **kwargs)
    legacy = _context_encoder(ce_insert_type="embed", **kwargs)
    pair = th.tensor([30.0, 98.0])

    qet = encoder._embed_ce_values(pair, None, 2, encoder.ace_max, encoder.ace_embedding)
    old = legacy._embed_ce_values(pair, None, 2, legacy.ace_max, legacy.ace_embedding)

    assert isinstance(encoder.ace_embedding, QETEnergyEncoder)
    assert not th.allclose(qet[0], qet[1], atol=1e-3)
    # 128 - 30 = 98: the ace_max 100 table (the template default before 2026-09-28) aliases
    # common QTOF energies.
    th.testing.assert_close(old[0], old[1], atol=1e-4, rtol=0.0)


def test_masster_qet_model_trains_the_energy_encoder() -> None:
    batch = _build_batch()
    model = _build_model(batch, ce_insert_type="qet")
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([25.0, 170.0]),
    )

    (-output["pred_logprobs"].mean()).backward()

    encoder = model.context_encoder.nce_embedding
    assert isinstance(encoder, QETEnergyEncoder)
    assert encoder.linear.weight.grad is not None
    assert th.isfinite(encoder.linear.weight.grad).all()
    assert encoder.linear.weight.grad.abs().sum() > 0


def test_shared_qet_embeds_merged_readings_and_trains() -> None:
    model = _build(ce_insert_type="qet", ce_insert_merge=True, nce_mean=NCE_MEAN, nce_std=NCE_STD)
    nce = th.tensor([20.0, 30.0, 40.0, -1.0])

    out = model.embed_ce(nce, th.tensor([0, 0, 0, 1]), 2)

    per_reading = model.ce_embedder(model.ce_transform(nce))
    th.testing.assert_close(out, th.stack((per_reading[:3].mean(0), per_reading[3])))
    out.square().sum().backward()
    assert model.ce_embedder.linear.weight.grad.abs().sum() > 0


@pytest.mark.parametrize(
    "kwargs", [{"std": 0.0}, {"mean": -250.0}, {"out_dim": 0}, {"hidden_dim": 0}]
)
def test_qet_encoder_rejects_invalid_settings(kwargs: dict) -> None:
    settings = {"mean": NCE_MEAN, "std": NCE_STD, "out_dim": 4} | kwargs
    with pytest.raises(ValueError):
        QETEnergyEncoder(**settings)


def test_qet_linear_mode_is_standardized_ev_with_the_missing_flag() -> None:
    encoder = QETEnergyEncoder(20.0, 15.0, out_dim=1, offset=None)
    with th.no_grad():
        encoder.linear.weight.copy_(th.tensor([[1.0, 0.0]]))
        encoder.linear.bias.zero_()

    z = encoder(th.tensor([5.0, 20.0, 35.0, 250.0, -1.0])).squeeze(1)

    th.testing.assert_close(z, th.tensor([-1.0, 0.0, 1.0, 230.0 / 15.0, 0.0]))


def test_masster_qet_keeps_the_nce_offset_and_makes_ace_linear() -> None:
    encoder = _context_encoder(ce_insert_type="qet", use_nce=True, use_ace=True)

    assert encoder.nce_embedding.offset == pytest.approx(QET_ENERGY_OFFSET)
    assert encoder.ace_embedding.offset is None


def test_shared_qet_keeps_the_nce_offset_and_makes_ace_linear() -> None:
    model = _build(ce_insert_type="qet", use_ace=True, ace_max=100.0)

    assert model.ce_embedder.offset == pytest.approx(QET_ENERGY_OFFSET)
    assert model.ce_embedder_ace.offset is None
    # Linear in eV: equal eV steps give equal embedding steps.
    e = model.ce_embedder_ace(model.ace_transform(th.tensor([10.0, 20.0, 30.0])))
    th.testing.assert_close(e[1] - e[0], e[2] - e[1])


def test_qet_keeps_its_linear_checkpoint_keys() -> None:
    # Checkpoints trained with "qet" (n1, jxczo2kf) store exactly these tensors.
    encoder = QETEnergyEncoder(NCE_MEAN, NCE_STD, out_dim=8)

    assert set(encoder.state_dict()) == {"linear.weight", "linear.bias"}
    assert encoder.output is None


@pytest.mark.parametrize("path", PATHS)
def test_qet_mlp_embeds_through_a_hidden_layer(path: str) -> None:
    module, embed = PATHS[path]("qet_mlp")

    out = embed(th.tensor([-1.0, 76.0, 180.0]))

    assert isinstance(module, QETEnergyEncoder)
    assert module.output is not None
    width = module.output.out_features
    # Hidden width is ce_insert_size, the same as the output width.
    assert tuple(module.linear.weight.shape) == (width, 2)
    assert tuple(module.output.weight.shape) == (width, width)
    assert out.shape == (3, width)
    assert not th.allclose(out[1], out[2], atol=1e-3)


def test_qet_mlp_is_nonlinear_on_the_energy_axis() -> None:
    th.manual_seed(0)
    encoder = QETEnergyEncoder(NCE_MEAN, NCE_STD, out_dim=8, hidden_dim=16)
    energies = th.tensor([20.0, 60.0, 160.0])
    e20, e60, e160 = encoder(energies)
    z20, z60, z160 = _z_reader()(energies).squeeze(1)

    # Unlike "qet", the midpoint energy leaves the segment between its neighbours.
    t = (z60 - z20) / (z160 - z20)
    assert (e60 - (e20 + t * (e160 - e20))).abs().max() > 1e-3


def test_masster_qet_mlp_model_trains_both_layers() -> None:
    batch = _build_batch()
    model = _build_model(batch, ce_insert_type="qet_mlp")
    output = model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 1]),
        spec_nce=th.tensor([25.0, 170.0]),
    )

    (-output["pred_logprobs"].mean()).backward()

    encoder = model.context_encoder.nce_embedding
    assert isinstance(encoder, QETEnergyEncoder)
    for layer in (encoder.linear, encoder.output):
        assert layer.weight.grad is not None
        assert th.isfinite(layer.weight.grad).all()
        assert layer.weight.grad.abs().sum() > 0
