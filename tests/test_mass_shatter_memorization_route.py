"""Step 1 options of the MASShatter memorization-route plan.

``docs/plans/masshatter/masshatter_memorization_route_plan.md`` adds four default-off options, one
per arm: removing the boundary block (``r0``), building it from a local cut-site encoder (``r1``),
dropout on the layers that form a candidate's key (``k1``), and a per-molecule memorization sink
used only in training (``m1``). Each must leave the model exactly unchanged at its default, act
only where it says, and keep every other parameter's initialization, so an arm differs from its
control by the option alone.
"""

from types import SimpleNamespace

import pytest
import torch as th

from fragnnet.model.base_model import CutBondEncoder
from fragnnet.model.bond_fragment_heads import CandidateIntensityTransformerHead
from fragnnet.model.mass_shatter_model import MassShatterModel
from fragnnet.model.nn_blocks import GINE, GINE_GSA
from fragnnet.pl_model.mass_shatter_pl import MassShatterPL
from tests.test_mass_shatter_model import _build_batch, _build_model

UNIQUE_KWARGS = {"max_masks": 4, "max_enumerated_masks": 8, "unique_fragment_mode": True}


@pytest.fixture
def batch():
    """Two small molecules, so per-spectrum effects can be told apart."""
    return _build_batch(("CCOC", "CCO"))


def _forward(model: MassShatterModel, batch, **kwargs) -> dict[str, th.Tensor]:
    return model(
        mol_pyg=batch,
        spec_prec_type=th.tensor([0, 0]),
        spec_nce=th.tensor([10.0, 40.0]),
        **kwargs,
    )


def _head(**kwargs) -> CandidateIntensityTransformerHead:
    th.manual_seed(17)
    return CandidateIntensityTransformerHead(
        fragment_embedding_dim=6,
        context_dim=0,
        model_dim=8,
        num_layers=1,
        num_heads=2,
        feedforward_dim=16,
        num_inducing_points=2,
        num_ion_modes=1,
        num_h_shifts=3,
        dropout=0.0,
        use_base_probability_feature=False,
        unique_fragment_tokens=True,
        **kwargs,
    )


def _head_inputs(num_components: int = 5) -> tuple[tuple, dict]:
    th.manual_seed(3)
    embeddings = tuple(th.randn(num_components, 6) for _ in range(3))
    zeros = th.zeros(num_components)
    batch_indices = th.tensor([0, 0, 0, 1, 1])[:num_components]
    args = (*embeddings, zeros, zeros, batch_indices, th.arange(num_components))
    return args, {"batch_size": 2, "num_masks": num_components}


# 1. CutBondEncoder hidden-layer dropout (arm k1).


def test_cut_bond_encoder_dropout_keeps_state_dict_keys() -> None:
    """Dropout lives in forward, so checkpoints written before it existed still load."""
    plain = CutBondEncoder(atom_dim=4, hidden_dim=8, out_dim=4)
    dropped = CutBondEncoder(atom_dim=4, hidden_dim=8, out_dim=4, dropout=0.3)

    assert list(dropped.state_dict()) == ["net.0.weight", "net.0.bias", "net.2.weight", "net.2.bias"]
    dropped.load_state_dict(plain.state_dict(), strict=True)


def test_cut_bond_encoder_without_dropout_matches_its_mlp_and_draws_nothing() -> None:
    th.manual_seed(1)
    encoder = CutBondEncoder(atom_dim=4, hidden_dim=8, out_dim=4).train()
    inside, outside = th.randn(6, 4), th.randn(6, 4)
    rng_before = th.get_rng_state()

    output = encoder(inside, outside)

    assert th.equal(th.get_rng_state(), rng_before)
    assert th.equal(output, encoder.net(th.cat([inside, outside, inside * outside], dim=-1)))


def test_cut_bond_encoder_dropout_is_training_only() -> None:
    th.manual_seed(1)
    plain = CutBondEncoder(atom_dim=4, hidden_dim=64, out_dim=4)
    dropped = CutBondEncoder(atom_dim=4, hidden_dim=64, out_dim=4, dropout=0.5)
    dropped.load_state_dict(plain.state_dict())
    inside, outside = th.randn(6, 4), th.randn(6, 4)

    assert th.equal(dropped.eval()(inside, outside), plain.eval()(inside, outside))
    th.manual_seed(2)
    assert not th.allclose(dropped.train()(inside, outside), plain.train()(inside, outside))


def test_cut_bond_encoder_rejects_certain_dropout() -> None:
    with pytest.raises(ValueError, match="dropout must be in"):
        CutBondEncoder(atom_dim=4, hidden_dim=8, out_dim=4, dropout=1.0)


# 2. Dropout on the projected candidate token (arm k1).


def test_intensity_input_dropout_is_training_only_and_adds_no_parameters() -> None:
    """Read on the ion/H logits: the residual heads start at zero, whatever the token holds."""
    args, kwargs = _head_inputs()
    plain, dropped = _head(), _head(input_dropout=0.5)
    assert list(dropped.state_dict()) == list(plain.state_dict())
    dropped.load_state_dict(plain.state_dict())

    assert th.equal(dropped.eval()(*args, **kwargs)[2], plain.eval()(*args, **kwargs)[2])
    th.manual_seed(4)
    assert not th.allclose(dropped.train()(*args, **kwargs)[2], plain.train()(*args, **kwargs)[2])


def test_intensity_input_dropout_at_zero_draws_no_random_numbers() -> None:
    args, kwargs = _head_inputs()
    head = _head().train()
    rng_before = th.get_rng_state()

    head(*args, **kwargs)

    assert th.equal(th.get_rng_state(), rng_before)


def test_key_dropout_model_matches_its_control_at_evaluation(batch) -> None:
    """Both k1 rates add no parameters, so at eval the arm is its control."""
    kwargs = {**UNIQUE_KWARGS, "boundary_bond_encoder": True}
    control = _build_model(batch, **kwargs).eval()
    arm = _build_model(
        batch,
        boundary_encoder_dropout=0.3,
        intensity_input_dropout=0.3,
        **kwargs,
    ).eval()

    assert arm.boundary_bond_encoder.dropout == 0.3
    assert arm.intensity_head.input_dropout == 0.3
    assert th.equal(_forward(arm, batch)["pred_logprobs"], _forward(control, batch)["pred_logprobs"])


# 3. Removing the boundary block (arm r0).


@pytest.mark.parametrize("training", [True, False])
def test_boundary_block_off_feeds_zeros_in_training_and_evaluation(
    batch,
    training: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(batch, candidate_boundary_block=False, **UNIQUE_KWARGS)
    model.train(training)
    captured: list[th.Tensor] = []
    real_forward = model.intensity_head.forward

    def _spy(fragment, complement, boundary, *args, **kwargs):
        captured.append(boundary)
        return real_forward(fragment, complement, boundary, *args, **kwargs)

    monkeypatch.setattr(model.intensity_head, "forward", _spy)
    output = _forward(model, batch)

    assert captured[-1].shape == (captured[-1].shape[0], 16)
    assert th.equal(captured[-1], th.zeros_like(captured[-1]))
    assert th.isfinite(output["pred_logprobs"]).all()


@pytest.mark.parametrize(
    "block_option",
    [
        {"boundary_bond_encoder": True},
        {"candidate_boundary_dropout": 0.2},
    ],
)
def test_boundary_block_off_rejects_options_that_act_on_it(batch, block_option: dict) -> None:
    with pytest.raises(ValueError, match="removes the boundary block"):
        _build_model(batch, candidate_boundary_block=False, **UNIQUE_KWARGS, **block_option)


# 4. A local cut-site encoder for the boundary block (arm r1).


def test_boundary_local_encoder_is_plain_gine(batch) -> None:
    """The shared encoder may carry global attention; the cut-site encoder never does."""
    model = _build_model(
        batch,
        precursor_gnn_type="GINE_GSA",
        boundary_bond_encoder=True,
        boundary_local_layers=1,
        **UNIQUE_KWARGS,
    )

    assert isinstance(model.precursor_encoder.gnn, GINE_GSA)
    assert isinstance(model.boundary_local_encoder.gnn, GINE)
    assert model.boundary_local_encoder.num_layers == 1


def test_boundary_local_encoder_feeds_the_cut_bond_encoder(
    batch,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _build_model(
        batch,
        precursor_gnn_type="GINE_GSA",
        boundary_bond_encoder=True,
        boundary_local_layers=1,
        **UNIQUE_KWARGS,
    ).eval()
    captured: list[th.Tensor] = []
    real_boundary = model._boundary_embeddings

    def _spy(mol_pyg, atom_embeddings, *args):
        captured.append(atom_embeddings)
        return real_boundary(mol_pyg, atom_embeddings, *args)

    monkeypatch.setattr(model, "_boundary_embeddings", _spy)
    _forward(model, batch)
    local = model.boundary_local_encoder(batch.x, batch.batch, batch.edge_index, batch.edge_attr)
    shared = model.precursor_encoder(batch.x, batch.batch, batch.edge_index, batch.edge_attr)

    assert th.allclose(captured[-1], local)
    assert not th.allclose(captured[-1], shared)


def test_boundary_local_encoder_leaves_every_other_initialization_unchanged(batch) -> None:
    kwargs = {**UNIQUE_KWARGS, "boundary_bond_encoder": True}
    control = _build_model(batch, **kwargs).state_dict()
    arm = _build_model(batch, boundary_local_layers=1, **kwargs).state_dict()

    new_keys = set(arm) - set(control)
    assert new_keys and all(key.startswith("boundary_local_encoder.") for key in new_keys)
    assert set(control) <= set(arm)
    assert all(th.equal(arm[key], control[key]) for key in control)


def test_boundary_local_layers_require_the_cut_bond_encoder(batch) -> None:
    with pytest.raises(ValueError, match="require boundary_bond_encoder"):
        _build_model(batch, boundary_local_layers=1, **UNIQUE_KWARGS)


# 5. The memorization sink (arm m1).


def _sink_model(batch, **kwargs) -> MassShatterModel:
    return _build_model(
        batch,
        memorization_sink_dim=3,
        memorization_sink_num_molecules=10,
        **UNIQUE_KWARGS,
        **kwargs,
    )


def test_memorization_sink_starts_as_the_shared_model(batch) -> None:
    """Zero rows and a forked RNG: at init the sink changes neither outputs nor other weights."""
    control = _build_model(batch, **UNIQUE_KWARGS).train()
    model = _sink_model(batch).train()
    control_state, arm_state = control.state_dict(), model.state_dict()

    assert set(arm_state) - set(control_state) == {
        "memorization_sink.weight",
        "intensity_head.sink_projection.weight",
    }
    assert all(th.equal(arm_state[key], control_state[key]) for key in control_state)
    assert th.equal(model.memorization_sink.weight, th.zeros(10, 3))
    with_sink = _forward(model, batch, memorization_sink_indices=th.tensor([3, 7]))
    assert th.equal(with_sink["pred_logprobs"], _forward(control, batch)["pred_logprobs"])


def test_memorization_sink_moves_only_the_molecule_whose_row_is_set(batch) -> None:
    model = _sink_model(batch).train()
    indices = th.tensor([3, 7])
    before = _forward(model, batch, memorization_sink_indices=indices)
    with th.no_grad():
        model.memorization_sink.weight[3] = th.tensor([2.0, -1.0, 0.5])
    after = _forward(model, batch, memorization_sink_indices=indices)

    first = before["pred_batch_idxs"] == 0
    assert th.equal(after["pred_batch_idxs"], before["pred_batch_idxs"])
    assert not th.allclose(after["pred_logprobs"][first], before["pred_logprobs"][first])
    assert th.equal(after["pred_logprobs"][~first], before["pred_logprobs"][~first])


def test_memorization_sink_is_unused_without_indices(batch) -> None:
    """Evaluation passes no indices, so it scores the shared model whatever the rows hold."""
    model = _sink_model(batch).eval()
    control = _build_model(batch, **UNIQUE_KWARGS).eval()
    with th.no_grad():
        model.memorization_sink.weight.normal_()

    assert th.equal(_forward(model, batch)["pred_logprobs"], _forward(control, batch)["pred_logprobs"])


def test_memorization_sink_gradient_reaches_only_the_batch_molecules(batch) -> None:
    model = _sink_model(batch).train()

    _forward(model, batch, memorization_sink_indices=th.tensor([3, 7]))["pred_logprobs"].sum().backward()

    row_has_grad = model.memorization_sink.weight.grad.abs().sum(dim=1) > 0
    assert row_has_grad.tolist() == [index in (3, 7) for index in range(10)]


def test_memorization_sink_rejects_out_of_range_and_unexpected_indices(batch) -> None:
    with pytest.raises(ValueError, match="memorization_sink_indices must lie in"):
        _forward(_sink_model(batch), batch, memorization_sink_indices=th.tensor([3, 10]))
    with pytest.raises(ValueError, match="model without a sink"):
        _forward(
            _build_model(batch, **UNIQUE_KWARGS),
            batch,
            memorization_sink_indices=th.tensor([3, 7]),
        )


@pytest.mark.parametrize(
    ("options", "message"),
    [
        ({"memorization_sink_dim": 3}, "set together"),
        ({"memorization_sink_num_molecules": 10}, "set together"),
        (
            {"memorization_sink_dim": 3, "memorization_sink_num_molecules": 10},
            "requires unique_fragment_mode",
        ),
    ],
)
def test_memorization_sink_rejects_incomplete_configurations(
    batch,
    options: dict,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _build_model(batch, **options)


def _sink_pl(*, sink_dim: int, training: bool, epoch: int) -> SimpleNamespace:
    hparams = SimpleNamespace(
        mass_shatter_memorization_sink_dim=sink_dim,
        mass_shatter_memorization_sink_start_epoch=4,
    )
    return SimpleNamespace(hparams=hparams, training=training, current_epoch=epoch)


@pytest.mark.parametrize(
    ("sink_dim", "training", "epoch"),
    [(0, True, 9), (16, False, 9), (16, True, 3)],
)
def test_pl_passes_no_sink_indices_outside_sink_training(
    sink_dim: int,
    training: bool,
    epoch: int,
) -> None:
    """No sink configured, evaluation, and the epochs before the start all run the shared model."""
    pl = _sink_pl(sink_dim=sink_dim, training=training, epoch=epoch)
    batch_kwargs = {"mol_id": [12, 7], "spec_prec_type": th.tensor([0, 0])}

    assert MassShatterPL._memorization_sink_indices(pl, batch_kwargs) is None


def test_pl_passes_each_spectrums_mol_id_during_sink_training() -> None:
    pl = _sink_pl(sink_dim=16, training=True, epoch=4)
    batch_kwargs = {"mol_id": [12, 7, 12], "spec_prec_type": th.tensor([0, 0, 0])}

    indices = MassShatterPL._memorization_sink_indices(pl, batch_kwargs)

    assert indices.dtype == th.long
    assert indices.device == batch_kwargs["spec_prec_type"].device
    assert indices.tolist() == [12, 7, 12]
