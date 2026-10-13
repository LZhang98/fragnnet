"""Step 1 options of the MASSter memorization-route plan.

``docs/plans/masster/masster_memorization_route_plan.md`` adds four default-off options, one
per arm: removing the boundary block (``r0``), building it from a local cut-site encoder (``r1``),
dropout on the layers that form a candidate's key (``k1``), and a per-molecule memorization sink
used only in training (``m1``). Later arms (``gm1``-``gm6``) add training-only noise to the
shared encoder's global-attention message, ``am1``/``de1`` mask atoms or drop bonds in its
inputs, ``sm1`` swaps the allocator's per-candidate set message for its set's mean, and ``sd1``
hides random candidates from the keys the allocator summarizes its set from. Each must
leave the model exactly unchanged at its default, act only where it says, and
keep every other parameter's initialization, so an arm differs from its control by the option
alone.
"""

from types import SimpleNamespace

import pytest
import torch as th

from fragnnet import runner
from fragnnet.layers.cut_bond_encoder import CutBondEncoder
from fragnnet.layers.gine_gsa import GINE_GSA
from fragnnet.layers.gnn import GINE
from fragnnet.models.masster.candidate_transformer import (
    CandidateIntensityTransformerHead,
    _drop_set_keys,
    _InducedCandidateAttentionLayer,
    _substitute_set_message,
)
from fragnnet.models.masster.masster_model import MassterModel
from fragnnet.models.masster.masster_pl import MassterPL
from tests.test_masster_model import _build_batch, _build_model

UNIQUE_KWARGS = {"max_masks": 4, "max_enumerated_masks": 8, "unique_fragment_mode": True}


@pytest.fixture
def batch():
    """Two small molecules, so per-spectrum effects can be told apart."""
    return _build_batch(("CCOC", "CCO"))


def _forward(model: MassterModel, batch, **kwargs) -> dict[str, th.Tensor]:
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


def _sink_model(batch, **kwargs) -> MassterModel:
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


# 5b. Sink dropout (arm m2): each training spectrum runs with or without its sink row.


def _sink_dropout_outputs(batch, dropout: float) -> tuple[MassterModel, th.Tensor, th.Tensor]:
    """Return a sink model with rows 3 and 7 set, its sink-on output and its shared output."""
    model = _sink_model(batch, memorization_sink_dropout=dropout).train()
    with th.no_grad():
        model.memorization_sink.weight[3] = th.tensor([2.0, -1.0, 0.5])
        model.memorization_sink.weight[7] = th.tensor([-1.5, 0.5, 1.0])
    model.memorization_sink_dropout = 0.0
    sink_on = _forward(model, batch, memorization_sink_indices=th.tensor([3, 7]))["pred_logprobs"]
    model.memorization_sink_dropout = dropout
    shared = _forward(model, batch)["pred_logprobs"]
    return model, sink_on, shared


def test_memorization_sink_dropout_adds_no_parameters(batch) -> None:
    """m2 differs from m1 by a probability alone, so m1 checkpoints and initialization carry over."""
    m1_state = _sink_model(batch).state_dict()
    m2_state = _sink_model(batch, memorization_sink_dropout=0.5).state_dict()

    assert list(m2_state) == list(m1_state)
    assert all(th.equal(m2_state[key], m1_state[key]) for key in m1_state)


def test_memorization_sink_dropout_runs_each_spectrum_with_or_without_its_row(batch) -> None:
    """Every spectrum matches exactly the sink-on or the shared output, drawn per spectrum."""
    model, sink_on, shared = _sink_dropout_outputs(batch, dropout=0.5)
    first = _forward(model, batch)["pred_batch_idxs"] == 0
    assert not th.allclose(sink_on[first], shared[first])
    assert not th.allclose(sink_on[~first], shared[~first])

    patterns = set()
    for seed in range(24):
        th.manual_seed(seed)
        out = _forward(model, batch, memorization_sink_indices=th.tensor([3, 7]))["pred_logprobs"]
        kept = []
        for rows in (first, ~first):
            is_on = th.equal(out[rows], sink_on[rows])
            assert is_on != th.equal(out[rows], shared[rows])
            kept.append(is_on)
        patterns.add(tuple(kept))

    # 24 draws at 0.5: all four keep patterns appear, so the two spectra are drawn independently.
    assert patterns == {(True, True), (True, False), (False, True), (False, False)}


def test_memorization_sink_dropout_is_training_only(batch) -> None:
    """In eval mode the rows are never dropped; evaluation passes no indices in any case."""
    model, sink_on, _ = _sink_dropout_outputs(batch, dropout=0.9)
    model.eval()
    for seed in range(5):
        th.manual_seed(seed)
        out = _forward(model, batch, memorization_sink_indices=th.tensor([3, 7]))["pred_logprobs"]
        assert th.equal(out, sink_on)


def test_memorization_sink_dropout_rejects_invalid_settings(batch) -> None:
    with pytest.raises(ValueError, match="requires memorization_sink_dim"):
        _build_model(batch, memorization_sink_dropout=0.5, **UNIQUE_KWARGS)
    for dropout in (-0.1, 1.0):
        with pytest.raises(ValueError, match=r"must lie in \[0, 1\)"):
            _sink_model(batch, memorization_sink_dropout=dropout)


def _sink_pl(*, sink_dim: int, training: bool, epoch: int) -> SimpleNamespace:
    hparams = SimpleNamespace(
        masster_memorization_sink_dim=sink_dim,
        masster_memorization_sink_start_epoch=4,
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

    assert MassterPL._memorization_sink_indices(pl, batch_kwargs) is None


def test_pl_passes_each_spectrums_mol_id_during_sink_training() -> None:
    pl = _sink_pl(sink_dim=16, training=True, epoch=4)
    batch_kwargs = {"mol_id": [12, 7, 12], "spec_prec_type": th.tensor([0, 0, 0])}

    indices = MassterPL._memorization_sink_indices(pl, batch_kwargs)

    assert indices.dtype == th.long
    assert indices.device == batch_kwargs["spec_prec_type"].device
    assert indices.tolist() == [12, 7, 12]


# 6. Noise on the shared encoder's global-attention message (arms gm1-gm6).

GLOBAL_MESSAGE_OPTIONS = [
    {"global_message_substitution": 0.3},
    {"global_message_noise_std": 0.5},
    {"global_message_swap": 0.3, "global_message_swap_unit": "atom"},
    # Two molecules at 0.3 leave both unswapped in about a quarter of draws.
    {"global_message_swap": 0.9, "global_message_swap_unit": "molecule"},
    {"global_message_swap": 0.3, "global_message_swap_unit": "feature"},
]


@pytest.mark.parametrize("option", GLOBAL_MESSAGE_OPTIONS)
def test_global_message_noise_reaches_only_the_shared_encoder(batch, option: dict) -> None:
    model = _build_model(
        batch,
        precursor_gnn_type="GINE_GSA",
        separate_proposal_encoder=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        **UNIQUE_KWARGS,
        **option,
    )
    shared, proposal = model.precursor_encoder.gnn, model.proposal_encoder.gnn

    assert shared.global_message_substitution == option.get("global_message_substitution", 0.0)
    assert shared.global_message_noise_std == option.get("global_message_noise_std", 0.0)
    assert shared.global_message_swap == option.get("global_message_swap", 0.0)
    assert shared.global_message_swap_unit == option.get("global_message_swap_unit", "atom")
    assert proposal.global_message_substitution == 0.0
    assert proposal.global_message_noise_std == 0.0
    assert proposal.global_message_swap == 0.0


@pytest.mark.parametrize("option", GLOBAL_MESSAGE_OPTIONS)
def test_global_message_noise_model_is_its_control_at_evaluation(batch, option: dict) -> None:
    """No parameters and no initialization draws: at eval the arm is its control."""
    kwargs = {**UNIQUE_KWARGS, "precursor_gnn_type": "GINE_GSA"}
    control, arm = _build_model(batch, **kwargs), _build_model(batch, **kwargs, **option)
    control_state, arm_state = control.state_dict(), arm.state_dict()

    assert list(arm_state) == list(control_state)
    assert all(th.equal(arm_state[key], control_state[key]) for key in control_state)
    assert th.equal(
        _forward(arm.eval(), batch)["pred_logprobs"],
        _forward(control.eval(), batch)["pred_logprobs"],
    )
    th.manual_seed(5)
    trained_arm = _forward(arm.train(), batch)["pred_logprobs"]
    th.manual_seed(5)
    trained_control = _forward(control.train(), batch)["pred_logprobs"]
    assert trained_arm.shape != trained_control.shape or not th.allclose(
        trained_arm, trained_control
    )


@pytest.mark.parametrize("option", GLOBAL_MESSAGE_OPTIONS)
def test_global_message_noise_requires_global_attention(batch, option: dict) -> None:
    with pytest.raises(ValueError, match="require precursor_gnn_type GINE_GSA"):
        _build_model(batch, precursor_gnn_type="GINE", **UNIQUE_KWARGS, **option)


# 7. Structural noise on the shared encoder's inputs (arms am1 and de1).

ENCODER_NOISE_OPTIONS = [{"encoder_atom_mask": 0.3}, {"encoder_bond_dropout": 0.3}]


@pytest.mark.parametrize("option", ENCODER_NOISE_OPTIONS)
def test_encoder_noise_reaches_only_the_shared_encoder(batch, option: dict) -> None:
    model = _build_model(
        batch,
        precursor_gnn_type="GINE_GSA",
        separate_proposal_encoder=True,
        proposal_selection_only=True,
        use_base_probability_feature=False,
        **UNIQUE_KWARGS,
        **option,
    )
    shared, proposal = model.precursor_encoder, model.proposal_encoder

    assert shared.atom_mask == option.get("encoder_atom_mask", 0.0)
    assert shared.gnn.bond_dropout == option.get("encoder_bond_dropout", 0.0)
    assert proposal.atom_mask == 0.0
    assert proposal.gnn.bond_dropout == 0.0
    assert "atom_mask_embedding" not in proposal.state_dict()


@pytest.mark.parametrize("option", ENCODER_NOISE_OPTIONS)
def test_encoder_noise_model_is_its_control_at_evaluation(batch, option: dict) -> None:
    """Every control parameter initializes as in the control; the atom mask adds one vector."""
    kwargs = {**UNIQUE_KWARGS, "precursor_gnn_type": "GINE_GSA"}
    control, arm = _build_model(batch, **kwargs), _build_model(batch, **kwargs, **option)
    control_state, arm_state = control.state_dict(), arm.state_dict()

    added = [key for key in arm_state if key not in control_state]
    expected_added = (
        ["precursor_encoder.atom_mask_embedding"] if "encoder_atom_mask" in option else []
    )
    assert added == expected_added
    assert [key for key in arm_state if key in control_state] == list(control_state)
    assert all(th.equal(arm_state[key], control_state[key]) for key in control_state)
    assert th.equal(
        _forward(arm.eval(), batch)["pred_logprobs"],
        _forward(control.eval(), batch)["pred_logprobs"],
    )
    th.manual_seed(5)
    trained_arm = _forward(arm.train(), batch)["pred_logprobs"]
    th.manual_seed(5)
    trained_control = _forward(control.train(), batch)["pred_logprobs"]
    assert trained_arm.shape != trained_control.shape or not th.allclose(
        trained_arm, trained_control
    )


def test_encoder_bond_dropout_requires_global_attention(batch) -> None:
    with pytest.raises(ValueError, match="encoder_bond_dropout requires precursor_gnn_type"):
        _build_model(batch, precursor_gnn_type="GINE", **UNIQUE_KWARGS, encoder_bond_dropout=0.3)


def test_encoder_atom_mask_works_without_global_attention(batch) -> None:
    """The mask lives in the GNN wrapper, so a plain GINE encoder gets it too."""
    kwargs = {**UNIQUE_KWARGS, "precursor_gnn_type": "GINE"}
    control = _build_model(batch, **kwargs)
    arm = _build_model(batch, **kwargs, encoder_atom_mask=0.3)

    assert isinstance(arm.precursor_encoder.gnn, GINE)
    assert th.equal(
        _forward(arm.eval(), batch)["pred_logprobs"],
        _forward(control.eval(), batch)["pred_logprobs"],
    )


# 8. The allocator's set: substituting its message (arm sm1) and dropping its keys (arm sd1).

# (head and layer keyword, MassterModel keyword, config key)
SET_OPTIONS = [
    pytest.param(
        "set_message_substitution",
        "intensity_set_message_substitution",
        "masster_set_message_substitution",
        id="sm1",
    ),
    pytest.param(
        "set_key_dropout", "intensity_set_key_dropout", "masster_set_key_dropout", id="sd1"
    ),
]


def _set_messages() -> tuple[th.Tensor, th.Tensor]:
    """Two candidate sets: four real candidates, and two real candidates padded to four."""
    th.manual_seed(11)
    updates = th.randn(2, 4, 3)
    padding = th.tensor([[False, False, False, False], [False, False, True, True]])
    return updates, padding


def test_set_message_substitution_swaps_in_each_sets_own_detached_mean() -> None:
    """Each real candidate independently keeps its message or takes its own set's mean."""
    updates, padding = _set_messages()
    updates.requires_grad_(True)
    set_means = (updates[0].mean(dim=0), updates[1, :2].mean(dim=0))
    patterns = set()

    for seed in range(24):
        th.manual_seed(seed)
        substituted = _substitute_set_message(updates, padding, 0.5)
        (gradient,) = th.autograd.grad(substituted.sum(), updates)
        replaced = []
        for molecule in range(2):
            for candidate in range(4):
                row = substituted[molecule, candidate]
                if padding[molecule, candidate]:
                    assert th.equal(row, updates[molecule, candidate])
                    continue
                kept = th.equal(row, updates[molecule, candidate])
                assert kept or th.allclose(row, set_means[molecule], atol=1e-6)
                # The mean is detached, so a replaced candidate passes no gradient back.
                assert th.equal(gradient[molecule, candidate], th.full((3,), float(kept)))
                replaced.append(not kept)
        patterns.add(tuple(replaced))

    assert len(patterns) > 2
    assert all(any(pattern[slot] for pattern in patterns) for slot in range(6))
    assert all(any(not pattern[slot] for pattern in patterns) for slot in range(6))


def test_set_key_dropout_hides_real_candidates_and_keeps_every_set_nonempty() -> None:
    """Padding stays hidden, each real candidate is drawn alone, and no set loses every key."""
    padding = th.tensor(
        [
            [False, False, False, False],
            [False, False, True, True],
            [False, True, True, True],
        ]
    )
    real = ~padding
    emptied_rows = 0
    patterns = set()

    for seed in range(48):
        th.manual_seed(seed)
        drawn = real & (th.rand(padding.shape) < 0.6)
        th.manual_seed(seed)
        key_mask = _drop_set_keys(padding, 0.6)

        assert key_mask[padding].all()
        for row in range(3):
            if drawn[row][real[row]].all():
                # Every real key was drawn: the first stays visible, the rest stay hidden.
                emptied_rows += 1
                assert not key_mask[row, 0]
                assert key_mask[row][real[row]][1:].all()
            else:
                assert th.equal(key_mask[row], padding[row] | drawn[row])
        patterns.add(tuple(key_mask[real].tolist()))

    assert emptied_rows > 0
    # The one-candidate set is never hidden; every other real slot is hidden in some draws.
    assert all(not pattern[6] for pattern in patterns)
    assert all(any(pattern[slot] for pattern in patterns) for slot in range(6))
    assert all(any(not pattern[slot] for pattern in patterns) for slot in range(6))


def test_set_key_dropout_hides_candidates_from_the_set_summary() -> None:
    """A hidden candidate cannot move the inducing tokens that summarize its set."""
    th.manual_seed(2)
    layer = _InducedCandidateAttentionLayer(8, 2, 16, 0.0, set_key_dropout=0.5).train()
    tokens, inducing = th.randn(1, 6, 8), th.randn(1, 3, 8)
    padding = th.zeros(1, 6, dtype=th.bool)

    def key_mask(seed: int) -> th.Tensor:
        th.manual_seed(seed)
        return _drop_set_keys(padding, 0.5)[0]

    seed = next(seed for seed in range(100) if 0 < int(key_mask(seed).sum()) < 6)
    hidden = key_mask(seed)

    def summary(candidate_tokens: th.Tensor) -> th.Tensor:
        th.manual_seed(seed)
        return layer(candidate_tokens, inducing, padding)[1]

    reference = summary(tokens)
    for index in range(6):
        perturbed = tokens.clone()
        perturbed[0, index] += 1.0
        assert th.allclose(summary(perturbed), reference) == bool(hidden[index])


@pytest.mark.parametrize(("head_key", "model_key", "config_key"), SET_OPTIONS)
def test_set_option_is_training_only_and_adds_no_parameters(
    head_key: str, model_key: str, config_key: str
) -> None:
    """Read on the ion/H logits: the residual heads start at zero, whatever the token holds."""
    args, kwargs = _head_inputs()
    plain, arm = _head(), _head(**{head_key: 0.5})
    assert list(arm.state_dict()) == list(plain.state_dict())
    arm.load_state_dict(plain.state_dict())

    assert th.equal(arm.eval()(*args, **kwargs)[2], plain.eval()(*args, **kwargs)[2])
    th.manual_seed(4)
    assert not th.allclose(arm.train()(*args, **kwargs)[2], plain.train()(*args, **kwargs)[2])


@pytest.mark.parametrize(("head_key", "model_key", "config_key"), SET_OPTIONS)
def test_set_option_draws_nothing_when_off_or_evaluating(
    head_key: str, model_key: str, config_key: str
) -> None:
    args, kwargs = _head_inputs()
    off_in_training = _head(**{head_key: 0.0}).train()
    on_in_evaluation = _head(**{head_key: 0.5}).eval()
    for head in (off_in_training, on_in_evaluation):
        rng_before = th.get_rng_state()
        head(*args, **kwargs)
        assert th.equal(th.get_rng_state(), rng_before)


@pytest.mark.parametrize("probability", [-0.1, 1.0])
@pytest.mark.parametrize(("head_key", "model_key", "config_key"), SET_OPTIONS)
def test_set_option_rejects_invalid_probabilities(
    head_key: str, model_key: str, config_key: str, probability: float
) -> None:
    with pytest.raises(ValueError, match=f"{head_key} must be in"):
        _head(**{head_key: probability})


@pytest.mark.parametrize(("head_key", "model_key", "config_key"), SET_OPTIONS)
def test_set_option_requires_induced_attention(
    head_key: str, model_key: str, config_key: str
) -> None:
    with pytest.raises(ValueError, match=f"{head_key} requires attention_type='induced'"):
        _head(attention_type="full", **{head_key: 0.3})


@pytest.mark.parametrize(("head_key", "model_key", "config_key"), SET_OPTIONS)
def test_set_option_model_is_its_control_at_evaluation(
    batch, head_key: str, model_key: str, config_key: str
) -> None:
    """No parameters and no initialization draws: at eval the arm is its control."""
    control = _build_model(batch, **UNIQUE_KWARGS)
    arm = _build_model(batch, **UNIQUE_KWARGS, **{model_key: 0.5})
    control_state, arm_state = control.state_dict(), arm.state_dict()

    assert [getattr(layer, head_key) for layer in arm.intensity_head.layers] == [0.5, 0.5]
    assert list(arm_state) == list(control_state)
    assert all(th.equal(arm_state[key], control_state[key]) for key in control_state)
    assert th.equal(
        _forward(arm.eval(), batch)["pred_logprobs"],
        _forward(control.eval(), batch)["pred_logprobs"],
    )
    th.manual_seed(5)
    trained_arm = _forward(arm.train(), batch)["pred_logprobs"]
    th.manual_seed(5)
    trained_control = _forward(control.train(), batch)["pred_logprobs"]
    assert trained_arm.shape != trained_control.shape or not th.allclose(
        trained_arm, trained_control
    )


@pytest.mark.parametrize(("head_key", "model_key", "config_key"), SET_OPTIONS)
def test_pl_passes_the_set_option_to_every_allocator_layer(
    head_key: str, model_key: str, config_key: str
) -> None:
    config = runner.load_config("config", None, None, model_type="masster")
    # The small MASSter build of tests/test_masster_backend_compatibility.py, two allocator layers.
    config.update(
        model_type="masster",
        compile=False,
        sparse_cosine_similarity=True,
        oos_tolerance_multiple=1,
        gaussian_renormalize=True,
        mol_hidden_size=16,
        mol_num_layers=1,
        masster_context_dim=8,
        masster_metadata_embedding_dim=4,
        masster_head_hidden_dim=16,
        masster_intensity_transformer_dim=16,
        masster_intensity_transformer_num_layers=2,
        masster_intensity_transformer_feedforward_dim=32,
        masster_intensity_transformer_num_inducing_points=4,
        masster_unique_fragment_mode=True,
        masster_selected_mask_limit=2,
        masster_unique_fragment_target=2,
        masster_ranked_mask_budget=2,
        masster_max_h_transfer=1,
    )
    config[config_key] = 0.4

    layers = MassterPL(**config).model.intensity_head.layers

    assert [getattr(layer, head_key) for layer in layers] == [0.4, 0.4]
